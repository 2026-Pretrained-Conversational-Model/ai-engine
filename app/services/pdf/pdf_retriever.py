"""
app/services/pdf/pdf_retriever.py
---------------------------------
역할: (이미 해석된) 사용자 쿼리를 기반으로,
      현재 세션의 활성 PDF에서 상위 k개의 청크를 반환

      벡터는 세션별 FAISS 저장소에 있는 것을 재사용하며,
      절대 다시 계산하지 않음 (요구사항 #13)

변경사항 (baseline 반영):
    [수정] EmbeddingSingleton.embed_query() 사용
          기존 model.encode([query]) → embed_query()로 통일
    [수정] get_context() 추가
          Answer Generator에 넘겨줄 출처 포함 context 문자열 반환
          형식: [출처 N: 파일명, p.페이지]\n텍스트\n\n---\n\n...      

TODO:
    [ ] 하이브리드 검색 적용 (BM25 + Dense Retrieval)
    [ ] Cross-encoder를 활용한 top-k 재정렬(Re-ranking)
"""
from typing import List
from app.schemas.session import Session
from app.schemas.pdf import PdfChunk
from app.services.embedding.embedding_singleton import EmbeddingSingleton
from app.services.embedding.faiss_store import FaissStore
from app.core.config import settings


def retrieve_relevant(session: Session, query: str) -> List[PdfChunk]:
    if not session.pdf_state.active_pdf:
        return []

    # [수정] EmbeddingSingleton.embed_query() 사용
    # 기존: model = EmbeddingSingleton.get()
    #       qvec = model.encode([query], normalize_embeddings=True)
    # 변경: EmbeddingSingleton.embed_query(query)
    #       → normalize_embeddings=True 강제 적용으로 통일    

    qvec = EmbeddingSingleton.embed_query(query).reshape(1, -1)
    chunk_ids = FaissStore.instance().search(
        session.session_meta.session_id, qvec, top_k=settings.RAG_TOP_K
    )
    by_id = {c.chunk_id: c for c in session.pdf_state.pdf_index.chunks}
    return [by_id[cid] for cid in chunk_ids if cid in by_id]

# get_context() 추가
# Answer Generator(prompt_builder)에 넘겨줄 context 문자열 반환
# 변경: 출처(파일명, 페이지) 포함 포맷 문자열로 변환해서 반환
#       → augmentation.get_pdf_context_string()에서 호출
def get_context(session: Session, query: str) -> str:
    """
    [기능] Answer Generator에 넘겨줄 context 문자열 반환
    - 전체 파이프라인에서 Retriever의 최종 출력물
    - 출처(파일명, 페이지) 포함 → 답변 근거 추적 가능
    - 관련 chunk 없으면 빈 문자열 반환

    Args:
        session: 현재 세션 객체
        query  : multiturn_resolver가 해석한 쿼리

    Returns:
        str: 출처 포함 context 문자열
             형식: [출처 N: 파일명, p.페이지]\n텍스트\n\n---\n\n...

    사용 예시:
        context = get_context(session, resolved_query)
        # → augmentation.get_pdf_context_string() → prompt_builder.build()로 전달
    """
    chunks = retrieve_relevant(session, query)
    if not chunks:
        return ""
    
    parts = []
    for i, chunk in enumerate(chunks, 1):
        source = getattr(chunk, "source", "unknown")
        page   = getattr(chunk, "page", "?")
        parts.append(f"[출처 {i}: {source}, p.{page}]\n{chunk.text}")

    return "\n\n---\n\n".join(parts)