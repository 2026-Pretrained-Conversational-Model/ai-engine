"""
app/services/pdf/pdf_retriever.py
---------------------------------
역할: (이미 해석된) 사용자 쿼리를 기반으로,
      현재 세션의 활성 PDF에서 상위 k개의 청크를 반환

      벡터는 세션별 FAISS 저장소에 있는 것을 재사용하며,
      절대 다시 계산하지 않음 (요구사항 #13)

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
    model = EmbeddingSingleton.get()
    qvec = model.encode([query], normalize_embeddings=True)
    chunk_ids = FaissStore.instance().search(
        session.session_meta.session_id, qvec, top_k=settings.RAG_TOP_K
    )
    by_id = {c.chunk_id: c for c in session.pdf_state.pdf_index.chunks}
    return [by_id[cid] for cid in chunk_ids if cid in by_id]
