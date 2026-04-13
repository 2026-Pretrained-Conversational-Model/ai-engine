"""
app/services/pdf/pdf_indexer.py
-------------------------------
역할: 청크 텍스트를 임베딩하고,
      세션별 FAISS 인덱스에 등록

      임베딩은 여기서 한 번만 계산되고,
      이후 동일 세션의 모든 검색 요청에서 재사용됨

TODO:
    [ ] 32개 단위로 배치 임베딩 처리
    [ ] 세션 재개를 위해 필요 시 FAISS를 디스크에 저장
"""
from typing import List
from app.schemas.pdf import PdfChunk
from app.services.embedding.embedding_singleton import EmbeddingSingleton
from app.services.embedding.faiss_store import FaissStore


def index_chunks(session_id: str, chunks: List[PdfChunk]) -> None:
    if not chunks:
        return
    model = EmbeddingSingleton.get()
    texts = [c.text for c in chunks]
    vectors = model.encode(texts, normalize_embeddings=True)
    FaissStore.instance().add(session_id, vectors, [c.chunk_id for c in chunks])
    for c in chunks:
        c.embedding_ref = f"vec://{session_id}/{c.chunk_id}"
