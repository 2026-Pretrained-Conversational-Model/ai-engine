"""
app/services/pdf/pdf_indexer.py
-------------------------------
역할: 청크 텍스트를 임베딩하고,
      세션별 FAISS 인덱스에 등록

      임베딩은 여기서 한 번만 계산되고,
      이후 동일 세션의 모든 검색 요청에서 재사용됨.

추가 baseline 기능:
- file_hash 기반 embedding cache 재사용
- cache hit 시에는 임베딩 재계산 없이 FAISS에만 재등록

TODO:
    [ ] 32개 단위로 배치 임베딩 처리
    [ ] 세션 재개를 위해 필요 시 FAISS를 디스크에 저장
    [ ] 파일 일부만 바뀐 경우 delta index만 갱신
"""
from typing import List

import numpy as np

from app.schemas.pdf import PdfChunk
from app.services.embedding.embedding_singleton import EmbeddingSingleton
from app.services.embedding.faiss_store import FaissStore
from app.services.pdf.document_cache import DocumentCache


def index_chunks(
    session_id: str,
    chunks: List[PdfChunk],
    file_hash: str = "",
    allow_cache_write: bool = True,
) -> None:
    if not chunks:
        return

    cache = DocumentCache.instance()
    vectors: np.ndarray

    cache_entry = cache.get_embeddings(file_hash) if file_hash else None
    if cache_entry is not None:
        vectors = cache_entry.vectors
    else:
        model = EmbeddingSingleton.get()
        texts = [c.text for c in chunks]
        vectors = model.encode(texts, normalize_embeddings=True)
        if file_hash and allow_cache_write:
            cache.set_embeddings(file_hash, np.asarray(vectors, dtype="float32"), [c.chunk_id for c in chunks])

    FaissStore.instance().add(session_id, np.asarray(vectors, dtype="float32"), [c.chunk_id for c in chunks])
    for c in chunks:
        c.embedding_ref = f"vec://{session_id}/{c.chunk_id}"
