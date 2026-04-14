"""
app/services/pdf/pdf_indexer.py
-------------------------------
역할: 청크 텍스트를 임베딩하고,
      세션별 FAISS 인덱스에 등록

      임베딩은 여기서 한 번만 계산되고,
      이후 동일 세션의 모든 검색 요청에서 재사용됨.

변경사항 (baseline 반영):
    [수정] model.encode() → EmbeddingSingleton.embed_documents() 로 교체
          normalize_embeddings=True + 진행바 포함된 배치 임베딩
    [수정] FaissStore.add()에 file_hash 인자 전달
          faiss_store 내부 캐시 저장/로드와 연결

추가 baseline 기능:
- file_hash 기반 embedding cache 재사용
- cache hit 시에는 임베딩 재계산 없이 FAISS에만 재등록

TODO:
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
        texts = [c.text for c in chunks]

        # EmbeddingSingleton.embed_documents() 사용
        # 변경: EmbeddingSingleton.embed_documents(texts)
        #       → normalize_embeddings=True + 진행바 포함 배치 임베딩으로 통일
        vectors = EmbeddingSingleton.embed_documents(texts)

        if file_hash and allow_cache_write:
            cache.set_embeddings(
                file_hash,
                np.asarray(vectors, dtype="float32"),
                [c.chunk_id for c in chunks],
            )

    # FaissStore.add()에 file_hash 전달
    # 변경: file_hash 추가 → faiss_store 내부 캐시 저장/로드 처리
    FaissStore.instance().add(
        session_id,
        np.asarray(vectors, dtype="float32"),
        [c.chunk_id for c in chunks],
        file_hash=file_hash,
    )

    for c in chunks:
        c.embedding_ref = f"vec://{session_id}/{c.chunk_id}"
