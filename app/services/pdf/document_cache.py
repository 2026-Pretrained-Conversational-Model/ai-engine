"""
app/services/pdf/document_cache.py
---------------------------------
역할: PDF 전처리 결과 캐시.

현재 baseline에서는 파일 해시(file_hash)를 key로 사용하여
다음 결과를 재사용한다.
- parser 결과(page texts)
- chunk 결과
- document summary
- embedding vectors

이 캐시는 '같은 PDF 재업로드' 시 중복 계산을 줄이는 목적이다.
세션 간 공유 캐시이므로, 세션 메모리(JSON)와는 별개로 유지한다.

TODO:
    [ ] 파일 버전/모델 버전(embedding model, parser version) 포함한 캐시 키 도입
    [ ] 디스크 기반 persistent cache로 확장
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np

from app.schemas.pdf import PdfChunk, DocSummary


@dataclass
class ParsedPdfCacheEntry:
    page_count: int
    pages: List[str]


@dataclass
class ChunkCacheEntry:
    chunks: List[PdfChunk]


@dataclass
class SummaryCacheEntry:
    summary: DocSummary


@dataclass
class EmbeddingCacheEntry:
    vectors: np.ndarray
    chunk_ids: List[str]


class DocumentCache:
    _instance: "DocumentCache | None" = None

    def __init__(self) -> None:
        self._parsed: Dict[str, ParsedPdfCacheEntry] = {}
        self._chunked: Dict[str, ChunkCacheEntry] = {}
        self._summary: Dict[str, SummaryCacheEntry] = {}
        self._embeddings: Dict[str, EmbeddingCacheEntry] = {}

    @classmethod
    def instance(cls) -> "DocumentCache":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def get_parsed(self, file_hash: str) -> Optional[ParsedPdfCacheEntry]:
        return self._parsed.get(file_hash)

    def set_parsed(self, file_hash: str, page_count: int, pages: List[str]) -> None:
        self._parsed[file_hash] = ParsedPdfCacheEntry(page_count=page_count, pages=list(pages))

    def get_chunks(self, file_hash: str) -> Optional[ChunkCacheEntry]:
        return self._chunked.get(file_hash)

    def set_chunks(self, file_hash: str, chunks: List[PdfChunk]) -> None:
        self._chunked[file_hash] = ChunkCacheEntry(chunks=[c.model_copy(deep=True) for c in chunks])

    def get_summary(self, file_hash: str) -> Optional[SummaryCacheEntry]:
        return self._summary.get(file_hash)

    def set_summary(self, file_hash: str, summary: DocSummary) -> None:
        self._summary[file_hash] = SummaryCacheEntry(summary=summary.model_copy(deep=True))

    def get_embeddings(self, file_hash: str) -> Optional[EmbeddingCacheEntry]:
        return self._embeddings.get(file_hash)

    def set_embeddings(self, file_hash: str, vectors: np.ndarray, chunk_ids: List[str]) -> None:
        self._embeddings[file_hash] = EmbeddingCacheEntry(vectors=vectors, chunk_ids=list(chunk_ids))
