"""
app/services/embedding/faiss_store.py
-------------------------------------
역할: 세션별 벡터 인덱스 레지스트리.

기본 구현은 FAISS를 사용하되, 개발 환경에서 faiss-cpu가 설치되지 않은 경우에도
baseline이 완전히 죽지 않도록 numpy 기반 fallback 인덱스를 함께 제공한다.

TODO:
    [ ] 청크 수가 5k를 초과하는 세션에 대해 IVF/HNSW 인덱스로 전환
    [ ] 백그라운드 메모리 사용량 측정 추가
"""
from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)

try:  # pragma: no cover - import availability depends on runtime image
    import faiss  # type: ignore
    _FAISS_AVAILABLE = True
except Exception:  # pragma: no cover
    faiss = None
    _FAISS_AVAILABLE = False


class _NumpyIndex:
    """FAISS가 없는 개발 환경용 cosine/IP 검색 대체 구현."""

    def __init__(self) -> None:
        self.vectors = np.empty((0, settings.EMBEDDING_DIM), dtype="float32")

    @property
    def ntotal(self) -> int:
        return int(self.vectors.shape[0])

    def add(self, vectors: np.ndarray) -> None:
        vectors = np.asarray(vectors, dtype="float32")
        if self.vectors.size == 0:
            self.vectors = vectors
        else:
            self.vectors = np.vstack([self.vectors, vectors])

    def search(self, qvec: np.ndarray, top_k: int):
        qvec = np.asarray(qvec, dtype="float32")
        scores = np.matmul(self.vectors, qvec[0])
        order = np.argsort(-scores)[:top_k]
        return scores[order][None, :], order[None, :]


class FaissStore:
    _instance: "FaissStore | None" = None

    def __init__(self) -> None:
        self._indexes: Dict[str, Tuple[object, List[str]]] = {}
        if not _FAISS_AVAILABLE:
            logger.warning("faiss is not available; using numpy fallback index")

    @classmethod
    def instance(cls) -> "FaissStore":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _new_index(self):
        if _FAISS_AVAILABLE:
            return faiss.IndexFlatIP(settings.EMBEDDING_DIM)
        return _NumpyIndex()

    def reset(self, session_id: str) -> None:
        """기존 세션 인덱스를 비우고 새 문서 인덱싱을 준비한다."""
        self._indexes[session_id] = (self._new_index(), [])

    def add(self, session_id: str, vectors: np.ndarray, chunk_ids: List[str]) -> None:
        if session_id not in self._indexes:
            self.reset(session_id)
        index, ids = self._indexes[session_id]
        index.add(np.asarray(vectors, dtype="float32"))
        ids.extend(chunk_ids)

    def search(self, session_id: str, qvec: np.ndarray, top_k: int) -> List[str]:
        if session_id not in self._indexes:
            return []
        index, ids = self._indexes[session_id]
        if index.ntotal == 0:
            return []
        _, I = index.search(np.asarray(qvec, dtype="float32"), top_k)
        return [ids[i] for i in I[0] if 0 <= i < len(ids)]

    def drop(self, session_id: str) -> None:
        self._indexes.pop(session_id, None)
        logger.info("Dropped vector index for session=%s", session_id)
