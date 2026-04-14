"""
app/services/embedding/faiss_store.py
-------------------------------------
역할: 세션별 벡터 인덱스 레지스트리.

기본 구현은 FAISS를 사용하되, 개발 환경에서 faiss-cpu가 설치되지 않은 경우에도
baseline이 완전히 죽지 않도록 numpy 기반 fallback 인덱스를 함께 제공한다.

변경사항 (baseline 반영):
    [수정] md5 해시 기반 캐시 추가
          PDF 파일 내용 + chunk_size + chunk_overlap 조합으로 캐시 키 생성
          → 동일 PDF 재임베딩 방지
          → chunk_size 등 파라미터 변경 시 자동으로 새 캐시 생성
    [수정] _save_cache(), _load_cache() 추가
          .faiss + .pkl 파일로 디스크 저장/로드
    [수정] add()에 file_hash 인자 추가
          캐시 저장/로드와 연결

TODO:
    [ ] 청크 수가 5k를 초과하는 세션에 대해 IVF/HNSW 인덱스로 전환
    [ ] 백그라운드 메모리 사용량 측정 추가
"""
from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
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

        # [수정] md5 캐시 디렉토리 초기화
        # chunk_size 변경 시 이 디렉토리 삭제 후 재실행 필요
        self._cache_dir = Path(getattr(settings, "FAISS_CACHE_DIR", "./faiss_cache"))
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def instance(cls) -> "FaissStore":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _new_index(self):
        if _FAISS_AVAILABLE:
            return faiss.IndexFlatIP(settings.EMBEDDING_DIM)
        return _NumpyIndex()

    # md5 해시 기반 캐시 키 생성
    # PDF 내용 + chunk_size + chunk_overlap 조합
    # → 파라미터 변경 시 해시값이 달라져 자동으로 새 캐시 생성
    @staticmethod
    def get_file_hash(pdf_path: str) -> str:
        """
        [기능] PDF 파일 내용의 md5 해시 반환
        - chunk_size, chunk_overlap 포함 → 파라미터 변경 시 캐시 자동 무효화
        """
        with open(pdf_path, "rb") as f:
            content_hash = hashlib.md5(f.read()).hexdigest()
        return f"{content_hash}_{settings.PDF_CHUNK_SIZE}_{settings.PDF_CHUNK_OVERLAP}"

    def _index_path(self, file_hash: str) -> Path:
        return self._cache_dir / f"{file_hash}.faiss"

    def _chunk_ids_path(self, file_hash: str) -> Path:
        return self._cache_dir / f"{file_hash}.pkl"

    def _is_cached(self, file_hash: str) -> bool:
        return self._index_path(file_hash).exists() and self._chunk_ids_path(file_hash).exists()

    # [수정] 캐시 저장
    def _save_cache(self, file_hash: str, index, chunk_ids: List[str]) -> None:
        if _FAISS_AVAILABLE:
            faiss.write_index(index, str(self._index_path(file_hash)))
        with open(self._chunk_ids_path(file_hash), "wb") as f:
            pickle.dump(chunk_ids, f)
        logger.info("FAISS cache saved: %s", file_hash[:8])

    # [수정] 캐시 로드
    def _load_cache(self, session_id: str, file_hash: str) -> bool:
        if not self._is_cached(file_hash):
            return False
        if _FAISS_AVAILABLE:
            index = faiss.read_index(str(self._index_path(file_hash)))
        else:
            index = _NumpyIndex()
        with open(self._chunk_ids_path(file_hash), "rb") as f:
            chunk_ids = pickle.load(f)
        self._indexes[session_id] = (index, chunk_ids)
        logger.info("FAISS cache loaded: %s", file_hash[:8])
        return True    

    def reset(self, session_id: str) -> None:
        """기존 세션 인덱스를 비우고 새 문서 인덱싱을 준비한다."""
        self._indexes[session_id] = (self._new_index(), [])

    # file_hash 인자 추가
    # 변경: add(session_id, vectors, chunk_ids, file_hash="")
    #       → file_hash 있으면 캐시 로드 시도 → 없으면 새로 빌드 후 저장
    def add(
        self,
        session_id: str,
        vectors: np.ndarray,
        chunk_ids: List[str],
        file_hash: str = "",  # [수정] 캐시 연결용 file_hash 추가
    ) -> None:
        if session_id not in self._indexes:
            # [수정] 캐시 있으면 로드 후 리턴 (재임베딩 불필요)
            if file_hash and self._load_cache(session_id, file_hash):
                return
            self.reset(session_id)

        index, ids = self._indexes[session_id]
        index.add(np.asarray(vectors, dtype="float32"))
        ids.extend(chunk_ids)

        # [수정] 캐시 저장
        if file_hash:
            self._save_cache(file_hash, index, ids)

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
