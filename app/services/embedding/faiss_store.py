"""
app/services/embedding/faiss_store.py
-------------------------------------
역할: 세션별 FAISS 인덱스 레지스트리
      {session_id: (index, [chunk_ids])} 형태로 관리

      인덱스는 첫 PDF 업로드 시 한 번 생성되고,
      이후 해당 세션의 모든 검색 요청에서 재사용됨
      → 요청마다 재생성하지 않음

      세션이 삭제될 때 함께 제거됨

TODO:
    [ ] 청크 수가 5k를 초과하는 세션에 대해 IVF 인덱스로 전환
    [ ] 백그라운드 메모리 사용량 측정 추가
"""
from typing import Dict, List, Tuple
import numpy as np
import faiss

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)


class FaissStore:
    _instance: "FaissStore | None" = None

    def __init__(self) -> None:
        self._indexes: Dict[str, Tuple[faiss.Index, List[str]]] = {}

    @classmethod
    def instance(cls) -> "FaissStore":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def add(self, session_id: str, vectors: np.ndarray, chunk_ids: List[str]) -> None:
        if session_id not in self._indexes:
            index = faiss.IndexFlatIP(settings.EMBEDDING_DIM)
            self._indexes[session_id] = (index, [])
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
        logger.info("Dropped FAISS index for session=%s", session_id)
