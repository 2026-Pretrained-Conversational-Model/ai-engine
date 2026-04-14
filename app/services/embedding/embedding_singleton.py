"""
app/services/embedding/embedding_singleton.py
---------------------------------------------
역할: 
    - 임베딩 모델을 프로세스당 단 한 번만 로딩하고,
      모든 호출자가 동일한 인스턴스를 사용하도록 제공
    
    - 임베딩 벡터나 FAISS를 여러 번 불러오지 않도록 설정

변경사항 (baseline 반영):
    [수정] embed_documents() 추가
          배치 임베딩 + normalize_embeddings=True + 진행바
    [수정] embed_query() 추가
          단일 쿼리 임베딩 + normalize_embeddings=True
          → 문서 임베딩과 동일 기준으로 코사인 유사도 계산 가능

TODO:
    [ ] 시작 시 블로킹을 방지하기 위해 비동기 워밍업으로 전환
    [ ] EMBEDDING_DEVICE=cpu이고 OOM 발생 시 더 작은 모델로 fallback 추가
"""
from typing import List
import numpy as np
from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)


class EmbeddingSingleton:
    _model = None

    @classmethod
    def warmup(cls) -> None:
        if cls._model is not None:
            return
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model: %s", settings.EMBEDDING_MODEL_NAME)
        cls._model = SentenceTransformer(
            settings.EMBEDDING_MODEL_NAME,
            device=settings.EMBEDDING_DEVICE,
        )
        logger.info("Embedding model ready")

    @classmethod
    def get(cls):
        if cls._model is None:
            cls.warmup()
        return cls._model

    # embed_documents() 추가
    # 변경: EmbeddingSingleton.embed_documents()로 통일
    #       → normalize_embeddings=True 강제 적용
    #       → show_progress_bar=True로 진행 상황 확인 가능
    @classmethod
    def embed_documents(cls, texts: List[str]) -> np.ndarray:
        """
        [기능] 텍스트 리스트를 임베딩 행렬로 변환 (배치 처리)

        Args:
            texts: 임베딩할 텍스트 리스트 (chunk 텍스트들)

        Returns:
            np.ndarray: shape (len(texts), dim)의 float32 벡터 행렬
        """
        model = cls.get()
        vectors = model.encode(
            texts,
            normalize_embeddings=True,  # 코사인 유사도를 위한 정규화
            show_progress_bar=True,     # 진행바 표시
        )
        return np.array(vectors, dtype=np.float32)

    # embed_query() 추가
    # 변경: EmbeddingSingleton.embed_query()로 통일
    #       → 문서 임베딩과 동일한 normalize 기준 보장
    @classmethod
    def embed_query(cls, query: str) -> np.ndarray:
        """
        [기능] 단일 쿼리를 임베딩 벡터로 변환 (검색 시 사용)

        Args:
            query: 검색 쿼리 문자열

        Returns:
            np.ndarray: shape (dim,)의 float32 벡터
        """
        model = cls.get()
        vector = model.encode(
            [query],
            normalize_embeddings=True,  # 코사인 유사도를 위한 정규화
        )
        return np.array(vector[0], dtype=np.float32)