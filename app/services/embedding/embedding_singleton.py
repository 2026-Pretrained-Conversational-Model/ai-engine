"""
app/services/embedding/embedding_singleton.py
---------------------------------------------
역할: 
    - 임베딩 모델을 프로세스당 단 한 번만 로딩하고,
      모든 호출자가 동일한 인스턴스를 사용하도록 제공
    
    - 임베딩 벡터나 FAISS를 여러 번 불러오지 않도록 설정

TODO:
    [ ] 시작 시 블로킹을 방지하기 위해 비동기 워밍업으로 전환
    [ ] EMBEDDING_DEVICE=cpu이고 OOM 발생 시 더 작은 모델로 fallback 추가
"""
from typing import Optional
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
