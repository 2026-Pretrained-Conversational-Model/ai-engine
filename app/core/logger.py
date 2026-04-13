"""
app/core/logger.py
------------------
역할: 중앙 집중식 로거 생성 팩토리
      모든 모듈은 `get_logger(__name__)`를 통해 로거를 사용

TODO:
    [ ] session_id 바인딩을 지원하는 structlog로 전환
    [ ] 운영 환경에서 CloudWatch로 로그 전송
"""
import logging
from app.core.config import settings


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        logger.addHandler(h)
        logger.setLevel(settings.LOG_LEVEL)
    return logger
