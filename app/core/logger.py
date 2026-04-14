"""
app/core/logger.py
------------------
역할: 중앙 집중식 로거 생성 팩토리
      모든 모듈은 `get_logger(__name__)`를 통해 로거를 사용

TODO: 김예슬
    [완] session_id 바인딩을 지원하는 structlog로 전환
    [미완] 운영 환경에서 CloudWatch로 로그 전송
"""
import logging
import json
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


# ===============================
# 🔥 추가: 구조화 로그 헬퍼
# ===============================
def _safe_json(data):
    try:
        return json.dumps(data, ensure_ascii=False, default=str)
    except Exception:
        return str(data)


def log_stage_start(logger, stage: str, **kwargs):
    logger.info(f"[START] {stage} | {_safe_json(kwargs)}")


def log_stage_end(logger, stage: str, **kwargs):
    logger.info(f"[END] {stage} | {_safe_json(kwargs)}")


def log_state(logger, title: str, **kwargs):
    logger.info(f"[STATE] {title} | {_safe_json(kwargs)}")


def log_memory(logger, session, label: str):
    try:
        memory = getattr(session, "memory_state", None)
        pdf = getattr(session, "pdf_state", None)
        runtime = getattr(session, "runtime_state", None)

        payload = {
            "label": label,
            "memory": memory.model_dump() if hasattr(memory, "model_dump") else str(memory),
            "pdf": pdf.model_dump() if hasattr(pdf, "model_dump") else str(pdf),
            "runtime": runtime.model_dump() if hasattr(runtime, "model_dump") else str(runtime),
        }

        logger.info(f"[MEMORY] {_safe_json(payload)}")

    except Exception as e:
        logger.warning(f"memory logging failed: {e}")