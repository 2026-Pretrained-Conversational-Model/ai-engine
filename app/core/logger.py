import logging
import json
import time
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


def _safe_json(data):
    try:
        return json.dumps(data, ensure_ascii=False, default=str)
    except Exception:
        return str(data)


def log_state(logger, label: str, **kwargs):
    payload = dict(kwargs)
    payload["ts"] = time.time()
    logger.info(f"[STATE] {label} | {_safe_json(payload)}")


def log_stage_start(logger, stage: str, **kwargs):
    start_time = time.perf_counter()
    logger.info(f"[START] {stage} | {_safe_json(kwargs)}")
    return start_time


def log_stage_end(logger, stage: str, start_time: float | None = None, **kwargs):
    payload = dict(kwargs)

    if start_time is not None:
        payload["latency_ms"] = round((time.perf_counter() - start_time) * 1000, 2)

    logger.info(f"[END] {stage} | {_safe_json(payload)}")


def log_memory(logger, session, label: str):
    try:
        conversation = getattr(session, "conversation", None)
        pdf = getattr(session, "pdf_state", None)
        runtime = getattr(session, "runtime_state", None)

        summary = getattr(conversation, "summary", None) if conversation else None
        recent_messages = getattr(conversation, "recent_messages", []) if conversation else []

        payload = {
            "label": label,
            "summary": summary.model_dump() if hasattr(summary, "model_dump") else str(summary),
            "recent_messages_count": len(recent_messages),
            "pdf": pdf.model_dump() if hasattr(pdf, "model_dump") else str(pdf),
            "runtime": runtime.model_dump() if hasattr(runtime, "model_dump") else str(runtime),
        }

        logger.info(f"[MEMORY] {_safe_json(payload)}")

    except Exception as e:
        logger.warning(f"memory logging failed: {e}")