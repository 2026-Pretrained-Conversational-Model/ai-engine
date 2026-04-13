"""
app/services/session/memory_monitor.py
--------------------------------------
역할: 현재 Session 데이터가 차지하는 메모리 크기를 측정하고
      SESSION_MAX_BYTES 초과 여부를 판단

      전략: JSON으로 직렬화한 뒤 UTF-8 바이트 길이를 측정
            → 가볍지만 비교적 정확하며,
              recent_messages + summary + chunk 텍스트를 모두 포함

TODO:
    [ ] FAISS 벡터 크기도 측정에 포함 (현재는 미포함)
    [ ] 80% 도달 시 소프트 경고 추가 (프론트에서 사용자 안내 가능하도록)
"""
from app.schemas.session import Session
from app.core.config import settings


def measure_bytes(session: Session) -> int:
    return len(session.model_dump_json().encode("utf-8"))


def is_over_limit(session: Session) -> bool:
    return measure_bytes(session) >= settings.SESSION_MAX_BYTES
