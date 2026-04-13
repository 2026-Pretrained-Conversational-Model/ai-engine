"""
app/services/session/session_creator.py
---------------------------------------
역할: 기본값(빈 메모리 + 설정 기반 정책)을 가진 새로운 Session 객체 생성
      순수 함수로 동작하며 I/O는 수행하지 않음

TODO:
    [ ] Node.js에서 전달받는 초기 힌트(예: 사용자 페르소나) 선택적으로 반영
"""
from app.schemas.session import Session, SessionMeta
from app.schemas.runtime import CleanupPolicy
from app.core.config import settings


def create_session(session_id: str) -> Session:
    return Session(
        session_meta=SessionMeta(session_id=session_id),
        cleanup_policy=CleanupPolicy(
            expire_after_minutes=settings.SESSION_IDLE_TIMEOUT_MIN,
            delete_local_file_on_expire=settings.DELETE_FILE_ON_EXPIRE,
        ),
    )
