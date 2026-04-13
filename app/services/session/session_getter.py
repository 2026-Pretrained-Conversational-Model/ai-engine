"""
app/services/session/session_getter.py
--------------------------------------
역할: session_id를 기반으로 Session을 조회하고,
      없으면 새로 생성

      항상 `last_accessed_at` 값을 갱신하여
      idle-timeout 로직이 정상 동작하도록 함

TODO:
    [ ] Node.js에서 발급한 패턴과 일치하지 않는 session_id는 거부
"""
from datetime import datetime
from app.schemas.session import Session
from app.services.session.session_manager import SessionManager
from app.services.session.session_creator import create_session


async def get_or_create(session_id: str) -> Session:
    mgr = SessionManager.instance()
    session = await mgr.get(session_id)
    if session is None:
        session = create_session(session_id)
        await mgr.save(session)
    session.session_meta.last_accessed_at = datetime.utcnow()
    return session
