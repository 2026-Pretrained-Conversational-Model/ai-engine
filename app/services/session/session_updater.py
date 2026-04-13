"""
app/services/session/session_updater.py
---------------------------------------
역할: 변경된(Session이 수정된) 데이터를 저장소에 반영
      향후 dirty-checking, write-through 메트릭 등을
      중앙에서 처리할 수 있도록 한 곳으로 모음
"""
from app.schemas.session import Session
from app.services.session.session_manager import SessionManager


async def save_session(session: Session) -> None:
    await SessionManager.instance().save(session)
