"""
app/services/conversation/message_appender.py
---------------------------------------------
역할: 새로운 메시지를 recent_messages에 추가하고,
      설정된 윈도우 크기(RECENT_MESSAGES_WINDOW)에 맞게 잘라냄

      순수 데이터 변경 로직이며 I/O는 없음
"""
from app.schemas.session import Session
from app.schemas.conversation import Message
from app.core.config import settings
from app.core.constants import Role


def append_message(session: Session, role: Role, text: str) -> None:
    session.conversation.recent_messages.append(Message(role=role, text=text))
    window = settings.RECENT_MESSAGES_WINDOW
    if len(session.conversation.recent_messages) > window:
        session.conversation.recent_messages = (
            session.conversation.recent_messages[-window:]
        )
