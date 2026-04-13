"""
app/services/conversation/recent_window.py
------------------------------------------
역할: recent_messages를 LLM 입력에 적합한 형태로 가공하는
      읽기 전용 유틸 함수
"""
from typing import List
from app.schemas.session import Session
from app.schemas.conversation import Message


def get_recent(session: Session) -> List[Message]:
    return list(session.conversation.recent_messages)


def render_for_prompt(messages: List[Message]) -> str:
    return "\n".join(f"{m.role.value}: {m.text}" for m in messages)
