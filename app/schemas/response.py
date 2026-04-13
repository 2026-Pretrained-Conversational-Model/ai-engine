"""
app/schemas/response.py
-----------------------
역할: Node.js로 반환되는 응답 DTO 정의

      `expired=True`는 사용자가 새 채팅을 시작하도록
      Node.js가 유도하는 트리거로 사용됨
"""
from typing import Optional
from pydantic import BaseModel
from app.core.constants import AnswerType


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    answer_type: AnswerType
    expired: bool = False
    expire_reason: Optional[str] = None    # "memory_limit" | "idle_timeout"
