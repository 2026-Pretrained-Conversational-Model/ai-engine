"""
app/schemas/conversation.py
---------------------------
역할: 세션 메모리의 `conversation` 블록을 위한 Pydantic 모델 정의
"""
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field

from app.core.constants import Role


class Message(BaseModel):
    role: Role
    text: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class StructuredSummary(BaseModel):
    goal: str = ""
    established_facts: List[str] = Field(default_factory=list)
    current_focus: str = ""
    unresolved_questions: List[str] = Field(default_factory=list)


class Summary(BaseModel):
    narrative: str = ""
    structured: StructuredSummary = Field(default_factory=StructuredSummary)


class Conversation(BaseModel):
    recent_messages: List[Message] = Field(default_factory=list)
    summary: Summary = Field(default_factory=Summary)
    current_topic: str = ""
    last_user_intent: str = ""
