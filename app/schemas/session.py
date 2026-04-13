"""
app/schemas/session.py
----------------------
역할: 최상위 Session 메모리 모델 정의
      session_id를 키로 하는 인메모리 저장소에 그대로 저장되는 핵심 데이터 구조

      설계 문서 Section 3의 JSON 구조를 1:1로 반영

TODO:
    [ ] 향후 스키마 마이그레이션을 위한 `version` 필드 추가
    [ ] 로그 출력 시 원본 chunk 텍스트를 제외하는 커스텀 serializer 추가
"""
from datetime import datetime
from pydantic import BaseModel, Field

from app.core.constants import SessionStatus
from app.schemas.conversation import Conversation
from app.schemas.pdf import PdfState
from app.schemas.runtime import RuntimeState, CleanupPolicy


class SessionMeta(BaseModel):
    session_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_accessed_at: datetime = Field(default_factory=datetime.utcnow)
    status: SessionStatus = SessionStatus.ACTIVE


class Session(BaseModel):
    session_meta: SessionMeta
    conversation: Conversation = Field(default_factory=Conversation)
    runtime_state: RuntimeState = Field(default_factory=RuntimeState)
    pdf_state: PdfState = Field(default_factory=PdfState)
    cleanup_policy: CleanupPolicy = Field(default_factory=CleanupPolicy)
