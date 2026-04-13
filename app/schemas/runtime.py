"""
app/schemas/runtime.py
----------------------
역할: `runtime_state` 및 `cleanup_policy` 블록 정의
"""
from typing import Optional
from pydantic import BaseModel
from app.core.constants import ResourceType, AnswerType


class RuntimeState(BaseModel):
    active_resource_type: ResourceType = ResourceType.NONE
    active_pdf_id: Optional[str] = None
    last_answer_type: Optional[AnswerType] = None


class CleanupPolicy(BaseModel):
    expire_after_minutes: int = 30
    delete_local_file_on_expire: bool = True
