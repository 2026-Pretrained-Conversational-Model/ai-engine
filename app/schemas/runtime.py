"""
app/schemas/runtime.py
----------------------
역할: `runtime_state` 및 `cleanup_policy` 블록 정의
"""
from typing import Optional
from pydantic import BaseModel
from app.core.constants import ResourceType, AnswerType, IngestStatus, RouterDecision


class RuntimeState(BaseModel):
    active_resource_type: ResourceType = ResourceType.NONE
    active_pdf_id: Optional[str] = None
    last_answer_type: Optional[AnswerType] = None

    # ---- Orchestration state -------------------------------------------------
    # 라우터와 파이프라인이 현재 PDF 준비 상태를 보고
    # "지금 바로 답변할지 / 인덱싱 완료까지 기다릴지"를 결정하는 데 사용한다.
    pdf_ingest_status: IngestStatus = IngestStatus.IDLE
    last_router_decision: Optional[RouterDecision] = None
    last_error: str = ""


class CleanupPolicy(BaseModel):
    expire_after_minutes: int = 30
    delete_local_file_on_expire: bool = True
