"""
app/schemas/rag_router.py
--------------------------
역할: QwenRagRouter의 입출력 스키마 정의.

    router.py에서 독립 FastAPI 앱으로 운영되던 스키마를
    ai-engine 구조에 맞게 편입한 모듈이다.

    - RagDecision  : NEED_RAG / NO_RAG Enum
    - RagReasonCode: 판정 근거 코드 Enum
    - RagRouterResponse: 모델 출력 검증 스키마
    - RagRequest   : /rag/decide 엔드포인트 입력 스키마
"""
from __future__ import annotations

import json
import re
from enum import Enum

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class RagDecision(str, Enum):
    """
    RAG 검색 필요 여부.

    Values:
        NEED_RAG: 외부 문서 검색 또는 지식베이스 조회가 필요함.
        NO_RAG  : 현재 질의만으로 답변 가능하거나 첨부 파일 분석 경로.
    """
    NEED_RAG = "NEED_RAG"
    NO_RAG   = "NO_RAG"


class RagReasonCode(str, Enum):
    """
    RAG 필요 여부 판단의 이유 코드.

    Values:
        EXTERNAL_KNOWLEDGE_REQUIRED: 외부 문서 또는 지식 검색 필요.
        ENOUGH_CONTEXT_IN_QUERY    : 현재 질의만으로 충분한 맥락 존재.
        FILE_REQUIRED              : 첨부 파일 자체가 핵심 근거, 파일 분석 우선.
        AMBIGUOUS_BUT_NOT_RAG      : 모호하지만 현 단계에서 RAG 불필요.
    """
    EXTERNAL_KNOWLEDGE_REQUIRED = "EXTERNAL_KNOWLEDGE_REQUIRED"
    ENOUGH_CONTEXT_IN_QUERY     = "ENOUGH_CONTEXT_IN_QUERY"
    FILE_REQUIRED               = "FILE_REQUIRED"
    AMBIGUOUS_BUT_NOT_RAG       = "AMBIGUOUS_BUT_NOT_RAG"


# ---------------------------------------------------------------------------
# Response schema
# ---------------------------------------------------------------------------

class RagRouterResponse(BaseModel):
    """
    QwenRagRouter의 RAG 판단 결과 스키마.

    Attributes:
        decision       : NEED_RAG 또는 NO_RAG.
        confidence     : 0.0 ~ 1.0 판단 신뢰도.
        reason_code    : 판단 근거 코드.
        retrieval_query: NEED_RAG일 때 사용할 검색 질의. NO_RAG이면 빈 문자열.

    Class Methods:
        from_json_text: 모델 raw 출력에서 JSON 블록을 추출해 객체로 변환한다.
    """
    decision        : RagDecision  = Field(..., description="RAG 검색 필요 여부")
    confidence      : float        = Field(..., ge=0.0, le=1.0, description="판정 신뢰도")
    reason_code     : RagReasonCode = Field(..., description="판정 이유 코드")
    retrieval_query : str          = Field(default="", description="RAG가 필요할 때만 사용할 검색용 질의")

    @classmethod
    def from_json_text(cls, text: str) -> "RagRouterResponse":
        """모델의 raw 문자열 출력에서 JSON 블록을 추출하고 객체로 변환한다."""
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError(f"No JSON object found in model output: {text}")
        data = json.loads(match.group(0))
        return cls.model_validate(data)


# ---------------------------------------------------------------------------
# Request schema (FastAPI /rag/decide 엔드포인트용)
# ---------------------------------------------------------------------------

class RagRequest(BaseModel):
    """
    /rag/decide 엔드포인트의 HTTP 요청 스키마.

    Attributes:
        current_query        : 현재 사용자 질의.
        has_attached_pdf     : PDF 파일 첨부 여부.
        has_attached_image   : 이미지 파일 첨부 여부.
        conversation_summary : 이전 대화 요약.
        last_action          : 직전 시스템 액션.
    """
    current_query        : str  = Field(...,      description="사용자 질의")
    has_attached_pdf     : bool = Field(False,    description="PDF 첨부 여부")
    has_attached_image   : bool = Field(False,    description="이미지 첨부 여부")
    conversation_summary : str  = Field("",       description="이전 대화 요약")
    last_action          : str  = Field("NONE",   description="직전 액션")
