"""
app/core/constants.py
---------------------
역할: 서비스 전반에서 사용하는 Enum 형태의 상수 정의
      코드 내 매직 문자열 사용을 방지
"""
from enum import Enum


class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ResourceType(str, Enum):
    NONE = "none"
    PDF = "pdf"
    IMAGE = "image"


class AnswerType(str, Enum):
    MULTITURN_ONLY = "multiturn_only"
    MULTITURN_WITH_PDF = "multiturn_with_pdf_context"
    MULTITURN_WITH_VLM = "multiturn_with_vlm"


class SessionStatus(str, Enum):
    ACTIVE = "active"
    EXPIRED = "expired"
    CLOSED = "closed"


class ModelKind(str, Enum):
    LLM = "llm"
    VLM = "vlm"
