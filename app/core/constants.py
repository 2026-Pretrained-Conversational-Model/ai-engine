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


class RouterDecision(str, Enum):
    """
    라우터가 현재 턴에서 수행할 다음 행동.

    - DIRECT_ANSWER:
        외부 문서 검색 없이 recent turns + memory summary만으로 답변 가능.
    - RETRIEVE_DOC:
        현재 질문이 문서/첨부/PDF의 내용 자체를 묻고 있어 즉시 검색이 필요.
    - SEARCH_PREP_THEN_RETRIEVE:
        '그거', '아까 표', '이 논문'처럼 참조 해소/질의 정규화 후 검색 필요.
    - ASK_CLARIFICATION:
        첨부도 없고 문맥도 부족하여 바로 답변하면 오답 위험이 큰 경우.
    """

    DIRECT_ANSWER = "direct_answer"
    RETRIEVE_DOC = "retrieve_doc"
    SEARCH_PREP_THEN_RETRIEVE = "search_prep_then_retrieve"
    ASK_CLARIFICATION = "ask_clarification"


class IngestStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    READY = "ready"
    FAILED = "failed"
