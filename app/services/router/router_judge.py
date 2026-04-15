"""
app/services/router/router_judge.py
------------------------------------
역할: 현재 턴의 다음 경로(RouterDecision)를 결정한다.

변경점 (QwenRagRouter 통합):
    - 이전 버전: LocalModelRegistry.generate()에 단일 텍스트 프롬프트 전달.
    - 이번 버전: LocalModelRegistry에 "router" role이 등록돼 있으면
                 QwenRagRouter 스타일의 JSON 출력(RagRouterResponse)을 파싱해
                 RagDecision → RouterDecision으로 매핑.
                 등록되지 않은 경우 기존 휴리스틱 폴백 유지.

매핑 규칙 (RagDecision → RouterDecision):
    NEED_RAG
        + active_pdf 또는 has_attachment → RETRIEVE_DOC
            retrieval_query에 참조 표현 포함 시 → SEARCH_PREP_THEN_RETRIEVE 승격
        + 그 외                          → DIRECT_ANSWER (벡터 DB 등 외부 RAG 없음)
    NO_RAG
        + FILE_REQUIRED reason           → RETRIEVE_DOC  (파일 분석 우선)
        + 모호 + 문맥 부족               → ASK_CLARIFICATION
        + 그 외                          → DIRECT_ANSWER

TODO:
    [ ] LLM judge 로그를 auto-label 데이터로 축적하여 소형 classifier 학습
    [ ] confidence < threshold 시 휴리스틱으로 강등하는 옵션
    [ ] 출력 파싱 실패 시 재시도 로직 (최대 2회)
"""
from __future__ import annotations

import asyncio

from app.core.constants import RouterDecision
from app.core.logger import get_logger
from app.schemas.rag_router import RagDecision, RagReasonCode, RagRouterResponse
from app.schemas.session import Session
from app.services.llm.local_registry import LocalModelRegistry
from app.services.router.qwen_rag_router import build_combined_prompt, build_user_prompt

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# 참조 표현 힌트 (SEARCH_PREP_THEN_RETRIEVE 승격 판단용)
# ---------------------------------------------------------------------------

_REFERENCE_HINTS = [
    "그거", "그 부분", "그 논문", "그 표", "그 그림", "아까", "방금", "이거", "저거",
    "that", "it", "those", "above", "previous",
]

_DOC_HINTS = [
    "논문", "pdf", "문서", "페이지", "표", "figure", "table", "section", "캡션", "첨부",
    "파일", "본문", "이 문서", "이 논문", "이 pdf",
]


# ---------------------------------------------------------------------------
# Heuristic baseline (fallback)
# ---------------------------------------------------------------------------

def _judge_route_heuristic(
    session: Session, query: str, has_attachment: bool = False
) -> RouterDecision:
    """LocalModelRegistry에 "router"가 없을 때 사용하는 규칙 기반 폴백."""
    q = query.strip().lower()
    active_pdf = session.pdf_state.active_pdf is not None
    summary_exists = bool(session.pdf_state.doc_summary.one_line)

    if active_pdf or has_attachment:
        if any(token in q for token in _DOC_HINTS):
            return RouterDecision.RETRIEVE_DOC
        if any(token in q for token in _REFERENCE_HINTS):
            return RouterDecision.SEARCH_PREP_THEN_RETRIEVE

    if any(token in q for token in _REFERENCE_HINTS):
        if not session.conversation.recent_messages and not summary_exists:
            return RouterDecision.DIRECT_ANSWER
        return RouterDecision.DIRECT_ANSWER

    if len(q) <= 6 and not active_pdf and not session.conversation.recent_messages:
        return RouterDecision.DIRECT_ANSWER

    return RouterDecision.DIRECT_ANSWER


# ---------------------------------------------------------------------------
# RagRouterResponse → RouterDecision 매핑
# ---------------------------------------------------------------------------

def _map_to_router_decision(
    rag_response: RagRouterResponse,
    session: Session,
    query: str,
    has_attachment: bool,
) -> RouterDecision:
    """
    QwenRagRouter의 RagRouterResponse를 ai-engine RouterDecision으로 변환한다.

    매핑 규칙:
        NEED_RAG + (active_pdf or has_attachment) → RETRIEVE_DOC
            retrieval_query에 참조 표현 포함 시 → SEARCH_PREP_THEN_RETRIEVE 승격
        NEED_RAG + 파일 없음 → DIRECT_ANSWER  (외부 벡터 DB 미지원)
        NO_RAG  + FILE_REQUIRED → RETRIEVE_DOC (파일 분석 우선)
        NO_RAG  + AMBIGUOUS_BUT_NOT_RAG + 문맥 부족 → ASK_CLARIFICATION
        NO_RAG  + 그 외 → DIRECT_ANSWER
    """
    active_pdf = session.pdf_state.active_pdf is not None
    has_doc = active_pdf or has_attachment

    if rag_response.decision == RagDecision.NEED_RAG:
        if has_doc:
            rq = (rag_response.retrieval_query or "").lower()
            if any(hint in rq for hint in _REFERENCE_HINTS):
                return RouterDecision.SEARCH_PREP_THEN_RETRIEVE
            elif any(rag_response.reason_code == RagReasonCode.EXTERNAL_KNOWLEDGE_REQUIRED):
                return RouterDecision.RETRIEVE_DOC
            elif any(hint in rq for hint in _DOC_HINTS):
                return RouterDecision.RETRIEVE_DOC
            return RouterDecision.RETRIEVE_DOC
        elif rag_response.reason_code == RagReasonCode.ENOUGH_CONTEXT_IN_QUERY:
            return RouterDecision.DIRECT_ANSWER
        # 문서 없이 NEED_RAG → 외부 RAG 미지원, 직접 답변
        logger.debug("NEED_RAG but no document available, falling back to DIRECT_ANSWER")
        return RouterDecision.DIRECT_ANSWER

    # NO_RAG 분기
    if rag_response.reason_code == RagReasonCode.FILE_REQUIRED and has_doc:
        return RouterDecision.RETRIEVE_DOC

    if rag_response.reason_code == RagReasonCode.AMBIGUOUS_BUT_NOT_RAG:
        summary_exists = bool(session.pdf_state.doc_summary.one_line)
        if not session.conversation.recent_messages and not summary_exists:
<<<<<<< HEAD
            return RouterDecision.ASK_CLARIFICATION
=======
            return RouterDecision.DIRECT_ANSWER

    if rag_response.reason_code == RagReasonCode.ENOUGH_CONTEXT_IN_QUERY:
        return RouterDecision.ASK_CLARIFICATION
>>>>>>> 89e0739f4460f066daa51bb1b57bf1f6a3d1602d
        
    return RouterDecision.DIRECT_ANSWER


# ---------------------------------------------------------------------------
# LLM judge (QwenRagRouter via LocalModelRegistry)
# ---------------------------------------------------------------------------

def _generate_rag_response_via_registry(
    session: Session,
    query: str,
    has_attachment: bool,
) -> RagRouterResponse:
    """
    LocalModelRegistry의 "router" 모델로 RagRouterResponse를 생성한다.

    LocalModelRegistry.generate()는 단일 문자열 프롬프트만 지원하므로
    build_combined_prompt()로 system + user 내용을 합쳐서 전달한다.
    """
    active_pdf = session.pdf_state.active_pdf
    conversation_summary = (
        session.pdf_state.doc_summary.one_line
        or session.conversation.current_topic
        or ""
    )
    last_action = (
        session.runtime_state.last_router_decision.value
        if session.runtime_state.last_router_decision
        else "NONE"
    )

    user_prompt = build_user_prompt(
        current_query=query,
        has_attached_pdf=bool(active_pdf) or has_attachment,
        has_attached_image=False,
        conversation_summary=conversation_summary,
        last_action=last_action,
    )
    combined_prompt = build_combined_prompt(user_prompt)

    raw = LocalModelRegistry.generate("router", combined_prompt, max_new_tokens=60)
    logger.debug("Qwen router raw output: %r", raw[:200])
    return RagRouterResponse.from_json_text(raw)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def judge_route(
    session: Session, query: str, has_attachment: bool = False
) -> RouterDecision:
    """
    현재 턴의 RouterDecision을 결정한다.

    1. LocalModelRegistry에 "router" role이 등록된 경우:
       → QwenRagRouter 스타일 프롬프트로 LLM judge 수행 후 RouterDecision으로 매핑.
    2. 등록되지 않은 경우:
       → 규칙 기반 휴리스틱 폴백.

    Args:
        session       : 현재 세션 객체.
        query         : 사용자 질의 문자열.
        has_attachment: 현재 턴에 첨부 파일이 있는지 여부.

    Returns:
        RouterDecision: 파이프라인이 수행할 다음 경로.
    """
    if not LocalModelRegistry.has("router"):
        logger.debug("No 'router' model registered, using heuristic fallback")
        return _judge_route_heuristic(session, query, has_attachment)

    try:
        rag_response = await asyncio.to_thread(
            _generate_rag_response_via_registry, session, query, has_attachment
        )
        decision = _map_to_router_decision(rag_response, session, query, has_attachment)
        logger.info(
            "Qwen router judge: rag=%s confidence=%.2f reason=%s → route=%s",
            rag_response.decision.value,
            rag_response.confidence,
            rag_response.reason_code.value,
            decision.value,
        )
        return decision

    except Exception as exc:
        logger.warning("Qwen router judge failed (%s), falling back to heuristic", exc)

    return _judge_route_heuristic(session, query, has_attachment)
