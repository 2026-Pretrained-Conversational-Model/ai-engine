"""
app/services/router/router_judge.py
------------------------------------
역할: 현재 턴의 다음 경로(RouterDecision)를 결정한다.

v8 변경:
- 로직 자체는 유지. 매핑 규칙은 동일.
- 로그 메시지를 명확화: NEED_RAG → DIRECT_ANSWER로 강등되는 경로에
  "no_doc_attached"라는 강등 사유(downgrade reason)를 명시해서
  "NEED_RAG라더니 왜 direct_answer로 가지?" 혼동을 제거.
- 매핑 결과를 단일 라인 INFO 로그로 일관되게 출력.

매핑 규칙 (요약):
    NEED_RAG + (active_pdf or has_attachment) → RETRIEVE_DOC
        retrieval_query에 참조 표현 포함 시 → SEARCH_PREP_THEN_RETRIEVE 승격
    NEED_RAG + 문서 없음 → DIRECT_ANSWER  (downgraded: no_doc_attached)
    NO_RAG  + FILE_REQUIRED + 문서 있음 → RETRIEVE_DOC
    NO_RAG  + 그 외 → DIRECT_ANSWER
"""
from __future__ import annotations

import asyncio
from typing import Tuple

from app.core.constants import RouterDecision
from app.core.logger import get_logger
from app.schemas.rag_router import RagDecision, RagReasonCode, RagRouterResponse
from app.schemas.session import Session
from app.services.llm.local_registry import LocalModelRegistry
from app.services.router.qwen_rag_router import build_combined_prompt, build_user_prompt

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Hint tables (used only by the heuristic fallback path)
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
# Heuristic baseline
# ---------------------------------------------------------------------------

def _judge_route_heuristic(
    session: Session, query: str, has_attachment: bool = False
) -> RouterDecision:
    """LocalModelRegistry에 'router' 모델이 없을 때의 규칙 기반 폴백."""
    q = query.strip().lower()
    active_pdf = session.pdf_state.active_pdf is not None

    if active_pdf or has_attachment:
        if any(token in q for token in _DOC_HINTS):
            return RouterDecision.RETRIEVE_DOC
        if any(token in q for token in _REFERENCE_HINTS):
            return RouterDecision.SEARCH_PREP_THEN_RETRIEVE

    return RouterDecision.DIRECT_ANSWER


# ---------------------------------------------------------------------------
# RagRouterResponse → RouterDecision 매핑
# ---------------------------------------------------------------------------

def _map_to_router_decision(
    rag_response: RagRouterResponse,
    session: Session,
    query: str,
    has_attachment: bool,
) -> Tuple[RouterDecision, str]:
    """
    매핑 결과와 함께 사람이 읽을 수 있는 설명 문자열을 같이 리턴한다.

    Returns:
        (decision, reason_text)
    """
    active_pdf = session.pdf_state.active_pdf is not None
    has_doc = active_pdf or has_attachment

    if rag_response.decision == RagDecision.NEED_RAG:
        if has_doc:
            rq = (rag_response.retrieval_query or "").lower()
            if any(hint in rq for hint in _REFERENCE_HINTS):
                return (
                    RouterDecision.SEARCH_PREP_THEN_RETRIEVE,
                    "need_rag+doc+reference",
                )
            return RouterDecision.RETRIEVE_DOC, "need_rag+doc"
        # 문서 없이 NEED_RAG → 외부 벡터 DB가 없으므로 어쩔 수 없이 강등
        return (
            RouterDecision.DIRECT_ANSWER,
            "downgraded:need_rag_but_no_doc_attached",
        )

    # NO_RAG 분기
    if rag_response.reason_code == RagReasonCode.FILE_REQUIRED and has_doc:
        return RouterDecision.RETRIEVE_DOC, "no_rag+file_required+doc"

    return RouterDecision.DIRECT_ANSWER, f"no_rag:{rag_response.reason_code.value}"


# ---------------------------------------------------------------------------
# LLM judge (QwenRagRouter via LocalModelRegistry)
# ---------------------------------------------------------------------------

def _generate_rag_response_via_registry(
    session: Session,
    query: str,
    has_attachment: bool,
) -> RagRouterResponse:
    """LocalModelRegistry의 'router' 모델로 RagRouterResponse를 생성한다."""
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
# Public entry
# ---------------------------------------------------------------------------

async def judge_route(
    session: Session, query: str, has_attachment: bool = False
) -> RouterDecision:
    if not LocalModelRegistry.has("router"):
        decision = _judge_route_heuristic(session, query, has_attachment)
        logger.info("router heuristic: route=%s (no LLM router registered)", decision.value)
        return decision

    try:
        rag_response = await asyncio.to_thread(
            _generate_rag_response_via_registry, session, query, has_attachment
        )
        decision, reason_text = _map_to_router_decision(
            rag_response, session, query, has_attachment
        )
        logger.info(
            "router LLM judge: rag=%s confidence=%.2f reason_code=%s "
            "→ route=%s (%s)",
            rag_response.decision.value,
            rag_response.confidence,
            rag_response.reason_code.value,
            decision.value,
            reason_text,
        )
        return decision

    except Exception as exc:
        logger.warning(
            "router LLM judge failed (%s), falling back to heuristic", exc
        )
        decision = _judge_route_heuristic(session, query, has_attachment)
        logger.info("router heuristic (fallback): route=%s", decision.value)
        return decision
