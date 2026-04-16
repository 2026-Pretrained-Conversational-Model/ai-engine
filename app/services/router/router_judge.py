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

v13 변경 (sagemaker 연동):
- LLM_BACKEND == "sagemaker"면 LocalModelRegistry 대신 LLMClient(role="router") 호출.
- LLM_BACKEND == "local"면 기존 registry 경로 유지 (Colab 호환).
- 어느 경로든 실패 시 heuristic 폴백.

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

from app.core.config import settings
from app.core.constants import RouterDecision
from app.core.logger import get_logger
from app.schemas.rag_router import RagDecision, RagReasonCode, RagRouterResponse
from app.schemas.session import Session
from app.services.llm.llm_client import LLMClient
from app.services.llm.local_registry import LocalModelRegistry
from app.services.router.qwen_rag_router import (
    RAG_ROUTER_SYSTEM_PROMPT,
    build_combined_prompt,
    build_user_prompt,
)

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
# Heuristic fallback (when LLM 판단 불가)
# ---------------------------------------------------------------------------

def _judge_route_heuristic(
    session: Session, query: str, has_attachment: bool
) -> RouterDecision:
    """LocalModelRegistry에 'router' 모델이 없을 때의 규칙 기반 폴백."""
    active_pdf = session.pdf_state.active_pdf
    q_lower = query.lower()

    has_doc_hint = any(hint in q_lower for hint in _DOC_HINTS)
    has_ref_hint = any(hint in q_lower for hint in _REFERENCE_HINTS)

    if active_pdf or has_attachment:
        if has_doc_hint or has_ref_hint:
            if has_ref_hint:
                return RouterDecision.SEARCH_PREP_THEN_RETRIEVE
            return RouterDecision.RETRIEVE_DOC

    return RouterDecision.DIRECT_ANSWER


# ---------------------------------------------------------------------------
# Mapping: RagRouterResponse → RouterDecision
# ---------------------------------------------------------------------------

def _map_to_router_decision(
    rag_response: RagRouterResponse,
    session: Session,
    query: str,
    has_attachment: bool,
) -> Tuple[RouterDecision, str]:
    active_pdf = session.pdf_state.active_pdf
    has_doc = bool(active_pdf) or has_attachment

    if rag_response.decision == RagDecision.NEED_RAG:
        if has_doc:
            q_lower = query.lower()
            retrieval_lower = (rag_response.retrieval_query or "").lower()
            has_ref = (
                any(hint in q_lower for hint in _REFERENCE_HINTS)
                or any(hint in retrieval_lower for hint in _REFERENCE_HINTS)
            )
            if has_ref:
                return (
                    RouterDecision.SEARCH_PREP_THEN_RETRIEVE,
                    "need_rag+doc+reference",
                )
            return RouterDecision.RETRIEVE_DOC, "need_rag+doc"

        return (
            RouterDecision.DIRECT_ANSWER,
            "downgraded:need_rag_but_no_doc_attached",
        )

    # NO_RAG 분기
    if rag_response.reason_code == RagReasonCode.FILE_REQUIRED and has_doc:
        return RouterDecision.RETRIEVE_DOC, "no_rag+file_required+doc"

    return RouterDecision.DIRECT_ANSWER, f"no_rag:{rag_response.reason_code.value}"


# ---------------------------------------------------------------------------
# LLM judge via LocalModelRegistry (LLM_BACKEND=local 경로)
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
    logger.debug("Qwen router raw output (local): %r", raw[:200])
    return RagRouterResponse.from_json_text(raw)


# ---------------------------------------------------------------------------
# LLM judge via LLMClient (LLM_BACKEND=sagemaker 경로) — v13 추가
# ---------------------------------------------------------------------------

async def _generate_rag_response_via_llm_client(
    session: Session,
    query: str,
    has_attachment: bool,
) -> RagRouterResponse:
    """
    LLMClient.generate(role='router')로 RagRouterResponse를 생성한다.
    sagemaker_backend가 system/user 분리해서 엔드포인트 chat template에 주입한다.
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

    raw = await LLMClient.generate(
        prompt=user_prompt,
        role="router",
        max_new_tokens=60,
        system=RAG_ROUTER_SYSTEM_PROMPT,
    )
    logger.debug("Qwen router raw output (sagemaker): %r", raw[:200])
    return RagRouterResponse.from_json_text(raw)


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------

async def judge_route(
    session: Session, query: str, has_attachment: bool = False
) -> RouterDecision:
    """
    백엔드 모드에 따라 적절한 경로로 라우터 판단:
      - LLM_BACKEND=local   → LocalModelRegistry (없으면 heuristic)
      - LLM_BACKEND=sagemaker → LLMClient(role='router') (실패 시 heuristic)
    """
    backend = (settings.LLM_BACKEND or "").lower()

    # ---- local 모드: 기존 로직 유지 ----
    if backend == "local":
        if not LocalModelRegistry.has("router"):
            decision = _judge_route_heuristic(session, query, has_attachment)
            logger.info("router heuristic: route=%s (no local LLM router)", decision.value)
            return decision

        try:
            rag_response = await asyncio.to_thread(
                _generate_rag_response_via_registry, session, query, has_attachment
            )
        except Exception as exc:
            logger.warning("router local LLM failed (%s), heuristic fallback", exc)
            decision = _judge_route_heuristic(session, query, has_attachment)
            logger.info("router heuristic (fallback): route=%s", decision.value)
            return decision

        decision, reason_text = _map_to_router_decision(
            rag_response, session, query, has_attachment
        )
        logger.info(
            "router LLM judge (local): rag=%s confidence=%.2f reason_code=%s → route=%s (%s)",
            rag_response.decision.value,
            rag_response.confidence,
            rag_response.reason_code.value,
            decision.value,
            reason_text,
        )
        return decision

    # ---- sagemaker 모드 (기본): LLMClient 경유 ----
    try:
        rag_response = await _generate_rag_response_via_llm_client(
            session, query, has_attachment
        )
    except Exception as exc:
        logger.warning("router sagemaker LLM failed (%s), heuristic fallback", exc)
        decision = _judge_route_heuristic(session, query, has_attachment)
        logger.info("router heuristic (fallback): route=%s", decision.value)
        return decision

    decision, reason_text = _map_to_router_decision(
        rag_response, session, query, has_attachment
    )
    logger.info(
        "router LLM judge (sagemaker): rag=%s confidence=%.2f reason_code=%s → route=%s (%s)",
        rag_response.decision.value,
        rag_response.confidence,
        rag_response.reason_code.value,
        decision.value,
        reason_text,
    )
    return decision
