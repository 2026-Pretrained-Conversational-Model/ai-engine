"""
app/services/router/router_judge.py
----------------------------------
역할: 현재 턴의 다음 경로(RouterDecision)를 결정한다.

변경점:
- 이전 버전: 완전 규칙 기반 (keyword hints)
- 이번 버전: LocalModelRegistry에 "router" role이 등록돼 있으면 LLM judge 사용,
             없으면 기존 휴리스틱 폴백.
             함수 시그니처가 async로 변경됨 (pipeline.py의 호출 부분 1줄도 수정 필요).

LLM judge 프롬프트는 few-shot 없이 label-only 출력을 유도하는 baseline.
운영에서 교체하고 싶으면 _ROUTER_SYSTEM/_ROUTER_USER만 바꾸면 된다.

TODO:
    [ ] LLM judge 로그를 auto-label 데이터로 축적하여 소형 classifier 학습
    [ ] 출력이 label이 아닌 경우 재시도 로직
"""
from __future__ import annotations

import asyncio

from app.core.constants import RouterDecision
from app.core.logger import get_logger
from app.schemas.session import Session
from app.services.llm.local_registry import LocalModelRegistry

logger = get_logger(__name__)


_DOC_HINTS = [
    "논문", "pdf", "문서", "페이지", "표", "figure", "table", "section", "캡션", "첨부",
    "파일", "본문", "이 문서", "이 논문", "이 pdf",
]
_REFERENCE_HINTS = [
    "그거", "그 부분", "그 논문", "그 표", "그 그림", "아까", "방금", "이거", "저거",
    "that", "it", "those", "above", "previous",
]


# --- Heuristic baseline (fallback) -------------------------------------------

def _judge_route_heuristic(
    session: Session, query: str, has_attachment: bool = False
) -> RouterDecision:
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
            return RouterDecision.ASK_CLARIFICATION
        return RouterDecision.DIRECT_ANSWER

    if len(q) <= 6 and not active_pdf and not session.conversation.recent_messages:
        return RouterDecision.ASK_CLARIFICATION

    return RouterDecision.DIRECT_ANSWER


# --- LLM judge ---------------------------------------------------------------

_ROUTER_PROMPT = """You are a routing classifier for a multi-turn chatbot that optionally receives PDF attachments.

Choose exactly ONE label for the user's current query, from this set:
- DIRECT_ANSWER: answer from conversation memory alone, no document retrieval needed
- RETRIEVE_DOC: user is directly asking about content in the attached document; run RAG
- SEARCH_PREP_THEN_RETRIEVE: query uses references ("그거", "that", "아까 본 표") that must be resolved before retrieval
- ASK_CLARIFICATION: conversation context is too thin to answer safely; ask the user to clarify

Session state:
- has_active_pdf: {has_pdf}
- pdf_file_name: {pdf_name}
- recent_messages_count: {recent_count}
- current_topic: {topic}

User query: {query}

Output ONLY the label string. No explanation."""


def _parse_label(text: str) -> RouterDecision | None:
    upper = (text or "").strip().upper()
    # 우선순위: 길이가 긴 라벨부터 매칭 (SEARCH_PREP_THEN_RETRIEVE가 RETRIEVE_DOC보다 먼저)
    ordered = sorted(
        list(RouterDecision), key=lambda d: len(d.value), reverse=True
    )
    for decision in ordered:
        if decision.value.upper() in upper:
            return decision
    return None


async def judge_route(
    session: Session, query: str, has_attachment: bool = False
) -> RouterDecision:
    # router 모델이 없으면 바로 휴리스틱
    if not LocalModelRegistry.has("router"):
        return _judge_route_heuristic(session, query, has_attachment)

    pdf = session.pdf_state.active_pdf
    prompt = _ROUTER_PROMPT.format(
        has_pdf=bool(pdf) or has_attachment,
        pdf_name=pdf.file_name if pdf else "-",
        recent_count=len(session.conversation.recent_messages),
        topic=session.conversation.current_topic or "-",
        query=query,
    )

    try:
        raw = await asyncio.to_thread(
            LocalModelRegistry.generate, "router", prompt, 24
        )
        label = _parse_label(raw)
        if label is not None:
            logger.info("router LLM judged: %s (raw=%r)", label.value, raw[:60])
            return label
        logger.warning("router LLM returned unparseable output: %r", raw[:120])
    except Exception as e:
        logger.exception("router LLM failed, falling back to heuristic: %s", e)

    return _judge_route_heuristic(session, query, has_attachment)
