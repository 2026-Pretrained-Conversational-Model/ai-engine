"""
app/services/memory/memory_state_generator.py
----------------------------------------------
역할: 매 턴 종료 후 대화 내용을 읽고 memory_state JSON을 생성하여
      세션의 conversation.summary (narrative + structured)에 반영한다.

이 모듈이 v8에서 메모리 갱신의 **유일한** 진입점이다.
narrative_updater / structured_updater는 모두 no-op으로 비워졌다.

LocalModelRegistry에서 모델을 찾는 우선순위:
    1) "memory" role
    2) "summary" role  (g34634/qwen2.5-3b-memory-summary-v1 같은 모델이 등록된 경우)
둘 다 없으면 graceful no-op.

학습된 모델(g34634/qwen2.5-3b-memory-summary-v1)이 기대하는 출력 포맷:
{
  "memory_state": {
    "key_facts": ["fact1", "fact2"],
    "unresolved_refs": ["..."],
    "topic": "...",
    "turn_count": 3
  },
  "memory_summary": "한 문장 요약"
}

이 포맷이 narrative + structured 두 슬롯을 동시에 채운다:
    - narrative ← memory_summary
    - structured.established_facts ← key_facts
    - structured.current_focus ← topic
    - structured.unresolved_questions ← unresolved_refs
    - current_topic ← topic
"""
from __future__ import annotations

import asyncio
import json
import re
from typing import Optional

from app.core.logger import get_logger
from app.schemas.session import Session
from app.schemas.conversation import StructuredSummary
from app.services.llm.local_registry import LocalModelRegistry

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Prompt — g34634/qwen2.5-3b-memory-summary-v1 학습 포맷에 최대한 맞춤
# ---------------------------------------------------------------------------

MEMORY_SYSTEM_PROMPT = """당신은 멀티턴 대화 시스템에서 메모리 상태를 생성하는 모델입니다.

입력으로 다음 정보가 주어집니다:
1) 이전 메모리 요약 (previous memory summary)
2) 이전 구조화된 메모리 (previous structured memory)
3) 최근 대화 내용 (recent conversation turns)

당신의 역할은 메모리를 새로 만드는 것이 아니라, 기존 메모리를 "업데이트"하는 것입니다.

아래의 출력 형식을 반드시 지켜서 JSON만 출력하세요:

{
  "memory_state": {
    "goal": "사용자의 전체적인 목적 (명확하지 않으면 기존 값을 유지하거나 빈 값)",
    "key_facts": ["fact1", "fact2"],
    "unresolved_refs": ["아직 해결되지 않은 지시어 또는 질문"],
    "topic": "현재 대화의 핵심 주제",
    "turn_count": <대화 턴 수>,
    "last_resolved_anchor": "'그거', '이거' 등이 가리키는 핵심 대상"
  },
  "memory_summary": "지금까지 대화를 한 문장으로 요약"
}

규칙:
- 반드시 유효한 JSON만 출력하세요. 설명, 문장, 코드블록 등을 추가하지 마세요.
- 무조건 한국어로 출력하세요. 
- 기존 메모리에서 여전히 유효한 정보는 유지하세요.
- 명확히 틀린 정보가 아니면 삭제하지 마세요.
- key_facts에는 오래 유지해야 하는 핵심 정보만 넣으세요. (단순 문장 요약 금지)
- unresolved_refs에는 아직 해결되지 않은 참조나 질문만 넣으세요.
- memory_summary는 전체 대화를 반영해야 합니다. 단순 key_facts 나열이 아니라, 대화의 흐름과 맥락을 반영해야 하며 길어질 경우 600자 이내로 요약하세요."""

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# def _build_conversation_text(session: Session) -> str:
#     """recent_messages를 'A:' / 'B:' 형식 대화 텍스트로 변환."""
#     messages = session.conversation.recent_messages
#     if not messages:
#         return ""
#     lines = []
#     for m in messages:
#         speaker = "A" if m.role.value == "user" else "B"
#         # 너무 긴 발화는 자른다 (요약 입력이 폭주하지 않게)
#         text = m.text if len(m.text) <= 600 else (m.text[:600] + "…")
#         lines.append(f"{speaker}: {text}")
#     return "\n".join(lines)

def _build_memory_input(session: Session) -> str:
    summary = session.conversation.summary
    messages = session.conversation.recent_messages

    lines = []

    lines.append("[Previous Memory Summary]")
    lines.append(summary.narrative or "(none)")
    lines.append("")

    lines.append("[Previous Structured Memory]")
    lines.append(f"Goal: {summary.structured.goal or '-'}")
    lines.append(
        "Established Facts: "
        + (", ".join(summary.structured.established_facts) or "-")
    )
    lines.append(f"Current Focus: {summary.structured.current_focus or '-'}")
    lines.append(
        "Unresolved Questions: "
        + (", ".join(summary.structured.unresolved_questions) or "-")
    )
    lines.append(f"Current Topic: {session.conversation.current_topic or '-'}")
    lines.append("")

    lines.append("[Recent Conversation]")
    if messages:
        for m in messages:
            speaker = "A" if m.role.value == "user" else "B"
            text = m.text if len(m.text) <= 600 else (m.text[:600] + "…")
            lines.append(f"{speaker}: {text}")
    else:
        lines.append("(empty)")

    return "\n".join(lines)

def _parse_memory_state(raw: str) -> Optional[dict]:
    if not raw:
        return None
    match = _JSON_RE.search(raw)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _resolve_role() -> Optional[str]:
    if LocalModelRegistry.has("memory"):
        return "memory"
    if LocalModelRegistry.has("summary"):
        return "summary"
    return None


def _apply_to_session(session: Session, parsed: dict) -> None:
    memory_state = parsed.get("memory_state", {}) or {}
    memory_summary = parsed.get("memory_summary", "") or ""

    if isinstance(memory_summary, str) and memory_summary.strip():
        session.conversation.summary.narrative = memory_summary.strip()[:800]

    goal_raw = memory_state.get("goal") or session.conversation.summary.structured.goal or ""
    key_facts_raw = memory_state.get("key_facts") or []
    unresolved_raw = memory_state.get("unresolved_refs") or []
    topic_raw = memory_state.get("topic") or ""
    anchor_raw = memory_state.get("last_resolved_anchor") or ""

    goal = str(goal_raw).strip()[:200]
    key_facts = [str(f).strip()[:200] for f in key_facts_raw if str(f).strip()][:8]
    unresolved = [str(r).strip()[:200] for r in unresolved_raw if str(r).strip()][:5]
    topic = str(topic_raw).strip()[:200]
    anchor = str(anchor_raw).strip()[:200]

    session.conversation.summary.structured = StructuredSummary(
        goal=goal,
        established_facts=key_facts,
        current_focus=topic,
        unresolved_questions=unresolved,
    )

    if topic:
        session.conversation.current_topic = topic[:60]

    if anchor:
        session.conversation.last_resolved_anchor = anchor
        session.conversation.last_referenced_item = anchor
# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------

async def update_memory_state(session: Session) -> None:
    """
    Memory State Generator 모델을 호출하여 세션 메모리(narrative + structured)를
    한 번에 갱신한다.

    호출 시점: response_finalizer 안에서 fire-and-forget으로 background 실행.
    이렇게 해서 사용자에게 응답 latency가 summary 모델 호출 시간만큼 늘어나지 않게 한다.
    """
    role = _resolve_role()
    if role is None:
        return  # 등록 없음 → graceful no-op

    logger.info("[MEMORY_UPDATE] update_memory_state_enter")

    conversation_text = _build_memory_input(session)
    if not conversation_text:
        return
    logger.info(
    "[MEMORY_UPDATE] memory_input_built | chars=%d preview=%s",
    len(conversation_text),
    conversation_text[:500].replace("\n", "\\n"),
)
    # user 메시지 본문 — system은 LocalModelRegistry가 chat template에 분리 주입
    user_prompt = f"Conversation:\n{conversation_text}"

    try:
        raw = await asyncio.to_thread(
            LocalModelRegistry.generate,
            role,
            user_prompt,
            256,
            MEMORY_SYSTEM_PROMPT,
        )
        parsed = _parse_memory_state(raw)
        if parsed:
            logger.info(
                "[MEMORY_UPDATE] apply_start | parsed=%s",
                json.dumps(parsed, ensure_ascii=False),
            )
            _apply_to_session(session, parsed)
            logger.info(
                "memory_state updated via role=%s: topic=%r facts=%d unresolved=%d",
                role,
                (parsed.get("memory_state") or {}).get("topic", ""),
                len((parsed.get("memory_state") or {}).get("key_facts") or []),
                len((parsed.get("memory_state") or {}).get("unresolved_refs") or []),
            )
            # apply 이후
            logger.info(
                "[MEMORY_UPDATE] apply_done | narrative=%s facts=%s focus=%s unresolved=%s",
                session.conversation.summary.narrative,
                session.conversation.summary.structured.established_facts,
                session.conversation.summary.structured.current_focus,
                session.conversation.summary.structured.unresolved_questions,
            )
        else:
            logger.warning(
                "memory_state_generator (role=%s): unparseable output: %r",
                role,
                (raw or "")[:200],
            )
    except Exception as e:
        logger.exception("memory_state_generator failed: %s", e)
