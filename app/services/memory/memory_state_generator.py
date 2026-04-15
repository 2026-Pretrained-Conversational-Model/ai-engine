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

MEMORY_SYSTEM_PROMPT = """You are a Memory State Generator in a multi-turn dialogue system.
Given a conversation, extract and output a structured memory state as JSON.

Output format (strictly follow this):
{
  "memory_state": {
    "key_facts": ["fact1", "fact2"],
    "unresolved_refs": ["any unclear references or pronouns"],
    "topic": "main topic of the conversation",
    "turn_count": <number of turns>
  },
  "memory_summary": "One concise sentence summarizing the conversation so far."
}

Rules:
- Output only valid JSON. No explanation, no markdown fences, no extra text.
- key_facts must be facts that the USER has stated or that have been agreed upon. Do NOT invent facts.
- If the user gave their name, profession, preference, etc., put it in key_facts.
- memory_summary should be in the SAME language as the conversation (Korean if Korean).
- topic should be a short noun phrase in the conversation's language.
- Do not write narrative fiction. Only summarize what was actually said."""


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_conversation_text(session: Session) -> str:
    """recent_messages를 'A:' / 'B:' 형식 대화 텍스트로 변환."""
    messages = session.conversation.recent_messages
    if not messages:
        return ""
    lines = []
    for m in messages:
        speaker = "A" if m.role.value == "user" else "B"
        # 너무 긴 발화는 자른다 (요약 입력이 폭주하지 않게)
        text = m.text if len(m.text) <= 600 else (m.text[:600] + "…")
        lines.append(f"{speaker}: {text}")
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
    """파싱된 memory_state JSON을 세션에 반영."""
    memory_state = parsed.get("memory_state", {}) or {}
    memory_summary = parsed.get("memory_summary", "") or ""

    # 1) narrative 갱신
    if isinstance(memory_summary, str) and memory_summary.strip():
        session.conversation.summary.narrative = memory_summary.strip()[:800]

    # 2) structured 갱신
    key_facts_raw = memory_state.get("key_facts") or []
    unresolved_raw = memory_state.get("unresolved_refs") or []
    topic_raw = memory_state.get("topic") or ""

    key_facts = [str(f).strip()[:200] for f in key_facts_raw if str(f).strip()][:8]
    unresolved = [str(r).strip()[:200] for r in unresolved_raw if str(r).strip()][:5]
    topic = str(topic_raw).strip()[:200]

    # goal은 memory_state_generator가 직접 안 쓰는 필드이므로 기존 값 보존
    prev_goal = session.conversation.summary.structured.goal
    session.conversation.summary.structured = StructuredSummary(
        goal=prev_goal,
        established_facts=key_facts,
        current_focus=topic,
        unresolved_questions=unresolved,
    )

    # 3) current_topic 동기화
    if topic:
        session.conversation.current_topic = topic[:60]


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

    conversation_text = _build_conversation_text(session)
    if not conversation_text:
        return

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
            _apply_to_session(session, parsed)
            logger.info(
                "memory_state updated via role=%s: topic=%r facts=%d unresolved=%d",
                role,
                (parsed.get("memory_state") or {}).get("topic", ""),
                len((parsed.get("memory_state") or {}).get("key_facts") or []),
                len((parsed.get("memory_state") or {}).get("unresolved_refs") or []),
            )
        else:
            logger.warning(
                "memory_state_generator (role=%s): unparseable output: %r",
                role,
                (raw or "")[:200],
            )
    except Exception as e:
        logger.exception("memory_state_generator failed: %s", e)
