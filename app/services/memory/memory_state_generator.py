"""
app/services/memory/memory_state_generator.py
----------------------------------------------
역할: 3턴 단위로 summary(narrative + structured)를 재생성한다.

v12 설계 — 입력 구성:
    기존 summary(narrative + structured)  +
    직전 N턴 전체 대화 (user + assistant 쌍)
    → Memory LLM이 이 둘을 보고 summary 업데이트

    N은 settings.MEMORY_UPDATE_WINDOW_TURNS (기본 3).
    user 턴 기준 3이면 user 3개 + assistant 3개 = 6개 메시지를 뽑음.

왜 "직전 3턴 전체"를 넣나:
    3턴에 한 번씩만 LLM을 돌리므로, 그 한 번에 3턴치 정보를 몰아서 넣어야
    놓치는 턴이 없음. 1턴씩 잘라서 3번 돌리는 게 아니라, 3턴 뭉치를 1번에
    LLM에게 "이 흐름을 기존 summary에 녹여줘"라고 위임하는 구조.

LocalModelRegistry role 우선순위:
    1) "memory"
    2) "summary"  (g34634/qwen2.5-3b-memory-summary-v1 등)
    둘 다 없으면 no-op.
"""
from __future__ import annotations

import asyncio
import json
import re
from typing import List, Optional

from app.core.config import settings
from app.core.logger import get_logger
from app.schemas.session import Session
from app.schemas.conversation import StructuredSummary, Message
from app.services.llm.local_registry import LocalModelRegistry

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# System prompt (Korean, fits g34634 training format)
# ---------------------------------------------------------------------------

MEMORY_SYSTEM_PROMPT = """당신은 멀티턴 대화 시스템에서 세션 메모리를 업데이트하는 역할입니다.

입력으로 다음이 주어집니다:
1) 이전까지의 메모리 요약 (previous summary)
2) 이번에 갱신해야 할 직전 N턴의 대화 (user + assistant 쌍)

당신의 일은 이전 메모리에 이번 N턴의 핵심을 녹여서 "업데이트된 메모리"를 JSON으로 출력하는 것입니다.

출력 형식 (반드시 이 JSON만 출력. 설명/코드펜스 금지):
{
  "memory_state": {
    "goal": "사용자의 전체 목적 (없으면 기존 값 유지 또는 빈 문자열)",
    "key_facts": ["사용자가 알려준 이름/직업/선호/결정 등 장기 기억할 사실"],
    "unresolved_refs": ["아직 해결되지 않은 지시어나 미완 질문"],
    "topic": "현재 대화의 세부 주제",
    "turn_count": <user 턴 수>,
    "last_resolved_anchor": "'그거', '이거' 등이 가리키는 대상"
  },
  "memory_summary": "전체 대화의 흐름을 1~2문장으로 요약 (한국어, 최대 400자)"
}

규칙:
- 반드시 유효한 JSON만 출력. 앞뒤에 다른 텍스트 절대 금지.
- 반드시 한국어로만 작성.
- 이전 메모리의 유효한 정보는 유지. 새 턴에서 정정되면 그때만 수정.
- key_facts는 "오래 유지할 핵심 사실"만. 단순 문장 요약 X.
- 이번 3턴에서 사용자가 이름/직업/선호를 말했으면 반드시 key_facts에 넣을 것."""


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_role() -> Optional[str]:
    if LocalModelRegistry.has("memory"):
        return "memory"
    if LocalModelRegistry.has("summary"):
        return "summary"
    return None


def _select_window_messages(session: Session, window_turns: int) -> List[Message]:
    """
    직전 N턴(user 기준)에 해당하는 메시지를 뽑는다.
    recent_messages를 뒤에서부터 훑어 user 메시지를 N개 만날 때까지 수집.
    그 결과를 시간순(과거→최신)으로 재정렬해서 반환.
    """
    msgs = session.conversation.recent_messages
    if not msgs:
        return []

    collected: List[Message] = []
    user_seen = 0
    for m in reversed(msgs):
        collected.append(m)
        if m.role.value == "user":
            user_seen += 1
            if user_seen >= window_turns:
                break
    collected.reverse()
    return collected


def _build_memory_input(session: Session) -> str:
    """
    Memory LLM의 user 메시지 본문.
    [Previous Summary] + [Last N Turns] 두 블록.
    """
    summary = session.conversation.summary
    window_turns = max(1, settings.MEMORY_UPDATE_WINDOW_TURNS)
    window_msgs = _select_window_messages(session, window_turns)

    lines: list[str] = []

    # ---- Previous summary ----
    lines.append("[Previous Summary - Narrative]")
    lines.append(summary.narrative or "(none)")
    lines.append("")
    lines.append("[Previous Summary - Structured]")
    lines.append(f"- goal: {summary.structured.goal or '-'}")
    lines.append(
        "- established_facts: "
        + (", ".join(summary.structured.established_facts) or "-")
    )
    lines.append(f"- current_focus: {summary.structured.current_focus or '-'}")
    lines.append(
        "- unresolved_questions: "
        + (", ".join(summary.structured.unresolved_questions) or "-")
    )
    lines.append("")

    # ---- Last N turns (this is the "3턴 내용" that must be included) ----
    lines.append(f"[Last {window_turns} User Turns — update memory based on these]")
    if window_msgs:
        for m in window_msgs:
            role = "user" if m.role.value == "user" else "assistant"
            text = m.text if len(m.text) <= 600 else (m.text[:600] + "…")
            lines.append(f"{role}: {text}")
    else:
        lines.append("(empty)")
    lines.append("")

    lines.append("위 정보를 바탕으로 memory_state JSON을 출력하세요.")
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


def _apply_to_session(session: Session, parsed: dict) -> None:
    memory_state = parsed.get("memory_state", {}) or {}
    memory_summary = parsed.get("memory_summary", "") or ""

    if isinstance(memory_summary, str) and memory_summary.strip():
        session.conversation.summary.narrative = memory_summary.strip()[:800]

    goal_raw = memory_state.get("goal") or session.conversation.summary.structured.goal or ""
    key_facts_raw = memory_state.get("key_facts") or []
    unresolved_raw = memory_state.get("unresolved_refs") or []
    topic_raw = memory_state.get("topic") or ""

    goal = str(goal_raw).strip()[:200]
    key_facts = [str(f).strip()[:200] for f in key_facts_raw if str(f).strip()][:8]
    unresolved = [str(r).strip()[:200] for r in unresolved_raw if str(r).strip()][:5]
    topic = str(topic_raw).strip()[:200]

    session.conversation.summary.structured = StructuredSummary(
        goal=goal,
        established_facts=key_facts,
        current_focus=topic,
        unresolved_questions=unresolved,
    )

    if topic:
        session.conversation.current_topic = topic[:60]


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------

async def update_memory_state(session: Session) -> None:
    """
    response_finalizer가 fire-and-forget으로 호출.

    동작:
    - 직전 N턴(user 기준) + 기존 summary를 합친 프롬프트로 Memory LLM 호출.
    - JSON 파싱 성공 시 session.conversation.summary에 반영.
    - 실패/미등록이면 graceful no-op (기존 summary 그대로 유지).
    """
    role = _resolve_role()
    if role is None:
        logger.debug("memory_state: no 'memory'/'summary' model registered, skip")
        return

    user_prompt = _build_memory_input(session)

    logger.info(
        "[MEMORY_GEN] role=%s window_turns=%d input_len=%d",
        role,
        settings.MEMORY_UPDATE_WINDOW_TURNS,
        len(user_prompt),
    )

    try:
        raw = await asyncio.to_thread(
            LocalModelRegistry.generate,
            role,
            user_prompt,
            300,                      # memory 출력 JSON 최대 토큰
            MEMORY_SYSTEM_PROMPT,
        )
        parsed = _parse_memory_state(raw)
        if parsed:
            _apply_to_session(session, parsed)
            logger.info(
                "[MEMORY_GEN] ok | topic=%r facts=%d unresolved=%d",
                (parsed.get("memory_state") or {}).get("topic", ""),
                len((parsed.get("memory_state") or {}).get("key_facts") or []),
                len((parsed.get("memory_state") or {}).get("unresolved_refs") or []),
            )
        else:
            logger.warning(
                "[MEMORY_GEN] unparseable output: %r", (raw or "")[:200]
            )
    except Exception as e:
        logger.exception("[MEMORY_GEN] failed: %s", e)
