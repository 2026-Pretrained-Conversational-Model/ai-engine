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

v13 변경 (sagemaker 연동):
- LLM_BACKEND == "sagemaker"면 LLMClient(role="memory") 호출.
  sagemaker_backend가 SAGEMAKER_SUMMARY_ENDPOINT로 라우팅.
- LLM_BACKEND == "local"면 기존 LocalModelRegistry 경로 유지.
  둘 다 안 되는 경우에만 no-op.

LocalModelRegistry role 우선순위 (local 모드에서만 사용):
    1) "memory"
    2) "summary"
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
from app.services.llm.llm_client import LLMClient
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

출력 형식:
{
  "memory_state": {
    "goal": "",
    "key_facts": [],
    "unresolved_refs": [],
    "topic": "",
    "turn_count": 0,
    "last_resolved_anchor": ""
  },
  "memory_summary": ""
}

규칙:
- goal은 실제 사용자 목표가 있을 때만 작성
- 없으면 반드시 "" (빈 문자열)
- 설명문, 지시문을 절대 값으로 넣지 마라
- key_facts에는 장기 기억할 사실만 넣어라
- 최근 대화 요약문을 그대로 넣지 마라
"""

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

# memory 출력 JSON 최대 토큰
_MEMORY_MAX_NEW_TOKENS = 300


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_local_role() -> Optional[str]:
    """local 모드일 때 사용 가능한 registry role 선택."""
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
# LLM generation dispatchers
# ---------------------------------------------------------------------------

async def _generate_via_local(user_prompt: str) -> Optional[str]:
    """LocalModelRegistry 경로. 등록된 role이 없으면 None."""
    role = _resolve_local_role()
    if role is None:
        logger.debug("memory_state (local): no 'memory'/'summary' model registered, skip")
        return None

    logger.info(
        "[MEMORY_GEN] backend=local role=%s window_turns=%d input_len=%d",
        role, settings.MEMORY_UPDATE_WINDOW_TURNS, len(user_prompt),
    )
    return await asyncio.to_thread(
        LocalModelRegistry.generate,
        role,
        user_prompt,
        _MEMORY_MAX_NEW_TOKENS,
        MEMORY_SYSTEM_PROMPT,
    )


async def _generate_via_sagemaker(user_prompt: str) -> str:
    """LLMClient(role='memory') → SAGEMAKER_SUMMARY_ENDPOINT."""
    logger.info(
        "[MEMORY_GEN] backend=sagemaker role=memory window_turns=%d input_len=%d",
        settings.MEMORY_UPDATE_WINDOW_TURNS, len(user_prompt),
    )
    return await LLMClient.generate(
        prompt=user_prompt,
        role="memory",
        max_new_tokens=_MEMORY_MAX_NEW_TOKENS,
        system=MEMORY_SYSTEM_PROMPT,
    )


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
    user_prompt = _build_memory_input(session)
    backend = (settings.LLM_BACKEND or "").lower()

    try:
        if backend == "local":
            raw = await _generate_via_local(user_prompt)
            if raw is None:
                return
        else:
            # sagemaker 모드 (기본)
            raw = await _generate_via_sagemaker(user_prompt)

        # 에러 sentinel 필터링 — sagemaker_backend가 실패 시 "[LLM invocation error...]" 리턴함
        if not raw or raw.startswith("[LLM invocation error") or raw.startswith("[HTTP backend error"):
            logger.warning("[MEMORY_GEN] backend returned error sentinel: %r", raw[:200] if raw else raw)
            return

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
