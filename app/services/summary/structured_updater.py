"""
app/services/summary/structured_updater.py
------------------------------------------
역할: structured summary(JSON) 증분 갱신.
      goal / established_facts / current_focus / unresolved_questions 4개 필드.

변경점:
- 이전 버전: no-op stub
- 이번 버전: LocalModelRegistry에 "summary" role이 등록돼 있으면 LLM 호출.
             모델 응답을 JSON으로 파싱해서 StructuredSummary 필드에 반영.
             파싱 실패 시 기존 값 유지.
"""
from __future__ import annotations

import asyncio
import json
import re

from app.core.logger import get_logger
from app.schemas.session import Session
from app.schemas.conversation import StructuredSummary
from app.services.llm.local_registry import LocalModelRegistry

logger = get_logger(__name__)


_STRUCTURED_PROMPT = """You maintain a structured JSON summary of a multi-turn chatbot conversation.

Previous structured summary:
{prev_json}

Latest turns:
{turns}

Return an UPDATED JSON object with EXACTLY these keys:
- "goal": one sentence describing the overall goal of the conversation
- "established_facts": list of short facts already agreed upon (dedup)
- "current_focus": one phrase describing the sub-topic currently being discussed
- "unresolved_questions": list of open questions the user still needs answered

Rules:
- Output valid JSON only, no markdown fences, no extra text.
- Keep each list to at most 5 items.
- Use Korean for the field values when the conversation is in Korean."""


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> dict | None:
    if not text:
        return None
    match = _JSON_RE.search(text)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


async def update_structured(session: Session) -> None:
    if not LocalModelRegistry.has("summary"):
        return

    recent = session.conversation.recent_messages[-4:]
    if not recent:
        return

    turns_text = "\n".join(f"{m.role.value}: {m.text}" for m in recent)
    prev = session.conversation.summary.structured
    prev_json = json.dumps(
        {
            "goal": prev.goal,
            "established_facts": prev.established_facts,
            "current_focus": prev.current_focus,
            "unresolved_questions": prev.unresolved_questions,
        },
        ensure_ascii=False,
    )
    prompt = _STRUCTURED_PROMPT.format(prev_json=prev_json, turns=turns_text)

    try:
        raw = await asyncio.to_thread(
            LocalModelRegistry.generate, "summary", prompt, 400
        )
        parsed = _extract_json(raw)
        if not parsed:
            logger.warning("structured updater: unparseable output %r", (raw or "")[:120])
            return

        session.conversation.summary.structured = StructuredSummary(
            goal=str(parsed.get("goal", prev.goal))[:200],
            established_facts=[str(x)[:200] for x in (parsed.get("established_facts") or [])][:5],
            current_focus=str(parsed.get("current_focus", prev.current_focus))[:200],
            unresolved_questions=[
                str(x)[:200] for x in (parsed.get("unresolved_questions") or [])
            ][:5],
        )
    except Exception as e:
        logger.exception("structured updater failed: %s", e)
