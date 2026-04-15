"""
app/services/memory/memory_state_generator.py
----------------------------------------------
역할: 매 턴 종료 후 대화 내용을 읽고 memory_state JSON을 생성하여
      세션의 conversation.summary에 반영한다.

파이프라인에서의 위치:
    사용자 입력 → Router → LLM/VLM → [Memory State Generator] → 세션 저장

LocalModelRegistry에 "memory" role로 등록된 모델을 사용한다.
등록되어 있지 않으면 graceful no-op으로 동작한다.

출력 형식:
{
  "memory_state": {
    "key_facts": ["fact1", "fact2"],
    "unresolved_refs": ["..."],
    "topic": "...",
    "turn_count": 3
  },
  "memory_summary": "한 문장 요약"
}
"""
from __future__ import annotations

import asyncio
import json
import re
from typing import Optional

from app.core.logger import get_logger
from app.schemas.session import Session
from app.schemas.conversation import StructuredSummary

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are a Memory State Generator in a multi-turn dialogue system.
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

Output only valid JSON. No explanation, no markdown."""

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _generate_with_system(role: str, system: str, user: str, max_tokens: int) -> str:
    """system + user 메시지를 chat template으로 구성해서 모델 호출"""
    from app.services.llm.local_registry import LocalModelRegistry
    import torch

    entry = LocalModelRegistry.get(role)
    tok = entry.tokenizer
    model = entry.model

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    input_ids = tok.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(entry.device)

    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )

    new_tokens = out[0, input_ids.shape[1]:]
    return tok.decode(new_tokens, skip_special_tokens=True).strip()


def _build_conversation_text(session: Session) -> str:
    """recent_messages를 대화 텍스트로 변환"""
    messages = session.conversation.recent_messages
    if not messages:
        return ""
    lines = []
    for m in messages:
        role = "A" if m.role.value == "user" else "B"
        lines.append(f"{role}: {m.text}")
    return "\n".join(lines)


def _parse_memory_state(raw: str) -> Optional[dict]:
    """모델 출력에서 JSON 파싱"""
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
    """파싱된 memory_state를 세션에 반영"""
    memory_state = parsed.get("memory_state", {})
    memory_summary = parsed.get("memory_summary", "")

    if memory_summary:
        session.conversation.summary.narrative = memory_summary[:800]

    key_facts  = memory_state.get("key_facts", [])
    unresolved = memory_state.get("unresolved_refs", [])
    topic      = memory_state.get("topic", "")

    session.conversation.summary.structured = StructuredSummary(
        goal=session.conversation.summary.structured.goal,
        established_facts=[str(f)[:200] for f in key_facts[:5]],
        current_focus=str(topic)[:200],
        unresolved_questions=[str(r)[:200] for r in unresolved[:5]],
    )

    if topic:
        session.conversation.current_topic = str(topic)[:60]


async def update_memory_state(session: Session) -> None:
    """
    Memory State Generator 모델을 호출하여 세션 메모리를 업데이트한다.
    "memory" role이 등록되어 있지 않으면 graceful no-op.
    """
    from app.services.llm.local_registry import LocalModelRegistry

    if not LocalModelRegistry.has("memory"):
        return

    conversation_text = _build_conversation_text(session)
    if not conversation_text:
        return

    try:
        raw = await asyncio.to_thread(
            _generate_with_system,
            "memory",
            SYSTEM_PROMPT,
            f"Conversation:\n{conversation_text}",
            256,
        )
        parsed = _parse_memory_state(raw)
        if parsed:
            _apply_to_session(session, parsed)
            logger.info(
                "memory_state updated: topic=%s facts=%d",
                parsed.get("memory_state", {}).get("topic", ""),
                len(parsed.get("memory_state", {}).get("key_facts", [])),
            )
        else:
            logger.warning(
                "memory_state_generator: unparseable output: %r", (raw or "")[:120]
            )
    except Exception as e:
        logger.exception("memory_state_generator failed: %s", e)