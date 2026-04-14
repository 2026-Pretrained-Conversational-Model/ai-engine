"""
app/services/summary/narrative_updater.py
-----------------------------------------
역할: (기존 내러티브 요약 + 최근 메시지 + 새로운 턴)을 기반으로
      내러티브 요약을 점진적으로 업데이트
      (설계 문서 Section 5 참고)

현재 baseline 전략:
- 매 턴 전체를 다시 요약하지 않고,
  최근 user/assistant 교환을 한 줄 trace로 압축한다.
- summary는 메모리 절약을 위해 길이 제한을 둔다.

TODO: 김예슬
    [ ] 요약 프롬프트를 사용하여 LLM 클라이언트와 연동
    [ ] 변화량(delta)이 토큰 기준 이하일 경우 업데이트 생략
    
    변경점:
    - 이전 버전: no-op stub
    - 이번 버전: LocalModelRegistry에 "summary" role이 등록돼 있으면 LLM 호출,
                없으면 기존과 동일하게 no-op (summary 필드 유지).

"""
from __future__ import annotations

import asyncio

from app.core.logger import get_logger
from app.schemas.session import Session
from app.services.llm.local_registry import LocalModelRegistry

logger = get_logger(__name__)


_NARRATIVE_PROMPT = """You maintain a rolling narrative summary of a multi-turn chatbot conversation.

Previous narrative:
{prev}

Latest turns:
{turns}

Write the UPDATED narrative summary in 2-3 Korean sentences. Keep the main thread. Do not invent facts. Output only the summary text, nothing else."""


async def update_narrative(session: Session) -> None:
    if not LocalModelRegistry.has("summary"):
        return  # graceful no-op
    # recent = session.conversation.recent_messages[-2:] [이전 버전]
    recent = session.conversation.recent_messages[-4:] 
    if not recent:
        return
    
    # 이전 버전: 간단한 delta 생성 후 기존 narrative에 덧붙이는 방식
    # parts = [f"{m.role.value}: {m.text[:120]}" for m in recent]
    # delta = " | ".join(parts)
    # existing = session.conversation.summary.narrative.strip()
    

    turns_text = "\n".join(f"{m.role.value}: {m.text}" for m in recent)
    prompt = _NARRATIVE_PROMPT.format(
        prev=session.conversation.summary.narrative or "(none)",
        turns=turns_text,
    )

    try:
        new_text = await asyncio.to_thread(
            LocalModelRegistry.generate, "summary", prompt, 220
        )
        if new_text:
            
            # session.conversation.summary.narrative = merged[-1000:] [이전 버전]

            # 너무 길어지지 않도록 상한 — SESSION_MAX_BYTES 방어
            session.conversation.summary.narrative = new_text.strip()[:800]
    except Exception as e:
        logger.exception("narrative updater failed: %s", e)



# async def update_narrative(session: Session) -> None:


#     parts = [f"{m.role.value}: {m.text[:120]}" for m in recent]
#     delta = " | ".join(parts)
#     existing = session.conversation.summary.narrative.strip()

#     merged = f"{existing} || {delta}" if existing else delta
#     # 무한히 커지지 않도록 최근 1000자만 유지한다.
#     session.conversation.summary.narrative = merged[-1000:]

