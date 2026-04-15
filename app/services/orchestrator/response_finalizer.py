"""
app/services/orchestrator/response_finalizer.py
-----------------------------------------------
역할: LLM 호출 이후 후처리 단계.
      1. assistant 메시지를 recent_messages에 추가
      2. memory_state_generator 호출 (narrative + structured 갱신)
      3. 세션 저장
      4. 메모리 제한 검사 → 초과 시 세션 삭제(purge)

v8.1 변경 (메모리 비어있는 문제 해결):
- v8에서 fire-and-forget (asyncio.create_task)으로 분리했는데,
  Colab의 run_until_complete 모델에서는 pipeline_run이 리턴되면
  이벤트 루프가 멈추면서 background task가 실행 기회를 못 잡았음.
  → 모든 턴에서 narrative="", established_facts=[] 인 상태가 유지.

- 해법: memory 갱신을 다시 await으로 되돌림.
  이전(v7)에는 narrative + structured 두 번 순차 호출이라 21~50초 걸렸지만,
  v8에서 memory_state_generator 단일 호출(1회)로 통합했으므로
  await해도 5~8초. 이전의 1/6 수준.

  FastAPI 운영 환경에서 latency를 더 줄이고 싶으면
  MEMORY_UPDATE_ASYNC=true 설정으로 fire-and-forget 모드를 켤 수 있게 옵션을 남겨둠.
"""
from __future__ import annotations

import asyncio
from typing import Optional, Set, Tuple

from app.core.config import settings
from app.core.logger import get_logger
from app.core.constants import Role
from app.schemas.session import Session
from app.services.conversation.message_appender import append_message
from app.services.session.session_updater import save_session
from app.services.session.memory_monitor import is_over_limit
from app.services.session.session_cleaner import purge_session
from app.services.session.session_getter import get_or_create
import json
logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Background task registry (fire-and-forget 모드에서만 사용)
# ---------------------------------------------------------------------------

_background_tasks: Set[asyncio.Task] = set()


def _track(task: asyncio.Task) -> None:
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


async def _run_memory_update(session_id: str) -> None:
    """
    memory_state_generator를 1회 호출해서 narrative + structured를 갱신.
    세션을 다시 로드 → 갱신 → save.
    """
    try:
        from app.services.memory.memory_state_generator import update_memory_state

        session = await get_or_create(session_id)
        logger.info(
            "[MEMORY_UPDATE] finalize_start | recent_messages=%d summary_narrative_len=%d facts=%d",
            len(session.conversation.recent_messages),
            len(session.conversation.summary.narrative or ""),
            len(session.conversation.summary.structured.established_facts or []),
        )

        await update_memory_state(session)

        logger.info(
            "[MEMORY_UPDATE] finalize_done | summary=%s",
            json.dumps(
                {
                    "narrative": session.conversation.summary.narrative,
                    "structured": {
                        "goal": session.conversation.summary.structured.goal,
                        "established_facts": session.conversation.summary.structured.established_facts,
                        "current_focus": session.conversation.summary.structured.current_focus,
                        "unresolved_questions": session.conversation.summary.structured.unresolved_questions,
                    },
                },
                ensure_ascii=False,
            ),
        )

        await save_session(session)
        logger.debug("memory refresh done: session=%s", session_id)
    except Exception as e:
        logger.exception(
            "memory refresh failed: session=%s err=%s", session_id, e
        )


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------

async def finalize(
    session: Session, assistant_text: str
) -> Tuple[bool, Optional[str]]:
    """
    LLM 호출 직후의 마무리 단계.

    Returns:
        (expired, expire_reason)
    """
    # 1) assistant 턴 append
    append_message(session, Role.ASSISTANT, assistant_text)

    # 2) memory 갱신 — 기본은 await, 설정으로 fire-and-forget 가능
    memory_async = getattr(settings, "MEMORY_UPDATE_ASYNC", False)

    if memory_async:
        # 운영 FastAPI: fire-and-forget (이벤트 루프가 계속 돌아서 괜찮음)
        await save_session(session)
        try:
            loop = asyncio.get_running_loop()
            bg_task = loop.create_task(
                _run_memory_update(session.session_meta.session_id)
            )
            _track(bg_task)
        except RuntimeError:
            logger.debug("no running loop; skipping background memory refresh")
    else:
        # Colab / 테스트: await해서 메모리가 확실히 채워진 뒤 반환
        await save_session(session)
        await _run_memory_update(session.session_meta.session_id)
        # _run_memory_update가 내부에서 save_session을 또 부르므로
        # session 객체를 다시 로드하지 않아도 저장은 된 상태임.

    # 3) 메모리 캡 체크 (최신 세션 기준)
    session = await get_or_create(session.session_meta.session_id)
    if is_over_limit(session):
        await purge_session(session.session_meta.session_id, reason="memory_limit")
        return True, "memory_limit"

    return False, None
