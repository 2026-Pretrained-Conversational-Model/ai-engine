"""
app/services/orchestrator/response_finalizer.py
-----------------------------------------------
역할: LLM 호출 이후 후처리 단계.
      1. assistant 메시지를 recent_messages에 추가
      2. 세션 저장 (응답 반환 직전 빠른 commit)
      3. **memory_state_generator를 fire-and-forget으로 background 실행**
         → 사용자 응답 latency에 summary 모델 호출 시간이 포함되지 않게 함
      4. 메모리 제한 검사 → 초과 시 세션 삭제(purge)

v8 변경 (이게 FINALIZE 21~50s latency를 해결):
- 이전: refresh_summary() (narrative + structured 두 번 동기 호출) → save → 반환.
        한 턴마다 summary 모델이 2번 직렬로 돌아 응답 시간을 크게 늘렸음.
- 현재: assistant append → save → memory 갱신은 asyncio.create_task로 detach.
        background task는 다음 턴이 시작되기 전까지 끝나면 됨.
        실패해도 사용자 응답에는 영향 없음 (로그만 남김).

주의:
- background task는 BackgroundTaskRegistry에 보관해 GC로부터 보호한다.
- 세션이 purge되면 task는 그대로 살아있다가 finally에서 무해하게 종료된다
  (LocalModelRegistry는 process-global이므로 세션 purge와 무관).
"""
from __future__ import annotations

import asyncio
import weakref
from typing import Optional, Set, Tuple

from app.core.logger import get_logger
from app.core.constants import Role
from app.schemas.session import Session
from app.services.conversation.message_appender import append_message
from app.services.session.session_updater import save_session
from app.services.session.memory_monitor import is_over_limit
from app.services.session.session_cleaner import purge_session
from app.services.session.session_getter import get_or_create

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Background task registry — fire-and-forget memory update를 GC로부터 보호
# ---------------------------------------------------------------------------

_background_tasks: Set[asyncio.Task] = set()


def _track(task: asyncio.Task) -> None:
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


async def _memory_refresh_task(session_id: str) -> None:
    """
    background에서 돌아가는 memory 갱신 작업.
    세션을 다시 로드해서(append된 assistant 메시지 포함) memory_state_generator를 1회 호출.
    """
    try:
        # local import — 순환 import 방지
        from app.services.memory.memory_state_generator import update_memory_state

        session = await get_or_create(session_id)
        await update_memory_state(session)
        await save_session(session)
        logger.debug("background memory refresh done: session=%s", session_id)
    except Exception as e:
        # 사용자 응답엔 이미 영향이 없으므로 로그만 남기고 죽지 않는다.
        logger.exception("background memory refresh failed: session=%s err=%s", session_id, e)


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------

async def finalize(session: Session, assistant_text: str) -> Tuple[bool, Optional[str]]:
    """
    LLM 호출 직후의 마무리 단계.

    Returns:
        (expired, expire_reason)
    """
    # 1) assistant 턴 append + 즉시 save (다음 턴이 일관된 상태를 보도록)
    append_message(session, Role.ASSISTANT, assistant_text)
    await save_session(session)

    # 2) memory 갱신은 background로 분리 (latency 절감의 핵심)
    #    - update_memory_state는 LocalModelRegistry에 'memory'/'summary' 등록이
    #      없으면 즉시 no-op으로 끝나므로 cost 0.
    try:
        loop = asyncio.get_running_loop()
        bg_task = loop.create_task(_memory_refresh_task(session.session_meta.session_id))
        _track(bg_task)
    except RuntimeError:
        # 이벤트 루프가 없는 경우(테스트 등)는 그냥 스킵
        logger.debug("no running loop; skipping background memory refresh")

    # 3) 메모리 캡 체크 (assistant 메시지까지 포함된 상태 기준)
    if is_over_limit(session):
        await purge_session(session.session_meta.session_id, reason="memory_limit")
        return True, "memory_limit"

    return False, None
