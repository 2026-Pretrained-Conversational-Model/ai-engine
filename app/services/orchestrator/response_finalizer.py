"""
app/services/orchestrator/response_finalizer.py
-----------------------------------------------
역할: LLM 호출 이후 후처리 단계.
      1. assistant 메시지를 recent_messages에 추가  (매 턴)
      2. 세션 저장                                     (매 턴)
      3. user 턴 % N == 0 이면 memory update task를  (3턴마다)
         fire-and-forget으로 스케줄
      4. 메모리 제한 검사 → 초과 시 세션 삭제

v12 설계 — 한 줄 요약:
    memory read는 prompt_builder가 매 턴 수행 (summary 필드 조회).
    memory WRITE(LLM 호출해서 요약 재생성)만 3턴마다 1번, 병렬.

환경 분기 제거:
- 이전엔 MEMORY_UPDATE_ASYNC 설정으로 Colab=await / FastAPI=background 분기.
- v12부터 무조건 create_task()로 background. Colab에서도 작동하려면
  노트북이 "영구 이벤트 루프 헬퍼"를 써야 함 (notebooks/colab_loop_helper.py 참고).
- create_task가 실패(이벤트 루프 없음)하면 경고만 남기고 skip. 다음 3턴 주기에 또 시도.

Race 방지:
- 같은 session_id에 대한 memory update는 asyncio.Lock으로 직렬화.
- 다른 세션은 병렬 가능.
"""
from __future__ import annotations

import asyncio
import json
from typing import Dict, Optional, Set, Tuple

from app.core.config import settings
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
# Per-session lock (race 방지)
# ---------------------------------------------------------------------------

_session_locks: Dict[str, asyncio.Lock] = {}


def _lock_for(session_id: str) -> asyncio.Lock:
    lock = _session_locks.get(session_id)
    if lock is None:
        lock = asyncio.Lock()
        _session_locks[session_id] = lock
    return lock


# ---------------------------------------------------------------------------
# Background task registry (GC 보호)
# ---------------------------------------------------------------------------

_background_tasks: Set[asyncio.Task] = set()


def _track(task: asyncio.Task) -> None:
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


# ---------------------------------------------------------------------------
# Policy: user 턴 기준 N회마다
# ---------------------------------------------------------------------------

def _user_turn_count(session: Session) -> int:
    """recent_messages 안의 user role 메시지 수."""
    return sum(
        1 for m in session.conversation.recent_messages if m.role.value == "user"
    )


def _should_update_memory(session: Session) -> tuple[bool, str]:
    every_n = max(1, settings.MEMORY_UPDATE_EVERY_N_TURNS)
    user_turns = _user_turn_count(session)
    if user_turns > 0 and user_turns % every_n == 0:
        return True, f"every_{every_n}_turns(user_turn={user_turns})"
    return False, f"skip(user_turn={user_turns}, every={every_n})"


# ---------------------------------------------------------------------------
# Memory update work
# ---------------------------------------------------------------------------

async def _run_memory_update(session_id: str) -> None:
    """
    직전 N턴 내용 + 기존 summary로 memory_state_generator 호출.
    세션별 lock으로 같은 세션 동시 갱신 방지.
    """
    lock = _lock_for(session_id)
    async with lock:
        try:
            from app.services.memory.memory_state_generator import update_memory_state

            session = await get_or_create(session_id)
            logger.info(
                "[MEMORY_UPDATE] start | sid=%s recent=%d prev_narrative_len=%d prev_facts=%d",
                session_id,
                len(session.conversation.recent_messages),
                len(session.conversation.summary.narrative or ""),
                len(session.conversation.summary.structured.established_facts or []),
            )

            await update_memory_state(session)

            logger.info(
                "[MEMORY_UPDATE] done  | sid=%s summary=%s",
                session_id,
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
                )[:600],
            )
            await save_session(session)
        except Exception as e:
            logger.exception(
                "memory refresh failed: session=%s err=%s", session_id, e
            )


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------

async def finalize(
    session: Session,
    assistant_text: str,
    **_ignored,  # 하위호환 (pipeline에서 user_text / pdf_just_attached 주던 거 무시)
) -> Tuple[bool, Optional[str]]:
    """
    매 턴 호출. assistant append + save + (조건부) memory update task 스케줄.

    Returns:
        (expired, expire_reason)
    """
    # 1) assistant append + 저장 (이건 매 턴 항상)
    append_message(session, Role.ASSISTANT, assistant_text)
    await save_session(session)

    # 2) 3턴 정책 판정
    should_update, reason = _should_update_memory(session)
    logger.info(
        "[MEMORY_POLICY] sid=%s should_update=%s reason=%s",
        session.session_meta.session_id,
        should_update,
        reason,
    )

    # 3) 업데이트 필요하면 fire-and-forget으로 스케줄 (Colab/FastAPI 동일)
    if should_update:
        try:
            loop = asyncio.get_running_loop()
            bg_task = loop.create_task(
                _run_memory_update(session.session_meta.session_id)
            )
            _track(bg_task)
            logger.info(
                "[MEMORY_POLICY] sid=%s scheduled background task (id=%s)",
                session.session_meta.session_id,
                id(bg_task),
            )
        except RuntimeError as e:
            # 이벤트 루프가 살아있지 않은 환경 (예: Colab에서 영구 루프 헬퍼를 안 썼을 때).
            # 메모리 업데이트는 skip되지만 답변은 정상 반환됨.
            logger.warning(
                "[MEMORY_POLICY] cannot schedule background task: %s. "
                "Colab이면 notebooks의 영구 이벤트 루프 헬퍼를 사용하세요.",
                e,
            )

    # 4) 메모리 캡 체크
    session = await get_or_create(session.session_meta.session_id)
    if is_over_limit(session):
        await purge_session(session.session_meta.session_id, reason="memory_limit")
        _session_locks.pop(session.session_meta.session_id, None)
        return True, "memory_limit"

    return False, None
