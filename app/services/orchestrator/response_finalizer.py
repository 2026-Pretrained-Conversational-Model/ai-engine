"""
app/services/orchestrator/response_finalizer.py
-----------------------------------------------
역할: LLM 호출 이후 후처리 단계:
      1. assistant 메시지를 recent_messages에 추가
      2. narrative + structured 요약 갱신
      3. 세션 저장
      4. 메모리 제한 검사 → 초과 시 세션 삭제(purge)

      (expired, expire_reason)을 반환

TODO:
    [ ] 지연 시간 감소를 위해 summary 갱신을 비동기 fire-and-forget 방식으로 처리
"""
from typing import Tuple, Optional
from app.schemas.session import Session
from app.core.constants import Role
from app.services.conversation.message_appender import append_message
from app.services.summary.summary_orchestrator import refresh_summary
from app.services.session.session_updater import save_session
from app.services.session.memory_monitor import is_over_limit
from app.services.session.session_cleaner import purge_session


async def finalize(session: Session, assistant_text: str) -> Tuple[bool, Optional[str]]:
    append_message(session, Role.ASSISTANT, assistant_text)
    await refresh_summary(session)
    await save_session(session)

    if is_over_limit(session):
        await purge_session(session.session_meta.session_id, reason="memory_limit")
        return True, "memory_limit"
    return False, None
