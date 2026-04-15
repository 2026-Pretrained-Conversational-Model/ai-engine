"""
app/services/summary/summary_orchestrator.py
--------------------------------------------
역할: (이전) narrative + structured 갱신을 함께 호출.
      (현재 v8) no-op 래퍼. 호환성을 위해 시그니처만 유지.

v8: 메모리 갱신은 모두 memory_state_generator로 일원화되었음.
    response_finalizer는 더 이상 refresh_summary()를 호출하지 않음.
"""
from app.schemas.session import Session
from app.services.summary.narrative_updater import update_narrative
from app.services.summary.structured_updater import update_structured


async def refresh_summary(session: Session) -> None:
    # 두 함수 모두 no-op. 호출돼도 아무 일도 일어나지 않는다.
    await update_narrative(session)
    await update_structured(session)
