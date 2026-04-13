"""
app/services/summary/summary_orchestrator.py
--------------------------------------------
역할: narrative + structured 업데이트를 함께 실행
      각 턴마다, assistant 응답이 추가된 이후 한 번 호출됨

TODO:
    [ ] 비용 절감을 위해 스로틀링 추가 (예: 한 턴씩 건너뛰기)
"""
from app.schemas.session import Session
from app.services.summary.narrative_updater import update_narrative
from app.services.summary.structured_updater import update_structured


async def refresh_summary(session: Session) -> None:
    await update_narrative(session)
    await update_structured(session)
