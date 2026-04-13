"""
app/services/summary/structured_updater.py
------------------------------------------
역할: 구조화된 요약 필드 업데이트:
      goal / established_facts / current_focus / unresolved_questions

TODO:
    [ ] StructuredSummary 형식(JSON)에 맞는 LLM 호출 구현
    [ ] established_facts 추가 시 중복 제거 처리
"""
from app.schemas.session import Session


async def update_structured(session: Session) -> None:
    # TODO: call LLM with JSON-mode prompt
    pass
