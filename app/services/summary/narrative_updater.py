"""
app/services/summary/narrative_updater.py
-----------------------------------------
역할: (기존 내러티브 요약 + 최근 메시지 + 새로운 턴)을 기반으로
      내러티브 요약을 점진적으로 업데이트
      (설계 문서 Section 5 참고)

TODO:
    [ ]  요약 프롬프트를 사용하여 LLM 클라이언트와 연동
    [ ] 변화량(delta)이 토큰 기준 이하일 경우 업데이트 생략
"""
from app.schemas.session import Session


async def update_narrative(session: Session) -> None:
    # TODO: call LLM. For now, append a minimal trace.
    # Keep narrative bounded so it doesn't blow up SESSION_MAX_BYTES.
    pass
