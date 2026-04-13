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

TODO:
    [ ] 요약 프롬프트를 사용하여 LLM 클라이언트와 연동
    [ ] 변화량(delta)이 토큰 기준 이하일 경우 업데이트 생략
"""
from app.schemas.session import Session


async def update_narrative(session: Session) -> None:
    recent = session.conversation.recent_messages[-2:]
    if not recent:
        return

    parts = [f"{m.role.value}: {m.text[:120]}" for m in recent]
    delta = " | ".join(parts)
    existing = session.conversation.summary.narrative.strip()

    merged = f"{existing} || {delta}" if existing else delta
    # 무한히 커지지 않도록 최근 1000자만 유지한다.
    session.conversation.summary.narrative = merged[-1000:]
