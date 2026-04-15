"""
app/services/summary/structured_updater.py
------------------------------------------
역할: (이전) summary 모델로 4-필드 JSON 구조화 요약 갱신.
      (현재 v8) **no-op**. memory_state_generator가 일을 인계.

v8 결정:
- 이전 버전은 영문 4-필드 JSON 프롬프트를 summary 모델에 보냈는데,
  학습 포맷이 달라 모델이 prompt 일부를 echo하는 등 거의 항상 파싱 실패했다.
  ("structured updater: unparseable output ..." 경고가 매 턴 떴음.)
- v8부터는 memory_state_generator가 단일 호출로 narrative + structured를 함께
  채우며, 그 출력 포맷은 summary 모델이 실제로 학습한 memory_state JSON과 동일.
- 본문은 빈 채로 두되, 시그니처와 export는 유지한다.
"""
from __future__ import annotations

from app.schemas.session import Session


async def update_structured(session: Session) -> None:
    # 의도적 no-op. memory_state_generator가 structured까지 함께 갱신한다.
    return
