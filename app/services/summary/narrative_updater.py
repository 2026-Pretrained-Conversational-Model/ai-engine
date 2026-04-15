"""
app/services/summary/narrative_updater.py
-----------------------------------------
역할: (이전) summary 모델로 narrative 줄글 요약 갱신.
      (현재 v8) **no-op**. 이유는 아래 참고.

v8 결정:
- 이 모듈이 호출하던 summary 모델(g34634/qwen2.5-3b-memory-summary-v1)은
  Memory State Generator JSON 포맷으로 학습된 모델이지, "narrative 줄글 요약"
  포맷으로 학습된 모델이 아니다. 따라서 영문 instruction 프롬프트에 대해
  환각이 심하게 일어났다 ("They become friends" 등).
- 이제 narrative 갱신은 memory_state_generator가 단일 호출(JSON)로 처리하고
  그 결과로 narrative + structured 둘 다 채운다.
- 이 파일은 호환을 위해 함수만 유지하고 본문은 비워둔다. 향후 별도 학습된
  요약 모델이 등록되면 다시 활성화할 수 있도록 hook은 남겼다.
"""
from __future__ import annotations

from app.schemas.session import Session


async def update_narrative(session: Session) -> None:
    # 의도적 no-op. memory_state_generator가 narrative까지 함께 갱신한다.
    return
