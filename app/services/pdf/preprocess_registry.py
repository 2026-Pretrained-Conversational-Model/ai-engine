"""
app/services/pdf/preprocess_registry.py
--------------------------------------
역할: PDF 전처리(background ingest) 작업 레지스트리.

왜 필요한가?
- 업로드 직후 PDF 파싱/청킹/임베딩/FAISS 구축은 시간이 걸린다.
- 반면 memory 갱신과 router 판정은 먼저 수행할 수 있다.
- 따라서 같은 세션에서 이미 시작된 전처리 작업을 재사용하고,
  필요할 때만 await 하기 위한 경량 레지스트리가 필요하다.

이 구현은 프로세스 메모리 내 baseline이다.
실운영에서는 Celery/RQ/SQS 기반 워커로 분리하는 것이 더 안전하다.

TODO:
    [ ] multi-process 환경에서 Redis 기반 task registry로 교체
    [ ] 완료된 task/result에 TTL 적용
"""
from __future__ import annotations

import asyncio
from typing import Dict, Optional


class PreprocessRegistry:
    _instance: "PreprocessRegistry | None" = None

    def __init__(self) -> None:
        self._tasks: Dict[str, asyncio.Task] = {}

    @classmethod
    def instance(cls) -> "PreprocessRegistry":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def get(self, session_id: str) -> Optional[asyncio.Task]:
        task = self._tasks.get(session_id)
        if task and task.done():
            # 완료된 task는 다음 조회에서 자동 정리한다.
            self._tasks.pop(session_id, None)
        return task

    def set(self, session_id: str, task: asyncio.Task) -> None:
        self._tasks[session_id] = task

    def clear(self, session_id: str) -> None:
        self._tasks.pop(session_id, None)
