"""
app/storage/memory_store.py
---------------------------
역할: 프로세스 전역에서 사용하는 인메모리 딕셔너리 {session_id: Session}
      동일한 session_id에 대해 동시에 들어오는 WS 메시지 간 충돌을 방지하기 위해
      asyncio.Lock으로 감싸서 관리함

TODO:
    [ ] 시간 기반 eviction이 필요해지면 dict 대신 TTL 캐시로 교체
    [ ] Prometheus 메트릭 노출: active_sessions, bytes_used
"""
import asyncio
from typing import Dict, Optional
from app.schemas.session import Session


class MemoryStore:
    def __init__(self) -> None:
        self._store: Dict[str, Session] = {}
        self._lock = asyncio.Lock()

    async def get(self, session_id: str) -> Optional[Session]:
        async with self._lock:
            return self._store.get(session_id)

    async def set(self, session_id: str, session: Session) -> None:
        async with self._lock:
            self._store[session_id] = session

    async def delete(self, session_id: str) -> None:
        async with self._lock:
            self._store.pop(session_id, None)

    async def all_ids(self) -> list[str]:
        async with self._lock:
            return list(self._store.keys())

    async def clear(self) -> None:
        async with self._lock:
            self._store.clear()
