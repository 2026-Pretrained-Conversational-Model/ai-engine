"""
app/services/session/session_manager.py
---------------------------------------
역할: MemoryStore를 감싸는 싱글톤 파사드
      다른 서비스들은 세션을 읽고/쓰기 위해 반드시 이 클래스를 통해야 하며,
      MemoryStore를 직접 접근하면 안 됨

      이를 통해 추후 Redis 등으로 교체하는 작업을 쉽게 만들 수 있음

TODO:
    [ ] 메트릭 수집을 위한 훅 추가
    [ ] on_evict 콜백 레지스트리 추가 (FAISS 정리 연동용)
"""
from typing import Optional
from app.schemas.session import Session
from app.storage.memory_store import MemoryStore
from app.core.logger import get_logger

logger = get_logger(__name__)


class SessionManager:
    _instance: "SessionManager | None" = None

    def __init__(self) -> None:
        self._store = MemoryStore()

    @classmethod
    def instance(cls) -> "SessionManager":
        if cls._instance is None:
            cls._instance = cls()
            logger.info("SessionManager initialized")
        return cls._instance

    async def get(self, session_id: str) -> Optional[Session]:
        return await self._store.get(session_id)

    async def save(self, session: Session) -> None:
        await self._store.set(session.session_meta.session_id, session)

    async def delete(self, session_id: str) -> None:
        await self._store.delete(session_id)

    def purge_all(self) -> None:
        # called on shutdown
        import asyncio
        try:
            asyncio.get_event_loop().run_until_complete(self._store.clear())
        except RuntimeError:
            pass
