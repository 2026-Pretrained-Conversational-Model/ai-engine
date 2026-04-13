"""
app/services/session/session_cleaner.py
---------------------------------------
역할: 세션을 완전히 제거(hard purge)
      → 인메모리 데이터 + 로컬 파일 + FAISS 인덱스를 모두 삭제

      메모리 제한 초과 또는 idle timeout 발생 시 호출됨
      (요구사항 #11: 세션 만료 시 메모리/파일 즉시 삭제)

TODO:
    [ ] Node.js와 상태 동기화를 위해 "session.purged" 이벤트 발행
"""
from app.services.session.session_manager import SessionManager
from app.services.embedding.faiss_store import FaissStore
from app.storage.file_store import remove_session_files
from app.core.logger import get_logger

logger = get_logger(__name__)


async def purge_session(session_id: str, reason: str) -> None:
    logger.info("Purging session=%s reason=%s", session_id, reason)
    # 1. drop FAISS index for this session
    FaissStore.instance().drop(session_id)
    # 2. remove uploaded files on disk
    remove_session_files(session_id)
    # 3. drop in-memory session blob
    await SessionManager.instance().delete(session_id)
