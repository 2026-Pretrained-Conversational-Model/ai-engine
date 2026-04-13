"""
app/storage/file_store.py
-------------------------
역할: 업로드된 PDF를 위한 로컬 파일 시스템 유틸
      각 세션은 고유한 하위 디렉토리를 가짐:
      {LOCAL_FILE_DIR}/{session_id}/
      따라서 정리는 `rmtree(session_dir)`로 간단히 처리 가능

TODO:
    [ ] 파일 쓰기 전에 디스크 용량(quota) 검사 추가
    [ ] 고아 디렉토리 정리를 위한 백그라운드 스위퍼 추가
"""
import os
import shutil
from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)


def ensure_storage_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def session_dir(session_id: str) -> str:
    path = os.path.join(settings.LOCAL_FILE_DIR, session_id)
    os.makedirs(path, exist_ok=True)
    return path


def save_bytes(session_id: str, file_name: str, data: bytes) -> str:
    target = os.path.join(session_dir(session_id), file_name)
    with open(target, "wb") as f:
        f.write(data)
    return target


def remove_session_files(session_id: str) -> None:
    path = os.path.join(settings.LOCAL_FILE_DIR, session_id)
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)
        logger.info("Removed local files for session=%s", session_id)
