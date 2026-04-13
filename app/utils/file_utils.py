"""
app/utils/file_utils.py
-----------------------
역할: storage / pdf 서비스에서 공통으로 사용하는
      간단한 파일 시스템 유틸 함수 모음
"""
import os


def safe_size(path: str) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def exists(path: str) -> bool:
    return os.path.exists(path)
