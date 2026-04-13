"""
app/utils/time_utils.py
-----------------------
역할: idle-timeout 및 정리(cleanup) 로직에서 사용되는 시간 관련 유틸 함수
"""
from datetime import datetime, timedelta


def now() -> datetime:
    return datetime.utcnow()


def minutes_since(ts: datetime) -> float:
    return (now() - ts).total_seconds() / 60.0


def is_idle(last_accessed: datetime, max_minutes: int) -> bool:
    return minutes_since(last_accessed) >= max_minutes
