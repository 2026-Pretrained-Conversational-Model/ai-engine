"""
app/services/conversation/topic_tracker.py
------------------------------------------
역할: 최근 대화를 기반으로 `current_topic`(현재 주제) 유지 및 업데이트

TODO:
    [ ] LLM 기반 주제 변화(drift) 감지
    [ ] 디버깅을 위한 주제 히스토리 저장
"""
from app.schemas.session import Session


def update_topic(session: Session, user_text: str) -> None:
    if not session.conversation.current_topic.strip():
        session.conversation.current_topic = user_text.strip()[:60]
