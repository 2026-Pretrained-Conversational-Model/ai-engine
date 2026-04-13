"""
app/services/conversation/intent_extractor.py
---------------------------------------------
역할: 현재 사용자 메시지를 1줄의 `last_user_intent`로 요약

      현재는 간단한 휴리스틱 방식 사용,
      이후에는 LLM 기반으로 전환 예정

TODO:
    [ ] 소형 LLM 호출로 휴리스틱 대체 (또는 메인 LLM 재사용)
    [ ] (session, message_hash) 기준으로 intent 캐싱
"""
from app.schemas.session import Session


def extract_intent(session: Session, user_text: str) -> str:
    # naive baseline: first sentence, capped
    first = user_text.strip().split("\n")[0]
    intent = first[:120]
    session.conversation.last_user_intent = intent
    return intent
