"""
app/services/conversation/multiturn_resolver.py
-----------------------------------------------
역할: 현재 사용자 질의를 재작성하여
      지시 표현("그거", "아까 말한 것", "it", "that one")을
      recent_messages + summary를 기반으로 명확하게 해석

      결과는 retrieval 및 prompt builder에 전달됨

      ※ baseline에서는 완전한 LLM 재작성 대신,
         최근 assistant/user 발화를 이용한 가벼운 치환만 수행한다.
         이 모듈은 추후 Search Prep 단계의 실제 LLM rewriter로 대체 가능하다.

TODO:
    [ ] resolver 전용 few-shot LLM prompt 적용
    [ ] `resolved_query`와 `clarification_needed`를 함께 반환하도록 확장
"""
from app.schemas.session import Session


REFERENCE_TOKENS = ["그거", "그 부분", "그 논문", "그 표", "아까", "that", "it"]


def resolve(session: Session, user_text: str) -> str:
    text = user_text.strip()
    lowered = text.lower()

    recent = (
        session.conversation.recent_messages[-4:]
        if session.conversation.recent_messages
        else []
    )
    last_context = " ".join(m.text for m in recent).strip()

    anchor_candidates = [
        session.conversation.last_resolved_anchor,
        session.conversation.last_referenced_item,
        session.conversation.summary.structured.current_focus,
        session.conversation.current_topic,
    ]
    anchor = next((a for a in anchor_candidates if a and a.strip()), "")

    if any(token in lowered for token in REFERENCE_TOKENS):
        if anchor:
            return f"{text} (reference: {anchor[:200]})"
        if last_context:
            return f"{text} (reference: {last_context[:200]})"

    return text