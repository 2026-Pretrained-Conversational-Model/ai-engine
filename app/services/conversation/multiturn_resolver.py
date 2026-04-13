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


def resolve(session: Session, user_text: str) -> str:
    text = user_text.strip()
    lowered = text.lower()

    # 최근 메시지 중 가장 마지막 assistant/user 발화를 참조 후보로 사용한다.
    recent = session.conversation.recent_messages[-2:] if session.conversation.recent_messages else []
    last_context = " ".join(m.text for m in recent).strip()

    # resolved_refs가 있으면 우선 사용한다.
    resolved_refs = session.conversation.summary.structured.established_facts
    ref_hint = resolved_refs[-1] if resolved_refs else session.conversation.current_topic

    if any(token in lowered for token in ["그거", "그 부분", "그 논문", "그 표", "아까", "that", "it"]):
        anchor = ref_hint or last_context
        if anchor:
            return f"{text} (reference: {anchor[:200]})"

    return text
