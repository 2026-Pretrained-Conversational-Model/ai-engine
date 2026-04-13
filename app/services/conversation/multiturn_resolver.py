"""
app/services/conversation/multiturn_resolver.py
-----------------------------------------------
역할: 현재 사용자 질의를 재작성하여
      지시 표현("그거", "아까 말한 것", "it", "that one")을
      recent_messages + summary를 기반으로 명확하게 해석

      결과는 retrieval 및 prompt builder에 전달됨

      ※ 멀티턴의 핵심 로직
         (요구사항 #7 / #9: 멀티턴 우선)

TODO:
    [ ] 간결한 resolver 전용 프롬프트로 LLM 호출,
         (session, msg_hash) 기준 캐싱
    [ ] 검색 성능 향상을 위한 한국어 조사 제거 처리
    [ ] `resolved_query`와 `clarification_needed` 플래그 함께 반환
"""
from app.schemas.session import Session


def resolve(session: Session, user_text: str) -> str:
    """
    Returns a context-resolved version of `user_text`.
    Currently a no-op pass-through; replace with LLM call.
    """
    # TODO: implement real resolution
    return user_text
