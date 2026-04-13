"""
app/services/orchestrator/search_prep.py
--------------------------------------
역할: 독립 Rewriter 모델을 두지 않는 baseline용 검색 준비 단계.

현재 수행하는 일:
- current query의 참조 표현을 간단히 정규화
- active_document / current_topic을 질의 뒤에 덧붙여 검색 질의를 보강

TODO:
    [ ] 실제 LLM rewriter로 교체
    [ ] 한국어 조사/불용어 제거 및 query expansion 추가
"""
from app.schemas.session import Session


def prepare_search_query(session: Session, resolved_query: str) -> str:
    query = resolved_query.strip()
    additions: list[str] = []

    if session.pdf_state.active_pdf:
        additions.append(f"document:{session.pdf_state.active_pdf.file_name}")
    if session.conversation.current_topic:
        additions.append(f"topic:{session.conversation.current_topic[:80]}")

    if additions:
        return f"{query} {' '.join(additions)}"
    return query
