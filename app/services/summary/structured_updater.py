"""
app/services/summary/structured_updater.py
------------------------------------------
역할: 구조화된 요약 필드 업데이트:
      goal / established_facts / current_focus / unresolved_questions

현재 baseline 전략:
- goal          : last_user_intent를 복사
- current_focus : current_topic을 복사
- established_facts:
    active PDF, 최근 문서 요약, 최근 assistant 답변 일부를 facts로 적재
- unresolved_questions:
    마지막 user turn이 질문형이면 해당 내용을 유지

TODO:
    [ ] StructuredSummary 형식(JSON)에 맞는 실제 LLM 호출 구현
    [ ] established_facts 추가 시 신뢰도/출처 정보를 함께 보관
"""
from app.schemas.session import Session


async def update_structured(session: Session) -> None:
    structured = session.conversation.summary.structured

    structured.goal = session.conversation.last_user_intent or structured.goal
    structured.current_focus = session.conversation.current_topic or structured.current_focus

    facts = list(structured.established_facts)
    if session.pdf_state.active_pdf:
        facts.append(f"active_pdf={session.pdf_state.active_pdf.file_name}")
    if session.pdf_state.doc_summary.one_line:
        facts.append(f"doc_summary={session.pdf_state.doc_summary.one_line[:120]}")
    if session.conversation.recent_messages:
        last_assistant = next(
            (m.text for m in reversed(session.conversation.recent_messages) if m.role.value == "assistant"),
            "",
        )
        if last_assistant:
            facts.append(f"last_answer={last_assistant[:120]}")

    # 중복 제거 + 최근 5개 유지
    deduped = []
    for fact in facts:
        if fact and fact not in deduped:
            deduped.append(fact)
    structured.established_facts = deduped[-5:]

    unresolved = []
    last_user = next(
        (m.text for m in reversed(session.conversation.recent_messages) if m.role.value == "user"),
        "",
    )
    if "?" in last_user or "뭐" in last_user or "왜" in last_user or "어떻게" in last_user:
        unresolved.append(last_user[:160])
    structured.unresolved_questions = unresolved[-3:]
