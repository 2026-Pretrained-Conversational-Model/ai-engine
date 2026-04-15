"""
app/services/llm/prompt_builder.py
----------------------------------
역할: 최종 Answer LLM 입력 프롬프트를 구성.

입력 우선순위:
1. system instruction
2. session memory (narrative + structured summary)
3. recent turns
4. document summary (있으면)
5. retrieved context chunks (있으면)
6. current user question

왜 document summary를 별도 블록으로 넣는가?
- 라우터가 RAG가 필요하다고 판단한 경우 chunk 검색 결과를 붙이기 전에
  문서 전체의 큰 주제를 짧게 알려주면 모델이 검색 결과를 해석하기 쉬워진다.
- 단, direct answer 경로에서는 문서 요약이 비어 있을 수 있으므로 optional이다.

TODO:
    [ ] 토큰 예산을 고려한 트렁케이션 처리
    [ ] 모델별 템플릿 분리 (LLM vs VLM)
"""
from typing import List
from app.schemas.session import Session
from app.schemas.pdf import PdfChunk
from app.services.conversation.recent_window import get_recent, render_for_prompt


SYSTEM_INSTRUCTION = (
    "You are a helpful assistant. Always respond in the same language as the user's question. "
    "Always interpret the user's question on top "
    "of the multi-turn conversation context. If PDF context is provided, use "
    "it as supporting evidence; otherwise rely on the conversation alone. "
    "If the evidence is insufficient, say what is missing instead of pretending."
)


def build_prompt(
    session: Session,
    user_text: str,
    pdf_chunks: List[PdfChunk],
) -> str:
    summary = session.conversation.summary
    parts: list[str] = [
        "[System Instruction]",
        SYSTEM_INSTRUCTION,
        "",
        "[Session Summary - Narrative]",
        summary.narrative or "(none)",
        "",
        "[Session Summary - Structured]",
        f"- Goal: {summary.structured.goal}",
        f"- Established Facts: {', '.join(summary.structured.established_facts) or '-'}",
        f"- Current Focus: {summary.structured.current_focus or '-'}",
        f"- Unresolved: {', '.join(summary.structured.unresolved_questions) or '-'}",
        "",
        "[Recent Messages]",
        render_for_prompt(get_recent(session)) or "(empty)",
        "",
    ]

    if session.pdf_state.doc_summary.one_line:
        parts.extend([
            "[Document Summary]",
            session.pdf_state.doc_summary.one_line,
            "",
        ])

    if pdf_chunks:
        parts.append("[Retrieved PDF Context]")
        for c in pdf_chunks:
            parts.append(f"- (p.{c.page}) {c.text}")
        parts.append("")

    parts.append("[Current User Question]")
    parts.append(user_text)
    return "\n".join(parts)
