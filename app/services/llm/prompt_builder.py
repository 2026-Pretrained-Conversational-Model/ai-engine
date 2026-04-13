"""
app/services/llm/prompt_builder.py
----------------------------------
역할: 설계 문서 Section 7에 따라 최종 프롬프트를 구성

    [System Instruction]
    [세션 요약 - 내러티브]
    [세션 요약 - 구조화]
    [최근 메시지]
    [PDF 컨텍스트]   (존재할 경우에만 포함)
    [현재 사용자 질문]

TODO:
    [ ] 토큰 예산을 고려한 트렁케이션 처리
         (가장 오래된 recent부터 제거 → 그 다음 narrative 축소)
    [ ] 모델별 템플릿 분리 (LLM vs VLM)
"""
from typing import List
from app.schemas.session import Session
from app.schemas.pdf import PdfChunk
from app.services.conversation.recent_window import get_recent, render_for_prompt


SYSTEM_INSTRUCTION = (
    "You are a helpful assistant. Always interpret the user's question on top "
    "of the multi-turn conversation context. If PDF context is provided, use "
    "it as supporting evidence; otherwise rely on the conversation alone."
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

    if pdf_chunks:
        parts.append("[PDF Context]")
        for c in pdf_chunks:
            parts.append(f"- (p.{c.page}) {c.text}")
        parts.append("")

    parts.append("[Current User Question]")
    parts.append(user_text)
    return "\n".join(parts)
