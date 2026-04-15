"""
app/services/llm/prompt_builder.py
----------------------------------
역할: Answer LLM 입력을 (system, user) 두 부분으로 빌드한다.

v11 변경 (PDF/중국어 문제 + latency 양쪽 동시 해결):

1) [Recent Messages] 윈도우를 settings.RECENT_MESSAGES_WINDOW로 **엄격 제한**.
   이전에는 session 쪽이 append만 하고 자르지 않아서 20턴 쌓이면 프롬프트가
   수천 자가 됐음 → prefill 시간 폭주.

2) [Retrieved PDF Context]의 각 chunk 앞에
   "원문 인용(자동 번역 아님)" 구분을 명시.
   그리고 system instruction에 "출력 언어는 항상 한국어, 인용은 짧게" 규칙 강화.

3) user 메시지 맨 위에 **한국어 앵커 문장**을 한 줄 추가.
   Qwen은 user 메시지 첫 토큰 언어에 크게 영향받아서, 한국어 문장이 먼저
   등장하면 중국어 전환 확률이 많이 떨어진다.

4) PDF chunk 개별 길이도 300자로 제한 (너무 긴 중국어/영어 chunk가
   그대로 프롬프트에 들어가는 걸 방지).
"""
from typing import List, Tuple

from app.core.config import settings
from app.schemas.session import Session
from app.schemas.pdf import PdfChunk


# ---------------------------------------------------------------------------
# System instruction
# ---------------------------------------------------------------------------

SYSTEM_INSTRUCTION_KO = """당신은 한국어 AI 어시스턴트입니다. 아래 규칙을 절대적으로 지키세요.

[언어 규칙 — 최우선]
- 응답은 처음부터 끝까지 반드시 한국어로만 작성합니다.
- 절대로 중국어(中文)로 전환하지 마세요. 한 글자라도 중국어가 섞이면 안 됩니다.
- PDF 컨텍스트가 영어나 다른 언어라도, 답변은 반드시 한국어로 합니다.
- 영어 고유명사(RAG, API, Transformer 등)는 그대로 쓸 수 있지만 문장은 한국어여야 합니다.
- 이 규칙은 답변이 아무리 길어져도 마지막까지 적용됩니다.

[답변 길이]
- 기본 150자 이내로 간결하게. 사용자가 "자세히/길게" 요청하면 최대 400자까지.
- 불필요한 인사, 사과, 자기소개 반복 금지.

[정확성]
- 확실하지 않은 약어/용어는 추측하지 말고 "확인이 필요합니다"라고 답합니다.
- 대화 맥락과 PDF 컨텍스트에 없는 사실을 만들어내지 마세요.
- 사용자가 알려준 정보(이름 등)는 그대로 기억해서 사용하세요.

[PDF 컨텍스트 규칙]
- [Retrieved PDF Context]가 주어지면 그것을 근거로만 답합니다.
- PDF 원문이 다른 언어라도, 답은 한국어로 번역해서 전달하세요.
- PDF 컨텍스트가 비어있으면 "문서에서 해당 내용을 찾지 못했습니다"라고 먼저 답합니다.

[멀티턴]
- [Session Summary]와 [Recent Messages]의 맥락을 먼저 파악한 뒤 답변합니다.
- "그거", "아까" 같은 지시어는 직전 대화에서 가리키는 대상을 확정합니다."""


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

_PDF_CHUNK_MAX_CHARS = 300  # chunk 하나당 프롬프트에 넣을 최대 길이


def _render_recent(session: Session) -> str:
    """최근 메시지를 window 제한 적용해 렌더. session 자체는 변경 안 함."""
    window = settings.RECENT_MESSAGES_WINDOW
    msgs = session.conversation.recent_messages[-window:]
    if not msgs:
        return "(empty)"
    lines = []
    for m in msgs:
        role = "user" if m.role.value == "user" else "assistant"
        text = m.text if len(m.text) <= 400 else (m.text[:400] + "…")
        lines.append(f"{role}: {text}")
    return "\n".join(lines)


def build_user_prompt(
    session: Session,
    user_text: str,
    pdf_chunks: List[PdfChunk],
) -> str:
    """answer LLM의 user 메시지 본문. system은 별도."""
    summary = session.conversation.summary

    # (3) 한국어 앵커 한 줄 — 첫 토큰 언어 고정용
    parts: list[str] = [
        "아래 정보를 참고해서 한국어로 답변해 주세요.",
        "",
        "[Session Summary - Narrative]",
        summary.narrative or "(none)",
        "",
        "[Session Summary - Structured]",
        f"- Goal: {summary.structured.goal or '-'}",
        f"- Established Facts: {', '.join(summary.structured.established_facts) or '-'}",
        f"- Current Focus: {summary.structured.current_focus or '-'}",
        f"- Unresolved: {', '.join(summary.structured.unresolved_questions) or '-'}",
        "",
        "[Recent Messages]",
        _render_recent(session),
        "",
    ]

    # Document Summary는 pdf_summarizer가 한국어로 만든 것만 들어감 (v11 안전)
    if session.pdf_state.doc_summary.one_line:
        parts.extend([
            "[Document Summary]",
            session.pdf_state.doc_summary.one_line,
            "",
        ])

    if pdf_chunks:
        parts.append("[Retrieved PDF Context] (원문 인용 — 다른 언어일 수 있음. 답변은 한국어로)")
        for c in pdf_chunks:
            text = c.text if len(c.text) <= _PDF_CHUNK_MAX_CHARS else (
                c.text[:_PDF_CHUNK_MAX_CHARS] + "…"
            )
            parts.append(f"- (p.{c.page}) {text}")
        parts.append("")

    parts.append("[Current User Question]")
    parts.append(user_text)
    parts.append("")
    parts.append("위 정보를 바탕으로 한국어로 답변하세요.")
    return "\n".join(parts)


def build_prompt(
    session: Session,
    user_text: str,
    pdf_chunks: List[PdfChunk],
) -> Tuple[str, str]:
    """
    (system, user) 튜플을 반환.
    호출자는 LLMClient.generate(prompt=user, system=system)으로 분리 전달.
    """
    return SYSTEM_INSTRUCTION_KO, build_user_prompt(session, user_text, pdf_chunks)
