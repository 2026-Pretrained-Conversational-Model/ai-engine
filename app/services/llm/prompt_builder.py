"""
app/services/llm/prompt_builder.py
----------------------------------
역할: Answer LLM 입력을 (system, user) 두 부분으로 빌드한다.

v8.1 변경 (중국어 전환 방지 + 답변 길이 제어):
- System instruction에 "중국어(Chinese) 사용 절대 금지" 명시 추가.
  Qwen2.5는 긴 답변 생성 중 중국어로 빠지는 경향이 있어서,
  단순히 "한국어로 답하라"만으로는 부족. 금지 조항을 별도로 명시해야 함.
- "200자 이내로 간결하게 답하라" 기본 가이드 추가.
  사용자가 "자세히" / "길게" 요청하면 모델이 알아서 늘리되,
  기본값을 짧게 잡아서 긴 생성 중 언어 전환이 일어나기 전에 끊기게 유도.
"""
from typing import List, Tuple

from app.schemas.session import Session
from app.schemas.pdf import PdfChunk
from app.services.conversation.recent_window import get_recent, render_for_prompt


# ---------------------------------------------------------------------------
# System instruction (Korean, strict, with Chinese prohibition)
# ---------------------------------------------------------------------------

SYSTEM_INSTRUCTION_KO = """당신은 한국어 AI 어시스턴트입니다. 아래 규칙을 절대적으로 지키세요.

[언어 규칙 — 최우선]
- 응답은 처음부터 끝까지 반드시 한국어로만 작성합니다.
- 절대 중국어(Chinese/中文)로 전환하지 마세요. 한 글자라도 중국어를 출력하면 안 됩니다.
- 영어 단어(RAG, API, Transformer 등)는 그대로 써도 되지만, 문장 자체는 한국어여야 합니다.
- 이 규칙은 답변이 아무리 길어져도 끝까지 적용됩니다.

[답변 길이 규칙]
- 기본적으로 200자 이내로 간결하게 답하세요.
- 사용자가 "자세히", "길게", "상세하게" 요청한 경우에만 더 길게 답합니다.
- 그래도 최대 500자를 넘기지 마세요.

[정확성 규칙]
- 확실하지 않은 약어, 전문 용어, 고유명사는 추측하지 마세요.
  모르면 "정확한 정의를 확인해야 합니다"라고 답합니다.
- 대화 맥락이나 PDF 컨텍스트에 없는 사실을 만들어내지 마세요.
- 사용자가 알려준 정보(이름, 직업 등)는 그대로 기억해서 사용하세요.

[멀티턴 규칙]
- [Session Summary]와 [Recent Messages]를 먼저 읽고 문맥을 파악한 뒤 답합니다.
- "그거", "아까" 같은 지시어는 직전 대화에서 가리키는 대상을 확정한 뒤 답합니다.

[PDF 컨텍스트 규칙]
- [Retrieved PDF Context]가 있으면 그것을 근거로 답하고, 없는 내용은 지어내지 마세요.
- PDF 컨텍스트가 비어있으면 "문서에서 해당 내용을 찾지 못했습니다"라고 먼저 말하세요.

[형식 규칙]
- 불필요한 사과, 인사, 자기소개 반복은 하지 마세요.
- 리스트보다 자연스러운 문장으로 답하는 것을 선호합니다."""


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def build_user_prompt(
    session: Session,
    user_text: str,
    pdf_chunks: List[PdfChunk],
) -> str:
    """answer LLM의 user 메시지 본문을 만든다. (system 부분은 별도)"""
    summary = session.conversation.summary
    parts: list[str] = [
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


def build_prompt(
    session: Session,
    user_text: str,
    pdf_chunks: List[PdfChunk],
) -> Tuple[str, str]:
    """
    (system, user) 튜플을 반환한다.
    파이프라인은 이 둘을 LLMClient.generate(prompt=user, system=system)으로 분리 전달한다.
    """
    return SYSTEM_INSTRUCTION_KO, build_user_prompt(session, user_text, pdf_chunks)
