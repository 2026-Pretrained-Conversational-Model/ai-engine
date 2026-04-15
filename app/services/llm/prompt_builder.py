"""
app/services/llm/prompt_builder.py
----------------------------------
역할: Answer LLM 입력을 (system, user) 두 부분으로 빌드한다.

v8 변경 (이게 한국어 일관성 + 환각 문제의 핵심 해결):
- 이전 버전: system instruction 영문 1문장을 user prompt 안에 [System Instruction] 블록으로
            "끼워넣어서" 모델에 전달했다. Qwen은 system role을 chat template로 받지 않으면
            instruction-following이 약해지고 한국어 → 중국어로 fallback하는 경향이 있다.
- 이번 버전:
    1) system instruction을 한국어로 강하게 다시 작성. "항상 한국어", "모르면 모른다고",
       "약어/전문용어 추측 금지" 명시.
    2) build_prompt()가 (system, user) 튜플을 반환하도록 시그니처 변경.
       호출자(pipeline.py)는 LLMClient.generate(system=..., prompt=...)로 분리 전달.
       LocalModelRegistry가 chat template의 system role로 주입한다.

호환성: 기존 호출자가 단일 문자열을 기대하는 경우를 위해 build_user_prompt()를 따로 둔다.
"""
from typing import List, Tuple

from app.schemas.session import Session
from app.schemas.pdf import PdfChunk
from app.services.conversation.recent_window import get_recent, render_for_prompt


# ---------------------------------------------------------------------------
# System instruction (Korean, strict)
# ---------------------------------------------------------------------------

SYSTEM_INSTRUCTION_KO = """당신은 한국어로 답하는 AI 어시스턴트입니다. 다음 규칙을 반드시 지키세요.

[언어 규칙]
- 응답은 항상 한국어로 작성합니다. 사용자가 한국어로 질문하면 절대 다른 언어로 답하지 않습니다.
- 영어/중국어 단어가 필요한 경우에도 문장 자체는 한국어로 유지합니다.
- 코드 블록 안의 식별자는 원어 그대로 두되, 설명은 한국어로 합니다.

[정확성 규칙]
- 확실하지 않은 약어, 전문 용어, 고유명사는 추측해서 풀어 쓰지 마세요. 모르면 "정확한 정의를 확인해야 합니다"라고 말합니다.
- 예: "RAG"가 무엇인지 100% 확실하지 않다면 멋대로 풀어 쓰지 말 것.
- 대화 맥락이나 PDF 컨텍스트에 없는 사실을 만들어내지 마세요.
- 사용자가 직전에 알려준 정보(이름, 직업 등)는 그대로 사용해야 합니다.

[멀티턴 규칙]
- [Session Summary]와 [Recent Messages]를 먼저 읽고, 사용자가 가리키는 대상이 무엇인지 파악한 뒤 답합니다.
- "그거", "아까 그거" 같은 지시어는 직전 대화의 어떤 것을 가리키는지 확정한 뒤 답합니다.

[PDF 컨텍스트 규칙]
- [Retrieved PDF Context]가 있으면 그것만을 근거로 답하고, 없는 내용을 지어내지 마세요.
- 페이지 번호가 있으면 "(p.N)" 형식으로 인용할 수 있습니다.
- PDF 컨텍스트가 비어있으면 "문서에서 해당 내용을 찾지 못했습니다"라고 말한 뒤 일반 지식으로 답할지 확인합니다.

[형식 규칙]
- 불필요한 사과, 인사, 자기소개 반복은 하지 마세요.
- 한 번에 너무 길게 답하지 말고, 사용자 질문에 정확히 대응하는 분량으로 답하세요."""


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

    이전 시그니처(단일 문자열 반환)와의 차이:
        prev: prompt = build_prompt(...)        # system이 user 안에 박혀있음
        now : sys, usr = build_prompt(...)      # 분리. chat template로 진입.
    """
    return SYSTEM_INSTRUCTION_KO, build_user_prompt(session, user_text, pdf_chunks)
