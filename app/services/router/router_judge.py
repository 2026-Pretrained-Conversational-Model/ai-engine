"""
app/services/router/router_judge.py
----------------------------------
역할: 현재 턴에서 어떤 후속 경로를 탈지 결정하는 라우터.

중요한 설계 포인트:
- 이 baseline은 '별도 학습된 router classifier'를 두지 않는다.
- 대신 lightweight LLM judge를 흉내내는 규칙 기반 baseline + TODO를 제공한다.
- 추후에는 같은 입출력 스키마를 유지한 채 실제 LLM judge 또는
  자동 라벨로 학습한 소형 classifier로 교체할 수 있다.

라우터는 PDF 전처리 완료를 기다리지 않고 먼저 판정한다.
다만 RETRIEVE_DOC / SEARCH_PREP_THEN_RETRIEVE가 선택되면
파이프라인이 이후 단계에서 ingest 완료까지 await 한다.

TODO:
    [ ] few-shot LLM judge 프롬프트 기반으로 교체
    [ ] LLM judge 로그를 축적하여 소형 classifier 학습 데이터로 전환
"""
from __future__ import annotations

from app.core.constants import RouterDecision
from app.schemas.session import Session


_DOC_HINTS = [
    "논문", "pdf", "문서", "페이지", "표", "figure", "table", "section", "캡션", "첨부",
    "파일", "본문", "이 문서", "이 논문", "이 pdf",
]
_REFERENCE_HINTS = [
    "그거", "그 부분", "그 논문", "그 표", "그 그림", "아까", "방금", "이거", "저거",
    "that", "it", "those", "above", "previous",
]
_CLARIFY_HINTS = ["뭐야?", "왜?", "다시", "설명해줘"]


def judge_route(session: Session, query: str, has_attachment: bool = False) -> RouterDecision:
    q = query.strip().lower()

    active_pdf = session.pdf_state.active_pdf is not None
    summary_exists = bool(session.pdf_state.doc_summary.one_line)

    # 1) 첨부 문서나 활성 PDF를 직접 참조하는 질문은 검색 우선.
    if active_pdf or has_attachment:
        if any(token in q for token in _DOC_HINTS):
            return RouterDecision.RETRIEVE_DOC
        if any(token in q for token in _REFERENCE_HINTS):
            return RouterDecision.SEARCH_PREP_THEN_RETRIEVE

    # 2) 문서가 없어도 강한 참조 표현만 있고 최근 문맥이 짧으면 clarification.
    if any(token in q for token in _REFERENCE_HINTS):
        if not session.conversation.recent_messages and not summary_exists:
            return RouterDecision.ASK_CLARIFICATION
        return RouterDecision.DIRECT_ANSWER

    # 3) 질문이 지나치게 짧고 첨부도 문맥도 없으면 clarification.
    if len(q) <= 6 and not active_pdf and not session.conversation.recent_messages:
        return RouterDecision.ASK_CLARIFICATION

    # 4) 기본값은 direct answer.
    return RouterDecision.DIRECT_ANSWER
