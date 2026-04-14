"""
app/services/orchestrator/augmentation.py
-----------------------------------------
역할: 선택적 PDF 보강(augmentation) 단계
      세션에 활성 PDF가 있을 경우,
      (해석된) 사용자 쿼리에 대해 관련 상위 k개의 청크를 조회

      PDF가 없거나 검색 결과가 없으면 [] 반환
      → 이 경우 파이프라인은 멀티턴 기반 처리만 수행

변경사항 (baseline 반영):
    [수정] get_pdf_context_string() 추가
          변경: context 문자열 반환 → prompt_builder에서 바로 사용 가능

TODO:
    [ ] 관련도가 낮은 청크를 제거하기 위한 score threshold 추가
    [ ] 하이브리드 검색 적용 (BM25 + Dense Retrieval)
"""
from typing import List
from app.schemas.session import Session
from app.schemas.pdf import PdfChunk
from app.services.pdf.pdf_retriever import retrieve_relevant, get_context


def maybe_attach_pdf_context(session: Session, resolved_query: str) -> List[PdfChunk]:
    if not session.pdf_state.active_pdf:
        return []
    return retrieve_relevant(session, resolved_query)

# get_pdf_context_string() 추가
# 변경: get_pdf_context_string() → 출처 포함 context 문자열 반환
#       → prompt_builder.build()에서 바로 사용 가능
def get_pdf_context_string(session: Session, resolved_query: str) -> str:
    """
    [기능] prompt_builder에 넘겨줄 context 문자열 반환
    - 활성 PDF 없으면 빈 문자열 반환
    - pdf_retriever.get_context() 호출 → 출처 포함 포맷

    Args:
        session       : 현재 세션 객체
        resolved_query: multiturn_resolver가 해석한 쿼리

    Returns:
        str: 출처 포함 context 문자열 또는 빈 문자열

    사용 예시:
        context = get_pdf_context_string(session, resolved_query)
        prompt  = prompt_builder.build(query, context)
    """
    if not session.pdf_state.active_pdf:
        return ""
    return get_context(session, resolved_query)
