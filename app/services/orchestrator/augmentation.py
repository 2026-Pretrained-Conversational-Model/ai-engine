"""
app/services/orchestrator/augmentation.py
-----------------------------------------
역할: 선택적 PDF 보강(augmentation) 단계
      세션에 활성 PDF가 있을 경우,
      (해석된) 사용자 쿼리에 대해 관련 상위 k개의 청크를 조회

      PDF가 없거나 검색 결과가 없으면 [] 반환
      → 이 경우 파이프라인은 멀티턴 기반 처리만 수행

TODO:
    [ ] 관련도가 낮은 청크를 제거하기 위한 score threshold 추가
"""
from typing import List
from app.schemas.session import Session
from app.schemas.pdf import PdfChunk
from app.services.pdf.pdf_retriever import retrieve_relevant


def maybe_attach_pdf_context(session: Session, resolved_query: str) -> List[PdfChunk]:
    if not session.pdf_state.active_pdf:
        return []
    return retrieve_relevant(session, resolved_query)
