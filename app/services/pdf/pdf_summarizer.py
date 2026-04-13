"""
app/services/pdf/pdf_summarizer.py
----------------------------------
역할: 새로 파싱된 PDF에 대해 doc_summary 생성
      (one_line + section_summaries 포함)
      LLM 클라이언트를 사용

TODO:
    [ ] 긴 PDF를 위한 map-reduce 요약 방식 적용
    [ ] 페이지 수가 3 미만일 경우 섹션 요약 생략
"""
from typing import List
from app.schemas.pdf import DocSummary, SectionSummary


async def summarize_document(pages: List[str]) -> DocSummary:
    # TODO: real LLM call. Stub keeps the pipeline runnable.
    one_line = (pages[0][:140] if pages else "")
    return DocSummary(one_line=one_line, section_summaries=[])
