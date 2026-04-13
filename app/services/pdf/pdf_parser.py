"""
app/services/pdf/pdf_parser.py
------------------------------
역할: 저장된 PDF에서 페이지별 텍스트를 추출 (pypdf 사용)

TODO:
    [ ] 스캔된 PDF를 감지하여 OCR로 분기 처리
    [ ] 섹션 단위 구조 추출 (현재는 단순 페이지 단위 텍스트만 반환)
"""
from typing import List, Tuple
from pypdf import PdfReader


def parse_pdf(file_path: str) -> Tuple[int, List[str]]:
    """Returns (page_count, [page_text, ...])."""
    reader = PdfReader(file_path)
    pages = [(p.extract_text() or "") for p in reader.pages]
    return len(pages), pages
