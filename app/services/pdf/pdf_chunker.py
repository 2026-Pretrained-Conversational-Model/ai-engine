"""
app/services/pdf/pdf_chunker.py
-------------------------------
역할: 페이지별 텍스트를 임베딩에 적합하도록
      겹침(overlap)을 포함한 청크로 분할

TODO:
    [ ] 문장 단위 경계를 고려하도록 개선 (현재는 문자 기반)
    [ ] 파서가 섹션을 지원하면 섹션 단위 청킹 적용
"""
from typing import List
from app.schemas.pdf import PdfChunk
from app.core.config import settings


def chunk_pages(file_id: str, pages: List[str]) -> List[PdfChunk]:
    chunks: List[PdfChunk] = []
    size = settings.PDF_CHUNK_SIZE
    overlap = settings.PDF_CHUNK_OVERLAP
    counter = 0
    for page_idx, text in enumerate(pages, start=1):
        if not text:
            continue
        i = 0
        while i < len(text):
            piece = text[i : i + size]
            counter += 1
            chunks.append(
                PdfChunk(
                    chunk_id=f"{file_id}_c{counter:04d}",
                    page=page_idx,
                    section="",
                    text=piece,
                )
            )
            i += size - overlap
    return chunks
