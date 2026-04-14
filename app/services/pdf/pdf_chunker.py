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
# RecursiveCharacterTextSplitter 추가
from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_pages(file_id: str, pages: List[str]) -> List[PdfChunk]:
    chunks: List[PdfChunk] = []
    counter = 0

    # [수정] RecursiveCharacterTextSplitter 사용
    # 변경: splitter.split_text(text)
    #       → separators 우선순위대로 자연스러운 경계에서 분리
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.PDF_CHUNK_SIZE,
        chunk_overlap=settings.PDF_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    for page_idx, text in enumerate(pages, start=1):
        if not text:
            continue
        
        # [수정] splitter로 페이지 텍스트 분할
        pieces = splitter.split_text(text)

        for piece in pieces:
            if not piece.strip():
                continue
            counter += 1
            chunks.append(
                PdfChunk(
                    chunk_id=f"{file_id}_c{counter:04d}",
                    page=page_idx,
                    section="",
                    text=piece,
                )
            )

    return chunks
