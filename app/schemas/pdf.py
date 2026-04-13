"""
app/schemas/pdf.py
------------------
역할: `pdf_state` 블록을 위한 Pydantic 모델 정의

      `pdf_index.chunks`는 메타데이터만 저장하며,
      실제 벡터는 세션별 FAISS 저장소에 별도로 보관됨
      (services/embedding/faiss_store.py 참고)

      → 따라서 세션 JSON에 벡터가 반복적으로 직렬화되지 않도록 함
"""
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field
from app.core.constants import IngestStatus


class PdfMeta(BaseModel):
    file_id: str
    file_name: str
    file_path: str
    uploaded_at: datetime
    file_size: int
    page_count: int
    mime_type: str = "application/pdf"
    file_hash: str = ""


class SectionSummary(BaseModel):
    section_title: str
    summary: str


class DocSummary(BaseModel):
    one_line: str = ""
    section_summaries: List[SectionSummary] = Field(default_factory=list)


class PdfChunk(BaseModel):
    chunk_id: str
    page: int
    section: str = ""
    text: str
    embedding_ref: Optional[str] = None   # vector lives in FAISS, this is the key


class PdfIndex(BaseModel):
    chunks: List[PdfChunk] = Field(default_factory=list)


class PdfState(BaseModel):
    active_pdf: Optional[PdfMeta] = None
    doc_summary: DocSummary = Field(default_factory=DocSummary)
    pdf_index: PdfIndex = Field(default_factory=PdfIndex)

    # ---- Ingestion state -----------------------------------------------------
    # 업로드된 PDF가 현재 어느 단계까지 준비되었는지 기록한다.
    # 라우터가 RAG 필요로 판단하면 READY가 될 때까지 기다린다.
    ingest_status: IngestStatus = IngestStatus.IDLE
    ingest_error: str = ""
