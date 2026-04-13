"""
app/api/endpoints/upload.py
---------------------------
역할: POST /upload — Node.js가 파일 경로 대신
      PDF 바이트를 직접 전달하는 경우(multipart) 처리

      업로드된 파일을 로컬 저장소에 저장하고,
      즉시 세션에 반영(ingest)

TODO:
    [ ] PDF가 아닌 mimetype은 거부
    [ ] 대용량 업로드 시 전체를 한 번에 읽지 않고 스트리밍 처리
"""
from fastapi import APIRouter, UploadFile, File, Form
from app.services.session.session_getter import get_or_create
from app.services.session.session_updater import save_session
from app.services.pdf.pdf_saver import save_pdf
from app.services.pdf.pdf_parser import parse_pdf
from app.services.pdf.pdf_chunker import chunk_pages
from app.services.pdf.pdf_summarizer import summarize_document
from app.services.pdf.pdf_indexer import index_chunks
from app.core.constants import ResourceType

router = APIRouter()


@router.post("")
async def upload(session_id: str = Form(...), file: UploadFile = File(...)) -> dict:
    data = await file.read()
    session = await get_or_create(session_id)

    meta = save_pdf(session_id, file.filename or "upload.pdf", data)
    page_count, pages = parse_pdf(meta.file_path)
    meta.page_count = page_count
    chunks = chunk_pages(meta.file_id, pages)

    session.pdf_state.active_pdf = meta
    session.pdf_state.pdf_index.chunks = chunks
    session.pdf_state.doc_summary = await summarize_document(pages)
    index_chunks(session_id, chunks)
    session.runtime_state.active_resource_type = ResourceType.PDF
    session.runtime_state.active_pdf_id = meta.file_id

    await save_session(session)
    return {"file_id": meta.file_id, "page_count": page_count}
