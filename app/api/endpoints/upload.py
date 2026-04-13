"""
app/api/endpoints/upload.py
---------------------------
역할: POST /upload — Node.js가 파일 경로 대신
      PDF 바이트를 직접 전달하는 경우(multipart) 처리

정책:
- /upload는 문서를 세션에 연결하고 즉시 background ingest를 시작한다.
- 기본 응답은 빠른 ack를 반환하며, 필요 시 `wait_until_ready=true`로
  완전한 인덱싱 완료까지 기다리는 동기 모드도 지원한다.

TODO:
    [ ] PDF가 아닌 mimetype은 거부
    [ ] 대용량 업로드 시 전체를 한 번에 읽지 않고 스트리밍 처리
"""
from fastapi import APIRouter, UploadFile, File, Form
from app.services.session.session_getter import get_or_create
from app.services.pdf.pdf_ingest import attach_pdf_to_session, ensure_pdf_ingest_started, wait_for_pdf_ready

router = APIRouter()


@router.post("")
async def upload(
    session_id: str = Form(...),
    file: UploadFile = File(...),
    wait_until_ready: bool = Form(False),
) -> dict:
    data = await file.read()
    session = await get_or_create(session_id)

    meta = await attach_pdf_to_session(session, file.filename or "upload.pdf", data)
    await ensure_pdf_ingest_started(session)

    if wait_until_ready:
        await wait_for_pdf_ready(session_id)
        session = await get_or_create(session_id)
        return {
            "file_id": meta.file_id,
            "page_count": session.pdf_state.active_pdf.page_count if session.pdf_state.active_pdf else 0,
            "ingest_status": session.pdf_state.ingest_status,
            "ready": True,
        }

    return {
        "file_id": meta.file_id,
        "page_count": 0,
        "ingest_status": "running",
        "ready": False,
    }
