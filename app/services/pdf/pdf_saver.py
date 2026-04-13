"""
app/services/pdf/pdf_saver.py
-----------------------------
역할: 업로드된 PDF를 세션별 로컬 디렉토리에 저장하고
      PdfMeta 정보를 생성

TODO:
    [ ] 매직 바이트 검사 (%PDF-로 시작하는지 확인)
    [ ] 설정된 최대 용량을 초과하는 파일은 거부
"""
import os
from datetime import datetime
from app.schemas.pdf import PdfMeta
from app.storage.file_store import save_bytes
from app.utils.id_generator import new_pdf_id


def save_pdf(session_id: str, file_name: str, data: bytes, file_hash: str = "") -> PdfMeta:
    path = save_bytes(session_id, file_name, data)
    return PdfMeta(
        file_id=new_pdf_id(),
        file_name=file_name,
        file_path=path,
        uploaded_at=datetime.utcnow(),
        file_size=os.path.getsize(path),
        page_count=0,           # filled in by parser
        mime_type="application/pdf",
        file_hash=file_hash,
    )
