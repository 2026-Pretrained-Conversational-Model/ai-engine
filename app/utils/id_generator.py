"""
app/utils/id_generator.py
-------------------------
역할: PDF와 청크에 대한 안정적인 고유 ID 생성
      세션 ID는 Node.js에서 전달되며, 여기서 생성하지 않음
"""
import uuid


def new_pdf_id() -> str:
    return f"pdf_{uuid.uuid4().hex[:8]}"
