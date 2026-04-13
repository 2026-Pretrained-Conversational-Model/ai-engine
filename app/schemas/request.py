"""
app/schemas/request.py
----------------------
역할: Node.js 오케스트레이터로부터 들어오는 요청 DTO 정의
"""
from typing import Optional
from pydantic import BaseModel


class ChatRequest(BaseModel):
    session_id: str
    user_text: str
    file_path: Optional[str] = None      # set when Node.js already saved an upload
    file_name: Optional[str] = None
    image_b64: Optional[str] = None      # for VLM path


class UploadRequest(BaseModel):
    session_id: str
    file_name: str
    # actual bytes come via multipart in the endpoint
