"""
app/api/endpoints/chat.py
-------------------------
역할: POST /chat — 메인 턴 처리 엔드포인트
      Node.js가 WebSocket 세션의 모든 사용자 메시지마다 호출

      응답으로 답변과 expired 플래그를 반환

TODO:
    [ ] 추후 Node→Python 통신이 HTTP에서 WebSocket으로 변경될 경우
         WebSocket 버전 추가
"""
from fastapi import APIRouter
from app.schemas.request import ChatRequest
from app.schemas.response import ChatResponse
from app.services.orchestrator.pipeline import run

router = APIRouter()


@router.post("", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    return await run(req)
