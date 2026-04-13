"""
app/api/endpoints/session.py
----------------------------
역할: 세션 생명주기를 명시적으로 제어하는 엔드포인트

      - DELETE /session/{id}  → 수동 삭제 (Node.js가 WS 종료 시 호출)
      - GET    /session/{id}  → 디버깅용 세션 조회
"""
from fastapi import APIRouter, HTTPException
from app.services.session.session_manager import SessionManager
from app.services.session.session_cleaner import purge_session

router = APIRouter()


@router.delete("/{session_id}")
async def close_session(session_id: str) -> dict:
    await purge_session(session_id, reason="ws_closed")
    return {"ok": True}


@router.get("/{session_id}")
async def get_session(session_id: str) -> dict:
    s = await SessionManager.instance().get(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="session not found")
    return s.model_dump(mode="json")
