"""
app/api/router.py
-----------------
역할: 모든 엔드포인트 라우터를 하나의 APIRouter로 통합
"""
from fastapi import APIRouter
from app.api.endpoints import chat, upload, session, rag_router

api_router = APIRouter()
api_router.include_router(chat.router,       prefix="/chat",    tags=["chat"])
api_router.include_router(upload.router,     prefix="/upload",  tags=["upload"])
api_router.include_router(session.router,    prefix="/session", tags=["session"])
api_router.include_router(rag_router.router, prefix="/rag",     tags=["rag-router"])
