"""
app/api/endpoints/rag_router.py
--------------------------------
역할: QwenRagRouter를 HTTP API로 노출하는 엔드포인트 모듈.

    기존 router.py(독립 FastAPI 앱)의 /health, /rag/decide 엔드포인트를
    ai-engine API 구조에 맞게 통합한 모듈이다.

    QwenRagRouter는 매 요청마다 재생성하지 않고,
    LocalModelRegistry에 "router" role이 등록돼 있으면 그것을 재사용한다.
    등록되지 않은 경우 이 엔드포인트는 503을 반환한다.

Endpoints:
    GET  /rag/health   - 라우터 모델 로딩 상태 확인
    POST /rag/decide   - RAG 필요 여부 판단
"""
from fastapi import APIRouter, HTTPException

from app.schemas.rag_router import RagRequest, RagRouterResponse
from app.services.llm.local_registry import LocalModelRegistry
from app.services.router.qwen_rag_router import (
    build_combined_prompt,
    build_user_prompt,
)

router = APIRouter()


@router.get("/health")
def rag_router_health():
    """
    라우터 모델 로딩 상태를 확인하는 헬스체크 엔드포인트.

    Returns:
        dict: {"status": "ok", "router_loaded": bool}
    """
    return {
        "status": "ok",
        "router_loaded": LocalModelRegistry.has("router"),
    }


@router.post("/decide", response_model=RagRouterResponse)
def decide_rag_api(request: RagRequest) -> RagRouterResponse:
    """
    현재 질의에 대해 RAG 필요 여부를 판단하는 엔드포인트.

    LocalModelRegistry에 "router" role이 등록된 모델을 사용한다.
    모델이 등록되지 않은 경우 503을 반환한다.

    Args:
        request: RagRequest 형식의 요청 객체.

    Returns:
        RagRouterResponse: decision, confidence, reason_code, retrieval_query 포함.

    Raises:
        HTTPException 503: "router" 모델이 로드되지 않은 경우.
        HTTPException 500: 추론 중 예외가 발생한 경우.
    """
    if not LocalModelRegistry.has("router"):
        raise HTTPException(
            status_code=503,
            detail=(
                "Router model is not loaded. "
                "Register a model via LocalModelRegistry.register('router', ...) "
                "before calling this endpoint."
            ),
        )

    try:
        user_prompt = build_user_prompt(
            current_query=request.current_query,
            has_attached_pdf=request.has_attached_pdf,
            has_attached_image=request.has_attached_image,
            conversation_summary=request.conversation_summary,
            last_action=request.last_action,
        )
        combined_prompt = build_combined_prompt(user_prompt)
        raw = LocalModelRegistry.generate("router", combined_prompt, max_new_tokens=60)
        return RagRouterResponse.from_json_text(raw)

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
