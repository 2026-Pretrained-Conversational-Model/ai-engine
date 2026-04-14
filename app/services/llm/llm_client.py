"""
app/services/llm/llm_client.py
------------------------------
역할: AWS SageMaker 엔드포인트를 통한 텍스트 LLM 호출.

baseline 추가 정책:
- 엔드포인트가 비어 있으면 로컬 heuristic fallback 응답을 사용한다.
- 이렇게 하면 인프라가 준비되지 않은 개발 단계에서도 전체 파이프라인을 검증할 수 있다.

TODO:
    [ ] InvokeEndpointWithResponseStream을 통한 스트리밍 지원
    [ ] ThrottlingException 발생 시 지수 백오프 기반 재시도
    [ ] 비용 및 토큰 사용량 로깅
"""
from __future__ import annotations

from app.core.config import settings
from app.services.llm.backends.local_backend import LocalLLMBackend
from app.services.llm.backends.sagemaker_backend import SageMakerLLMBackend


class LLMClient:
    _local_backend = None
    _sagemaker_backend = None

    @classmethod
    def _backend(cls):
        if settings.LLM_BACKEND == "local":
            if cls._local_backend is None:
                cls._local_backend = LocalLLMBackend()
            return cls._local_backend

        if cls._sagemaker_backend is None:
            cls._sagemaker_backend = SageMakerLLMBackend()
        return cls._sagemaker_backend

    @classmethod
    async def generate(
        cls,
        prompt: str,
        role: str = "answer",
        max_new_tokens: int = 1024,
    ) -> str:
        backend = cls._backend()
        return await backend.generate(
            prompt=prompt,
            role=role,
            max_new_tokens=max_new_tokens,
        )