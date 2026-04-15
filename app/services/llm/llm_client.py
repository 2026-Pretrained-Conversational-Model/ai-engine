"""
app/services/llm/llm_client.py
------------------------------
역할: LLM 호출 파사드. settings.LLM_BACKEND에 따라 local/sagemaker 백엔드 선택.

v8 변경:
- generate(system=...) 시그니처 추가. 호출자가 명시적으로 system을 줄 수 있다.
- 안 주면 백엔드(LocalModelRegistry)에 등록된 default_system 사용.
"""
from __future__ import annotations

from typing import Optional

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
        system: Optional[str] = None,
    ) -> str:
        backend = cls._backend()
        return await backend.generate(
            prompt=prompt,
            role=role,
            max_new_tokens=max_new_tokens,
            system=system,
        )

    @classmethod
    def reset(cls) -> None:
        """노트북 재실행 안전: 백엔드 캐시 리셋."""
        cls._local_backend = None
        cls._sagemaker_backend = None
