"""
app/services/llm/backends/base.py
---------------------------------
역할: LLMBackend / VLMBackend 추상 인터페이스.

구체 구현은 local_backend.py, sagemaker_backend.py 참고.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class LLMBackend(ABC):
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        role: str = "answer",
        max_new_tokens: int = 1024,
    ) -> str:
        ...


class VLMBackend(ABC):
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        image_b64: str,
        max_new_tokens: int = 1024,
    ) -> str:
        ...
