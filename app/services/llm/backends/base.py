"""
app/services/llm/backends/base.py
---------------------------------
역할: LLMBackend / VLMBackend 추상 인터페이스.

v8 변경:
- generate()가 system 파라미터를 받는다. None이면 백엔드 default 사용.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional


class LLMBackend(ABC):
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        role: str = "answer",
        max_new_tokens: int = 1024,
        system: Optional[str] = None,
    ) -> str:
        ...


class VLMBackend(ABC):
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        image_b64: str,
        max_new_tokens: int = 1024,
        system: Optional[str] = None,
    ) -> str:
        ...
