"""
app/services/llm/backends/local_backend.py
------------------------------------------
역할: LocalModelRegistry에 등록된 HuggingFace 모델을 in-process로 호출.

v8 변경:
- system 파라미터를 LocalModelRegistry.generate()로 그대로 전달.
- 'answer' role 폴백은 유지하되, 폴백 시 system도 같이 적용 가능.
"""
from __future__ import annotations

import asyncio
from typing import Optional

from app.core.logger import get_logger
from app.services.llm.backends.base import LLMBackend, VLMBackend
from app.services.llm.local_registry import LocalModelRegistry

logger = get_logger(__name__)


class LocalLLMBackend(LLMBackend):
    async def generate(
        self,
        prompt: str,
        role: str = "answer",
        max_new_tokens: int = 1024,
        system: Optional[str] = None,
    ) -> str:
        target_role = role
        if not LocalModelRegistry.has(target_role):
            if LocalModelRegistry.has("answer"):
                logger.warning(
                    "LocalLLMBackend: role='%s' not registered, falling back to 'answer'",
                    role,
                )
                target_role = "answer"
            else:
                raise RuntimeError(
                    f"LocalLLMBackend: no model for role='{role}' and no 'answer' fallback"
                )

        try:
            return await asyncio.to_thread(
                LocalModelRegistry.generate,
                target_role,
                prompt,
                max_new_tokens,
                system,
            )
        except Exception as e:
            logger.exception("LocalLLMBackend generate failed (role=%s): %s", target_role, e)
            return f"[local backend error: {e}]"


class LocalVLMBackend(VLMBackend):
    async def generate(
        self,
        prompt: str,
        image_b64: str,
        max_new_tokens: int = 1024,
        system: Optional[str] = None,
    ) -> str:
        if LocalModelRegistry.has("vlm"):
            return await asyncio.to_thread(
                LocalModelRegistry.generate, "vlm", prompt, max_new_tokens, system
            )
        logger.warning(
            "LocalVLMBackend: no 'vlm' model registered; using 'answer' (image ignored)"
        )
        if not LocalModelRegistry.has("answer"):
            return "[no local model registered]"
        return await asyncio.to_thread(
            LocalModelRegistry.generate, "answer", prompt, max_new_tokens, system
        )
