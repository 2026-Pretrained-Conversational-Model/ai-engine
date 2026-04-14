"""
app/services/llm/backends/local_backend.py
------------------------------------------
역할: LocalModelRegistry에 등록된 HuggingFace 모델을 in-process로 호출.

주의:
- model.generate()는 동기·블로킹이므로 asyncio.to_thread로 감싸서
  FastAPI 이벤트 루프를 막지 않는다.
- 요청된 role이 등록되어 있지 않으면 'answer'로 폴백한다.
  (router/summary 모델을 따로 로드하지 않아도 answer 모델만 있으면 돌아가도록.)
"""
from __future__ import annotations

import asyncio

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
                LocalModelRegistry.generate, target_role, prompt, max_new_tokens
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
    ) -> str:
        # 이번 baseline은 이미지 인코딩까지는 다루지 않는다.
        # 'vlm' role이 등록돼 있으면 그대로 텍스트 프롬프트만 넣고,
        # 없으면 answer 모델로 폴백 (이미지는 무시, 경고 로그).
        if LocalModelRegistry.has("vlm"):
            return await asyncio.to_thread(
                LocalModelRegistry.generate, "vlm", prompt, max_new_tokens
            )
        logger.warning("LocalVLMBackend: no 'vlm' model registered; using 'answer' (image ignored)")
        if not LocalModelRegistry.has("answer"):
            return "[no local model registered]"
        return await asyncio.to_thread(
            LocalModelRegistry.generate, "answer", prompt, max_new_tokens
        )
