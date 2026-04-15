"""
app/services/llm/backends/sagemaker_backend.py
---------------------------------------------
역할: 기존 llm_client/vlm_client의 boto3 SageMaker 호출 로직을 이관.

운영 환경(LLM_BACKEND=sagemaker)에서 사용된다.
"""
from __future__ import annotations

import json

from app.core.config import settings
from app.core.logger import get_logger
from app.services.llm.backends.base import LLMBackend, VLMBackend

logger = get_logger(__name__)


class _SageMakerRuntimeMixin:
    _client = None

    @classmethod
    def _runtime(cls):
        if cls._client is None:
            import boto3  # 필요 시점에만 import (local 모드에서는 import 비용 회피)
            cls._client = boto3.client("sagemaker-runtime", region_name=settings.AWS_REGION)
        return cls._client

class SageMakerLLMBackend(_SageMakerRuntimeMixin, LLMBackend):
    def _endpoint_for_role(self, role: str) -> str:
        if role == "answer":
            return settings.SAGEMAKER_ANSWER_ENDPOINT
        if role == "router":
            return settings.SAGEMAKER_ROUTER_ENDPOINT
        if role in ("summary", "memory"):
            return settings.SAGEMAKER_SUMMARY_ENDPOINT
        raise ValueError(f"Unsupported role: {role}")

    async def generate(
        self,
        prompt: str,
        role: str = "answer",
        max_new_tokens: int = 1024,
        system: str | None = None,
    ) -> str:
        endpoint_name = self._endpoint_for_role(role)

        payload = {
            "system": system or "",
            "user": prompt,
            "max_new_tokens": max_new_tokens,
            "role": role,
        }

        try:
            resp = self._runtime().invoke_endpoint(
                EndpointName=endpoint_name,
                ContentType="application/json",
                Accept="application/json",
                Body=json.dumps(payload),
            )
            body = json.loads(resp["Body"].read().decode("utf-8"))
            return body.get("text", str(body))
        except Exception as e:
            logger.exception("SageMaker LLM invoke failed: %s", e)
            return "[LLM invocation error]"



class SageMakerVLMBackend(_SageMakerRuntimeMixin, VLMBackend):
    async def generate(
        self,
        prompt: str,
        image_b64: str,
        max_new_tokens: int = 1024,
    ) -> str:
        payload = {
            "inputs": {"text": prompt, "image": image_b64},
            "parameters": {"max_new_tokens": max_new_tokens, "temperature": 0.3},
        }
        try:
            resp = self._runtime().invoke_endpoint(
                EndpointName=settings.SAGEMAKER_VLM_ENDPOINT,
                ContentType="application/json",
                Body=json.dumps(payload),
            )
            body = json.loads(resp["Body"].read())
            return body.get("generated_text", str(body))
        except Exception as e:
            logger.exception("SageMaker VLM invoke failed: %s", e)
            return "[VLM invocation error]"
