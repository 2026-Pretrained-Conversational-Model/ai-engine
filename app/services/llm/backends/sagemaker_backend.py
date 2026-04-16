"""
app/services/llm/backends/sagemaker_backend.py
---------------------------------------------
역할: SageMaker 또는 호환 HTTP 엔드포인트를 통한 LLM 호출.

두 가지 모드:
    1) boto3 모드 (운영):
       SAGEMAKER_BASE_URL이 비어있으면 boto3로 실제 SageMaker endpoint 호출.
       endpoint 이름은 settings.SAGEMAKER_ANSWER_ENDPOINT 등.

    2) HTTP 모드 (로컬 테스트):
       SAGEMAKER_BASE_URL이 설정되면 (예: http://localhost:9000)
       boto3 대신 해당 URL의 /generate 엔드포인트로 HTTP POST 호출.
       tools/dummy_sagemaker.py를 띄워놓으면 실제 모델 없이
       파이프라인 전체가 돌아가는지 테스트 가능.

요청 페이로드 (두 모드 동일):
    {
        "system": "...",
        "user": "...",
        "max_new_tokens": 200,
        "role": "answer"   // "answer" | "router" | "memory" | "summary"
    }

응답 페이로드 (두 모드 동일):
    { "text": "..." }
"""
from __future__ import annotations

import asyncio
import json
from typing import Optional
from urllib.request import Request, urlopen
from urllib.error import URLError

from app.core.config import settings
from app.core.logger import get_logger
from app.services.llm.backends.base import LLMBackend, VLMBackend

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# boto3 lazy init (운영 모드에서만 사용)
# ---------------------------------------------------------------------------

class _SageMakerRuntimeMixin:
    _client = None

    @classmethod
    def _runtime(cls):
        if cls._client is None:
            import boto3
            cls._client = boto3.client(
                "sagemaker-runtime", region_name=settings.AWS_REGION
            )
        return cls._client


# ---------------------------------------------------------------------------
# HTTP 호출 헬퍼 (로컬 테스트 모드)
# ---------------------------------------------------------------------------

def _http_post_sync(url: str, payload: dict) -> dict:
    """
    stdlib urllib으로 동기 HTTP POST. 추가 pip 의존성 없음.
    asyncio.to_thread()로 감싸서 이벤트 루프를 블로킹하지 않는다.
    """
    data = json.dumps(payload).encode("utf-8")
    req = Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with urlopen(req, timeout=settings.SAGEMAKER_TIMEOUT_SEC) as resp:
        return json.loads(resp.read().decode("utf-8"))


# ---------------------------------------------------------------------------
# LLM Backend
# ---------------------------------------------------------------------------

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
        system: Optional[str] = None,
    ) -> str:
        payload = {
            "system": system or "",
            "user": prompt,
            "max_new_tokens": max_new_tokens,
            "role": role,
        }

        base_url = getattr(settings, "SAGEMAKER_BASE_URL", "")

        # ---- HTTP 모드 (로컬 테스트) -----------------------------------------
        if base_url:
            url = f"{base_url.rstrip('/')}/generate"
            try:
                body = await asyncio.to_thread(_http_post_sync, url, payload)
                return body.get("text", str(body))
            except URLError as e:
                logger.error(
                    "HTTP backend call failed (url=%s role=%s): %s. "
                    "dummy_sagemaker가 실행 중인지 확인하세요.",
                    url, role, e,
                )
                return f"[HTTP backend error: {e}]"
            except Exception as e:
                logger.exception("HTTP backend unexpected error: %s", e)
                return f"[HTTP backend error: {e}]"

        # ---- boto3 모드 (운영) -----------------------------------------------
        endpoint_name = self._endpoint_for_role(role)
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


# ---------------------------------------------------------------------------
# VLM Backend (이미지 처리용 — 로컬 테스트에서는 폴백 메시지 반환)
# ---------------------------------------------------------------------------

class SageMakerVLMBackend(_SageMakerRuntimeMixin, VLMBackend):
    async def generate(
        self,
        prompt: str,
        image_b64: str,
        max_new_tokens: int = 1024,
        system: Optional[str] = None,
    ) -> str:
        base_url = getattr(settings, "SAGEMAKER_BASE_URL", "")

        # HTTP 모드에서는 VLM 미지원 — 더미 응답
        if base_url:
            return "[DUMMY VLM] 이미지 분석은 로컬 테스트에서 지원되지 않습니다."

        # boto3 모드
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
