"""
app/services/llm/backends/sagemaker_backend.py
---------------------------------------------
역할: SageMaker 또는 호환 HTTP 엔드포인트를 통한 LLM 호출.

서버(deploy/sagemaker_code/inference.py) 계약 — 반드시 지킨다:
    입력:  {"system": str, "user": str, "max_new_tokens": int}
    출력:  {"text": str}
    헤더:  Content-Type == "application/json"
           Accept       == "application/json"  (정확히 일치해야 함)

v13 변경:
- role 키를 서버 payload에서 제거 (서버는 무시하지만 계약 엄수).
- UTF-8 bytes로 인코딩해서 전송 (한국어 프롬프트 그대로).
- invoke_endpoint를 asyncio.to_thread로 감싸 이벤트 루프 블로킹 방지.
- ClientError에서 서버가 반환한 실제 에러 본문을 끌어와 로그에 남김.
- 응답이 non-JSON이거나 "text" 키가 없는 경우를 방어적으로 처리.

두 가지 모드:
    1) boto3 모드 (운영): SAGEMAKER_BASE_URL=""
    2) HTTP 모드 (로컬 테스트): SAGEMAKER_BASE_URL 설정
       tools/dummy_sagemaker.py를 띄워놓으면 실제 모델 없이
       파이프라인 전체가 돌아가는지 테스트 가능.
"""
from __future__ import annotations

import asyncio
import json
from typing import Optional
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

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
    """stdlib urllib으로 동기 HTTP POST. asyncio.to_thread()로 감싸서 호출."""
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(req, timeout=settings.SAGEMAKER_TIMEOUT_SEC) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as e:
        body_preview = ""
        try:
            body_preview = e.read().decode("utf-8", errors="replace")[:500]
        except Exception:
            pass
        logger.error("HTTP %s from %s: %s | body=%r", e.code, url, e.reason, body_preview)
        raise


# ---------------------------------------------------------------------------
# 에러 본문 추출 헬퍼
# ---------------------------------------------------------------------------

def _extract_boto_error(exc: Exception) -> str:
    """
    boto3 ClientError에서 서버가 돌려준 실제 상세 메시지를 뽑아 로그용 문자열로 정리.
    ModelError의 경우 CloudWatch 링크도 같이 찍어준다.
    """
    try:
        from botocore.exceptions import ClientError
        if isinstance(exc, ClientError):
            err = exc.response.get("Error", {}) or {}
            meta = exc.response.get("ResponseMetadata", {}) or {}
            parts = [
                f"code={err.get('Code')!r}",
                f"status={meta.get('HTTPStatusCode')}",
                f"msg={err.get('Message')!r}",
            ]
            # SageMaker ModelError는 OriginalStatusCode / OriginalMessage를 별도로 담아줌
            for k in ("OriginalStatusCode", "OriginalMessage", "LogStreamArn"):
                if k in exc.response:
                    parts.append(f"{k}={exc.response[k]!r}")
            return " | ".join(parts)
    except Exception:
        pass
    return f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# LLM Backend
# ---------------------------------------------------------------------------

class SageMakerLLMBackend(_SageMakerRuntimeMixin, LLMBackend):

    def _endpoint_for_role(self, role: str) -> str:
        """role → 실제 SageMaker endpoint 이름 매핑 (클라 측 라우팅)."""
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
        # 서버 contract에 정확히 맞는 페이로드만 구성.
        # role은 엔드포인트 선택에만 사용하고 body에는 안 넣는다.
        payload = {
            "system": system or "",
            "user": prompt or "",
            "max_new_tokens": int(max_new_tokens),
        }

        base_url = getattr(settings, "SAGEMAKER_BASE_URL", "")

        # ---- HTTP 모드 (로컬 테스트) -----------------------------------------
        if base_url:
            url = f"{base_url.rstrip('/')}/generate"
            try:
                body = await asyncio.to_thread(_http_post_sync, url, payload)
                text = body.get("text")
                if not isinstance(text, str):
                    logger.error("HTTP backend response missing 'text': %r", body)
                    return "[LLM invocation error: no text field]"
                return text
            except (URLError, HTTPError) as e:
                logger.error("HTTP backend failed (url=%s role=%s): %s", url, role, e)
                return "[LLM invocation error]"
            except Exception as e:
                logger.exception("HTTP backend unexpected error: %s", e)
                return "[LLM invocation error]"

        # ---- boto3 모드 (운영) -----------------------------------------------
        endpoint_name = self._endpoint_for_role(role)

        # UTF-8 bytes로 보내 한국어를 있는 그대로 전달. 서버 json.loads는 bytes OK.
        body_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")

        def _invoke():
            return self._runtime().invoke_endpoint(
                EndpointName=endpoint_name,
                ContentType="application/json",
                Accept="application/json",
                Body=body_bytes,
            )

        try:
            resp = await asyncio.to_thread(_invoke)
        except Exception as e:
            logger.error(
                "SageMaker invoke_endpoint failed | endpoint=%s role=%s | %s",
                endpoint_name, role, _extract_boto_error(e),
            )
            return "[LLM invocation error]"

        # 응답 본문 파싱 (방어적)
        try:
            raw_bytes = resp["Body"].read()
            raw_text = raw_bytes.decode("utf-8", errors="replace")
        except Exception as e:
            logger.exception("SageMaker response read failed: %s", e)
            return "[LLM invocation error: response read failed]"

        try:
            body = json.loads(raw_text)
        except json.JSONDecodeError:
            logger.error(
                "SageMaker response is not JSON | endpoint=%s | raw=%r",
                endpoint_name, raw_text[:500],
            )
            return "[LLM invocation error: non-JSON response]"

        text = body.get("text")
        if not isinstance(text, str):
            logger.error(
                "SageMaker response missing 'text' field | endpoint=%s | body=%r",
                endpoint_name, body,
            )
            return "[LLM invocation error: no text field]"

        return text


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
        body_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")

        def _invoke():
            return self._runtime().invoke_endpoint(
                EndpointName=settings.SAGEMAKER_VLM_ENDPOINT,
                ContentType="application/json",
                Accept="application/json",
                Body=body_bytes,
            )

        try:
            resp = await asyncio.to_thread(_invoke)
            body = json.loads(resp["Body"].read().decode("utf-8"))
            return body.get("generated_text") or body.get("text") or str(body)
        except Exception as e:
            logger.error("SageMaker VLM invoke failed: %s", _extract_boto_error(e))
            return "[VLM invocation error]"
