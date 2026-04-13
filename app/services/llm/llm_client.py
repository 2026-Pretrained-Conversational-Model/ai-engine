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
import json
import boto3
from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)


class LLMClient:
    _client = None

    @classmethod
    def _runtime(cls):
        if cls._client is None:
            cls._client = boto3.client(
                "sagemaker-runtime", region_name=settings.AWS_REGION
            )
        return cls._client

    @classmethod
    async def generate(cls, prompt: str, max_new_tokens: int = 1024) -> str:
        if not settings.SAGEMAKER_LLM_ENDPOINT:
            return cls._fallback(prompt)

        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_new_tokens,
                "temperature": 0.3,
                "do_sample": True,
            },
        }
        try:
            resp = cls._runtime().invoke_endpoint(
                EndpointName=settings.SAGEMAKER_LLM_ENDPOINT,
                ContentType="application/json",
                Body=json.dumps(payload),
            )
            body = json.loads(resp["Body"].read())
            if isinstance(body, list) and body and "generated_text" in body[0]:
                return body[0]["generated_text"]
            return body.get("generated_text", str(body))
        except Exception as e:
            logger.exception("LLM invoke failed: %s", e)
            return cls._fallback(prompt)

    @staticmethod
    def _fallback(prompt: str) -> str:
        """개발 환경용 간단한 응답.

        TODO:
            [ ] 실제 로컬 모델 서버 연동 또는 mock framework로 교체
        """
        marker = "[Current User Question]"
        if marker in prompt:
            question = prompt.split(marker, 1)[-1].strip()
        else:
            question = prompt[-300:]
        return (
            "[baseline-fallback-answer]\n"
            f"요청을 처리했지만 현재 원격 LLM 엔드포인트가 설정되지 않았습니다.\n"
            f"해석한 질문: {question[:300]}"
        )
