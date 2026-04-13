"""
app/services/llm/llm_client.py
------------------------------
역할: AWS SageMaker 엔드포인트를 통한 텍스트 LLM 호출
      해당 엔드포인트는 "zero-orchestration" 방식으로 동작하며,
      프롬프트와 파라미터만 전달하면 텍스트를 생성하여 반환함

      별도의 tool 호출이나 function calling은 사용하지 않음

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
            # Adjust to match your container's output format
            if isinstance(body, list) and body and "generated_text" in body[0]:
                return body[0]["generated_text"]
            return body.get("generated_text", str(body))
        except Exception as e:
            logger.exception("LLM invoke failed: %s", e)
            return "[LLM invocation error]"
