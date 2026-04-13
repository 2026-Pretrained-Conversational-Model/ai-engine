"""
app/services/llm/vlm_client.py
------------------------------
역할: 별도의 SageMaker 엔드포인트를 사용하는 Vision LLM 호출
      요청에 이미지가 포함된 경우에만 사용됨

TODO:
    [ ] 모델의 최대 크기에 맞게 이미지 사전 리사이즈
    [ ] 스트리밍 지원 추가
"""
import json
import boto3
from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)


class VLMClient:
    _client = None

    @classmethod
    def _runtime(cls):
        if cls._client is None:
            cls._client = boto3.client(
                "sagemaker-runtime", region_name=settings.AWS_REGION
            )
        return cls._client

    @classmethod
    async def generate(cls, prompt: str, image_b64: str, max_new_tokens: int = 1024) -> str:
        if not settings.SAGEMAKER_VLM_ENDPOINT:
            return "[baseline-vlm-fallback-answer]\n이미지 입력은 감지됐지만 VLM 엔드포인트가 설정되지 않았습니다."

        payload = {
            "inputs": {"text": prompt, "image": image_b64},
            "parameters": {"max_new_tokens": max_new_tokens, "temperature": 0.3},
        }
        try:
            resp = cls._runtime().invoke_endpoint(
                EndpointName=settings.SAGEMAKER_VLM_ENDPOINT,
                ContentType="application/json",
                Body=json.dumps(payload),
            )
            body = json.loads(resp["Body"].read())
            return body.get("generated_text", str(body))
        except Exception as e:
            logger.exception("VLM invoke failed: %s", e)
            return "[baseline-vlm-fallback-answer]\nVLM 호출 중 오류가 발생했습니다."
