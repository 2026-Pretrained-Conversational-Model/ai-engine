"""
app/services/router/qwen_rag_router.py
--------------------------------------
역할: Qwen2.5-3B-Instruct 모델을 사용해 RAG 필요 여부(NEED_RAG / NO_RAG)를
      판단하는 소형 라우터 클래스.

      기존 router.py(독립 FastAPI 앱)의 QwenRagRouter 로직을 그대로 이식하되,
      ai-engine의 LocalModelRegistry / 스키마 체계와 통합한다.

      사용 방법 (두 가지):

      [A] LocalModelRegistry 경유 (권장 - ai-engine 파이프라인 표준)
          model  = AutoModelForCausalLM.from_pretrained(MODEL_NAME, ...)
          tok    = AutoTokenizer.from_pretrained(MODEL_NAME, ...)
          LocalModelRegistry.register("router", model, tok, device="cuda",
                                      max_new_tokens=60)
          # 이후 router_judge.judge_route()가 자동으로 사용함.

      [B] 직접 인스턴스화 (독립 스크립트 / 테스트)
          router = QwenRagRouter()          # 내부에서 직접 모델 로드
          result = router.decide_rag("최신 논문 찾아줘")

      참고:
      - pipeline.py 는 router_judge.judge_route()를 통해 간접 호출하므로
        이 클래스를 직접 import할 필요가 없다.
      - LocalModelRegistry.generate()는 system prompt 없이 단일 user 메시지만
        지원하므로, registry 경유 시에는 system+user 메시지를 하나의 문자열로
        합쳐서 전달한다.

TODO:
    [ ] confidence threshold 설정값(config.py)으로 관리
    [ ] 출력 파싱 실패 시 재시도 로직 (최대 2회)
"""
from __future__ import annotations

from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
import os

transformers.logging.set_verbosity_error()
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

from app.core.logger import get_logger
from app.schemas.rag_router import RagRouterResponse

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

RAG_ROUTER_SYSTEM_PROMPT = """
You are a routing classifier.

Your only job is to decide whether the user's current request requires RAG retrieval.

Return exactly one decision:
- NEED_RAG
- NO_RAG

Decision policy:
- Choose NEED_RAG if answering correctly requires external documents, knowledge base lookup,
  uploaded document search, or retrieval over prior sources.
- Choose NO_RAG if the question can be answered directly from the current user query alone
  or from general reasoning without retrieval.
- If the question is mainly about analyzing an attached file itself, choose NO_RAG.
- If the request is ambiguous but does not yet justify retrieval, choose NO_RAG.

Constraints:
- Output valid JSON only.
- No markdown, no explanation, no extra text.
- confidence must be between 0 and 1.
- If decision is NO_RAG, retrieval_query must be "".
- If decision is NEED_RAG, retrieval_query should be short and retrieval-friendly.
- reason_code must be one of:
  EXTERNAL_KNOWLEDGE_REQUIRED
  ENOUGH_CONTEXT_IN_QUERY
  FILE_REQUIRED
  AMBIGUOUS_BUT_NOT_RAG
""".strip()

RAG_ROUTER_USER_TEMPLATE = """
Input:
current_query: {current_query}
has_attached_pdf: {has_attached_pdf}
has_attached_image: {has_attached_image}
conversation_summary: {conversation_summary}
last_action: {last_action}

Return this JSON schema exactly:
{{
  "decision": "NEED_RAG | NO_RAG",
  "confidence": 0.0,
  "reason_code": "",
  "retrieval_query": ""
}}
""".strip()


def build_user_prompt(
    current_query: str,
    has_attached_pdf: bool = False,
    has_attached_image: bool = False,
    conversation_summary: str = "",
    last_action: str = "NONE",
) -> str:
    """RAG 라우터용 user prompt 문자열을 생성한다."""
    return RAG_ROUTER_USER_TEMPLATE.format(
        current_query=current_query,
        has_attached_pdf=str(has_attached_pdf).lower(),
        has_attached_image=str(has_attached_image).lower(),
        conversation_summary=conversation_summary,
        last_action=last_action,
    )


def build_combined_prompt(user_prompt: str) -> str:
    """
    LocalModelRegistry.generate()는 system prompt를 별도로 지원하지 않으므로,
    system + user 내용을 하나의 문자열로 합친다.

    registry 경유 경로(router_judge.py)에서만 사용한다.
    """
    return f"{RAG_ROUTER_SYSTEM_PROMPT}\n\n{user_prompt}"


# ---------------------------------------------------------------------------
# QwenRagRouter
# ---------------------------------------------------------------------------

class QwenRagRouter:
    """
    Qwen2.5-3B-Instruct 모델을 직접 로드해 RAG 필요 여부를 판단하는 클래스.

    LocalModelRegistry를 사용하지 않는 독립 실행 경로(직접 인스턴스화)를 위해
    모델/토크나이저를 내부에서 관리한다.

    Args:
        model_name     : Hugging Face 모델 이름 (기본: Qwen/Qwen2.5-3B-Instruct).
        device_map     : 모델을 배치할 디바이스 설정 (기본: "auto").
        dtype          : torch dtype. None이면 GPU/CPU에 따라 자동 선택.
        max_new_tokens : generate 시 생성할 최대 토큰 수 (기본: 60).

    Attributes:
        model_name     : 로드된 모델 이름.
        max_new_tokens : 최대 생성 토큰 수.
        tokenizer      : AutoTokenizer 인스턴스.
        model          : AutoModelForCausalLM 인스턴스.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        device_map: str = "auto",
        dtype: Optional[torch.dtype] = None,
        max_new_tokens: int = 60,
    ):
        if dtype is None:
            if torch.cuda.is_available():
                dtype = (
                    torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                )
            else:
                dtype = torch.float32

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens

        logger.info("Loading QwenRagRouter: model=%s dtype=%s", model_name, dtype)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        self.model.eval()
        logger.info("QwenRagRouter loaded successfully: model=%s", model_name)

    def _generate(self, user_prompt: str) -> str:
        """
        system + user 메시지를 chat template으로 포맷한 뒤 모델 추론을 수행한다.

        Args:
            user_prompt: build_user_prompt()로 생성된 user 프롬프트 문자열.

        Returns:
            모델이 생성한 raw 출력 문자열 (JSON 형태 예상).
        """
        messages = [
            {"role": "system", "content": RAG_ROUTER_SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ]

        tokenized_inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )

        if hasattr(self.model, "device"):
            tokenized_inputs = {
                k: v.to(self.model.device) for k, v in tokenized_inputs.items()
            }

        with torch.inference_mode():
            outputs = self.model.generate(
                **tokenized_inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        prompt_len = tokenized_inputs["input_ids"].shape[-1]
        generated_ids = outputs[0][prompt_len:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return text.strip()

    def decide_rag(
        self,
        current_query: str,
        has_attached_pdf: bool = False,
        has_attached_image: bool = False,
        conversation_summary: str = "",
        last_action: str = "NONE",
    ) -> RagRouterResponse:
        """
        RAG 필요 여부를 판단하고 RagRouterResponse를 반환한다.

        Args:
            current_query        : 현재 사용자 질의.
            has_attached_pdf     : PDF 파일 첨부 여부.
            has_attached_image   : 이미지 파일 첨부 여부.
            conversation_summary : 이전 대화 요약.
            last_action          : 직전 시스템 액션.

        Returns:
            RagRouterResponse: 검증이 완료된 응답 객체.
        """
        user_prompt = build_user_prompt(
            current_query=current_query,
            has_attached_pdf=has_attached_pdf,
            has_attached_image=has_attached_image,
            conversation_summary=conversation_summary,
            last_action=last_action,
        )
        raw_text = self._generate(user_prompt)
        logger.debug("QwenRagRouter raw output: %r", raw_text[:200])
        return RagRouterResponse.from_json_text(raw_text)
