"""
app/services/llm/local_registry.py
----------------------------------
역할: Colab/로컬에서 로드한 HuggingFace 모델을 프로세스 전역에 등록하는 싱글톤.

지원 role:
    - "answer":  최종 답변 생성 LLM
    - "router":  RouterDecision 분류용 LLM (소형 권장)
    - "summary": narrative/structured summary 갱신용 LLM
    - "vlm":     (선택) 이미지+텍스트 VLM

노트북에서 사용 예:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct",
                                                  device_map="auto", torch_dtype="auto")

    from app.services.llm.local_registry import LocalModelRegistry
    LocalModelRegistry.register("answer", model, tok, device="cuda")

중요:
- 모델은 노트북 셀 1회 로드 → 모든 pipeline.run() 호출이 재사용한다.
- pipeline은 비동기지만 model.generate()는 동기이므로,
  호출부(local_backend.py)가 asyncio.to_thread로 감싸서 이벤트 루프를 블로킹하지 않는다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from app.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ModelEntry:
    model: Any
    tokenizer: Any
    device: str = "cuda"
    max_new_tokens: int = 512
    generate_kwargs: Dict[str, Any] = field(default_factory=dict)


class LocalModelRegistry:
    """role 문자열 → ModelEntry 매핑 싱글톤."""

    _entries: Dict[str, ModelEntry] = {}

    # ---- 등록 / 조회 ---------------------------------------------------------

    @classmethod
    def register(
        cls,
        role: str,
        model: Any,
        tokenizer: Any,
        device: str = "cuda",
        max_new_tokens: int = 512,
        generate_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        cls._entries[role] = ModelEntry(
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_new_tokens=max_new_tokens,
            generate_kwargs=generate_kwargs or {},
        )
        logger.info("Registered local model: role=%s device=%s", role, device)

    @classmethod
    def has(cls, role: str) -> bool:
        return role in cls._entries

    @classmethod
    def get(cls, role: str) -> Optional[ModelEntry]:
        return cls._entries.get(role)

    @classmethod
    def clear(cls) -> None:
        cls._entries.clear()

    @classmethod
    def list_roles(cls) -> list[str]:
        return list(cls._entries.keys())

    # ---- 실제 generate 호출 --------------------------------------------------

@classmethod
def generate(cls, role: str, prompt: str, max_new_tokens: Optional[int] = None) -> str:
    entry = cls._entries.get(role)
    if entry is None:
        raise RuntimeError(f"LocalModelRegistry: role='{role}' is not registered")

    import torch

    tok = entry.tokenizer
    model = entry.model
    device = entry.device

    if hasattr(tok, "apply_chat_template") and getattr(tok, "chat_template", None):
        messages = [{"role": "user", "content": prompt}]
        enc = tok.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
        )

        # case 1: Tensor 반환
        if isinstance(enc, torch.Tensor):
            input_ids = enc.to(device)
            attention_mask = torch.ones_like(input_ids)

        # case 2: BatchEncoding / dict 반환
        else:
            enc = enc.to(device)
            input_ids = enc["input_ids"]
            attention_mask = enc.get("attention_mask", torch.ones_like(input_ids))
    else:
        enc = tok(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        ).to(device)
        input_ids = enc["input_ids"]
        attention_mask = enc.get("attention_mask", torch.ones_like(input_ids))

    gen_kwargs: Dict[str, Any] = dict(entry.generate_kwargs)
    gen_kwargs.setdefault("max_new_tokens", max_new_tokens or entry.max_new_tokens)
    gen_kwargs.setdefault("do_sample", False)
    if tok.eos_token_id is not None:
        gen_kwargs.setdefault("pad_token_id", tok.eos_token_id)

    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **gen_kwargs,
        )

    prompt_len = input_ids.shape[1]
    new_tokens = out[0, prompt_len:]
    text = tok.decode(new_tokens, skip_special_tokens=True)
    return text.strip()