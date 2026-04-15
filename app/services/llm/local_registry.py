"""
app/services/llm/local_registry.py
----------------------------------
역할: Colab/로컬에서 로드한 HuggingFace 모델을 프로세스 전역에 등록하는 싱글톤.

지원 role:
    - "answer":  최종 답변 생성 LLM
    - "router":  RouterDecision 분류용 LLM (소형 권장)
    - "summary" / "memory": Memory State Generator용 LLM
    - "vlm":     (선택) 이미지+텍스트 VLM

이번 패치 핵심 변경 (v8):
    1) generate()가 system 파라미터를 받는다. chat template에 system role로 주입.
       이전엔 user 메시지 하나만 넣어서 Qwen이 한국어 instruction-following을 자주 깨뜨렸음.
    2) 등록 시 default_system 문자열을 같이 등록할 수 있다. 호출부에서 system을 안 주면 default 사용.
    3) generate_kwargs를 role별로 분리할 수 있게 그대로 유지. (answer는 약간 sampling, router/summary는 greedy)

호출자(local_backend.py, narrative/memory updater, router_judge)는
이 함수에 system을 명시적으로 전달하도록 함께 패치됨.
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
    default_system: str = ""
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
        default_system: str = "",
        generate_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        cls._entries[role] = ModelEntry(
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_new_tokens=max_new_tokens,
            default_system=default_system,
            generate_kwargs=generate_kwargs or {},
        )
        logger.info(
            "Registered local model: role=%s device=%s default_system=%d chars",
            role, device, len(default_system),
        )

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
    def generate(
        cls,
        role: str,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        system: Optional[str] = None,
    ) -> str:
        """
        동기 generation. asyncio.to_thread에서 호출될 것을 가정함.

        - system: chat template에 system role로 주입할 문자열.
                  None이면 등록 시 지정한 default_system 사용.
                  빈 문자열이면 system 메시지 생략.
        - chat template이 있으면 messages=[system, user]로 빌드.
          없으면 raw text 인코딩(이 경우 system은 prompt 앞에 prepend).
        - 입력 이후 새로 생성된 토큰만 디코딩해서 반환.
        """
        entry = cls._entries.get(role)
        if entry is None:
            raise RuntimeError(f"LocalModelRegistry: role='{role}' is not registered")

        import torch

        tok = entry.tokenizer
        model = entry.model
        device = entry.device

        effective_system = system if system is not None else entry.default_system

        gen_kwargs: Dict[str, Any] = dict(entry.generate_kwargs)
        gen_kwargs.setdefault("max_new_tokens", max_new_tokens or entry.max_new_tokens)
        gen_kwargs.setdefault("do_sample", False)
        if tok.eos_token_id is not None:
            gen_kwargs.setdefault("pad_token_id", tok.eos_token_id)

        if hasattr(tok, "apply_chat_template") and getattr(tok, "chat_template", None):
            messages: list[dict] = []
            if effective_system:
                messages.append({"role": "system", "content": effective_system})
            messages.append({"role": "user", "content": prompt})

            inputs = tok.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
                return_dict=True,
            ).to(device)

            with torch.no_grad():
                out = model.generate(**inputs, **gen_kwargs)

            prompt_len = inputs["input_ids"].shape[1]
            new_tokens = out[0, prompt_len:]
            text = tok.decode(new_tokens, skip_special_tokens=True)
            return text.strip()

        # chat_template 없는 경우: system + user 단순 결합
        flat = (effective_system + "\n\n" + prompt) if effective_system else prompt
        enc = tok(
            flat,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        ).to(device)

        with torch.no_grad():
            out = model.generate(**enc, **gen_kwargs)

        prompt_len = enc["input_ids"].shape[1]
        new_tokens = out[0, prompt_len:]
        text = tok.decode(new_tokens, skip_special_tokens=True)
        return text.strip()
