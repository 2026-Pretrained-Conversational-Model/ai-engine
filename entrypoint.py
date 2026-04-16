"""
entrypoint.py
-------------
RunPod GPU Docker 컨테이너 진입점.
start_server.py와 동일한 순서로 모델을 로드하고 FastAPI를 실행한다.

순서:
    1) 환경변수에서 모델 ID, 양자화 설정 읽기
    2) HuggingFace 모델 3종 로드 (answer, router, memory)
    3) LocalModelRegistry에 등록
    4) uvicorn으로 FastAPI 앱 실행
       (lifespan에서 EmbeddingSingleton.warmup + SessionManager 초기화)

환경변수 (모두 Dockerfile ENV에 기본값 있음, RunPod에서 오버라이드 가능):
    ANSWER_MODEL_ID   — 답변 LLM (기본: Qwen/Qwen2.5-7B-Instruct)
    ROUTER_MODEL_ID   — 라우터 LLM (기본: Qwen/Qwen2.5-3B-Instruct)
    SUMMARY_MODEL_ID  — 메모리 LLM (기본: Qwen/Qwen2.5-3B-Instruct)
    BASE_QWEN_TOKENIZER_ID — summary 모델용 토크나이저 (기본: Qwen/Qwen2.5-3B-Instruct)
    USE_4BIT          — "true" 이면 4bit NF4 양자화 (기본: true)
    HOST              — 바인드 주소 (기본: 0.0.0.0)
    PORT              — 포트 (기본: 8000)

주의: ai-orchestrator 앱 코드(app/)는 일절 수정하지 않는다.
      이 파일은 앱 바깥에서 모델만 미리 올려두는 역할.
"""

import os
import sys
import logging

# ─── 0. 로깅 설정 ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("entrypoint")

# ─── 1. 환경변수 확인 ────────────────────────────────────────────────────────
# Dockerfile ENV에서 이미 설정돼 있지만, 명시적으로 로그로 출력.
ANSWER_MODEL_ID        = os.environ.get("ANSWER_MODEL_ID",  "Qwen/Qwen2.5-7B-Instruct")
ROUTER_MODEL_ID        = os.environ.get("ROUTER_MODEL_ID",  "Qwen/Qwen2.5-3B-Instruct")
SUMMARY_MODEL_ID       = os.environ.get("SUMMARY_MODEL_ID", "Qwen/Qwen2.5-3B-Instruct")
BASE_QWEN_TOKENIZER_ID = os.environ.get("BASE_QWEN_TOKENIZER_ID", "Qwen/Qwen2.5-3B-Instruct")
USE_4BIT               = os.environ.get("USE_4BIT", "true").lower() == "true"

HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8000"))

logger.info("=" * 60)
logger.info("AI Orchestrator — GPU Entrypoint")
logger.info("=" * 60)
logger.info("ANSWER_MODEL_ID  : %s", ANSWER_MODEL_ID)
logger.info("ROUTER_MODEL_ID  : %s", ROUTER_MODEL_ID)
logger.info("SUMMARY_MODEL_ID : %s", SUMMARY_MODEL_ID)
logger.info("USE_4BIT         : %s", USE_4BIT)
logger.info("LLM_BACKEND      : %s", os.environ.get("LLM_BACKEND", "(not set)"))
logger.info("EMBEDDING_DEVICE : %s", os.environ.get("EMBEDDING_DEVICE", "(not set)"))
logger.info("HF_HOME          : %s", os.environ.get("HF_HOME", "(not set)"))

# ─── 2. GPU 확인 ─────────────────────────────────────────────────────────────
import torch

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
    logger.info("GPU detected: %s (%.1f GB VRAM)", gpu_name, gpu_mem)
else:
    logger.warning("No GPU detected! Models will run on CPU (very slow).")

# ─── 3. 모델 로드 (start_server.py와 동일) ───────────────────────────────────
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def _load(model_id: str | None, tokenizer_id: str | None = None):
    """HuggingFace 모델 + 토크나이저 로드. None이면 skip."""
    if model_id is None:
        return None, None

    tok_id = tokenizer_id or model_id
    logger.info("Loading model=%s  tokenizer=%s", model_id, tok_id)

    tok = AutoTokenizer.from_pretrained(tok_id, trust_remote_code=True)
    tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    kwargs = dict(device_map="auto", trust_remote_code=True)
    if USE_4BIT:
        logger.info("  → 4bit NF4 quantization enabled")
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    else:
        kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)

    model.config.pad_token_id = tok.pad_token_id
    model.config.eos_token_id = tok.eos_token_id
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.pad_token_id = tok.pad_token_id
        model.generation_config.eos_token_id = tok.eos_token_id

    model.eval()
    device = next(model.parameters()).device
    logger.info("  → loaded on %s", device)
    return model, tok


logger.info("─" * 60)
logger.info("Loading answer model...")
answer_model, answer_tok = _load(ANSWER_MODEL_ID)

logger.info("Loading router model...")
router_model, router_tok = _load(ROUTER_MODEL_ID)

logger.info("Loading summary/memory model...")
summary_model, summary_tok = _load(SUMMARY_MODEL_ID, tokenizer_id=BASE_QWEN_TOKENIZER_ID)

logger.info("─" * 60)
logger.info("All models loaded successfully.")

# ─── 4. LocalModelRegistry 등록 ─────────────────────────────────────────────
from app.services.llm.local_registry import LocalModelRegistry

LocalModelRegistry.clear()

if answer_model is not None:
    LocalModelRegistry.register(
        "answer", answer_model, answer_tok,
        device="cuda", max_new_tokens=800,
    )

if router_model is not None:
    LocalModelRegistry.register(
        "router", router_model, router_tok,
        device="cuda", max_new_tokens=60,
    )

if summary_model is not None:
    LocalModelRegistry.register(
        "memory", summary_model, summary_tok,
        device="cuda", max_new_tokens=400,
    )

logger.info("Registered roles: %s", LocalModelRegistry.list_roles())

# ─── 5. 메모리 사용 현황 ─────────────────────────────────────────────────────
if torch.cuda.is_available():
    alloc = torch.cuda.memory_allocated(0) / (1024 ** 3)
    reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)
    logger.info("GPU memory — allocated: %.2f GB, reserved: %.2f GB", alloc, reserved)

# ─── 6. FastAPI 앱 실행 ─────────────────────────────────────────────────────
# app.main의 lifespan이 EmbeddingSingleton.warmup() + SessionManager.instance()를
# 처리하므로 여기서 별도 호출 불필요.
from app.main import app
import uvicorn

logger.info("=" * 60)
logger.info("Starting uvicorn — http://%s:%d", HOST, PORT)
logger.info("=" * 60)

uvicorn.run(
    app,
    host=HOST,
    port=PORT,
    log_level="info",
    reload=False,
    workers=1,      # 모델 메모리 공유 불가, 반드시 단일 워커
)
