"""
app/core/config.py
------------------
역할: 환경 변수 기반 설정 단일 소스

TODO
김예슬[완]
- SAGEMAKER_BASE_URL: 비어있으면 boto3(운영), 설정하면 HTTP(로컬 테스트).
  예: SAGEMAKER_BASE_URL=http://localhost:9000
  → sagemaker_backend.py가 boto3 대신 http://localhost:9000/generate 호출.

"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # App
    APP_NAME: str = "ai-orchestrator"
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8888
    LOG_LEVEL: str = "INFO"

    # Session memory cap
    SESSION_MAX_BYTES: int = 10 * 1024 * 1024
    SESSION_IDLE_TIMEOUT_MIN: int = 30
    RECENT_MESSAGES_WINDOW: int = 6

    # File storage
    LOCAL_FILE_DIR: str = "/tmp/ai-orchestrator"
    DELETE_FILE_ON_EXPIRE: bool = True

    # ---- LLM backend ---------------------------------------------------------
    LLM_BACKEND: str = "local"   # "local" (Colab) or "sagemaker" (운영/테스트)

    AWS_REGION: str = "ap-northeast-2"
    SAGEMAKER_ANSWER_ENDPOINT: str = "ai-orchestrator-answer-v1"
    SAGEMAKER_ROUTER_ENDPOINT: str = "ai-orchestrator-router-v1"
    SAGEMAKER_SUMMARY_ENDPOINT: str = "ai-orchestrator-memory-v1"   # ← 오타 수정
    SAGEMAKER_VLM_ENDPOINT: str = "qwen25-vlm-async"
    SAGEMAKER_TIMEOUT_SEC: int = 120

    # 로컬 테스트용: 설정하면 boto3 대신 이 URL로 HTTP POST 호출
    # 비어있으면 → boto3 (운영)
    # http://localhost:9000 → tools/dummy_sagemaker.py로 HTTP (로컬 테스트)
    SAGEMAKER_BASE_URL: str = ""

    # Embedding
    EMBEDDING_MODEL_NAME: str = "jhgan/ko-sroberta-multitask"
    EMBEDDING_DEVICE: str = "cpu"
    EMBEDDING_DIM: int = 768

    # PDF / RAG
    PDF_CHUNK_SIZE: int = 800
    PDF_CHUNK_OVERLAP: int = 100
    RAG_TOP_K: int = 4

    # ---- memory update policy ------------------------------------------------
    MEMORY_UPDATE_EVERY_N_TURNS: int = 3
    MEMORY_UPDATE_WINDOW_TURNS: int = 3

    # answer 생성 토큰 상한
    ANSWER_MAX_NEW_TOKENS: int = 800


settings = Settings()
