"""
app/core/config.py
------------------
역할: 환경 변수 기반 설정 단일 소스
      프로세스 시작 시 1회 로드되고 settings로 주입됨

TODO : 김예슬 (완)
변경점(Colab 지원):
- LLM_BACKEND 추가: "sagemaker" | "local"
  - "sagemaker": 기존 boto3 엔드포인트 호출 (운영)
  - "local":     LocalModelRegistry에 등록된 HuggingFace 모델을 in-process 호출 (Colab/개발)
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # App
    APP_NAME: str = "ai-orchestrator"
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    LOG_LEVEL: str = "INFO"

    # Session memory cap
    SESSION_MAX_BYTES: int = 10 * 1024 * 1024
    SESSION_IDLE_TIMEOUT_MIN: int = 30
    RECENT_MESSAGES_WINDOW: int = 6

    # File storage
    LOCAL_FILE_DIR: str = "/tmp/ai-orchestrator"
    DELETE_FILE_ON_EXPIRE: bool = True

    # ---- LLM backend selector ------------------------------------------------
    # "sagemaker": 운영용. boto3로 SageMaker endpoint 호출.
    # "local":     개발/Colab용. LocalModelRegistry에 등록된 HF 모델 직접 호출.
    # LLM_BACKEND: str = "sagemaker"
    LLM_BACKEND: str = "local"

    # SageMaker (LLM_BACKEND=sagemaker 일 때만 사용)
    AWS_REGION: str = "ap-northeast-2"
    SAGEMAKER_LLM_ENDPOINT: str = ""
    SAGEMAKER_VLM_ENDPOINT: str = ""
    SAGEMAKER_TIMEOUT_SEC: int = 60

    # Embedding
    EMBEDDING_MODEL_NAME: str = "jhgan/ko-sroberta-multitask"
    EMBEDDING_DEVICE: str = "cpu"
    EMBEDDING_DIM: int = 768

    # PDF / RAG
    PDF_CHUNK_SIZE: int = 800
    PDF_CHUNK_OVERLAP: int = 100
    RAG_TOP_K: int = 4


settings = Settings()
