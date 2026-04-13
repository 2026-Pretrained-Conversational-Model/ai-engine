"""
app/core/config.py
------------------
역할: 환경 변수 기반 설정의 단일 진실 소스(Single Source of Truth)
      프로세스 시작 시 한 번 로드되며,
      `settings`를 통해 각 서비스에 주입됨

TODO:
    [ ] 환경별 설정 파일 추가 (.env.dev, .env.prod)
    [ ] 시작 시 SageMaker 엔드포인트 연결 가능 여부 검증
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

    # SageMaker
    AWS_REGION: str = "ap-northeast-2"
    SAGEMAKER_LLM_ENDPOINT: str = ""
    SAGEMAKER_VLM_ENDPOINT: str = ""
    SAGEMAKER_TIMEOUT_SEC: int = 60

    # Embedding
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-m3"
    EMBEDDING_DEVICE: str = "cpu"
    EMBEDDING_DIM: int = 1024

    # PDF / RAG
    PDF_CHUNK_SIZE: int = 800
    PDF_CHUNK_OVERLAP: int = 100
    RAG_TOP_K: int = 4


settings = Settings()
