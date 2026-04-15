"""
app/core/config.py
------------------
역할: 환경 변수 기반 설정 단일 소스

v12 변경:
- MEMORY_UPDATE_ASYNC 제거 (분기 없음).
  memory update는 Colab/FastAPI 둘 다에서 무조건 fire-and-forget(병렬).
  Colab에서 실행이 안 되던 문제는 "영구 이벤트 루프 헬퍼"(노트북 측)로 해결.
- MEMORY_UPDATE_EVERY_N_TURNS: user 턴 기준 3회마다 업데이트 LLM 호출.
- ANSWER_MAX_NEW_TOKENS: answer 생성 상한 (200).
- MEMORY_UPDATE_WINDOW_TURNS: 업데이트 LLM에 넣을 "직전 몇 턴" (기본 3).
  user 3턴이면 messages = user+assistant 6개가 들어감.
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
    RECENT_MESSAGES_WINDOW: int = 6  # prompt_builder에서 answer LLM에 넣을 최근 메시지 수

    # File storage
    LOCAL_FILE_DIR: str = "/tmp/ai-orchestrator"
    DELETE_FILE_ON_EXPIRE: bool = True

    # LLM backend
    LLM_BACKEND: str = "sagemaker"  # "local" or "sagemaker"
    AWS_REGION: str = "ap-northeast-2"
    SAGEMAKER_ANSWER_ENDPOINT: str = ""
    SAGEMAKER_ROUTER_ENDPOINT: str = ""
    SAGEMAKER_SUMMARY_ENDPOINT: str = ""
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

    # ---- v12: memory update policy ------------------------------------------
    # user 턴 기준 N회마다 memory LLM 호출 (3=3턴에 1번)
    MEMORY_UPDATE_EVERY_N_TURNS: int = 3
    # update LLM에 넣을 직전 turn 수 (user 기준). 3턴이면 user+assistant 6메시지.
    MEMORY_UPDATE_WINDOW_TURNS: int = 3

    # answer 생성 토큰 상한 (latency + 언어 드리프트 제어)
    ANSWER_MAX_NEW_TOKENS: int = 200


settings = Settings()
