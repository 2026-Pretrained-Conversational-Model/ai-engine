"""
app/main.py
-----------
역할: FastAPI 애플리케이션 생성 및 프로세스 전반의 시작/종료 관리
      - 라우터 연결
      - 싱글톤(임베딩 모델) 1회 로딩 (프로세스 단위)
        → FAISS / 벡터가 요청마다 다시 로드되지 않도록 함
      - 종료 시 로컬 파일 정리를 위한 shutdown 훅 등록

참고: 미들웨어 없음.
      이 서비스는 브라우저에서 직접 호출되지 않고,
      내부 Node.js 오케스트레이터에서만 호출됨

TODO:
    [ ] 진행 중 요청을 안전하게 처리하는 graceful shutdown 구현
    [ ] 요청 단위 structured logging (session_id 포함) 연결
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.core.config import settings
from app.core.logger import get_logger
from app.api.router import api_router
from app.services.embedding.embedding_singleton import EmbeddingSingleton
from app.services.session.session_manager import SessionManager
from app.storage.file_store import ensure_storage_dir

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ---- startup ----
    logger.info("Starting %s", settings.APP_NAME)
    ensure_storage_dir(settings.LOCAL_FILE_DIR)
    EmbeddingSingleton.warmup()           # load model ONCE
    SessionManager.instance()             # init global memory dict
    yield
    # ---- shutdown ----
    logger.info("Shutting down %s", settings.APP_NAME)
    SessionManager.instance().purge_all()


def create_app() -> FastAPI:
    app = FastAPI(title=settings.APP_NAME, lifespan=lifespan)
    app.include_router(api_router)
    return app


app = create_app()
