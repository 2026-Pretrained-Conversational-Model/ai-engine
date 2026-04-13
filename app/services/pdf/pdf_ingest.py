"""
app/services/pdf/pdf_ingest.py
-----------------------------
역할: PDF 준비 전체를 담당하는 오케스트레이터.

이 모듈이 구현하는 핵심 정책:
1. PDF 업로드 직후 parse/chunk/summary/index를 백그라운드 task로 시작한다.
2. memory 갱신과 router 판단은 이 task와 병렬로 먼저 진행할 수 있다.
3. 라우터가 RAG 필요로 판단한 경우에만 전처리 완료까지 기다린다.
4. 같은 세션에서 이미 돌고 있는 ingest task는 재사용한다.
5. 같은 파일(file_hash)이면 parser/summary/embedding 결과 캐시를 재사용한다.

TODO:
    [ ] OCR fallback 실제 구현
    [ ] chunking/embedding을 페이지 batch 단위로 더 세밀하게 병렬화
    [ ] 실패 시 재시도 정책(exponential backoff) 추가
"""
from __future__ import annotations

import asyncio
import hashlib
from typing import Optional

from app.core.constants import IngestStatus, ResourceType
from app.core.logger import get_logger
from app.schemas.pdf import PdfMeta
from app.schemas.session import Session
from app.services.embedding.embedding_singleton import EmbeddingSingleton
from app.services.pdf.document_cache import DocumentCache
from app.services.pdf.pdf_chunker import chunk_pages
from app.services.pdf.pdf_indexer import index_chunks
from app.services.pdf.pdf_parser import parse_pdf
from app.services.pdf.pdf_saver import save_pdf
from app.services.pdf.pdf_summarizer import summarize_document
from app.services.pdf.preprocess_registry import PreprocessRegistry
from app.services.session.session_updater import save_session
from app.services.embedding.faiss_store import FaissStore

logger = get_logger(__name__)


def compute_file_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


async def attach_pdf_to_session(session: Session, file_name: str, data: bytes) -> PdfMeta:
    """
    업로드된 PDF를 세션에 '활성 문서'로 연결만 먼저 수행한다.

    중요한 점:
    - 여기서는 파일 저장과 메타데이터 연결만 빠르게 끝낸다.
    - 무거운 parse/chunk/summary/index는 별도 background ingest task에서 처리한다.
    - 따라서 라우터는 PDF 전처리 완료를 기다리지 않고도 먼저 판단 가능하다.
    """
    file_hash = compute_file_hash(data)
    meta = save_pdf(session.session_meta.session_id, file_name, data, file_hash=file_hash)
    session.pdf_state.active_pdf = meta
    session.pdf_state.doc_summary.one_line = ""
    session.pdf_state.doc_summary.section_summaries = []
    session.pdf_state.pdf_index.chunks = []
    session.pdf_state.ingest_status = IngestStatus.RUNNING
    session.pdf_state.ingest_error = ""
    session.runtime_state.active_resource_type = ResourceType.PDF
    session.runtime_state.active_pdf_id = meta.file_id
    session.runtime_state.pdf_ingest_status = IngestStatus.RUNNING
    await save_session(session)
    return meta


async def ensure_pdf_ingest_started(session: Session) -> Optional[asyncio.Task]:
    """
    현재 세션의 active_pdf에 대해 ingest task가 필요하면 시작하고,
    이미 실행 중이면 그 task를 반환한다.
    """
    active_pdf = session.pdf_state.active_pdf
    if not active_pdf:
        return None

    registry = PreprocessRegistry.instance()
    existing = registry.get(session.session_meta.session_id)
    if existing:
        return existing

    task = asyncio.create_task(_ingest_pdf(session.session_meta.session_id))
    registry.set(session.session_meta.session_id, task)
    return task


async def wait_for_pdf_ready(session_id: str) -> None:
    """
    라우터가 문서 검색이 필요하다고 판단한 경우 호출한다.
    아직 돌고 있는 ingest task가 있으면 완료까지 기다린다.
    """
    task = PreprocessRegistry.instance().get(session_id)
    if task:
        await task


async def _ingest_pdf(session_id: str) -> None:
    registry = PreprocessRegistry.instance()
    try:
        from app.services.session.session_getter import get_or_create

        session = await get_or_create(session_id)
        active_pdf = session.pdf_state.active_pdf
        if not active_pdf:
            return

        file_hash = active_pdf.file_hash
        cache = DocumentCache.instance()

        # ---- 1) Parse PDF ----------------------------------------------------
        parsed = cache.get_parsed(file_hash)
        if parsed is None:
            page_count, pages = await asyncio.to_thread(parse_pdf, active_pdf.file_path)
            cache.set_parsed(file_hash, page_count, pages)
        else:
            page_count, pages = parsed.page_count, parsed.pages

        # 세션 쪽 메타데이터를 먼저 갱신해두면,
        # 이후 라우터/답변기가 '총 몇 페이지 문서인지' 정도는 사용할 수 있다.
        session = await get_or_create(session_id)
        if session.pdf_state.active_pdf:
            session.pdf_state.active_pdf.page_count = page_count
            await save_session(session)

        # ---- 2) Chunk build --------------------------------------------------
        chunk_entry = cache.get_chunks(file_hash)
        if chunk_entry is None:
            chunks = await asyncio.to_thread(chunk_pages, active_pdf.file_id, pages)
            cache.set_chunks(file_hash, chunks)
        else:
            chunks = [c.model_copy(deep=True) for c in chunk_entry.chunks]
            # cache에는 이전 file_id로 만든 chunk_id가 있을 수 있으므로,
            # 현재 active_pdf.file_id 기준으로 chunk_id를 다시 부여한다.
            for i, chunk in enumerate(chunks, start=1):
                chunk.chunk_id = f"{active_pdf.file_id}_c{i:04d}"
                chunk.embedding_ref = None

        # ---- 3) Summary ------------------------------------------------------
        summary_entry = cache.get_summary(file_hash)
        if summary_entry is None:
            summary = await summarize_document(pages)
            cache.set_summary(file_hash, summary)
        else:
            summary = summary_entry.summary.model_copy(deep=True)

        # ---- 4) Embedding + FAISS -------------------------------------------
        # 같은 세션에 새 PDF가 올라오면 이전 문서 인덱스는 폐기하고 재구축한다.
        FaissStore.instance().reset(session_id)
        embedding_entry = cache.get_embeddings(file_hash)
        if embedding_entry is None:
            # warmup은 호출 비용이 큰 편이므로 ingest path에서 미리 해둔다.
            EmbeddingSingleton.get()
            await asyncio.to_thread(index_chunks, session_id, chunks, file_hash, True)
        else:
            await asyncio.to_thread(index_chunks, session_id, chunks, file_hash, False)

        # ---- 5) Session commit -----------------------------------------------
        session = await get_or_create(session_id)
        session.pdf_state.pdf_index.chunks = chunks
        session.pdf_state.doc_summary = summary
        session.pdf_state.ingest_status = IngestStatus.READY
        session.pdf_state.ingest_error = ""
        session.runtime_state.pdf_ingest_status = IngestStatus.READY
        await save_session(session)
        logger.info("PDF ingest ready: session=%s file_id=%s", session_id, active_pdf.file_id)
    except Exception as e:  # pragma: no cover - defensive logging path
        logger.exception("PDF ingest failed for session=%s: %s", session_id, e)
        from app.services.session.session_getter import get_or_create
        session = await get_or_create(session_id)
        session.pdf_state.ingest_status = IngestStatus.FAILED
        session.pdf_state.ingest_error = str(e)
        session.runtime_state.pdf_ingest_status = IngestStatus.FAILED
        session.runtime_state.last_error = str(e)
        await save_session(session)
    finally:
        registry.clear(session_id)
