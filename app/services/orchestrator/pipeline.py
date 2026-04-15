"""
app/services/orchestrator/pipeline.py
-------------------------------------
역할: 전체 흐름을 담당하는 핵심 파이프라인.

변경점:
- judge_route()가 async 함수로 바뀜에 따라,
  router_task 생성 시 asyncio.to_thread 래핑을 제거하고
  직접 judge_route 코루틴을 create_task로 스케줄링한다.
  (resolve/intent/topic은 여전히 동기 함수라 to_thread 유지.)

기존 설계 포인트는 유지:
1. PDF 업로드가 있는 턴에서는 무거운 문서 전처리를 background ingest task로 시작.
2. 동시에 user message append / resolve / intent / topic / router 판정을 먼저 수행.
3. 라우터가 RAG 또는 document summary가 필요한 경로를 선택한 경우에만
   ingest task 완료까지 기다린다.
4. Memory State Generator만 학습 모듈로 가정하고,
   나머지(router / search prep / retriever / answer)는 비학습 baseline으로 유지.

변경점:
- judge_route()가 async 함수이므로 직접 create_task로 스케줄링
- turn / stage / memory 로그 추가
- 각 레이어 결과값을 추적 가능하게 로깅 강화
"""
from __future__ import annotations

import asyncio
from typing import List

from app.schemas.request import ChatRequest
from app.schemas.response import ChatResponse
from app.core.constants import Role, AnswerType, ModelKind, RouterDecision
from app.core.logger import (
    get_logger,
    log_stage_start,
    log_stage_end,
    log_state,
    log_memory,
)
from app.services.session.session_getter import get_or_create
from app.services.conversation.message_appender import append_message
from app.services.conversation.multiturn_resolver import resolve
from app.services.conversation.intent_extractor import extract_intent
from app.services.conversation.topic_tracker import update_topic
from app.services.llm.prompt_builder import build_prompt
from app.services.llm.llm_router import pick_model
from app.services.llm.llm_client import LLMClient
from app.services.llm.vlm_client import VLMClient
from app.services.orchestrator.response_finalizer import finalize
from app.services.orchestrator.search_prep import prepare_search_query
from app.services.pdf.pdf_ingest import (
    attach_pdf_to_session,
    ensure_pdf_ingest_started,
    wait_for_pdf_ready,
)
from app.services.pdf.pdf_retriever import retrieve_relevant
from app.services.router.router_judge import judge_route
from app.services.memory.memory_state_generator import update_memory_state

logger = get_logger(__name__)


async def run(req: ChatRequest) -> ChatResponse:
    log_stage_start(
        logger,
        "TURN",
        session_id=req.session_id,
        user_text=req.user_text,
        has_pdf=bool(req.file_path and req.file_name),
        has_image=bool(req.image_b64),
        file_name=req.file_name,
    )

    # 1) session load ----------------------------------------------------------
    log_stage_start(logger, "SESSION_LOAD", session_id=req.session_id)
    session = await get_or_create(req.session_id)
    log_stage_end(logger, "SESSION_LOAD", session_id=req.session_id)
    log_memory(logger, session, "after_session_load")

    # 2) optional PDF attach + background ingest start -------------------------
    ingest_task = None
    if req.file_path and req.file_name:
        log_stage_start(
            logger,
            "PDF_ATTACH_OR_INGEST",
            session_id=req.session_id,
            file_name=req.file_name,
            file_path=req.file_path,
        )

        current_name = (
            session.pdf_state.active_pdf.file_name
            if session.pdf_state.active_pdf
            else None
        )
        should_attach_new_pdf = current_name != req.file_name

        log_state(
            logger,
            "pdf_attach_decision",
            current_name=current_name,
            requested_name=req.file_name,
            should_attach_new_pdf=should_attach_new_pdf,
        )

        if should_attach_new_pdf:
            with open(req.file_path, "rb") as f:
                data = f.read()
            log_state(
                logger,
                "pdf_bytes_read",
                file_name=req.file_name,
                num_bytes=len(data),
            )
            await attach_pdf_to_session(session, req.file_name, data)
            log_state(logger, "pdf_attached", file_name=req.file_name)

        ingest_task = await ensure_pdf_ingest_started(session)
        log_stage_end(
            logger,
            "PDF_ATTACH_OR_INGEST",
            ingest_task_started=bool(ingest_task),
        )
        log_memory(logger, session, "after_pdf_attach")

    # 3) record current user turn immediately ---------------------------------
    log_stage_start(logger, "APPEND_USER_MESSAGE", user_text=req.user_text)
    append_message(session, Role.USER, req.user_text)
    log_stage_end(logger, "APPEND_USER_MESSAGE", appended=True)
    log_memory(logger, session, "after_user_append")

    # 4) memory-side lightweight updates + router judge in parallel ------------
    log_stage_start(logger, "PARALLEL_MEMORY_AND_ROUTER")

    resolved_task = asyncio.create_task(asyncio.to_thread(resolve, session, req.user_text))
    intent_task = asyncio.create_task(asyncio.to_thread(extract_intent, session, req.user_text))
    topic_task = asyncio.create_task(asyncio.to_thread(update_topic, session, req.user_text))

    has_attachment = bool(session.pdf_state.active_pdf)
    router_task = asyncio.create_task(
        judge_route(session, req.user_text, has_attachment)
    )

    resolved = await resolved_task
    intent_result = await intent_task
    topic_result = await topic_task
    router_decision = await router_task
    session.runtime_state.last_router_decision = router_decision

    log_state(logger, "resolved_result", resolved=resolved)
    log_state(logger, "intent_result", result=intent_result)
    log_state(logger, "topic_result", result=topic_result)
    log_state(logger, "router_decision", decision=str(router_decision))

    log_stage_end(
        logger,
        "PARALLEL_MEMORY_AND_ROUTER",
        resolved=resolved,
        router_decision=str(router_decision),
    )
    log_memory(logger, session, "after_memory_and_router")

    # 5) branch by router decision --------------------------------------------
    pdf_chunks: List = []
    answer_type = AnswerType.MULTITURN_ONLY

    if router_decision == RouterDecision.ASK_CLARIFICATION:
        log_stage_start(logger, "ASK_CLARIFICATION_BRANCH")
        answer = (
            "질문이 어떤 대상을 가리키는지 조금 더 필요해요. "
            "문서/표/그림/이전 대화 중 무엇을 말하는지 한 번만 더 구체적으로 적어주세요."
        )
        log_stage_end(
            logger,
            "ASK_CLARIFICATION_BRANCH",
            answer_preview=answer[:120],
        )
    else:
        if router_decision in {
            RouterDecision.RETRIEVE_DOC,
            RouterDecision.SEARCH_PREP_THEN_RETRIEVE,
        }:
            log_stage_start(
                logger,
                "RETRIEVE_BRANCH",
                router_decision=str(router_decision),
            )

            await wait_for_pdf_ready(req.session_id)
            session = await get_or_create(req.session_id)
            log_memory(logger, session, "after_pdf_ready")

            search_query = resolved
            if router_decision == RouterDecision.SEARCH_PREP_THEN_RETRIEVE:
                search_query = prepare_search_query(session, resolved)

            log_state(logger, "search_query", query=search_query)

            pdf_chunks = retrieve_relevant(session, search_query)
            answer_type = (
                AnswerType.MULTITURN_WITH_PDF if pdf_chunks else AnswerType.MULTITURN_ONLY
            )

            log_state(
                logger,
                "retrieved_chunks",
                count=len(pdf_chunks),
                preview=pdf_chunks[:1] if pdf_chunks else [],
                answer_type=str(answer_type),
            )

            log_stage_end(
                logger,
                "RETRIEVE_BRANCH",
                chunk_count=len(pdf_chunks),
            )

        log_stage_start(logger, "PROMPT_BUILD")
        prompt = build_prompt(session, resolved, pdf_chunks)
        log_stage_end(
            logger,
            "PROMPT_BUILD",
            prompt_length=len(prompt),
            prompt_preview=prompt[:300],
        )

        model = pick_model(req.image_b64)
        log_state(logger, "model_selected", model=str(model))

        if model == ModelKind.VLM:
            log_stage_start(logger, "VLM_GENERATE")
            answer = await VLMClient.generate(prompt, req.image_b64 or "")
            answer_type = AnswerType.MULTITURN_WITH_VLM
            log_stage_end(
                logger,
                "VLM_GENERATE",
                answer_length=len(answer),
                answer_preview=answer[:200],
                answer_type=str(answer_type),
            )
        else:
            log_stage_start(logger, "LLM_GENERATE", role="answer")
            answer = await LLMClient.generate(prompt, role="answer")
            log_stage_end(
                logger,
                "LLM_GENERATE",
                answer_length=len(answer),
                answer_preview=answer[:200],
                answer_type=str(answer_type),
            )

    # 6) finalize --------------------------------------------------------------
    log_stage_start(logger, "FINALIZE")
    session.runtime_state.last_answer_type = answer_type
    log_memory(logger, session, "before_finalize")

    expired, reason = await finalize(session, answer)

    log_stage_end(
        logger,
        "FINALIZE",
        expired=expired,
        expire_reason=reason,
        answer_type=str(answer_type),
    )
    log_memory(logger, session, "after_finalize")

    # 7) Memory State Generator -----------------------------------------------
    if not expired:
        log_stage_start(logger, "MEMORY_STATE_GENERATOR")
        session = await get_or_create(req.session_id)
        await update_memory_state(session)
        from app.services.session.session_updater import save_session
        await save_session(session)
        log_stage_end(
            logger,
            "MEMORY_STATE_GENERATOR",
            narrative_length=len(session.conversation.summary.narrative),
            topic=session.conversation.current_topic,
        )
        log_memory(logger, session, "after_memory_state_generator")

    response = ChatResponse(
        session_id=req.session_id,
        answer=answer,
        answer_type=answer_type,
        expired=expired,
        expire_reason=reason,
    )

    log_stage_end(
        logger,
        "TURN",
        session_id=req.session_id,
        answer_type=str(answer_type),
        expired=expired,
        answer_preview=answer[:200],
    )

    return response