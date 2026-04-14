"""
app/services/orchestrator/pipeline.py
-------------------------------------
역할: 전체 흐름을 담당하는 핵심 파이프라인.

변경점(이 커밋):
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
"""
from __future__ import annotations

import asyncio
from typing import List

from app.schemas.request import ChatRequest
from app.schemas.response import ChatResponse
from app.core.constants import Role, AnswerType, ModelKind, RouterDecision
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
from app.services.pdf.pdf_ingest import attach_pdf_to_session, ensure_pdf_ingest_started, wait_for_pdf_ready
from app.services.pdf.pdf_retriever import retrieve_relevant
from app.services.router.router_judge import judge_route


async def run(req: ChatRequest) -> ChatResponse:
    # 1) session load ----------------------------------------------------------
    session = await get_or_create(req.session_id)

    # 2) optional PDF attach + background ingest start -------------------------
    ingest_task = None
    if req.file_path and req.file_name:
        current_name = session.pdf_state.active_pdf.file_name if session.pdf_state.active_pdf else None
        should_attach_new_pdf = current_name != req.file_name
        if should_attach_new_pdf:
            with open(req.file_path, "rb") as f:
                data = f.read()
            await attach_pdf_to_session(session, req.file_name, data)
        ingest_task = await ensure_pdf_ingest_started(session)

    # 3) record current user turn immediately ---------------------------------
    append_message(session, Role.USER, req.user_text)

    # 4) memory-side lightweight updates + router judge in parallel ------------
    resolved_task = asyncio.create_task(asyncio.to_thread(resolve, session, req.user_text))
    intent_task = asyncio.create_task(asyncio.to_thread(extract_intent, session, req.user_text))
    topic_task = asyncio.create_task(asyncio.to_thread(update_topic, session, req.user_text))

    has_attachment = bool(session.pdf_state.active_pdf)
    # judge_route is now an async coroutine (it may call a local LLM under the hood)
    router_task = asyncio.create_task(
        judge_route(session, req.user_text, has_attachment)
    )

    resolved = await resolved_task
    await intent_task
    await topic_task
    router_decision = await router_task
    session.runtime_state.last_router_decision = router_decision

    # 5) branch by router decision --------------------------------------------
    pdf_chunks: List = []
    answer_type = AnswerType.MULTITURN_ONLY

    if router_decision == RouterDecision.ASK_CLARIFICATION:
        answer = (
            "질문이 어떤 대상을 가리키는지 조금 더 필요해요. "
            "문서/표/그림/이전 대화 중 무엇을 말하는지 한 번만 더 구체적으로 적어주세요."
        )
    else:
        if router_decision in {RouterDecision.RETRIEVE_DOC, RouterDecision.SEARCH_PREP_THEN_RETRIEVE}:
            await wait_for_pdf_ready(req.session_id)
            session = await get_or_create(req.session_id)

            search_query = resolved
            if router_decision == RouterDecision.SEARCH_PREP_THEN_RETRIEVE:
                search_query = prepare_search_query(session, resolved)

            pdf_chunks = retrieve_relevant(session, search_query)
            answer_type = AnswerType.MULTITURN_WITH_PDF if pdf_chunks else AnswerType.MULTITURN_ONLY

        prompt = build_prompt(session, resolved, pdf_chunks)
        model = pick_model(req.image_b64)
        if model == ModelKind.VLM:
            answer = await VLMClient.generate(prompt, req.image_b64 or "")
            answer_type = AnswerType.MULTITURN_WITH_VLM
        else:
            answer = await LLMClient.generate(prompt)

    # 6) finalize --------------------------------------------------------------
    session.runtime_state.last_answer_type = answer_type
    expired, reason = await finalize(session, answer)

    return ChatResponse(
        session_id=req.session_id,
        answer=answer,
        answer_type=answer_type,
        expired=expired,
        expire_reason=reason,
    )
