"""
app/services/orchestrator/pipeline.py
-------------------------------------
역할: 전체 흐름을 담당하는 핵심 파이프라인.

최종 baseline 설계 포인트:
1. PDF 업로드가 있는 턴에서는 무거운 문서 전처리(parse/chunk/summary/index)를
   background ingest task로 바로 시작한다.
2. 동시에 user message append / resolve / intent / topic / router 판정을 먼저 수행한다.
3. 라우터가 RAG 또는 document summary가 필요한 경로를 선택한 경우에만
   ingest task 완료까지 기다린다.
4. Memory State Generator만 학습 모듈로 가정하고,
   나머지(router / search prep / retriever / answer)는 비학습 baseline으로 유지한다.

이 구조 덕분에:
- 단순 정의 질문은 PDF 전처리 완료를 기다리지 않고 빠르게 DIRECT_ANSWER 가능
- 문서 검색이 필요한 질문만 준비 완료까지 await 후 RAG 수행

TODO:
    [ ] router 결과와 실제 answer 품질을 비교하여 auto-label 로그 저장
    [ ] 문서 준비 상태(progress %)를 Node.js에 실시간 push
    [ ] Search Prep 이후 query cache / retrieval cache를 실제 도입
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
    #    resolve/intent/topic은 무겁지 않지만 개념적으로는 Memory State Generator 입력을
    #    구성하는 단계다. PDF ingest와 동시에 먼저 수행할 수 있다.
    resolved_task = asyncio.create_task(asyncio.to_thread(resolve, session, req.user_text))
    intent_task = asyncio.create_task(asyncio.to_thread(extract_intent, session, req.user_text))
    topic_task = asyncio.create_task(asyncio.to_thread(update_topic, session, req.user_text))

    # 이미지가 있는 경우는 router와 별개로 최종 모델 선택에만 사용한다.
    has_attachment = bool(session.pdf_state.active_pdf)
    router_task = asyncio.create_task(
        asyncio.to_thread(judge_route, session, req.user_text, has_attachment)
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
            # 문서 검색이 필요한 경우에만 background ingest 완료를 기다린다.
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
