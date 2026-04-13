"""
app/services/orchestrator/pipeline.py
-------------------------------------
역할: 전체 흐름을 담당하는 핵심 파이프라인
      README.md의 mermaid 다이어그램과 동일한 구조
      모든 /chat 요청은 이곳으로 들어옴

      단계:
        1. 세션 조회 또는 생성 (get_or_create)
        2. (업로드가 있는 경우) PDF 파싱 → 청킹 → 요약 → 인덱싱
        3. 사용자 메시지 추가
        4. 멀티턴 기반으로 사용자 질의 해석
        5. 의도(intent) 및 주제(topic) 업데이트
        6. (선택적) PDF 보강(augmentation)
        7. 프롬프트 생성
        8. LLM 또는 VLM 라우팅 후 호출
        9. 후처리 (메시지 추가 + 요약 갱신 + 메모리 제한 검사)

TODO:
    [ ] 각 단계별 처리 시간 메트릭 추가
    [ ] WebSocket 연결 종료 시 진행 중인 LLM 호출 취소
"""
from app.schemas.request import ChatRequest
from app.schemas.response import ChatResponse
from app.core.constants import Role, ResourceType, AnswerType, ModelKind

from app.services.session.session_getter import get_or_create
from app.services.conversation.message_appender import append_message
from app.services.conversation.multiturn_resolver import resolve
from app.services.conversation.intent_extractor import extract_intent
from app.services.conversation.topic_tracker import update_topic
from app.services.orchestrator.augmentation import maybe_attach_pdf_context
from app.services.llm.prompt_builder import build_prompt
from app.services.llm.llm_router import pick_model
from app.services.llm.llm_client import LLMClient
from app.services.llm.vlm_client import VLMClient
from app.services.orchestrator.response_finalizer import finalize
from app.services.pdf.pdf_saver import save_pdf
from app.services.pdf.pdf_parser import parse_pdf
from app.services.pdf.pdf_chunker import chunk_pages
from app.services.pdf.pdf_summarizer import summarize_document
from app.services.pdf.pdf_indexer import index_chunks


async def run(req: ChatRequest) -> ChatResponse:
    # 1. session
    session = await get_or_create(req.session_id)

    # 2. (optional) ingest PDF that Node.js already saved on disk
    if req.file_path and req.file_name and not session.pdf_state.active_pdf:
        with open(req.file_path, "rb") as f:
            data = f.read()
        meta = save_pdf(req.session_id, req.file_name, data)
        page_count, pages = parse_pdf(meta.file_path)
        meta.page_count = page_count
        chunks = chunk_pages(meta.file_id, pages)
        session.pdf_state.active_pdf = meta
        session.pdf_state.pdf_index.chunks = chunks
        session.pdf_state.doc_summary = await summarize_document(pages)
        index_chunks(req.session_id, chunks)
        session.runtime_state.active_resource_type = ResourceType.PDF
        session.runtime_state.active_pdf_id = meta.file_id

    # 3. record user turn
    append_message(session, Role.USER, req.user_text)

    # 4-5. resolve + intent + topic
    resolved = resolve(session, req.user_text)
    extract_intent(session, req.user_text)
    update_topic(session, req.user_text)

    # 6. PDF augmentation (no-op if no active PDF)
    pdf_chunks = maybe_attach_pdf_context(session, resolved)

    # 7. prompt
    prompt = build_prompt(session, resolved, pdf_chunks)

    # 8. model call
    model = pick_model(req.image_b64)
    if model == ModelKind.VLM:
        answer = await VLMClient.generate(prompt, req.image_b64 or "")
        answer_type = AnswerType.MULTITURN_WITH_VLM
    else:
        answer = await LLMClient.generate(prompt)
        answer_type = (
            AnswerType.MULTITURN_WITH_PDF if pdf_chunks else AnswerType.MULTITURN_ONLY
        )
    session.runtime_state.last_answer_type = answer_type

    # 9. finalize + memory-cap check
    expired, reason = await finalize(session, answer)

    return ChatResponse(
        session_id=req.session_id,
        answer=answer,
        answer_type=answer_type,
        expired=expired,
        expire_reason=reason,
    )
