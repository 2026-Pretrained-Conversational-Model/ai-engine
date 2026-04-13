# AI 오케스트레이터 (FastAPI)

**PDF 기반 멀티턴 채팅을 위한 Python AI 오케스트레이터**
Node.js(WebSocket 게이트웨이)와 LLM/VLM 서버(AWS SageMaker) 사이에서 동작합니다.

```
[프론트엔드] ⇄ WS ⇄ [Node.js] ⇄ HTTP/WS ⇄ [FastAPI 오케스트레이터] ⇄ HTTP ⇄ [SageMaker LLM / VLM]
```

---

## 설계 원칙

1. **멀티턴 우선**

   * 모든 요청은 세션 컨텍스트를 기반으로 해석됩니다.
   * PDF/멀티모달은 별도 경로가 아니라 *보조(augmentation)* 역할입니다.

2. **세션별 인메모리 저장**

   * Redis 없이, 하나의 WebSocket 연결 = 하나의 세션 데이터 (글로벌 딕셔너리)

3. **세션 메모리 제한**

   * 메모리 초과 시 세션 종료
   * 메모리 + 로컬 PDF 삭제
   * 클라이언트에 새 채팅 시작 요청

4. **모델 엔드포인트 2개만 사용**

   * `LLM` (텍스트)
   * `VLM` (비전)
   * 둘 다 SageMaker에서 오케스트레이션 없이 직접 호출

5. **임베딩 + FAISS는 프로세스당 1회 로딩**

   * 싱글톤으로 관리, 모든 세션에서 재사용

6. **파일당 하나의 함수**

   * 협업을 위한 최대한의 모듈화

### 핵심 흐름
```get_or_create → (PDF 신규면 save→parse→chunk→summarize→index)
              → append_message(USER)
              → multiturn_resolver.resolve
              → intent + topic 갱신
              → augmentation (PDF 있으면 top-k retrieve)
              → prompt_builder.build
              → pick_model → LLM or VLM
              → finalize (assistant append + summary refresh + memory cap check)
              → ChatResponse(expired=?)
```

---

## 디렉토리 구조

```
app/
├── main.py                       # FastAPI 진입점, 앱 생성
├── core/                         # 설정, 로거, 상수
├── api/endpoints/                # chat / upload / session API
├── schemas/                      # Pydantic 모델 (세션 메모리 구조)
├── services/
│   ├── session/                  # 인메모리 저장소, 생명주기, 메모리 제한
│   ├── conversation/             # 메시지 저장, 의도 분석, 주제 추적, 멀티턴 처리
│   ├── summary/                  # 서술형 + 구조화된 요약 업데이트
│   ├── pdf/                      # 저장 / 파싱 / 청킹 / 요약 / 인덱싱 / 검색
│   ├── embedding/                # 임베딩 모델 싱글톤 + 세션별 FAISS
│   ├── llm/                      # SageMaker LLM/VLM 클라이언트 + 프롬프트 생성
│   └── orchestrator/             # 전체 파이프라인 연결
├── utils/                        # 유틸 함수
└── storage/                      # 글로벌 메모리 딕셔너리 + 로컬 파일 저장소
```

```
ai-orchestrator/
├── README.md                       # 전체 흐름 + mermaid + 실행법
├── .env.example                    # 모든 설정 (메모리캡/SageMaker/임베딩/RAG)
├── .gitignore
├── Dockerfile                      # docker-compose는 직접 작성
├── requirements.txt
├── run.sh
└── app/
    ├── main.py                     # FastAPI factory + lifespan (임베딩 워밍업)
    ├── core/      config / logger / constants
    ├── api/       router + endpoints/{chat, upload, session}
    ├── schemas/   request/response/session/conversation/pdf/runtime  (Pydantic v2)
    ├── storage/   memory_store (asyncio.Lock 보호) + file_store
    └── services/
        ├── session/         manager/creator/getter/updater/cleaner/memory_monitor
        ├── conversation/    appender/recent_window/intent/topic/multiturn_resolver
        ├── summary/         narrative/structured/orchestrator
        ├── pdf/             saver/parser/chunker/summarizer/indexer/retriever
        ├── embedding/       embedding_singleton + faiss_store  ← 프로세스 1회 로드
        ├── llm/             llm_client / vlm_client / prompt_builder / llm_router
        └── orchestrator/    pipeline / augmentation / response_finalizer
```

---

## 요청 처리 흐름 (턴 단위)

```
Node.js → POST /chat
   │
   ▼
session_manager.get_or_create(session_id)
   │
   ├── (새 파일일 경우)
   │     pdf_saver → pdf_parser → pdf_chunker
   │     → pdf_summarizer → pdf_indexer (FAISS)
   │
   ▼
multiturn_resolver  ("그거", "아까 말한 것" 해석)
   │
   ▼
augmentation.maybe_attach_pdf_context  (PDF가 있을 때만)
   │
   ▼
prompt_builder.build()
   │
   ▼
llm_router → llm_client 또는 vlm_client (SageMaker 호출)
   │
   ▼
response_finalizer → message_appender → summary_orchestrator
   │
   ▼
memory_monitor.check()
   │
   ├── (메모리 초과 시)
   │       session_cleaner.purge()
   │              │
   │              ▼
   │         response = {expired: true}
   ▼
Node.js로 응답 반환
```

---

## 실행 방법

```bash
cp .env.example .env
pip install -r requirements.txt
./run.sh
# 또는
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## Docker 실행

```bash
docker build -t ai-orchestrator .
docker run --env-file .env -p 8000:8000 ai-orchestrator
```

