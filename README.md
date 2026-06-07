# ai-engine — AI 오케스트레이터 (FastAPI)

> **Multiturn Memory Chat의 핵심 엔진.** 한 번의 사용자 메시지를 받아 세션 메모리를 읽고, 라우팅을 판단하고, 필요하면 PDF를 검색해, 답하고, 메모리를 갱신하는 **턴 파이프라인**입니다.

전체 프로젝트 개요와 다른 구성 저장소는 대표 저장소 [docs](https://github.com/2026-Pretrained-Conversational-Model/docs)를 참고하세요.

```
[프론트엔드] ⇄ WS ⇄ [Node.js 게이트웨이] ⇄ HTTP ⇄ ★[FastAPI 오케스트레이터]★ ⇄ HTTP ⇄ [SageMaker / 로컬 Qwen LLM·VLM]
```

이 저장소는 위 4계층에서 ★ 위치를 담당합니다. 브라우저가 직접 호출하지 않고 내부 게이트웨이에서만 호출되므로 CORS 미들웨어가 없습니다.

---

## 설계 원칙

1. **멀티턴 우선** — 모든 요청은 세션 컨텍스트로 해석하고, PDF/이미지는 보조(augmentation)입니다.
2. **학습 대상은 메모리 모델 하나** — 라우터·검색·답변은 비학습 baseline입니다.
3. **세션별 인메모리 저장** — Redis 없이 프로세스 전역 딕셔너리(asyncio.Lock 보호).
4. **세션 메모리 캡** — 초과 시 세션을 파기하고 `expired=true` 반환.
5. **임베딩/FAISS는 프로세스당 1회 로딩** — 싱글톤으로 모든 세션이 재사용.

---

## 턴 처리 흐름

```
POST /chat
  → 세션 로드
  → (PDF 신규 첨부 시) 저장 + 백그라운드 ingest 시작  ── 기다리지 않음
  → 지시어 해소 · 의도 추출 · 주제 갱신 · 라우터 판단   ── 병렬
  → 라우터 분기
       ASK_CLARIFICATION       : 모델 호출 없이 되묻기
       RETRIEVE / SEARCH_PREP  : PDF 준비 대기 → top-k 검색
       DIRECT_ANSWER           : 검색 없이 진행
  → 프롬프트 빌드(system+user 분리: 메모리·최근대화·검색결과)
  → LLM 또는 VLM 호출
  → finalize: assistant 기록 + 저장 + (3턴마다) 메모리 갱신(백그라운드) + 메모리 캡 검사
  → ChatResponse(answer, answer_type, expired)
```

---

## 사용 모델

| 역할 | 모델 | 비고 |
| --- | --- | --- |
| Answer | Qwen2.5-7B-Instruct | 4bit NF4 |
| Router | Qwen2.5-3B-Instruct | RAG 필요 판단 |
| Memory | `yeseul0-0/qwen2.5-3b-memory-summary-default_v0.3` | 메모리 요약 LoRA 파인튜닝 |
| Embedding | jhgan/ko-sroberta-multitask | 768차원 |

`LLM_BACKEND=local`이면 직접 로딩한 모델(LocalModelRegistry)을, `sagemaker`면 원격 엔드포인트를 사용합니다. 어느 경로든 실패 시 라우터는 휴리스틱으로, 메모리 갱신은 no-op으로 안전하게 폴백합니다.

---

## 디렉터리 구조

```
app/
├── main.py                  FastAPI 진입점 + lifespan(임베딩 워밍업)
├── core/                    config / logger / constants
├── api/endpoints/           chat / upload / session
├── schemas/                 Pydantic v2 (세션 메모리 구조)
├── storage/                 인메모리 세션 딕셔너리 + 로컬 파일 저장소
└── services/
    ├── session/             생성/조회/갱신/정리/메모리 모니터
    ├── conversation/        메시지 append / 지시어 해소 / 의도 / 주제
    ├── memory/              ★ Memory State Generator (학습 모델)
    ├── router/              RAG 라우터(LLM 판단 + 휴리스틱 폴백)
    ├── pdf/                 저장/파싱/청킹/요약/인덱싱/검색 + 캐시
    ├── embedding/           임베딩 싱글톤 + 세션별 FAISS
    ├── llm/                 LLM/VLM 클라이언트 + 프롬프트 빌더
    └── orchestrator/        파이프라인 / 검색준비 / 응답 finalize
```

---

## 실행

```bash
cp .env.example .env
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
# 또는
./run.sh
```

주요 환경 변수는 [.env.example](.env.example)에 정리되어 있습니다(메모리 캡, RAG 청크, 메모리 갱신 주기, LLM 백엔드 등).

### Colab / RunPod (LLM_BACKEND=local)
GPU 환경에서 7B+3B+3B를 4bit로 함께 올려 구동합니다. RunPod 기동은 [ai-pod-v1](https://github.com/2026-Pretrained-Conversational-Model/ai-pod-v1) 저장소를 참고하세요.

### Docker
```bash
docker build -t ai-orchestrator .
docker run --env-file .env -p 8000:8000 ai-orchestrator
```

---

## 담당 역할

- **김예슬 (팀장)** — 전체 턴 파이프라인·세션 메모리 구조 설계, 오케스트레이터 전반, 메모리 요약 모델 파인튜닝
- **진주용** — 라우터(RAG 필요 판단 및 경로 결정)
- **이지선** — 임베딩 싱글톤 / FAISS 검색(RAG retrieval baseline)
