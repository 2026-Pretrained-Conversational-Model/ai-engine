"""
notebooks/colab_loop_helper.py
------------------------------
Colab에서 pipeline.run()을 호출하면서 동시에
background fire-and-forget task (memory update 등)가 실제로 실행되도록 하는
영구 이벤트 루프 헬퍼.

문제 배경:
    Colab의 일반적 사용 패턴인 asyncio.run() / asyncio.run_until_complete()는
    코루틴이 끝나면 즉시 이벤트 루프를 종료한다. 이 때문에 finalize()가
    create_task()로 스케줄한 memory update task가 실행 기회를 못 잡고
    고아가 되어버린다.

해결:
    별도 daemon 스레드 하나를 띄워서 거기에 asyncio 이벤트 루프를 영구 실행.
    노트북의 chat() 호출은 run_coroutine_threadsafe()로 그 루프에 코루틴을
    제출하고 결과만 기다린다. pipeline.run()이 끝나도 루프는 살아있어서
    create_task()한 background task가 계속 실행됨.

사용법 (노트북 셀):

    # 앱 import 이후 이 모듈 import
    from notebooks.colab_loop_helper import chat, dump_session, shutdown_loop

    # 이후 chat() 호출만 하면 됨. 내부적으로 pipeline.run()을 돌리고
    # memory update background task도 정상 실행됨.
    r = chat(session_id="test-1", user_text="안녕, 내 이름은 김예슬이야")
    print(r["answer"])
"""
from __future__ import annotations

import asyncio
import threading
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Persistent event loop on a daemon thread
# ---------------------------------------------------------------------------

_loop: Optional[asyncio.AbstractEventLoop] = None
_loop_thread: Optional[threading.Thread] = None


def _ensure_loop() -> asyncio.AbstractEventLoop:
    global _loop, _loop_thread
    if _loop is not None and _loop.is_running():
        return _loop

    _loop = asyncio.new_event_loop()

    def _runner():
        asyncio.set_event_loop(_loop)
        _loop.run_forever()

    _loop_thread = threading.Thread(target=_runner, name="ai-orch-loop", daemon=True)
    _loop_thread.start()
    return _loop


def run_coro(coro) -> Any:
    """
    코루틴을 영구 루프에 제출하고 결과가 올 때까지 block.
    노트북에서 `result = run_coro(some_async_fn(...))` 형태로 사용.
    """
    loop = _ensure_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result()


# ---------------------------------------------------------------------------
# High-level chat() helper
# ---------------------------------------------------------------------------

def chat(
    session_id: str,
    user_text: str,
    file_path: Optional[str] = None,
    file_name: Optional[str] = None,
    image_b64: Optional[str] = None,
) -> Dict[str, Any]:
    """
    pipeline.run()을 영구 루프에서 실행하고 ChatResponse dict을 반환.
    """
    from app.schemas.request import ChatRequest
    from app.services.orchestrator.pipeline import run as pipeline_run

    req = ChatRequest(
        session_id=session_id,
        user_text=user_text,
        file_path=file_path,
        file_name=file_name,
        image_b64=image_b64,
    )
    resp = run_coro(pipeline_run(req))
    return resp.model_dump()


def dump_session(session_id: str) -> Dict[str, Any] | None:
    """
    세션 현재 상태 JSON으로 덤프 (디버깅용).
    memory update가 background에서 돌고 있으면 최신 summary까지 보고 싶을 때 사용.
    """
    from app.services.session.session_manager import SessionManager

    async def _get():
        s = await SessionManager.instance().get(session_id)
        return s.model_dump(mode="json") if s else None

    return run_coro(_get())


def wait_for_background(timeout: float = 30.0) -> None:
    """
    현재 pending인 background task가 모두 끝날 때까지 기다림 (테스트/디버깅용).
    memory update가 아직 안 끝났는데 다음 턴을 바로 이어가기 애매할 때 호출.
    """
    async def _gather():
        tasks = [t for t in asyncio.all_tasks(asyncio.get_running_loop())
                 if t is not asyncio.current_task() and not t.done()]
        if tasks:
            await asyncio.wait(tasks, timeout=timeout)

    run_coro(_gather())


def shutdown_loop() -> None:
    """노트북 세션 끝낼 때 정리용. 보통 안 불러도 됨 (daemon thread라 프로세스 종료 시 자동 정리)."""
    global _loop, _loop_thread
    if _loop is None:
        return
    _loop.call_soon_threadsafe(_loop.stop)
    if _loop_thread is not None:
        _loop_thread.join(timeout=3.0)
    _loop = None
    _loop_thread = None
