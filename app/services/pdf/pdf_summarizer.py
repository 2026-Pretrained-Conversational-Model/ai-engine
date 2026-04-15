"""
app/services/pdf/pdf_summarizer.py
----------------------------------
역할: 새로 파싱된 PDF에 대해 doc_summary를 생성한다.

v11 핵심 변경 (PDF 첨부 시 중국어/영어 답변 문제의 **진짜 원인** 해결):

기존 버전은 stub으로 `pages[0][:140]`를 날것 그대로 one_line에 박았다.
학술 PDF는 보통 영문, 중국어 PDF면 중국어 문자열이 그대로 들어갔고,
이게 prompt_builder의 [Document Summary] 블록에 그대로 올라가서
answer LLM이 "아 이 세션의 주 언어는 영어/중국어구나"라고 오판했다.

v11 방식:
1. summary LLM이 등록돼 있으면 그걸 써서 **한국어로** 한 줄 요약을 만든다
   (`summary` 또는 `memory` role 재사용).
2. LLM이 없으면 fallback으로 "[파일명] (총 N페이지)" 같은
   **메타정보만** 담은 안전한 한국어 문자열을 쓴다.
   **절대로 원문 페이지 텍스트를 그대로 prompt에 넣지 않는다.**

이 모듈은 PDF ingest 파이프라인(pdf_ingest.py)에서 호출되며,
결과는 session.pdf_state.doc_summary에 들어간다.
"""
from __future__ import annotations

import asyncio
import re
from typing import List

from app.core.logger import get_logger
from app.schemas.pdf import DocSummary, SectionSummary

logger = get_logger(__name__)


_SUMMARY_SYSTEM = (
    "당신은 문서 요약 도우미입니다. 주어진 PDF 본문 일부를 읽고, "
    "한국어로 한 문장(최대 80자)의 요약을 만드세요. "
    "원문이 어떤 언어이든 반드시 한국어로만 답하세요. "
    "설명이나 메타 코멘트 없이 요약 문장 하나만 출력하세요."
)


def _clean_text(text: str) -> str:
    """연속 공백/개행 정리. 프롬프트에 들어갈 수 있는 안전한 형태로."""
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _fallback_one_line(pages: List[str], file_name: str | None = None) -> str:
    """
    LLM이 없을 때의 안전 폴백.
    원문 언어가 무엇이든 상관없는 한국어 메타데이터 문자열만 반환한다.
    """
    page_count = len(pages)
    name = file_name or "업로드 문서"
    return f"첨부된 PDF 문서입니다. (파일: {name}, 총 {page_count}페이지)"


async def summarize_document(
    pages: List[str],
    file_name: str | None = None,
) -> DocSummary:
    """
    PDF 전체 요약을 생성한다.

    Args:
        pages: 페이지별 텍스트 리스트
        file_name: 원본 파일명 (폴백 메시지에 씀)

    Returns:
        DocSummary(one_line=..., section_summaries=[])
        one_line은 항상 한국어 문자열이거나 안전한 메타문자열.
    """
    if not pages:
        return DocSummary(one_line="", section_summaries=[])

    # local import: 순환 import 회피
    from app.services.llm.local_registry import LocalModelRegistry

    # summary 또는 memory role 모델이 있으면 LLM로 요약
    role = None
    if LocalModelRegistry.has("summary"):
        role = "summary"
    elif LocalModelRegistry.has("memory"):
        role = "memory"

    if role is None:
        one_line = _fallback_one_line(pages, file_name)
        logger.info("pdf_summarizer: no LLM available, using fallback one_line")
        return DocSummary(one_line=one_line, section_summaries=[])

    # 입력은 앞쪽 2페이지만 추출. abstract/intro가 담길 확률이 높음.
    sample_text = " ".join(pages[:2])
    sample_text = _clean_text(sample_text)[:1500]  # 너무 길면 자름

    user_prompt = (
        "다음은 PDF 문서의 앞부분 본문입니다. "
        "이 문서가 무엇에 관한 것인지 한국어로 한 문장(80자 이내)으로 요약해 주세요.\n\n"
        f"{sample_text}"
    )

    try:
        raw = await asyncio.to_thread(
            LocalModelRegistry.generate,
            role,
            user_prompt,
            100,           # 요약이므로 짧게
            _SUMMARY_SYSTEM,
        )
        one_line = _clean_text(raw)[:160] if raw else ""
        if not one_line:
            one_line = _fallback_one_line(pages, file_name)
        logger.info("pdf_summarizer: one_line=%r", one_line[:120])
        return DocSummary(one_line=one_line, section_summaries=[])
    except Exception as e:
        logger.exception("pdf_summarizer failed, using fallback: %s", e)
        return DocSummary(
            one_line=_fallback_one_line(pages, file_name),
            section_summaries=[],
        )
