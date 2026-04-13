"""
app/services/llm/llm_router.py
------------------------------
역할: 요청 형태에 따라 어떤 모델을 호출할지 결정 (LLM vs VLM)
      사용 가능한 백엔드는 두 가지뿐

      PDF 존재 여부는 모델 선택에 영향을 주지 않음
      → PDF 텍스트는 프롬프트에 포함되는 보조 정보일 뿐,
         VLM을 사용하지 않음

      실제 이미지가 포함된 경우에만 VLM을 사용

TODO:
    [ ] 요청 힌트를 통해 강제 모델 선택 기능 추가
"""
from app.core.constants import ModelKind


def pick_model(image_b64: str | None) -> ModelKind:
    return ModelKind.VLM if image_b64 else ModelKind.LLM
