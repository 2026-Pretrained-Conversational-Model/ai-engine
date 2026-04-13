"""
app/utils/token_counter.py
--------------------------
역할: 프롬프트 예산 관리를 위한 토큰 수 추정
      모든 모델마다 토크나이저를 불러오지 않고,
      문자 수 / 4 방식으로 근사 계산 (soft budget에 충분)

TODO:
    [ ] 정확도가 중요해질 경우 tiktoken 또는 모델별 토크나이저로 전환
"""

def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)
