"""
VQA アイテムのバリデーション
"""
from __future__ import annotations

# プロンプト断片として混入しやすいキーワード
_PROMPT_FRAGMENTS = [
    "VQAペア", "作成ルール", "JSONのみ", "出力形式", "質問ルール",
    "回答ルール", "バランスよく", "```", "question_id", "question_type",
]


def validate_vqa_item(question: str, answer: str) -> tuple[bool, str]:
    """
    VQAアイテムのバリデーションを行う。

    Returns:
        (is_valid, reason): is_valid=False の場合、reason に理由を格納
    """
    # ── 質問のチェック ──────────────────────────────────────────────
    if len(question) < 5:
        return False, f"質問が短すぎます ({len(question)}文字)"
    if len(question) > 100:
        return False, f"質問が長すぎます ({len(question)}文字)"
    if not (question.endswith("？") or question.endswith("?")):
        return False, "質問が「？」または「?」で終わっていません"

    for frag in _PROMPT_FRAGMENTS:
        if frag in question:
            return False, f"質問にプロンプト断片が含まれています: {frag!r}"

    # ── 回答のチェック ──────────────────────────────────────────────
    if len(answer) < 1:
        return False, "回答が空です"
    if len(answer) > 200:
        return False, f"回答が長すぎます ({len(answer)}文字)"

    for frag in _PROMPT_FRAGMENTS:
        if frag in answer:
            return False, f"回答にプロンプト断片が含まれています: {frag!r}"

    # ── 質問 ≠ 回答 ────────────────────────────────────────────────
    if question.strip() == answer.strip():
        return False, "質問と回答が同一です"

    return True, ""
