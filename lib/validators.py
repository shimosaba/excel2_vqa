"""
VQA アイテムのバリデーション
"""
from __future__ import annotations

# プロンプト断片として混入しやすいキーワード
_PROMPT_FRAGMENTS = [
    # 既存
    "VQAペア", "作成ルール", "JSONのみ", "出力形式", "質問ルール",
    "回答ルール", "バランスよく", "```", "question_id", "question_type",
    # 新規追加（プロンプト特有の連結パターン）
    "high|medium|low", "easy|medium|hard", "最低2問",
    "factual|comparative", "出力はJSON", "マークダウン記法",
]

_VALID_QUESTION_TYPES = {"factual", "comparative", "aggregation", "understanding", "reasoning"}
_VALID_DIFFICULTIES = {"easy", "medium", "hard"}


def validate_vqa_item(
    question: str,
    answer: str,
    confidence: str = "high",
    question_type: str = "",
    difficulty: str = "",
) -> tuple[bool, str]:
    """
    VQAアイテムのバリデーションを行う。

    Args:
        question: 質問文
        answer: 回答文
        confidence: 回答信頼度（high/medium/low）。low は除外。
        question_type: 質問タイプ（有効値チェック）
        difficulty: 難易度（有効値チェック）

    Returns:
        (is_valid, reason): is_valid=False の場合、reason に理由を格納
    """
    # ── confidence=low は除外 ────────────────────────────────────────
    if confidence == "low":
        return False, "confidence=low のため除外"

    # ── question_type の有効値チェック ───────────────────────────────
    if question_type and question_type not in _VALID_QUESTION_TYPES:
        return False, f"question_type が無効な値です: {question_type!r}"

    # ── difficulty の有効値チェック ──────────────────────────────────
    if difficulty and difficulty not in _VALID_DIFFICULTIES:
        return False, f"difficulty が無効な値です: {difficulty!r}"

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
    if not isinstance(answer, str):
        return False, f"回答が文字列ではありません ({type(answer).__name__})"
    if len(answer) < 1:
        return False, "回答が空です"
    if len(answer) > 300:
        return False, f"回答が長すぎます ({len(answer)}文字)"

    for frag in _PROMPT_FRAGMENTS:
        if frag in answer:
            return False, f"回答にプロンプト断片が含まれています: {frag!r}"

    # ── 質問 ≠ 回答 ────────────────────────────────────────────────
    if question.strip() == answer.strip():
        return False, "質問と回答が同一です"

    return True, ""
