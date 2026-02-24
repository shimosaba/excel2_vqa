"""
STEP 3: クロップ画像から VQA 質問文を生成
"""
from __future__ import annotations

import json
import re
import textwrap
from dataclasses import dataclass

from .config import Config
from .element_detector import CropInfo
from .vlm_backend import Qwen3VLBackend


@dataclass
class QuestionItem:
    question_id: int
    crop_info: CropInfo
    question: str
    question_type: str   # factual / comparative / aggregation / understanding / reasoning
    difficulty: str      # easy / medium / hard


# ── プロンプト ──────────────────────────────────────────────────────────────

_QUESTION_SYSTEM = textwrap.dedent("""\
    あなたはVQA(Visual Question Answering)データセット作成の専門家です。
    与えられた画像に対して、多様な観点から意味のある質問を生成します。
    出力はJSONのみで返してください。説明文は不要です。
""")

_QUESTION_PROMPT = textwrap.dedent("""\
    この画像（Excelシートの一部: {label}）に対して、
    VQAデータセット用の多様な質問を {num_q} 個生成してください。

    質問タイプを以下の5種からバランスよく混在させること:
    - factual: 「〜の値はいくつですか？」（数値・事実確認）
    - comparative: 「〜と〜を比べると、どちらが大きいですか？」（比較）
    - aggregation: 「合計/平均はいくらですか？」（集計）
    - understanding: 「このグラフ/テーブルは何を示していますか？」（内容理解）
    - reasoning: 「この傾向から何が読み取れますか？」（推論）

    制約:
    - 質問は日本語
    - 「？」で終わること
    - 5文字以上50文字以下

    出力形式 (JSONのみ、```なし):
    [
      {{
        "question_id": 1,
        "question": "質問文",
        "question_type": "factual|comparative|aggregation|understanding|reasoning",
        "difficulty": "easy|medium|hard"
      }},
      ...
    ]
""")


def generate_questions(
    crop_info: CropInfo,
    vlm: Qwen3VLBackend,
    config: Config,
) -> list[QuestionItem]:
    """
    クロップ画像から VQA 質問を生成する。

    Args:
        crop_info: クロップ情報
        vlm: VLM バックエンド
        config: 設定オブジェクト

    Returns:
        QuestionItem のリスト
    """
    prompt = _QUESTION_PROMPT.format(
        label=crop_info.label,
        num_q=config.questions_per_crop,
    )
    raw = vlm.infer(crop_info.crop_path, prompt, system_prompt=_QUESTION_SYSTEM)

    raw_clean = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    try:
        items = json.loads(raw_clean)
    except json.JSONDecodeError as e:
        print(f"  質問生成パース失敗 ({e})")
        print(f"  生レスポンス: {raw[:300]}")
        return []

    questions: list[QuestionItem] = []
    for item in items:
        q_text = item.get("question", "").strip()
        if not q_text:
            continue
        questions.append(QuestionItem(
            question_id=int(item.get("question_id", len(questions) + 1)),
            crop_info=crop_info,
            question=q_text,
            question_type=item.get("question_type", "factual"),
            difficulty=item.get("difficulty", "medium"),
        ))

    return questions
