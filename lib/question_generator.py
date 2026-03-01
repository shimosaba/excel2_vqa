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
    画像に表示されている情報のみを根拠にした質問を生成すること。外部知識・推定は使用しない。
""")

_QUESTION_PROMPT = textwrap.dedent("""\
    この画像（Excelシートの一部: {label}）に対して、
    VQAデータセット用の多様な質問を8〜15問生成してください。

    【重要】質問タイプ別の最低生成数:
    - factual（数値・事実確認）: 最低2問
      例: 「A列の3行目の値はいくつですか？」「セルB5には何と書かれていますか？」
    - comparative（比較）: 最低2問
      例: 「〇〇と△△を比べると、どちらが大きいですか？」「最大値はどの項目ですか？」
    - aggregation（集計）: 最低2問
      例: 「この列の合計はいくらですか？」「平均値はいくらですか？」「いくつの行がありますか？」
    - understanding（内容理解）: 最低2問
      例: 「このテーブルは何を示していますか？」「ヘッダー行にはどんな項目がありますか？」
    - reasoning（推論）: 最低2問
      例: 「この数値の傾向から何が読み取れますか？」「最も注目すべき変化はどれですか？」

    難易度の定義:
    - easy: 1つのセルから直接読み取れる（計算・比較不要）
    - medium: 複数セルの参照・比較・簡単な四則演算が必要
    - hard: 複数行/列の集計・傾向分析・多段階の推論が必要

    重複排除ルール:
    - 同じセル・同じ内容を対象とした質問を複数生成しない
    - 似た問い方・同じ答えになる質問は省く

    制約:
    - 質問は日本語
    - 「？」で終わること
    - 5文字以上100文字以下
    - この画像に表示されている情報のみを根拠にする

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


def _parse_json_array(text: str) -> list | None:
    """
    JSONの配列文字列をパースする。途中で切れている場合も完全なアイテムだけ救出する。

    Returns:
        パース成功時はリスト、失敗時は None
    """
    # まず通常のパースを試みる
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    # 途中で切れた場合: 完全なオブジェクトを正規表現で抽出して再構築
    # "[" から最後の完全な "}," or "}" までを切り出してパース
    bracket_pos = text.find("[")
    if bracket_pos == -1:
        return None

    # 最後の完全なオブジェクト末尾（"}," or "}") を探して切り詰める
    truncated = text[bracket_pos:]
    # 末尾が不完全なので、最後の完結した "}" を見つけて "]" を補完
    last_close = truncated.rfind("}")
    if last_close == -1:
        return None
    repaired = truncated[: last_close + 1] + "]"
    try:
        result = json.loads(repaired)
        if isinstance(result, list):
            print(f"  [情報] JSONが途中で切れていたため、完全な {len(result)} 件のみ救出しました")
            return result
    except json.JSONDecodeError:
        pass

    return None


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
    prompt = _QUESTION_PROMPT.format(label=crop_info.label)
    raw = vlm.infer(crop_info.crop_path, prompt, system_prompt=_QUESTION_SYSTEM)

    raw_clean = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    items = _parse_json_array(raw_clean)
    if items is None:
        print(f"  質問生成パース失敗")
        print(f"  生レスポンス: {raw[:300]}")
        return []

    VALID_TYPES = {"factual", "comparative", "aggregation", "understanding", "reasoning"}
    VALID_DIFFICULTIES = {"easy", "medium", "hard"}

    questions: list[QuestionItem] = []
    for item in items:
        q_text = item.get("question", "").strip()
        if not q_text:
            continue
        # 「？」で終わっていない場合は補完する
        if not (q_text.endswith("？") or q_text.endswith("?")):
            q_text = q_text.rstrip("。．.") + "？"

        q_type = item.get("question_type", "factual")
        if q_type not in VALID_TYPES:
            q_type = "factual"

        q_diff = item.get("difficulty", "medium")
        if q_diff not in VALID_DIFFICULTIES:
            q_diff = "medium"

        questions.append(QuestionItem(
            question_id=int(item.get("question_id", len(questions) + 1)),
            crop_info=crop_info,
            question=q_text,
            question_type=q_type,
            difficulty=q_diff,
        ))

    if len(questions) > config.question_limit:
        print(f"  [警告] 質問数 {len(questions)} がガードレール {config.question_limit} を超えたため切り詰めます")
        questions = questions[:config.question_limit]

    return questions
