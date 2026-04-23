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
    この画像（Excelシートの一部: {label}）に対して、VQAデータセット用の質問を生成してください。

    【★最重要: 画像の情報量に応じて質問数を決める】
    - 情報量が少ない画像（短い説明文1つ・ボタン数個・見出しのみ等）: **3〜5問**
    - 情報量が中程度（数行の説明文・小さい表等）: 5〜8問
    - 情報量が多い（大きな表・複数の指標・長い手順等）: 8〜15問
    **数合わせで水増ししないでください。本質的な質問だけを生成してください。**
    1つの短い文・1つの値しか無い画像に10問以上作るのは明らかな水増しで禁止です。

    【質問タイプ（任意・該当するものだけ）】
    - factual（数値・事実確認）: セルの値・表示テキストを直接問う
    - comparative（比較）: 複数の値を比べる（**2つ以上の比較対象がある場合のみ**）
    - aggregation（集計）: 合計・平均・件数など（**集計できる数値データがある場合のみ**）
    - understanding（内容理解）: 画像全体が何を示すか
    - reasoning（推論）: データから読み取れる傾向・示唆（**複数データがある場合のみ**）

    **該当する対象が画像に無ければそのタイプは作らないでください。無理に作ると重複や的外れな質問になります。**

    難易度:
    - easy: 1つのセル/テキストから直接読み取れる
    - medium: 複数セルの参照・比較・簡単な計算が必要
    - hard: 複数行列の集計・傾向分析・多段階の推論が必要

    【厳守: 重複禁止】
    - 同じ内容を問う質問を複数作らない（文字数クイズ・言い換えによる水増し禁止）
    - 答えが同じになる質問は1つだけにする
    - 「最初の単語は？」「最後の単語は？」のような些末な形式クイズは作らない

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
    seen_normalized: set[str] = set()
    dup_count = 0
    for item in items:
        q_text = item.get("question", "").strip()
        if not q_text:
            continue
        # 「？」で終わっていない場合は補完する
        if not (q_text.endswith("？") or q_text.endswith("?")):
            q_text = q_text.rstrip("。．.") + "？"

        # 重複排除: 正規化キーで比較（空白・句読点・疑問符を除去し、全角/半角を統一）
        normalized = re.sub(r"[\s?？。、,.!！]+", "", q_text).lower()
        if normalized in seen_normalized:
            dup_count += 1
            continue
        seen_normalized.add(normalized)

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

    if dup_count > 0:
        print(f"  重複質問 {dup_count} 件を除外しました")

    if len(questions) > config.question_limit:
        print(f"  [警告] 質問数 {len(questions)} がガードレール {config.question_limit} を超えたため切り詰めます")
        questions = questions[:config.question_limit]

    return questions
