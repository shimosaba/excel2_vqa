"""
STEP 4: 質問 + 画像から回答を生成 → VQA アイテム組立
"""
from __future__ import annotations

import json
import re
import textwrap
from dataclasses import dataclass

from .config import Config
from .question_generator import QuestionItem
from .vlm_backend import Qwen3VLBackend


@dataclass
class VQAItem:
    # 識別子
    id: str
    # ソース情報
    source_file: str
    sheet_name: str
    sheet_index: int
    sheet_image: str    # 相対パス（output_dir からの）
    crop_image: str     # 相対パス
    # 要素情報
    element_label: str
    element_description: str
    bbox: dict          # {"x": int, "y": int, "width": int, "height": int}
    # QA
    question: str
    question_type: str
    difficulty: str
    answer: str
    confidence: str     # high / medium / low
    reasoning: str


# ── プロンプト ──────────────────────────────────────────────────────────────

_ANSWER_SYSTEM = textwrap.dedent("""\
    あなたはExcelスプレッドシートの内容を正確に読み取り、質問に答える専門家です。
    画像を注意深く観察し、正確で簡潔な回答を提供してください。
    出力はJSONのみで返してください。説明文は不要です。
    【絶対ルール】この画像に表示されている情報のみ使用。外部知識・統計データ・一般常識は一切使用しない。
""")

_ANSWER_PROMPT = textwrap.dedent("""\
    この画像（{label}）に対して、以下の質問に答えてください。

    質問: {question}

    回答ルール:
    - この画像に表示されている情報のみを根拠にする。外部知識・統計・一般常識は一切使用しない
    - 短く明確に（1〜3文程度）
    - 数値は正確に
    - 不明な場合は「画像からは判断できません」と答える

    confidence の判定基準（正直に判定すること。high に偏らせない）:
    - high: 画像内のテキスト・数値をそのまま読み取れる。計算・解釈が不要
    - medium: 複数セルを参照・比較・計算した結果。読み取り自体は明確だが手順が必要
    - low: テキストが小さい・不鮮明・推測を含む・画像から直接読み取れない

    出力形式 (JSONのみ、```なし):
    {{
      "answer": "回答文",
      "confidence": "high|medium|low",
      "reasoning": "画像のどの部分からどう判断したかを1文で説明"
    }}
""")


def generate_answer(
    question: QuestionItem,
    vlm: Qwen3VLBackend,
    config: Config,
    entry_id: str,
    output_dir_path: str,
) -> VQAItem:
    """
    1つの質問に対する回答を生成し VQAItem を返す。

    Args:
        question: 質問アイテム
        vlm: VLM バックエンド
        config: 設定オブジェクト
        entry_id: データセットエントリID（例: "entry_0001"）
        output_dir_path: 出力ディレクトリパス（相対パス計算用）

    Returns:
        VQAItem
    """
    crop_info = question.crop_info
    prompt = _ANSWER_PROMPT.format(
        label=crop_info.label,
        question=question.question,
    )
    raw = vlm.infer(crop_info.crop_path, prompt, system_prompt=_ANSWER_SYSTEM)

    raw_clean = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    try:
        ans = json.loads(raw_clean)
    except json.JSONDecodeError:
        ans = {
            "answer": raw.strip()[:500],
            "confidence": "low",
            "reasoning": "パースエラー",
        }

    # output_dir からの相対パス
    from pathlib import Path
    out = Path(output_dir_path)
    try:
        sheet_rel = str(crop_info.sheet_image.relative_to(out))
    except ValueError:
        sheet_rel = crop_info.sheet_image.name
    try:
        crop_rel = str(crop_info.crop_path.relative_to(out))
    except ValueError:
        crop_rel = crop_info.crop_path.name

    return VQAItem(
        id=entry_id,
        source_file=crop_info.sheet_image.parent.parent.name,  # fallback
        sheet_name=crop_info.sheet_name,
        sheet_index=crop_info.sheet_index,
        sheet_image=sheet_rel,
        crop_image=crop_rel,
        element_label=crop_info.label,
        element_description=crop_info.description,
        bbox={
            "x": crop_info.bbox.x,
            "y": crop_info.bbox.y,
            "width": crop_info.bbox.width,
            "height": crop_info.bbox.height,
        },
        question=question.question,
        question_type=question.question_type,
        difficulty=question.difficulty,
        answer=ans.get("answer", ""),
        confidence=ans.get("confidence", "low"),
        reasoning=ans.get("reasoning", ""),
    )
