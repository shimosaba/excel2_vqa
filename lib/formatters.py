"""
VQA データセットを複数フォーマットで出力する
"""
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from .answer_generator import VQAItem


def _item_to_dict(item: VQAItem) -> dict[str, Any]:
    return {
        "id": item.id,
        "source_file": item.source_file,
        "sheet_name": item.sheet_name,
        "sheet_index": item.sheet_index,
        "sheet_image": item.sheet_image,
        "crop_image": item.crop_image,
        "element_label": item.element_label,
        "element_description": item.element_description,
        "bbox": item.bbox,
        "question": item.question,
        "question_type": item.question_type,
        "difficulty": item.difficulty,
        "answer": item.answer,
        "confidence": item.confidence,
        "reasoning": item.reasoning,
    }


def save_all(
    items: list[VQAItem],
    output_dir: Path,
    source_file: str,
    model_name: str,
    total_sheets: int,
    total_crops: int,
    elapsed_seconds: float,
) -> dict[str, Path]:
    """
    4形式のデータセットファイルを保存する。

    Returns:
        保存したファイルパスの辞書 {形式名: パス}
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, Path] = {}

    # ── 統計情報 ──────────────────────────────────────────────────────────
    by_type       = Counter(item.question_type for item in items)
    by_difficulty = Counter(item.difficulty for item in items)
    by_confidence = Counter(item.confidence for item in items)

    # ── 1. vqa_dataset.json ───────────────────────────────────────────────
    metadata = {
        "source_file": source_file,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": model_name,
        "total_sheets": total_sheets,
        "total_crops": total_crops,
        "total_qa_pairs": len(items),
        "elapsed_seconds": round(elapsed_seconds, 2),
        "statistics": {
            "by_type": dict(by_type),
            "by_difficulty": dict(by_difficulty),
            "by_confidence": dict(by_confidence),
        },
    }
    vqa_path = output_dir / "vqa_dataset.json"
    with open(vqa_path, "w", encoding="utf-8") as f:
        json.dump(
            {"metadata": metadata, "dataset": [_item_to_dict(i) for i in items]},
            f,
            ensure_ascii=False,
            indent=2,
        )
    saved["vqa_dataset"] = vqa_path

    # ── 2. dataset_llava.json  (LLaVA 形式) ──────────────────────────────
    llava_records = []
    for item in items:
        llava_records.append({
            "id": item.id,
            "image": item.crop_image,
            "conversations": [
                {"from": "human", "value": f"<image>\n{item.question}"},
                {"from": "gpt",   "value": item.answer},
            ],
        })
    llava_path = output_dir / "dataset_llava.json"
    with open(llava_path, "w", encoding="utf-8") as f:
        json.dump(llava_records, f, ensure_ascii=False, indent=2)
    saved["llava"] = llava_path

    # ── 3. dataset_sharegpt.json  (ShareGPT 形式) ─────────────────────────
    sharegpt_records = []
    for item in items:
        sharegpt_records.append({
            "id": item.id,
            "image": item.crop_image,
            "conversations": [
                {"role": "user",      "content": item.question},
                {"role": "assistant", "content": item.answer},
            ],
        })
    sharegpt_path = output_dir / "dataset_sharegpt.json"
    with open(sharegpt_path, "w", encoding="utf-8") as f:
        json.dump(sharegpt_records, f, ensure_ascii=False, indent=2)
    saved["sharegpt"] = sharegpt_path

    # ── 4. dataset_flat.jsonl  (フラット JSONL) ───────────────────────────
    flat_path = output_dir / "dataset_flat.jsonl"
    with open(flat_path, "w", encoding="utf-8") as f:
        for item in items:
            line = {
                "id": item.id,
                "image": item.crop_image,
                "question": item.question,
                "answer": item.answer,
                "question_type": item.question_type,
                "difficulty": item.difficulty,
                "confidence": item.confidence,
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    saved["flat"] = flat_path

    return saved
