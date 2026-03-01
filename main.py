"""
Excel → VQA Dataset 全自動パイプライン CLI
"""
from __future__ import annotations

import time
from pathlib import Path

import click
import openpyxl
from tqdm import tqdm

from lib.config import Config
from lib.vlm_backend import Qwen3VLBackend
from lib.excel_renderer import render_workbook
from lib.element_detector import detect_and_crop
from lib.question_generator import generate_questions
from lib.answer_generator import generate_answer
from lib.validators import validate_vqa_item
from lib.formatters import save_all


@click.command()
@click.argument("excel_file", type=click.Path(exists=True, path_type=Path))
@click.option("--output", "-o", default="output", show_default=True,
              help="出力ディレクトリ")
@click.option("--model", default="Qwen/Qwen3-VL-8B-Instruct-FP8", show_default=True,
              help="使用するモデル名")
@click.option("--dpi", default=0, show_default=True,
              help="シート画像の解像度 (DPI)。0=シートサイズに応じて自動設定")
@click.option("--crop-limit", default=30, show_default=True,
              help="クロップ数の絶対上限（ガードレール）。通常変更不要。")
@click.option("--question-limit", default=15, show_default=True,
              help="質問数の絶対上限（ガードレール）。通常変更不要。")
def main(
    excel_file: Path,
    output: str,
    model: str,
    dpi: int,
    crop_limit: int,
    question_limit: int,
) -> None:
    """
    Excel ファイルから VQA データセットを全自動生成する。

    \b
    処理フロー:
      1. Excel → シート別 PNG 画像化 (LibreOffice)
      2. シート画像から要素矩形を検出 → クロップ画像生成
      3. クロップ画像から質問文を生成
      4. 質問 + 画像から回答を生成 → VQA データセット保存
    """
    print("=" * 60)
    print("  Excel → VQA Dataset 全自動パイプライン")
    print("=" * 60)

    # ── 設定構築 ────────────────────────────────────────────────────────
    config = Config.from_env()
    config.model_name = model
    config.dpi = dpi
    config.crop_limit = crop_limit
    config.question_limit = question_limit
    config.output_dir = output

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # openpyxl でシート名を取得しておく
    wb = openpyxl.load_workbook(excel_file, read_only=True, data_only=True)
    sheet_names = wb.sheetnames
    wb.close()

    start_time = time.time()

    # ── STEP 0: VLM バックエンドをロード ──────────────────────────────
    vlm = Qwen3VLBackend(config)
    vlm.load()

    # ── STEP 1: Excel → シート画像 ────────────────────────────────────
    sheet_images = render_workbook(excel_file, output_dir, config)

    all_vqa_items = []
    all_crops_count = 0
    entry_counter = 0

    # ── シートごとに処理 ─────────────────────────────────────────────
    for sheet_idx, sheet_img in enumerate(
        tqdm(sheet_images, desc="シート処理", unit="sheet"), start=1
    ):
        sheet_name = sheet_names[sheet_idx - 1] if sheet_idx - 1 < len(sheet_names) else f"Sheet{sheet_idx}"

        # ── STEP 2: 要素検出 → クロップ ──────────────────────────────
        crop_infos = detect_and_crop(
            sheet_img, output_dir, sheet_idx, sheet_name, vlm, config
        )
        all_crops_count += len(crop_infos)

        for crop_info in tqdm(crop_infos, desc=f"  [{sheet_name}] クロップ処理", leave=False):
            print(f"\n[STEP 3] 質問生成: {crop_info.crop_path.name}  [{crop_info.label}]")

            # ── STEP 3: 質問生成 ─────────────────────────────────────
            questions = generate_questions(crop_info, vlm, config)
            if not questions:
                print("  質問を生成できませんでした、スキップ")
                continue
            print(f"  {len(questions)} 問を生成しました")

            print(f"[STEP 4] 回答生成: {crop_info.crop_path.name}")

            # ── STEP 4: 回答生成 ─────────────────────────────────────
            for q in questions:
                entry_counter += 1
                entry_id = f"entry_{entry_counter:04d}"

                vqa_item = generate_answer(
                    q, vlm, config,
                    entry_id=entry_id,
                    output_dir_path=str(output_dir),
                )
                # source_file を正確に設定
                vqa_item.source_file = excel_file.name

                # バリデーション
                is_valid, reason = validate_vqa_item(
                    vqa_item.question,
                    vqa_item.answer,
                    confidence=vqa_item.confidence,
                    question_type=vqa_item.question_type,
                    difficulty=vqa_item.difficulty,
                )
                if not is_valid:
                    print(f"  [{entry_id}] バリデーション除外: {reason} / 質問: {q.question[:60]!r}")
                    continue

                all_vqa_items.append(vqa_item)
                print(f"  [{entry_id}] {q.question[:60]}...")

    # ── データセット保存 ──────────────────────────────────────────────
    elapsed = time.time() - start_time
    saved_files = save_all(
        items=all_vqa_items,
        output_dir=output_dir,
        source_file=excel_file.name,
        model_name=config.model_name,
        total_sheets=len(sheet_images),
        total_crops=all_crops_count,
        elapsed_seconds=elapsed,
    )

    # ── 完了サマリー ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  パイプライン完了!")
    print("=" * 60)
    print(f"  入力ファイル     : {excel_file.name}")
    print(f"  シート数         : {len(sheet_images)}")
    print(f"  検出要素数       : {all_crops_count}")
    print(f"  QAペア数         : {len(all_vqa_items)}")
    print(f"  処理時間         : {elapsed:.1f}秒")
    print(f"  出力ディレクトリ : {output_dir.resolve()}")
    print()
    print("  保存ファイル:")
    for fmt, path in saved_files.items():
        print(f"    [{fmt}] {path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
