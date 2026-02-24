"""
STEP 1: Excel → シート別 PNG 画像化
LibreOffice でPDF変換 → pdf2image でページ別PNG化
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import openpyxl
from pdf2image import convert_from_path

from .config import Config


def render_workbook(excel_path: Path, output_dir: Path, config: Config) -> list[Path]:
    """
    Excel を1シートずつ PNG に変換して返す。

    Args:
        excel_path: 入力Excelファイルパス
        output_dir: 出力先ディレクトリ
        config: 設定オブジェクト

    Returns:
        生成したPNGファイルパスのリスト（シート順）
    """
    excel_path = Path(excel_path).resolve()
    sheets_dir = output_dir / "sheets"
    sheets_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[STEP 1] Excel → シート画像化: {excel_path.name}")

    # openpyxl でシート名を取得
    wb = openpyxl.load_workbook(excel_path, read_only=True, data_only=True)
    sheet_names = wb.sheetnames
    wb.close()

    with tempfile.TemporaryDirectory() as tmpdir:
        # LibreOffice で PDF 変換
        result = subprocess.run(
            [
                "libreoffice", "--headless",
                "--convert-to", "pdf",
                "--outdir", tmpdir,
                str(excel_path),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(f"LibreOffice 変換失敗:\n{result.stderr}")

        pdf_files = list(Path(tmpdir).glob("*.pdf"))
        if not pdf_files:
            raise RuntimeError("PDF が生成されませんでした")
        pdf_path = pdf_files[0]

        # PDF → ページ別 PNG
        pages = convert_from_path(str(pdf_path), dpi=config.dpi)

    image_paths: list[Path] = []
    for i, page in enumerate(pages):
        page_rgb = page.convert("RGB")
        img_path = sheets_dir / f"sheet_{i + 1:02d}.png"
        page_rgb.save(str(img_path), "PNG")
        image_paths.append(img_path)

        sheet_name = sheet_names[i] if i < len(sheet_names) else f"Sheet{i + 1}"
        print(f"  → sheet_{i + 1:02d}.png  [{sheet_name}]  "
              f"({page_rgb.width}x{page_rgb.height}px)")

    print(f"  合計 {len(image_paths)} シート画像を生成しました")
    return image_paths
