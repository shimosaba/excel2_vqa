"""
STEP 2: シート画像から要素矩形を検出 → クロップ画像生成

Qwen3-VL の座標系:
  - bbox_2d: [x1, y1, x2, y2] 形式
  - 値の範囲: 0〜1000 の相対座標（実ピクセルではない）
  - 変換: abs_x = val / 1000 * image_width
"""
from __future__ import annotations

import json
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from .config import Config
from .vlm_backend import Qwen3VLBackend


@dataclass
class BBox:
    x: int
    y: int
    width: int
    height: int


@dataclass
class CropInfo:
    crop_path: Path
    sheet_image: Path
    sheet_index: int
    sheet_name: str
    element_id: int
    label: str
    description: str
    bbox: BBox


# ── プロンプト ──────────────────────────────────────────────────────────────

_DETECT_SYSTEM = textwrap.dedent("""\
    あなたはExcelスプレッドシートのレイアウト解析の専門家です。
    与えられたシート画像を解析し、意味的にまとまりのある要素ブロックを検出します。
    要素ブロックの例: テーブル、グラフ、タイトル領域、KPIカード、注釈ブロック など
    出力はJSON配列のみを返してください。他のテキストは不要です。
""")

# Qwen3-VL のネイティブ bbox_2d 形式（0〜1000 相対座標）を使用
_DETECT_PROMPT = textwrap.dedent("""\
    このExcelシートの画像を解析し、意味的にまとまりのある要素ブロックを最大 {max_crops} 個検出してください。

    検出ルール:
    - テーブル、グラフ、タイトル、KPIエリアなど独立した情報単位を1ブロックとする
    - 余白・空白領域は含めない
    - ブロックは重複しないようにする

    出力形式 (JSONのみ、```なし):
    [
      {{
        "id": 1,
        "label": "要素の種類 (例: 売上テーブル, 月次グラフ, タイトル)",
        "description": "この要素が何を表しているかの簡単な説明",
        "bbox_2d": [x1, y1, x2, y2]
      }},
      ...
    ]

    bbox_2d の値は 0〜1000 の相対座標で指定してください (左上が原点)。
""")


def _parse_bbox_2d(region: dict, img_w: int, img_h: int) -> tuple[int, int, int, int] | None:
    """
    Qwen3-VL の bbox_2d（0〜1000 相対座標）を実ピクセル座標に変換する。

    Returns:
        (x1, y1, x2, y2) の実ピクセル座標、パース失敗時は None
    """
    bbox = region.get("bbox_2d")
    if not bbox or not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None

    # 0〜1000 の相対座標 → 実ピクセル変換（cookbook の変換式と同一）
    x1 = int(bbox[0] / 1000 * img_w)
    y1 = int(bbox[1] / 1000 * img_h)
    x2 = int(bbox[2] / 1000 * img_w)
    y2 = int(bbox[3] / 1000 * img_h)

    # x1 < x2, y1 < y2 を保証
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1

    # 画像境界にクランプ
    x1 = max(0, min(x1, img_w))
    y1 = max(0, min(y1, img_h))
    x2 = max(0, min(x2, img_w))
    y2 = max(0, min(y2, img_h))

    return x1, y1, x2, y2


def detect_and_crop(
    sheet_image_path: Path,
    output_dir: Path,
    sheet_idx: int,
    sheet_name: str,
    vlm: Qwen3VLBackend,
    config: Config,
) -> list[CropInfo]:
    """
    シート画像から要素矩形を検出し、クロップ画像を生成する。

    Args:
        sheet_image_path: シート画像のパス
        output_dir: 出力先ディレクトリ
        sheet_idx: シートのインデックス（1始まり）
        sheet_name: シート名
        vlm: VLM バックエンド
        config: 設定オブジェクト

    Returns:
        CropInfo のリスト
    """
    crops_dir = output_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    img = Image.open(sheet_image_path)
    w, h = img.size

    print(f"\n[STEP 2] 要素検出: {sheet_image_path.name}  ({w}x{h}px)")

    prompt = _DETECT_PROMPT.format(max_crops=config.max_crops)
    raw = vlm.infer(sheet_image_path, prompt, system_prompt=_DETECT_SYSTEM)

    # マークダウンのコードブロックを除去してJSONパース
    raw_clean = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    try:
        regions = json.loads(raw_clean)
    except json.JSONDecodeError as e:
        print(f"  JSON パース失敗 ({e}), スキップ")
        print(f"  生レスポンス: {raw[:300]}")
        return []

    crop_infos: list[CropInfo] = []
    for region in regions:
        coords = _parse_bbox_2d(region, w, h)
        if coords is None:
            print(f"  bbox_2d パース失敗, スキップ: {region}")
            continue

        x1, y1, x2, y2 = coords
        bw = x2 - x1
        bh = y2 - y1

        # 小さすぎる矩形はノイズとしてスキップ
        if bw < 30 or bh < 30:
            continue

        # パディングを加えてクロップ
        pad = 8
        cx1 = max(0, x1 - pad)
        cy1 = max(0, y1 - pad)
        cx2 = min(w, x2 + pad)
        cy2 = min(h, y2 + pad)

        crop_img = img.crop((cx1, cy1, cx2, cy2))
        elem_id = int(region.get("id", len(crop_infos) + 1))
        crop_name = f"sheet{sheet_idx:02d}_crop{elem_id:02d}.png"
        crop_path = crops_dir / crop_name
        crop_img.save(str(crop_path), "PNG")

        label = region.get("label", "unknown")
        description = region.get("description", "")

        crop_infos.append(CropInfo(
            crop_path=crop_path,
            sheet_image=sheet_image_path,
            sheet_index=sheet_idx,
            sheet_name=sheet_name,
            element_id=elem_id,
            label=label,
            description=description,
            bbox=BBox(x=x1, y=y1, width=bw, height=bh),
        ))
        print(f"  → {crop_name}  [{label}]  ({x1},{y1})-({x2},{y2})  {bw}x{bh}px")

    print(f"  {len(crop_infos)} 要素を検出・クロップしました")
    return crop_infos
