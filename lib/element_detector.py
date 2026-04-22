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
    あなたはExcelスプレッドシートのVQAデータセット作成のための領域検出専門家です。
    与えられたシート画像を解析し、質問・回答が成立する情報量を持つブロックを検出します。
    【検出すべき】データ表・グラフ・KPIエリア（数値含む）・複数行の説明文・手順リスト
    【重要】各ブロックのbboxには、必ず直上のタイトル行・見出しも含めてください。
    タイトルは後続のVQA生成で「何のデータか」を判断するための前提情報になります。
    出力はJSON配列のみを返してください。他のテキストは不要です。
""")

# Qwen3-VL のネイティブ bbox_2d 形式（0〜1000 相対座標）を使用
_DETECT_PROMPT = textwrap.dedent("""\
    このExcelシートの画像を解析し、VQAタスクとして質問・回答が成立する情報量を持つブロックを検出してください。

    【ブロック数の判断】
    - 情報量が少ないシート（タイトル・ボタン・説明文のみ等）: 意味のある領域をまとめてシート全体に近い1ブロックとして検出
    - 情報量が多いシート（複数テーブル・グラフ・KPI等）: テーブル・グラフ・説明文等の意味単位ごとに数ブロックに分割

    【検出してよい要素（情報量あり）】
    - データテーブル: ヘッダー行＋データ行を含む範囲
    - グラフ・チャート
    - KPIエリア: 数値・指標を含む複数セルの範囲
    - 説明文ブロック: 複数行にわたるテキスト
    - 手順・ステップリスト: 複数項目を含むもの

    【検出してはいけない要素】
    - ナビゲーションボタン・リンクだけの行
    - 装飾的な区切り線・空白領域・アイコン単体

    【★最重要: bboxにタイトル・見出しを必ず含める】
    クロップ画像は後続のVQA質問・回答生成にそのまま使われます。
    タイトルや見出しがないと「何のデータか」が分からず、質問・回答の質が低下します。
    そのため、各ブロックのbboxには以下を必ず含めてください:
    - テーブルの直上にあるタイトル行・見出し行（例: 「在庫リスト」「月次売上集計」等）
    - グラフの上部にあるタイトル
    - セクションの見出し
    タイトル・見出しだけを単独ブロックにするのではなく、必ず対応するコンテンツと一緒に1ブロックとして含めてください。

    【その他のbbox方針】
    - テーブルはタイトル行〜データ末尾まで必ず含める
    - 余白・空白領域は含めない
    - ブロックは重複しないようにする

    出力形式 (JSONのみ、```なし):
    [
      {{
        "id": 1,
        "label": "要素の種類 (例: 在庫リストテーブル, 月次グラフ, KPIエリア)",
        "description": "この要素が何を表しているかの簡単な説明",
        "bbox_2d": [x1, y1, x2, y2]
      }},
      ...
    ]

    bbox_2d の値は 0〜1000 の相対座標で指定してください (左上が原点)。
""")


def _parse_json_array(text: str) -> list | None:
    """
    JSONの配列文字列をパースする。途中で切れている場合も完全なアイテムだけ救出する。

    Returns:
        パース成功時はリスト、失敗時は None
    """
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    bracket_pos = text.find("[")
    if bracket_pos == -1:
        return None

    truncated = text[bracket_pos:]
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

    prompt = _DETECT_PROMPT
    raw = vlm.infer(sheet_image_path, prompt, system_prompt=_DETECT_SYSTEM)

    # マークダウンのコードブロックを除去してJSONパース
    raw_clean = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    regions = _parse_json_array(raw_clean)
    if regions is None:
        print(f"  JSON パース失敗, スキップ")
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

        # 小さすぎる矩形はノイズとしてスキップ（高さ100px未満はタイトル・ボタン・1行テキスト相当）
        if bw < 50 or bh < 100:
            continue

        # パディングを加えてクロップ
        pad = 8
        cx1 = max(0, x1 - pad)
        cy1 = max(0, y1 - pad)
        cx2 = min(w, x2 + pad)
        cy2 = min(h, y2 + pad)

        crop_img = img.crop((cx1, cy1, cx2, cy2))
        elem_id = len(crop_infos) + 1
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

    if len(crop_infos) > config.crop_limit:
        print(f"  [警告] 検出数 {len(crop_infos)} がガードレール {config.crop_limit} を超えたため切り詰めます")
        crop_infos = crop_infos[:config.crop_limit]

    print(f"  {len(crop_infos)} 要素を検出・クロップしました")
    return crop_infos
