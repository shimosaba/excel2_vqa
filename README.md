# excel2-vqa

Excel ファイルから VQA (Visual Question Answering) データセットを全自動生成するパイプライン CLI。

## 概要

Excelのシートを画像化し、Qwen3-VL (マルチモーダルLLM) を使って視覚的な要素を検出・分類し、QAペアを自動生成します。生成したデータセットは LLaVA / ShareGPT / JSONL など複数のフォーマットで保存されます。

```text
Excel (.xlsx)
    │
    ▼ LibreOffice → PDF → PNG (シート別)
    │
    ▼ Qwen3-VL による要素検出・クロップ
    │
    ▼ 質問文の自動生成
    │
    ▼ 回答の自動生成 + バリデーション
    │
    ▼ VQAデータセット保存 (JSON / LLaVA / ShareGPT / JSONL)
```

## 要件

- Python 3.10+
- LibreOffice (Excel → PDF 変換に使用)
- CUDA 対応 GPU (推奨)

### LibreOffice のインストール

```bash
# Ubuntu / Debian
sudo apt install libreoffice

# macOS
brew install libreoffice
```

## インストール

```bash
# リポジトリをクローン
git clone <repo_url>
cd excel2_vqa

# uv でセットアップ (推奨)
uv sync

# または pip でインストール
pip install -e .
```

## 使い方

```bash
uv run python main.py <excel_file> [OPTIONS]
```

### オプション

| オプション | デフォルト | 説明 |
| --------- | --------- | ---- |
| `EXCEL_FILE` | (必須) | 入力 Excel ファイルのパス |
| `--output`, `-o` | `output` | 出力ディレクトリ |
| `--model` | `Qwen/Qwen3-VL-2B-Instruct` | 使用するモデル名 |
| `--max-crops` | `10` | 1シートあたりの最大クロップ数 |
| `--questions-per-crop` | `5` | クロップ画像1枚あたりの質問数 |
| `--dpi` | `200` | シート画像の解像度 (DPI) |

### 実行例

```bash
# 基本的な使い方
uv run python main.py sample.xlsx

# 出力先と品質を指定
uv run python main.py sample.xlsx --output ./dataset --dpi 300 --max-crops 20

# 大きいモデルを使用
uv run python main.py sample.xlsx --model Qwen/Qwen3-VL-7B-Instruct
```

## 出力ファイル

実行後、指定した出力ディレクトリに以下のファイルが生成されます。

```text
output/
├── sheets/                  # シート別 PNG 画像
│   ├── sheet_01.png
│   └── sheet_02.png
├── crops/                   # 要素クロップ画像
│   ├── entry_0001_*.png
│   └── ...
├── vqa_dataset.json         # 全情報を含む標準形式
├── dataset_llava.json       # LLaVA 学習形式
├── dataset_sharegpt.json    # ShareGPT 形式
└── dataset_flat.jsonl       # フラット JSONL 形式
```

### `vqa_dataset.json` の構造

```json
{
  "metadata": {
    "source_file": "sample.xlsx",
    "generated_at": "2026-02-24T12:00:00",
    "model": "Qwen/Qwen3-VL-2B-Instruct",
    "total_sheets": 3,
    "total_crops": 25,
    "total_qa_pairs": 120,
    "elapsed_seconds": 180.5,
    "statistics": {
      "by_type": {"descriptive": 60, "factual": 40, "comparative": 20},
      "by_difficulty": {"easy": 50, "medium": 50, "hard": 20},
      "by_confidence": {"high": 80, "medium": 30, "low": 10}
    }
  },
  "dataset": [
    {
      "id": "entry_0001",
      "source_file": "sample.xlsx",
      "sheet_name": "Sheet1",
      "question": "この表の合計値はいくらですか？",
      "answer": "合計値は 12,500 円です。",
      "question_type": "factual",
      "difficulty": "easy",
      "confidence": "high"
    }
  ]
}
```

## 環境変数による設定

CLI オプションの代わりに環境変数でも設定できます。

| 環境変数 | 対応オプション |
| ------- | ------------ |
| `VQA_MODEL_NAME` | `--model` |
| `VQA_BACKEND` | バックエンド種別 |
| `VQA_DEVICE` | 推論デバイス (`cuda` / `cpu`) |
| `VQA_DTYPE` | データ型 (`bfloat16` / `float16` / `float32`) |
| `VQA_MAX_NEW_TOKENS` | 最大生成トークン数 |
| `VQA_TEMPERATURE` | 生成温度 |
| `VQA_DPI` | `--dpi` |
| `VQA_MAX_CROPS` | `--max-crops` |
| `VQA_QUESTIONS_PER_CROP` | `--questions-per-crop` |
| `VQA_OUTPUT_DIR` | `--output` |

## プロジェクト構造

```text
excel2_vqa/
├── main.py                  # CLI エントリーポイント
├── pyproject.toml           # プロジェクト設定・依存関係
├── lib/
│   ├── config.py            # 設定管理 (Config dataclass)
│   ├── vlm_backend.py       # Qwen3-VL 推論バックエンド
│   ├── excel_renderer.py    # Excel → PNG 変換 (LibreOffice)
│   ├── element_detector.py  # 要素検出・クロップ生成
│   ├── question_generator.py # 質問文生成
│   ├── answer_generator.py  # 回答生成・VQAItem定義
│   ├── validators.py        # QAペアのバリデーション
│   └── formatters.py        # データセット保存 (複数フォーマット)
└── sample.xlsx              # サンプル入力ファイル
```

## 依存ライブラリ

| ライブラリ | 用途 |
| --------- | ---- |
| `transformers` | Qwen3-VL モデルのロード・推論 |
| `qwen-vl-utils` | Qwen-VL 用画像前処理ユーティリティ |
| `torch` | 深層学習フレームワーク |
| `accelerate` | GPU メモリ最適化・device_map |
| `openpyxl` | Excel ファイルの読み込み |
| `pdf2image` | PDF → PNG 変換 |
| `pillow` | 画像処理 |
| `click` | CLI フレームワーク |
| `tqdm` | プログレスバー |

## ライセンス

MIT
