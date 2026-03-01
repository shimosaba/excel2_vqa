"""
設定管理モジュール
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class Config:
    # VLM モデル設定
    model_name: str = "Qwen/Qwen3-VL-8B-Instruct-FP8"
    backend: str = "transformers"
    device: str = "cuda"
    dtype: str = "bfloat16"

    # 推論設定
    max_new_tokens: int = 4096
    temperature: float = 0.1

    # 画像レンダリング設定
    dpi: int = 0                  # 0 = シートサイズに応じて自動計算

    # ガードレール（VLMが極端な数を返した場合のみ使用）
    crop_limit: int = 30          # クロップ数の絶対上限（VLMが異常値を返した場合の保護）
    question_limit: int = 15      # 質問数の絶対上限（絞って品質向上）

    # 出力設定
    output_dir: str = "output"

    @classmethod
    def from_env(cls) -> "Config":
        """環境変数でオーバーライドした設定を返す"""
        cfg = cls()
        if v := os.environ.get("VQA_MODEL_NAME"):
            cfg.model_name = v
        if v := os.environ.get("VQA_BACKEND"):
            cfg.backend = v
        if v := os.environ.get("VQA_DEVICE"):
            cfg.device = v
        if v := os.environ.get("VQA_DTYPE"):
            cfg.dtype = v
        if v := os.environ.get("VQA_MAX_NEW_TOKENS"):
            cfg.max_new_tokens = int(v)
        if v := os.environ.get("VQA_TEMPERATURE"):
            cfg.temperature = float(v)
        if v := os.environ.get("VQA_DPI"):
            cfg.dpi = int(v)
        if v := os.environ.get("VQA_CROP_LIMIT"):
            cfg.crop_limit = int(v)
        if v := os.environ.get("VQA_QUESTION_LIMIT"):
            cfg.question_limit = int(v)
        if v := os.environ.get("VQA_OUTPUT_DIR"):
            cfg.output_dir = v
        return cfg
