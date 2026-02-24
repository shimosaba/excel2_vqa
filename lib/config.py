"""
設定管理モジュール
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class Config:
    # VLM モデル設定
    model_name: str = "Qwen/Qwen3-VL-2B-Instruct"
    backend: str = "transformers"
    device: str = "cuda"
    dtype: str = "bfloat16"

    # 推論設定
    max_new_tokens: int = 1024
    temperature: float = 0.1

    # 画像レンダリング設定
    dpi: int = 200

    # パイプライン設定
    max_crops: int = 10
    questions_per_crop: int = 5

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
        if v := os.environ.get("VQA_MAX_CROPS"):
            cfg.max_crops = int(v)
        if v := os.environ.get("VQA_QUESTIONS_PER_CROP"):
            cfg.questions_per_crop = int(v)
        if v := os.environ.get("VQA_OUTPUT_DIR"):
            cfg.output_dir = v
        return cfg
