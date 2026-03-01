"""
VLM 推論バックエンド: Qwen3-VL (vLLM)
"""
from __future__ import annotations

import base64
from pathlib import Path
from typing import Optional

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

from .config import Config


class Qwen3VLBackend:
    """
    Qwen3-VL 推論エンジン（vLLM バックエンド）

    Usage:
        vlm = Qwen3VLBackend(config)
        vlm.load()
        text = vlm.infer(image_path, prompt, system_prompt)
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.llm: Optional[LLM] = None

    def load(self) -> None:
        """モデルを1回だけロードする"""
        if self.llm is not None:
            return

        print(f"[VLM] モデルをロード中: {self.config.model_name}")
        print(f"      dtype={self.config.dtype}, backend=vllm")

        self.llm = LLM(
            model=self.config.model_name,
            dtype=self.config.dtype,
            max_model_len=12000,
            gpu_memory_utilization=0.92,
            trust_remote_code=True,
            limit_mm_per_prompt={"image": 1},
        )
        print("[VLM] モデルのロード完了")

    def infer(
        self,
        image_path: Path,
        prompt: str,
        system_prompt: str = "",
    ) -> str:
        """
        画像 + テキストを入力してテキストを生成する。

        Args:
            image_path: 入力画像のパス
            prompt: ユーザープロンプト
            system_prompt: システムプロンプト（省略可）

        Returns:
            生成されたテキスト
        """
        if self.llm is None:
            raise RuntimeError("load() を先に呼び出してください")

        # 画像を base64 エンコード
        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")
        image_url = f"data:image/png;base64,{image_b64}"

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": prompt},
            ],
        })

        sampling_params = SamplingParams(
            max_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
        )

        outputs = self.llm.chat(
            messages=messages,
            sampling_params=sampling_params,
        )
        return outputs[0].outputs[0].text
