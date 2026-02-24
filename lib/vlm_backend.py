"""
VLM 推論バックエンド: Qwen3-VL-2B (Transformers)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor
from qwen_vl_utils import process_vision_info

from .config import Config


class Qwen3VLBackend:
    """
    Qwen3-VL 推論エンジン（シングルトンパターン）

    Usage:
        vlm = Qwen3VLBackend(config)
        vlm.load()
        text = vlm.infer(image_path, prompt, system_prompt)
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.model: Optional[Qwen3VLForConditionalGeneration] = None
        self.processor: Optional[Qwen3VLProcessor] = None

    def load(self) -> None:
        """モデルを1回だけロードする（device_map="auto" + bfloat16）"""
        if self.model is not None:
            return  # 既にロード済み

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(self.config.dtype, torch.bfloat16)

        print(f"[VLM] モデルをロード中: {self.config.model_name}")
        print(f"      dtype={self.config.dtype}, device_map=auto")

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.config.model_name,
            dtype=torch_dtype,
            device_map="auto",
        )
        self.processor = Qwen3VLProcessor.from_pretrained(self.config.model_name)
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
        if self.model is None or self.processor is None:
            raise RuntimeError("load() を先に呼び出してください")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": str(image_path.resolve()),
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        })

        # チャットテンプレートを適用してトークン化
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # 画像・動画を前処理
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        # 推論
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                do_sample=self.config.temperature > 0,
            )

        # 入力トークンを除いた生成部分のみをデコード
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output_text[0]
