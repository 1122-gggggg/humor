"""
JokeWriter — 基於 Llama 3 + LoRA 的脫口秀段子生成器

功能：
- 使用 Llama 3-8B-Instruct 作為基座模型
- LoRA 低秩微調，學習 Setup-Punchline 語法結構與長線伏筆（Callbacks）
- 支援 SFT (Supervised Fine-tuning) 訓練與推論
- W = W₀ + ΔW = W₀ + BA（LoRA 低秩分解）
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

logger = logging.getLogger(__name__)


# ── SFT 訓練資料格式 ─────────────────────────────────────────

JOKE_SYSTEM_PROMPT = """你是一位專業的脫口秀喜劇演員。你擅長觀察生活中的荒謬，
用巧妙的 Setup（鋪陳）引導觀眾進入一個預期，然後用出人意料的 Punchline（翻轉）
打破這個預期，製造笑聲。你的風格融合了自嘲、社會觀察與文字遊戲。"""

JOKE_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_prompt}<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{instruction}

情境：{context}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
{joke}<|eot_id|>"""


@dataclass
class JokeWriterConfig:
    """JokeWriter 配置"""
    base_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    target_modules: list[str] | None = None
    max_seq_length: int = 2048
    load_in_4bit: bool = True
    use_flash_attention: bool = True

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ]


class JokeWriter:
    """基於 Llama 3 + LoRA 的脫口秀段子生成器"""

    def __init__(self, config: JokeWriterConfig | None = None):
        self.config = config or JokeWriterConfig()
        self._model = None
        self._tokenizer = None
        self._is_trained = False

    @property
    def model(self):
        if self._model is None:
            self._load_base_model()
        return self._model

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._load_base_model()
        return self._tokenizer

    def _load_base_model(self):
        """載入基座模型 + LoRA"""
        logger.info(f"載入基座模型: {self.config.base_model}")

        # 量化配置
        bnb_config = None
        if self.config.load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

        # 載入 tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model,
            trust_remote_code=True,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # 載入模型
        model_kwargs: dict[str, Any] = {
            "torch_dtype": torch.bfloat16,
            "trust_remote_code": True,
        }
        if bnb_config:
            model_kwargs["quantization_config"] = bnb_config
        if self.config.use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"

        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            **model_kwargs,
        )

        # 套用 LoRA
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        self._model = get_peft_model(self._model, lora_config)
        trainable, total = self._model.get_nb_trainable_parameters()
        logger.info(
            f"LoRA 已套用: 可訓練參數 {trainable:,} / 總參數 {total:,} "
            f"({100 * trainable / total:.2f}%)"
        )

    def prepare_dataset(self, data_path: str | Path) -> Dataset:
        """
        準備 SFT 訓練資料集

        Args:
            data_path: JSON 資料路徑，格式：
                [{"setup_text": "...", "punchline_text": "...",
                  "full_text": "...", "humor_score": 0.8, "tags": [...]}]

        Returns:
            HuggingFace Dataset
        """
        with open(data_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        formatted = []
        for item in raw_data:
            setup = item.get("setup_text", "")
            punchline = item.get("punchline_text", "")
            full_text = item.get("full_text", f"{setup} {punchline}")
            humor_score = item.get("humor_score", 0.5)
            tags = item.get("tags", [])

            # 構建多樣化指令
            instructions = [
                "根據以下情境，寫一個脫口秀段子。",
                "請用幽默的方式回應以下情境。",
                "寫一個有 Setup 和 Punchline 的笑話。",
                "用脫口秀的風格講述以下主題。",
            ]

            import random
            instruction = random.choice(instructions)

            # 使用 chat template 格式
            text = JOKE_TEMPLATE.format(
                system_prompt=JOKE_SYSTEM_PROMPT,
                instruction=instruction,
                context=setup[:200],  # 截斷過長的 context
                joke=full_text,
            )

            formatted.append({
                "text": text,
                "humor_score": humor_score,
            })

        dataset = Dataset.from_list(formatted)
        logger.info(f"資料集準備完成: {len(dataset)} 筆")
        return dataset

    def train(
        self,
        dataset: Dataset,
        output_dir: str | Path = "checkpoints/joke_writer",
        num_epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 2e-4,
        warmup_ratio: float = 0.1,
        gradient_accumulation_steps: int = 4,
        eval_dataset: Dataset | None = None,
    ):
        """
        SFT 訓練

        Args:
            dataset: 訓練資料集
            output_dir: 模型輸出路徑
            num_epochs: 訓練輪數
            batch_size: 批次大小
            learning_rate: 學習率
            warmup_ratio: 預熱比例
            gradient_accumulation_steps: 梯度累積步數
            eval_dataset: 驗證資料集
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        training_args = SFTConfig(
            output_dir=str(output_dir),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_ratio=warmup_ratio,
            lr_scheduler_type="cosine",
            bf16=True,
            logging_steps=10,
            save_strategy="epoch",
            eval_strategy="epoch" if eval_dataset else "no",
            save_total_limit=3,
            max_seq_length=self.config.max_seq_length,
            dataset_text_field="text",
            report_to="none",
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )

        trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            eval_dataset=eval_dataset,
            processing_class=self.tokenizer,
        )

        logger.info("開始 SFT 訓練...")
        trainer.train()

        # 儲存 LoRA adapter
        adapter_path = output_dir / "lora_adapter"
        self.model.save_pretrained(str(adapter_path))
        self.tokenizer.save_pretrained(str(adapter_path))
        logger.info(f"LoRA adapter 已儲存: {adapter_path}")

        self._is_trained = True

    def load_adapter(self, adapter_path: str | Path):
        """載入已訓練的 LoRA adapter"""
        logger.info(f"載入 LoRA adapter: {adapter_path}")
        self._model = PeftModel.from_pretrained(
            self.model.base_model,
            str(adapter_path),
        )
        self._is_trained = True

    def generate(
        self,
        context: str,
        instruction: str = "根據以下情境，寫一個脫口秀段子。",
        max_new_tokens: int = 512,
        temperature: float = 0.8,
        top_p: float = 0.9,
        top_k: int = 50,
        num_return_sequences: int = 1,
        do_sample: bool = True,
    ) -> list[str]:
        """
        生成脫口秀段子

        Args:
            context: 情境/主題
            instruction: 指令
            max_new_tokens: 最大生成 token 數
            temperature: 溫度（越高越有創意）
            top_p: nucleus sampling 閾值
            top_k: top-k sampling
            num_return_sequences: 生成數量
            do_sample: 是否使用隨機採樣

        Returns:
            生成的段子列表
        """
        prompt = JOKE_TEMPLATE.format(
            system_prompt=JOKE_SYSTEM_PROMPT,
            instruction=instruction,
            context=context,
            joke="",  # 讓模型生成
        )
        # 移除末尾的 eot_id，讓模型繼續生成
        prompt = prompt.rsplit("<|eot_id|>", 1)[0]

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                num_return_sequences=num_return_sequences,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # 解碼並移除 prompt 部分
        results = []
        for output in outputs:
            full_text = self.tokenizer.decode(output, skip_special_tokens=True)
            # 提取助理回應
            if "assistant" in full_text:
                response = full_text.split("assistant")[-1].strip()
            else:
                response = full_text[len(prompt):].strip()
            results.append(response)

        return results
