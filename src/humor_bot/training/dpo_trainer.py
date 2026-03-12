"""
DPO 訓練器 — 直接偏好優化（PPO 的替代方案）

功能：
- 使用 TRL 的 DPOTrainer
- 直接從偏好對進行優化，不需要獨立訓練 Reward Model
- 參數 β 控制與 reference model 的差異程度
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOConfig, DPOTrainer

logger = logging.getLogger(__name__)


class DPOHumorTrainer:
    """DPO 幽默直接偏好優化訓練器"""

    def __init__(
        self,
        base_model: str = "meta-llama/Llama-3.1-8B-Instruct",
        sft_adapter_path: str = "",
        beta: float = 0.1,
        learning_rate: float = 5e-7,
        batch_size: int = 4,
        num_epochs: int = 2,
        max_length: int = 1024,
        max_prompt_length: int = 256,
        output_dir: str = "checkpoints/dpo",
        lora_r: int = 16,
        lora_alpha: int = 32,
    ):
        self.base_model = base_model
        self.sft_adapter_path = sft_adapter_path
        self.beta = beta
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length
        self.output_dir = Path(output_dir)
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha

    def prepare_dataset(self, data_path: str | Path) -> Dataset:
        """
        準備 DPO 訓練資料集

        Args:
            data_path: JSON 路徑, 格式:
                [{"prompt": "...", "chosen": "好笑文本", "rejected": "不好笑文本"}]

        Returns:
            HuggingFace Dataset
        """
        with open(data_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        dataset = Dataset.from_list(raw_data)
        logger.info(f"DPO 資料集: {len(dataset)} 筆偏好對")
        return dataset

    @staticmethod
    def build_dpo_dataset_from_jokes(
        jokes: list[dict],
        score_key: str = "humor_score",
        min_score_diff: float = 0.15,
    ) -> list[dict]:
        """
        從笑話資料建構 DPO 資料集

        將高分笑話配對低分笑話，產生 {prompt, chosen, rejected} 格式
        """
        sorted_jokes = sorted(jokes, key=lambda j: j.get(score_key, 0), reverse=True)
        n = len(sorted_jokes)
        top = sorted_jokes[:n // 3]
        bottom = sorted_jokes[2 * n // 3:]

        pairs = []
        for chosen_joke in top:
            for rejected_joke in bottom:
                diff = chosen_joke.get(score_key, 0) - rejected_joke.get(score_key, 0)
                if diff >= min_score_diff:
                    pairs.append({
                        "prompt": "請寫一個脫口秀段子：",
                        "chosen": chosen_joke.get("full_text", ""),
                        "rejected": rejected_joke.get("full_text", ""),
                    })
                    break  # 每個 chosen 只配一個 rejected

        logger.info(f"DPO 偏好對: {len(pairs)} 筆")
        return pairs

    def train(self, dataset: Dataset):
        """
        執行 DPO 訓練

        Args:
            dataset: 包含 prompt, chosen, rejected 的資料集
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 量化配置
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        # 載入 tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.base_model,
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # 載入模型
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        # LoRA 配置
        peft_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        # DPO 訓練配置
        training_args = DPOConfig(
            output_dir=str(self.output_dir),
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            beta=self.beta,
            max_length=self.max_length,
            max_prompt_length=self.max_prompt_length,
            bf16=True,
            logging_steps=10,
            save_strategy="epoch",
            save_total_limit=2,
            gradient_accumulation_steps=4,
            gradient_checkpointing=True,
            report_to="none",
        )

        # 建立 DPO Trainer
        trainer = DPOTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            processing_class=tokenizer,
            peft_config=peft_config,
        )

        logger.info("開始 DPO 訓練...")
        trainer.train()

        # 儲存
        final_path = self.output_dir / "final_adapter"
        trainer.save_model(str(final_path))
        tokenizer.save_pretrained(str(final_path))
        logger.info(f"DPO 訓練完成，模型已儲存: {final_path}")
