"""
PPO 訓練器 — 使用 TRL 的強化學習微調

功能：
- 使用 Expert Reward Model 作為獎勵信號
- 獎勵 = α × humor_score + β × safety_score - γ × KL_divergence
- 在保持語意通順的同時，最大化幽默得分與安全界線的平衡
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead

logger = logging.getLogger(__name__)


@dataclass
class PPOHumorConfig:
    """PPO 訓練配置"""
    # 模型
    policy_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    reward_model_path: str = "checkpoints/reward_model/final_model.pt"
    sft_adapter_path: str = ""  # SFT 訓練後的 LoRA adapter

    # PPO 超參數
    learning_rate: float = 1e-6
    batch_size: int = 4
    mini_batch_size: int = 2
    ppo_epochs: int = 4
    max_new_tokens: int = 256
    temperature: float = 0.8
    top_p: float = 0.9

    # 獎勵權重
    humor_weight: float = 0.6       # α
    safety_weight: float = 0.3      # β
    kl_penalty: float = 0.1         # γ

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32

    # 訓練
    total_episodes: int = 1000
    save_every: int = 100
    output_dir: str = "checkpoints/ppo"


class PPOHumorTrainer:
    """PPO 幽默強化學習訓練器"""

    def __init__(self, config: PPOHumorConfig | None = None):
        self.config = config or PPOHumorConfig()
        self._policy_model = None
        self._ref_model = None
        self._tokenizer = None
        self._reward_model = None
        self._ppo_trainer = None

    def setup(self):
        """初始化所有模型與訓練器"""
        logger.info("初始化 PPO 訓練環境...")

        # Tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.policy_model,
            trust_remote_code=True,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # 量化配置
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        # LoRA 配置
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        # Policy 模型（帶 Value Head）
        self._policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            self.config.policy_model,
            peft_config=lora_config,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        # 如果有 SFT adapter，先載入
        if self.config.sft_adapter_path:
            logger.info(f"載入 SFT adapter: {self.config.sft_adapter_path}")
            from peft import PeftModel
            self._policy_model.pretrained_model = PeftModel.from_pretrained(
                self._policy_model.pretrained_model,
                self.config.sft_adapter_path,
            )

        # 載入 Reward Model
        self._load_reward_model()

        # PPO 配置
        ppo_config = PPOConfig(
            learning_rate=self.config.learning_rate,
            batch_size=self.config.batch_size,
            mini_batch_size=self.config.mini_batch_size,
            ppo_epochs=self.config.ppo_epochs,
            log_with=None,
        )

        self._ppo_trainer = PPOTrainer(
            config=ppo_config,
            model=self._policy_model,
            tokenizer=self._tokenizer,
        )

        logger.info("PPO 訓練環境初始化完成")

    def _load_reward_model(self):
        """載入 Expert Reward Model"""
        from humor_bot.training.reward_model import HumorRewardModel

        rm_path = Path(self.config.reward_model_path)
        if rm_path.exists():
            logger.info(f"載入 Reward Model: {rm_path}")
            self._reward_model = HumorRewardModel(base_model="roberta-large")
            self._reward_model.load_state_dict(
                torch.load(str(rm_path), map_location="cuda")
            )
            self._reward_model.eval().cuda()
        else:
            logger.warning(f"Reward Model 不存在: {rm_path}，使用隨機獎勵")
            self._reward_model = None

    def compute_reward(self, query: str, response: str) -> float:
        """
        計算複合獎勵

        reward = α × humor_score + β × safety_score - γ × length_penalty
        """
        combined_text = f"{query} {response}"

        # 幽默分數（來自 Expert Reward Model）
        if self._reward_model is not None:
            from transformers import AutoTokenizer as AT
            rm_tokenizer = AT.from_pretrained("roberta-large")
            enc = rm_tokenizer(
                combined_text,
                max_length=512,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).to("cuda")

            with torch.no_grad():
                humor_score = self._reward_model(
                    enc["input_ids"], enc["attention_mask"]
                ).item()
        else:
            humor_score = 0.0

        # 安全分數（簡化版：基於長度和基本啟發式）
        safety_score = self._quick_safety_check(response)

        # 長度懲罰（鼓勵簡潔）
        length_penalty = max(0, (len(response) - 500) / 1000)

        reward = (
            self.config.humor_weight * humor_score
            + self.config.safety_weight * safety_score
            - 0.05 * length_penalty
        )

        return reward

    def _quick_safety_check(self, text: str) -> float:
        """快速安全性檢查（啟發式）"""
        # 高風險關鍵字（極簡版，實際應使用 Llama Guard）
        high_risk_terms = ["歧視", "仇恨", "暴力", "種族"]
        for term in high_risk_terms:
            if term in text:
                return -1.0
        return 1.0

    def train(self, prompts: list[str]):
        """
        PPO 訓練主迴圈

        Args:
            prompts: 用於生成段子的 prompt 列表
        """
        if self._ppo_trainer is None:
            self.setup()

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"開始 PPO 訓練: {len(prompts)} 個 prompts")

        for episode in range(0, len(prompts), self.config.batch_size):
            batch_prompts = prompts[episode:episode + self.config.batch_size]

            # Tokenize prompts
            query_tensors = [
                self._tokenizer.encode(p, return_tensors="pt").squeeze()
                for p in batch_prompts
            ]

            # 生成回應
            response_tensors = []
            for qt in query_tensors:
                gen = self._ppo_trainer.generate(
                    qt.unsqueeze(0),
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    do_sample=True,
                )
                response_tensors.append(gen.squeeze()[len(qt):])

            # 解碼回應
            responses = [
                self._tokenizer.decode(r, skip_special_tokens=True)
                for r in response_tensors
            ]

            # 計算獎勵
            rewards = [
                torch.tensor(self.compute_reward(p, r))
                for p, r in zip(batch_prompts, responses)
            ]

            # PPO 更新
            stats = self._ppo_trainer.step(query_tensors, response_tensors, rewards)

            # 紀錄
            mean_reward = sum(r.item() for r in rewards) / len(rewards)
            logger.info(
                f"Episode {episode + 1} | "
                f"Mean Reward: {mean_reward:.4f} | "
                f"KL: {stats.get('objective/kl', 0):.4f}"
            )

            # 儲存 checkpoint
            if (episode + 1) % self.config.save_every == 0:
                ckpt_path = output_dir / f"step_{episode + 1}"
                self._policy_model.save_pretrained(str(ckpt_path))
                logger.info(f"Checkpoint 已儲存: {ckpt_path}")

        # 儲存最終模型
        final_path = output_dir / "final"
        self._policy_model.save_pretrained(str(final_path))
        self._tokenizer.save_pretrained(str(final_path))
        logger.info(f"PPO 訓練完成，模型已儲存: {final_path}")
