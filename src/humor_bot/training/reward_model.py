"""
Expert Reward Model — 基於 RoBERTa-Large 的幽默獎勵模型

功能：
- 使用 RoBERTa-Large 作為判別式基底（分類任務更精準）
- 訓練 R(x, y) 預測笑聲分貝 / 幽默分數
- Bradley-Terry pairwise ranking loss
- 支援從笑聲標註自動建構偏好對

損失函數：
    L(θ) = -E_{(x, y_w, y_l) ~ D} [log(σ(R_θ(x, y_w) - R_θ(x, y_l)))]
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset as TorchDataset, DataLoader
from transformers import (
    AutoModel,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

logger = logging.getLogger(__name__)


# ── Reward Model 架構 ─────────────────────────────────────────

class HumorRewardModel(nn.Module):
    """
    基於 RoBERTa-Large 的幽默獎勵模型

    架構：RoBERTa-Large → Mean Pooling → MLP → Scalar Reward
    """

    def __init__(
        self,
        base_model: str = "roberta-large",
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model)
        encoder_dim = self.encoder.config.hidden_size  # 1024 for roberta-large

        self.reward_head = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),  # 輸出 scalar reward
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        前向傳播

        Returns:
            reward: (batch_size,) 的獎勵值
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Mean pooling（遮蔽 padding）
        hidden = outputs.last_hidden_state
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden.size()).float()
        pooled = (hidden * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)

        reward = self.reward_head(pooled).squeeze(-1)
        return reward


# ── 偏好資料集 ─────────────────────────────────────────────

@dataclass
class PreferencePair:
    """偏好對"""
    prompt: str         # 上下文 x
    chosen: str         # 好笑的回答 y_w
    rejected: str       # 不好笑的回答 y_l
    chosen_score: float    # 好笑程度
    rejected_score: float  # 不好笑程度


class PreferenceDataset(TorchDataset):
    """偏好對資料集"""

    def __init__(
        self,
        pairs: list[PreferencePair],
        tokenizer: AutoTokenizer,
        max_length: int = 512,
    ):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]

        # 編碼 chosen（好笑）
        chosen_text = f"{pair.prompt} {pair.chosen}"
        chosen_enc = self.tokenizer(
            chosen_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # 編碼 rejected（不好笑）
        rejected_text = f"{pair.prompt} {pair.rejected}"
        rejected_enc = self.tokenizer(
            rejected_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "chosen_input_ids": chosen_enc["input_ids"].squeeze(0),
            "chosen_attention_mask": chosen_enc["attention_mask"].squeeze(0),
            "rejected_input_ids": rejected_enc["input_ids"].squeeze(0),
            "rejected_attention_mask": rejected_enc["attention_mask"].squeeze(0),
        }


# ── 訓練器 ─────────────────────────────────────────────────

class RewardModelTrainer:
    """Expert Reward Model 訓練器"""

    def __init__(
        self,
        base_model: str = "roberta-large",
        learning_rate: float = 1e-5,
        batch_size: int = 8,
        num_epochs: int = 3,
        max_length: int = 512,
        device: str = "cuda",
    ):
        self.base_model = base_model
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.max_length = max_length
        self.device = device

        self.model: HumorRewardModel | None = None
        self.tokenizer: AutoTokenizer | None = None

    def _init_model(self):
        """初始化模型與 tokenizer"""
        if self.model is not None:
            return

        logger.info(f"初始化 Reward Model: {self.base_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        self.model = HumorRewardModel(base_model=self.base_model).to(self.device)

    @staticmethod
    def build_preference_pairs(
        jokes: list[dict],
        score_key: str = "humor_score",
        min_score_diff: float = 0.15,
        max_pairs: int = 10000,
    ) -> list[PreferencePair]:
        """
        從笑聲標註自動建構偏好對

        Args:
            jokes: 笑話列表, 每個含 "full_text" 和 "humor_score"
            score_key: 幽默分數的 key
            min_score_diff: 最小分數差距（避免噪音對）
            max_pairs: 最大偏好對數量

        Returns:
            PreferencePair 列表
        """
        # 按分數排序
        sorted_jokes = sorted(jokes, key=lambda j: j.get(score_key, 0), reverse=True)

        pairs = []
        # 從高分與低分的配對中抽樣
        n = len(sorted_jokes)
        top_third = sorted_jokes[:n // 3]
        bottom_third = sorted_jokes[2 * n // 3:]

        for chosen in top_third:
            for rejected in bottom_third:
                chosen_score = chosen.get(score_key, 0)
                rejected_score = rejected.get(score_key, 0)

                if chosen_score - rejected_score >= min_score_diff:
                    pairs.append(PreferencePair(
                        prompt="以下是一段脫口秀段子：",
                        chosen=chosen.get("full_text", ""),
                        rejected=rejected.get("full_text", ""),
                        chosen_score=chosen_score,
                        rejected_score=rejected_score,
                    ))

                    if len(pairs) >= max_pairs:
                        break
            if len(pairs) >= max_pairs:
                break

        random.shuffle(pairs)
        logger.info(f"建構偏好對: {len(pairs)} 對")
        return pairs

    def train(
        self,
        pairs: list[PreferencePair],
        output_dir: str | Path = "checkpoints/reward_model",
        eval_pairs: list[PreferencePair] | None = None,
    ):
        """
        訓練 Reward Model

        使用 Bradley-Terry pairwise ranking loss:
        L(θ) = -E[log(σ(R_θ(x, y_w) - R_θ(x, y_l)))]
        """
        self._init_model()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 建立資料集
        train_dataset = PreferenceDataset(pairs, self.tokenizer, self.max_length)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )

        # 優化器與排程器
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01,
        )
        total_steps = len(train_loader) * self.num_epochs
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * 0.1),
            num_training_steps=total_steps,
        )

        # 訓練迴圈
        logger.info("開始訓練 Reward Model...")
        self.model.train()

        for epoch in range(self.num_epochs):
            total_loss = 0.0
            correct = 0
            total = 0

            for batch_idx, batch in enumerate(train_loader):
                # 前向：chosen
                chosen_rewards = self.model(
                    input_ids=batch["chosen_input_ids"].to(self.device),
                    attention_mask=batch["chosen_attention_mask"].to(self.device),
                )

                # 前向：rejected
                rejected_rewards = self.model(
                    input_ids=batch["rejected_input_ids"].to(self.device),
                    attention_mask=batch["rejected_attention_mask"].to(self.device),
                )

                # Bradley-Terry Loss
                reward_diff = chosen_rewards - rejected_rewards
                loss = -torch.log(torch.sigmoid(reward_diff) + 1e-8).mean()

                # 反向傳播
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()
                correct += (reward_diff > 0).sum().item()
                total += reward_diff.size(0)

                if (batch_idx + 1) % 50 == 0:
                    avg_loss = total_loss / (batch_idx + 1)
                    accuracy = correct / total
                    logger.info(
                        f"Epoch {epoch + 1}/{self.num_epochs} | "
                        f"Step {batch_idx + 1}/{len(train_loader)} | "
                        f"Loss: {avg_loss:.4f} | Acc: {accuracy:.2%}"
                    )

            epoch_loss = total_loss / len(train_loader)
            epoch_acc = correct / total
            logger.info(
                f"Epoch {epoch + 1} 完成 | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2%}"
            )

            # 儲存 checkpoint
            ckpt_path = output_dir / f"epoch_{epoch + 1}.pt"
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": epoch_loss,
                "accuracy": epoch_acc,
            }, str(ckpt_path))
            logger.info(f"Checkpoint 已儲存: {ckpt_path}")

        # 儲存最終模型
        final_path = output_dir / "final_model.pt"
        torch.save(self.model.state_dict(), str(final_path))
        logger.info(f"最終模型已儲存: {final_path}")

    def predict(self, text: str) -> float:
        """預測單一文本的獎勵值"""
        self._init_model()
        self.model.eval()

        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        with torch.no_grad():
            reward = self.model(
                input_ids=enc["input_ids"].to(self.device),
                attention_mask=enc["attention_mask"].to(self.device),
            )

        return reward.item()

    def load_checkpoint(self, path: str | Path):
        """載入模型 checkpoint"""
        self._init_model()
        state_dict = torch.load(str(path), map_location=self.device)

        if "model_state_dict" in state_dict:
            self.model.load_state_dict(state_dict["model_state_dict"])
        else:
            self.model.load_state_dict(state_dict)

        logger.info(f"模型已從 {path} 載入")
