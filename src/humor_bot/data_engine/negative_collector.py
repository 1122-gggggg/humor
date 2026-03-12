"""
負面樣本收集器 — 解決倖存者偏差

功能：
1. 冷場偵測 (Bombing Detection)
   - 尋找「預期有笑聲但沒有」的段落
   - 偵測尷尬的沉默、禮貌性輕笑、觀眾無反應

2. 對比樣本建構
   - 同一演員的好段子 vs 冷場段子 → 偏好對 (DPO)
   - 結構相似但效果相反的段子 → Hard Negative

3. 冷場原因分析
   - 結構完整但不好笑（缺乏反轉）
   - 時機不對（韻律/停頓失敗）
   - 題材敏感（觀眾不敢笑）
   - 梗已過時（Context Gap）

學術價值：
- 知道「什麼時候不該講笑話」跟「怎麼講笑話」同等重要
- 負面樣本能顯著改善 Reward Model 的分界線
- 對比學習 (Contrastive Learning) 的基礎
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BombingSegment:
    """冷場段落"""
    id: str
    video_id: str
    start: float
    end: float
    text: str

    # 冷場指標
    bombing_type: str = ""        # silence / polite_laugh / awkward / heckle
    expected_laughter: bool = True  # 演員預期有笑聲嗎？
    actual_laughter_db: float = -60.0
    actual_laughter_duration: float = 0.0

    # 演員反應
    comedian_response: str = ""    # continued / acknowledged / pivoted / saved
    recovery_time: float = 0.0    # 冷場後多久恢復觀眾反應

    # 韻律特徵
    pre_bomb_pause: float = 0.0
    speech_rate_change: float = 0.0  # 冷場後語速變化

    # 原因分類
    failure_reason: str = ""      # no_reversal / bad_timing / sensitive / outdated / unclear


@dataclass
class ContrastPair:
    """對比樣本對"""
    positive_id: str
    negative_id: str
    positive_text: str
    negative_text: str
    positive_score: float
    negative_score: float
    similarity: float = 0.0     # 結構相似度
    pair_type: str = ""          # same_performer / same_topic / hard_negative


class NegativeSampleCollector:
    """負面樣本收集器"""

    # 冷場判定閾值
    BOMBING_DB_THRESHOLD = -45.0       # 笑聲分貝低於此為冷場
    BOMBING_DURATION_THRESHOLD = 0.3   # 笑聲短於此為冷場
    POLITE_LAUGH_DB = -35.0            # 禮貌笑聲的分貝範圍
    POLITE_LAUGH_MAX_DURATION = 1.0    # 禮貌笑聲的最長時間

    def __init__(
        self,
        bombing_db_threshold: float = -45.0,
        min_silence_for_bombing: float = 2.0,
    ):
        """
        Args:
            bombing_db_threshold: 冷場的笑聲分貝閾值
            min_silence_for_bombing: 冷場的最短沉默時間
        """
        self.bombing_db_threshold = bombing_db_threshold
        self.min_silence_for_bombing = min_silence_for_bombing

    def detect_bombing(
        self,
        transcript_segments: list[dict],
        laughter_events: list[dict],
        audio_features: list[dict],
    ) -> list[BombingSegment]:
        """
        偵測冷場段落

        策略：
        1. 找出有「段子結構」但「無笑聲反應」的段落
        2. 找出「笑聲極弱/極短」的段落（禮貌笑）
        3. 找出「觀眾長時間沉默」的區間

        Args:
            transcript_segments: 逐字稿片段
            laughter_events: 笑聲偵測結果
            audio_features: 音訊特徵

        Returns:
            BombingSegment 列表
        """
        logger.info("🥶 搜尋冷場段落...")
        bombings = []

        # 建立笑聲時間軸
        laughter_map = self._build_laughter_map(laughter_events)

        for i, seg in enumerate(transcript_segments):
            seg_start = seg.get("start", 0)
            seg_end = seg.get("end", 0)
            text = seg.get("text", "")

            # 跳過太短的段落
            if len(text) < 10 or (seg_end - seg_start) < 2.0:
                continue

            # 檢查此段落後 3 秒內是否有笑聲
            laughter_after = self._get_laughter_in_window(
                laughter_map, seg_end, seg_end + 3.0
            )

            if not laughter_after:
                # 完全沒有笑聲 → 潛在冷場
                bombing_type = "silence"
            else:
                # 有笑聲但很弱
                max_db = max(l.get("rms_db", -60) for l in laughter_after)
                max_dur = max(l.get("duration", 0) for l in laughter_after)

                if max_db < self.bombing_db_threshold:
                    bombing_type = "silence"
                elif max_db < self.POLITE_LAUGH_DB and max_dur < self.POLITE_LAUGH_MAX_DURATION:
                    bombing_type = "polite_laugh"
                else:
                    continue  # 正常笑聲，不是冷場

            # 判斷是否可能是為段子設計的片段
            # (有一定的文本長度且在語氣上像是 Punchline)
            if not self._looks_like_attempted_joke(text, transcript_segments, i):
                continue

            bombing = BombingSegment(
                id=f"{seg.get('video_id', 'unknown')}_{i:04d}_bomb",
                video_id=seg.get("video_id", ""),
                start=seg_start,
                end=seg_end,
                text=text,
                bombing_type=bombing_type,
                actual_laughter_db=max(
                    (l.get("rms_db", -60) for l in laughter_after), default=-60
                ),
                actual_laughter_duration=max(
                    (l.get("duration", 0) for l in laughter_after), default=0
                ),
            )
            bombings.append(bombing)

        logger.info(f"🥶 偵測到 {len(bombings)} 個冷場段落")
        return bombings

    def _build_laughter_map(self, events: list[dict]) -> list[dict]:
        """建立笑聲事件列表（按時間排序）"""
        return sorted(events, key=lambda e: e.get("start", 0))

    def _get_laughter_in_window(
        self, events: list[dict], start: float, end: float
    ) -> list[dict]:
        """取得指定時間窗口內的笑聲事件"""
        return [
            e for e in events
            if e.get("start", 0) >= start and e.get("start", 0) <= end
        ]

    def _looks_like_attempted_joke(
        self,
        text: str,
        all_segments: list[dict],
        current_idx: int,
    ) -> bool:
        """
        判斷文本是否像是「企圖講笑話」

        啟發式規則：
        - 有一定長度（> 15 字）
        - 前面有較長的 Setup 段落
        - 包含口語化的結尾語氣詞
        """
        if len(text) < 15:
            return False

        # 判斷是否有 Setup（前面有較長的段落）
        if current_idx > 0:
            prev_text = all_segments[current_idx - 1].get("text", "")
            if len(prev_text) > 10:
                return True

        # 口語化結尾（通常是 Punchline 的特徵）
        punchline_indicators = [
            "啊", "吧", "了", "嘛", "的", "耶", "欸",
            "！", "？", "...", "哈哈",
        ]
        for indicator in punchline_indicators:
            if text.endswith(indicator):
                return True

        return False

    def build_contrast_pairs(
        self,
        positive_candidates: list[dict],
        negative_candidates: list[dict] | list[BombingSegment],
        min_similarity: float = 0.3,
    ) -> list[ContrastPair]:
        """
        建構對比樣本對

        策略：
        1. 同一演員的成功 vs 失敗段子
        2. 結構相似但效果相反的段子（Hard Negative）

        Args:
            positive_candidates: 好笑的段子
            negative_candidates: 冷場的段子
            min_similarity: 最低結構相似度

        Returns:
            ContrastPair 列表
        """
        pairs = []

        for pos in positive_candidates:
            pos_vid = pos.get("video_id", "")
            pos_text = pos.get("full_text", pos.get("text", ""))
            pos_score = pos.get("humor_score", pos.get("expert_score", 0))

            for neg in negative_candidates:
                if isinstance(neg, BombingSegment):
                    neg_vid = neg.video_id
                    neg_text = neg.text
                    neg_score = 0.1
                else:
                    neg_vid = neg.get("video_id", "")
                    neg_text = neg.get("full_text", neg.get("text", ""))
                    neg_score = neg.get("humor_score", neg.get("expert_score", 0))

                # 同一演員的對比更有價值
                pair_type = "same_performer" if pos_vid == neg_vid else "cross_performer"

                # 結構相似度
                similarity = self._text_similarity(pos_text, neg_text)

                if similarity >= min_similarity:
                    pair_type = "hard_negative"

                pairs.append(ContrastPair(
                    positive_id=pos.get("id", ""),
                    negative_id=neg.id if isinstance(neg, BombingSegment) else neg.get("id", ""),
                    positive_text=pos_text,
                    negative_text=neg_text,
                    positive_score=float(pos_score) if pos_score else 0,
                    negative_score=float(neg_score) if neg_score else 0,
                    similarity=similarity,
                    pair_type=pair_type,
                ))

        # 按相似度排序（Hard Negative 優先）
        pairs.sort(key=lambda p: -p.similarity)

        logger.info(
            f"🔗 建構 {len(pairs)} 個對比對 | "
            f"Hard Negative: {sum(1 for p in pairs if p.pair_type == 'hard_negative')} | "
            f"Same Performer: {sum(1 for p in pairs if p.pair_type == 'same_performer')}"
        )

        return pairs

    def _text_similarity(self, text1: str, text2: str) -> float:
        """簡易文本相似度（字元 Jaccard）"""
        import re
        chars1 = set(re.sub(r'\s+', '', text1))
        chars2 = set(re.sub(r'\s+', '', text2))
        if not chars1 and not chars2:
            return 0.0
        intersection = chars1 & chars2
        union = chars1 | chars2
        return len(intersection) / len(union) if union else 0

    def save_bombings(
        self, bombings: list[BombingSegment], output_path: str | Path
    ) -> Path:
        """儲存冷場段落"""
        from dataclasses import asdict
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = [asdict(b) for b in bombings]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"冷場段落已儲存: {output_path} ({len(bombings)} 筆)")
        return output_path

    def save_contrast_pairs(
        self, pairs: list[ContrastPair], output_path: str | Path
    ) -> Path:
        """儲存對比對（DPO 格式）"""
        from dataclasses import asdict
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 轉換為 DPO 訓練格式
        dpo_data = []
        for p in pairs:
            dpo_data.append({
                "prompt": "請寫一個脫口秀段子：",
                "chosen": p.positive_text,
                "rejected": p.negative_text,
                "chosen_score": p.positive_score,
                "rejected_score": p.negative_score,
                "pair_type": p.pair_type,
                "similarity": p.similarity,
            })

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(dpo_data, f, ensure_ascii=False, indent=2)

        logger.info(f"對比對已儲存: {output_path} ({len(dpo_data)} 筆)")
        return output_path
