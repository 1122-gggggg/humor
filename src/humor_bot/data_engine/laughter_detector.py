"""
YAMNet 笑聲偵測模組

功能：
- 使用 Google YAMNet 預訓練模型偵測音訊中的笑聲與掌聲
- 計算每個偵測事件的時間範圍與信心分數
- 支援合併相鄰笑聲事件
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

logger = logging.getLogger(__name__)

# YAMNet 每幀 0.48 秒，以 0.48 秒步進
YAMNET_FRAME_DURATION = 0.48

# 目標偵測的 AudioSet 類別名稱
DEFAULT_TARGET_CLASSES = {
    "Laughter",
    "Baby laughter",
    "Giggle",
    "Snicker",
    "Belly laugh",
    "Chuckle, chortle",
    "Crowd",
    "Clapping",
}


@dataclass
class LaughterEvent:
    """笑聲/掌聲事件"""
    start: float              # 起始時間（秒）
    end: float                # 結束時間（秒）
    duration: float           # 持續時間（秒）
    event_class: str          # 事件類別名稱
    confidence: float         # 平均信心分數
    peak_confidence: float    # 最高信心分數
    frame_count: int          # 偵測到的幀數


class LaughterDetector:
    """基於 YAMNet 的笑聲偵測器"""

    def __init__(
        self,
        model_url: str = "https://tfhub.dev/google/yamnet/1",
        target_classes: set[str] | None = None,
        confidence_threshold: float = 0.3,
        min_duration_sec: float = 0.5,
        merge_gap_sec: float = 1.0,
    ):
        """
        Args:
            model_url: YAMNet 模型 URL
            target_classes: 要偵測的事件類別
            confidence_threshold: 最低信心閾值
            min_duration_sec: 最短事件持續時間（秒）
            merge_gap_sec: 相鄰事件合併的最大間距（秒）
        """
        self.model_url = model_url
        self.target_classes = target_classes or DEFAULT_TARGET_CLASSES
        self.confidence_threshold = confidence_threshold
        self.min_duration_sec = min_duration_sec
        self.merge_gap_sec = merge_gap_sec

        self._model = None
        self._class_names: list[str] = []
        self._target_indices: list[int] = []

    def _load_model(self):
        """載入 YAMNet 模型與類別名稱"""
        if self._model is not None:
            return

        logger.info("載入 YAMNet 模型...")
        self._model = hub.load(self.model_url)

        # 載入類別名稱
        class_map_path = self._model.class_map_path().numpy().decode("utf-8")
        with open(class_map_path, "r") as f:
            reader = csv.DictReader(f)
            self._class_names = [row["display_name"] for row in reader]

        # 找出目標類別的 index
        self._target_indices = [
            i for i, name in enumerate(self._class_names)
            if name in self.target_classes
        ]

        logger.info(
            f"YAMNet 載入完成: {len(self._class_names)} 類別, "
            f"{len(self._target_indices)} 個目標類別"
        )

    def _load_audio(self, audio_path: str | Path) -> np.ndarray:
        """載入音訊檔案為 16kHz mono float32"""
        import soundfile as sf

        audio_path = Path(audio_path)
        waveform, sr = sf.read(str(audio_path), dtype="float32")

        # 如果是多聲道，取平均
        if waveform.ndim > 1:
            waveform = np.mean(waveform, axis=1)

        # 如果取樣率不是 16kHz，重新取樣
        if sr != 16000:
            import librosa
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16000)

        return waveform

    def detect(self, audio_path: str | Path) -> list[LaughterEvent]:
        """
        偵測音訊中的笑聲與掌聲事件

        Args:
            audio_path: 音訊檔案路徑（WAV 格式）

        Returns:
            LaughterEvent 列表，按起始時間排序
        """
        self._load_model()

        waveform = self._load_audio(audio_path)
        logger.info(f"音訊長度: {len(waveform) / 16000:.1f} 秒")

        # YAMNet 推論
        scores, embeddings, spectrogram = self._model(waveform)
        scores = scores.numpy()  # shape: (num_frames, num_classes)

        # 提取目標類別的分數
        raw_events = self._extract_raw_events(scores)

        # 合併相鄰事件
        merged_events = self._merge_events(raw_events)

        # 過濾短事件
        filtered_events = [
            e for e in merged_events
            if e.duration >= self.min_duration_sec
        ]

        logger.info(
            f"偵測結果: {len(raw_events)} 原始事件 → "
            f"{len(merged_events)} 合併後 → "
            f"{len(filtered_events)} 最終事件"
        )

        return filtered_events

    def _extract_raw_events(self, scores: np.ndarray) -> list[LaughterEvent]:
        """從 YAMNet 分數矩陣提取原始事件"""
        events = []

        for frame_idx in range(scores.shape[0]):
            frame_time = frame_idx * YAMNET_FRAME_DURATION

            for class_idx in self._target_indices:
                score = float(scores[frame_idx, class_idx])
                if score >= self.confidence_threshold:
                    events.append(LaughterEvent(
                        start=frame_time,
                        end=frame_time + YAMNET_FRAME_DURATION,
                        duration=YAMNET_FRAME_DURATION,
                        event_class=self._class_names[class_idx],
                        confidence=score,
                        peak_confidence=score,
                        frame_count=1,
                    ))

        # 按時間排序
        events.sort(key=lambda e: (e.start, e.event_class))
        return events

    def _merge_events(self, events: list[LaughterEvent]) -> list[LaughterEvent]:
        """合併相鄰的同類別事件"""
        if not events:
            return []

        # 按類別分組
        by_class: dict[str, list[LaughterEvent]] = {}
        for e in events:
            # 將笑聲類別合併為一個大類
            key = self._normalize_class(e.event_class)
            by_class.setdefault(key, []).append(e)

        merged = []
        for cls, cls_events in by_class.items():
            cls_events.sort(key=lambda e: e.start)
            current = cls_events[0]

            for next_event in cls_events[1:]:
                gap = next_event.start - current.end
                if gap <= self.merge_gap_sec:
                    # 合併
                    total_conf = current.confidence * current.frame_count + next_event.confidence
                    new_count = current.frame_count + 1
                    current = LaughterEvent(
                        start=current.start,
                        end=next_event.end,
                        duration=next_event.end - current.start,
                        event_class=cls,
                        confidence=total_conf / new_count,
                        peak_confidence=max(current.peak_confidence, next_event.peak_confidence),
                        frame_count=new_count,
                    )
                else:
                    merged.append(current)
                    current = next_event

            merged.append(current)

        merged.sort(key=lambda e: e.start)
        return merged

    def _normalize_class(self, class_name: str) -> str:
        """將細分類別正規化為大類"""
        laughter_classes = {
            "Laughter", "Baby laughter", "Giggle",
            "Snicker", "Belly laugh", "Chuckle, chortle",
        }
        if class_name in laughter_classes:
            return "Laughter"
        return class_name

    def to_json(self, events: list[LaughterEvent], output_path: str | Path) -> Path:
        """儲存偵測結果為 JSON"""
        import json
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = [asdict(e) for e in events]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"偵測結果已儲存: {output_path}")
        return output_path
