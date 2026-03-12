"""
韻律分析模組 — Comedy is Timing

功能：
1. 基頻 (F0) 分析
   - 提取演員說話的音高曲線
   - Punchline 前的音高變化模式（通常先升後降）
   - 用於 TTS 情緒控制參數

2. 語速分析 (Speech Rate)
   - 每秒字數/音節數
   - Punchline 前的減速模式（build-up → slow delivery）

3. 停頓分析 (Pause Timing)
   - 自動偵測沉默段落
   - 「沉默」是幽默的一部分：comic pause
   - Setup 結尾的微停頓 + Punchline 後的長停頓（等笑聲）

4. 節奏特徵向量
   - 將韻律特徵編碼為向量
   - 可供 Reward Model 或 TTS 模型使用

學術參考：
- Warner (1999): "Prosodic correlates of humor in stand-up comedy"
- 韻律特徵與笑聲反應的相關性 r=0.52
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


@dataclass
class PauseEvent:
    """停頓事件"""
    start: float         # 開始時間（秒）
    end: float           # 結束時間（秒）
    duration: float      # 持續時間（秒）
    type: str            # "micro" (<0.3s) / "short" (0.3-1s) / "comic" (1-3s) / "long" (>3s)
    context: str = ""    # 停頓前後的文本


@dataclass
class ProsodyFeatures:
    """韻律特徵"""
    start: float
    end: float

    # F0 (基頻/音高)
    f0_mean: float = 0.0        # 平均基頻 (Hz)
    f0_std: float = 0.0         # 基頻標準差
    f0_range: float = 0.0       # 基頻範圍 (max - min)
    f0_slope: float = 0.0       # 基頻趨勢（正=升調，負=降調）
    f0_curve: list[float] = field(default_factory=list)

    # 語速
    speech_rate: float = 0.0    # 每秒音節數估計
    articulation_rate: float = 0.0  # 去除停頓的語速

    # 停頓
    pause_count: int = 0        # 停頓數量
    pause_total_duration: float = 0.0  # 停頓總時長
    pause_ratio: float = 0.0    # 停頓比例（停頓時間/總時間）
    longest_pause: float = 0.0  # 最長停頓

    # 能量
    energy_mean: float = 0.0    # 平均能量
    energy_std: float = 0.0     # 能量變化
    energy_contour: list[float] = field(default_factory=list)

    # 綜合節奏特徵向量（供 ML 使用）
    rhythm_vector: list[float] = field(default_factory=list)


@dataclass
class TimingAnalysis:
    """段子的時間節奏分析"""
    joke_id: str
    setup_prosody: ProsodyFeatures | None = None
    punchline_prosody: ProsodyFeatures | None = None
    pre_punchline_pause: float = 0.0   # Punchline 前的停頓時長
    post_punchline_pause: float = 0.0  # Punchline 後的停頓時長（等笑聲）
    setup_to_punch_speed_ratio: float = 0.0  # Setup vs Punchline 語速比
    f0_drop_at_punch: float = 0.0      # Punchline 時的音高下降
    timing_score: float = 0.0          # 節奏品質分數 (0-1)
    pauses: list[PauseEvent] = field(default_factory=list)


class ProsodyAnalyzer:
    """韻律分析器"""

    # 停頓分類閾值
    PAUSE_THRESHOLDS = {
        "micro": (0.1, 0.3),
        "short": (0.3, 1.0),
        "comic": (1.0, 3.0),
        "long": (3.0, float("inf")),
    }

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_length: int = 2048,
        hop_length: int = 512,
        silence_threshold_db: float = -40.0,
        min_pause_duration: float = 0.1,
    ):
        """
        Args:
            sample_rate: 音訊取樣率
            frame_length: FFT 視窗長度
            hop_length: 跳步長度
            silence_threshold_db: 靜音判定閾值 (dB)
            min_pause_duration: 最短停頓時間（秒）
        """
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.silence_threshold_db = silence_threshold_db
        self.min_pause_duration = min_pause_duration

    def extract_f0(
        self,
        audio: np.ndarray,
        sr: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        提取基頻 (F0) 曲線

        使用 PYIN 算法（比 YIN 更穩定，適合語音）

        Returns:
            (f0_values, timestamps) — f0 為 0 表示無聲段
        """
        f0, voiced_flag, voiced_prob = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz("C2"),   # ~65 Hz
            fmax=librosa.note_to_hz("C7"),   # ~2093 Hz
            sr=sr,
            frame_length=self.frame_length,
            hop_length=self.hop_length,
        )

        times = librosa.times_like(f0, sr=sr, hop_length=self.hop_length)

        # NaN → 0
        f0 = np.nan_to_num(f0, nan=0.0)

        return f0, times

    def detect_pauses(
        self,
        audio: np.ndarray,
        sr: int,
    ) -> list[PauseEvent]:
        """
        偵測停頓/沉默段落

        使用 RMS 能量低於閾值來判定靜音
        """
        rms = librosa.feature.rms(
            y=audio, frame_length=self.frame_length, hop_length=self.hop_length
        )[0]
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)
        times = librosa.times_like(rms, sr=sr, hop_length=self.hop_length)

        # 找出靜音段
        is_silent = rms_db < self.silence_threshold_db
        pauses = []
        in_pause = False
        pause_start = 0.0

        for i, (t, silent) in enumerate(zip(times, is_silent)):
            if silent and not in_pause:
                in_pause = True
                pause_start = t
            elif not silent and in_pause:
                in_pause = False
                duration = t - pause_start
                if duration >= self.min_pause_duration:
                    pause_type = self._classify_pause(duration)
                    pauses.append(PauseEvent(
                        start=pause_start,
                        end=t,
                        duration=duration,
                        type=pause_type,
                    ))

        # 處理結尾的靜音
        if in_pause:
            duration = times[-1] - pause_start
            if duration >= self.min_pause_duration:
                pauses.append(PauseEvent(
                    start=pause_start,
                    end=float(times[-1]),
                    duration=duration,
                    type=self._classify_pause(duration),
                ))

        return pauses

    def _classify_pause(self, duration: float) -> str:
        """分類停頓類型"""
        for ptype, (low, high) in self.PAUSE_THRESHOLDS.items():
            if low <= duration < high:
                return ptype
        return "long"

    def analyze_segment(
        self,
        audio_path: str | Path,
        start: float,
        end: float,
    ) -> ProsodyFeatures:
        """
        分析指定音訊片段的韻律特徵

        Args:
            audio_path: 音訊路徑
            start: 起始時間（秒）
            end: 結束時間（秒）

        Returns:
            ProsodyFeatures
        """
        audio, sr = sf.read(
            str(audio_path),
            start=int(start * self.sample_rate),
            stop=int(end * self.sample_rate),
        )

        if len(audio) == 0:
            return ProsodyFeatures(start=start, end=end)

        # 確保 mono
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        duration = end - start

        # F0 分析
        f0, f0_times = self.extract_f0(audio, sr)
        voiced_f0 = f0[f0 > 0]

        f0_mean = float(np.mean(voiced_f0)) if len(voiced_f0) > 0 else 0
        f0_std = float(np.std(voiced_f0)) if len(voiced_f0) > 0 else 0
        f0_range = float(np.ptp(voiced_f0)) if len(voiced_f0) > 0 else 0

        # F0 趨勢（線性迴歸斜率）
        f0_slope = 0.0
        if len(voiced_f0) > 2:
            x = np.arange(len(voiced_f0))
            coeffs = np.polyfit(x, voiced_f0, 1)
            f0_slope = float(coeffs[0])

        # 停頓分析
        pauses = self.detect_pauses(audio, sr)
        pause_total = sum(p.duration for p in pauses)
        speech_time = duration - pause_total

        # 能量
        rms = librosa.feature.rms(y=audio, frame_length=self.frame_length, hop_length=self.hop_length)[0]
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)

        # 語速估計（基於能量的音節偵測）
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
        syllable_count = len(librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr))
        speech_rate = syllable_count / duration if duration > 0 else 0
        articulation_rate = syllable_count / speech_time if speech_time > 0 else 0

        # 節奏特徵向量（供 ML 使用）
        rhythm_vector = [
            f0_mean / 300.0,        # 正規化到 ~0-1
            f0_std / 100.0,
            f0_slope / 50.0,
            speech_rate / 10.0,
            pause_total / duration if duration > 0 else 0,
            len(pauses) / 10.0,
            float(np.mean(rms_db)) / -60.0,
            float(np.std(rms_db)) / 20.0,
        ]

        return ProsodyFeatures(
            start=start,
            end=end,
            f0_mean=f0_mean,
            f0_std=f0_std,
            f0_range=f0_range,
            f0_slope=f0_slope,
            f0_curve=f0.tolist()[:100],  # 截斷避免太大
            speech_rate=speech_rate,
            articulation_rate=articulation_rate,
            pause_count=len(pauses),
            pause_total_duration=pause_total,
            pause_ratio=pause_total / duration if duration > 0 else 0,
            longest_pause=max((p.duration for p in pauses), default=0),
            energy_mean=float(np.mean(rms_db)),
            energy_std=float(np.std(rms_db)),
            energy_contour=rms_db.tolist()[:100],
            rhythm_vector=rhythm_vector,
        )

    def analyze_joke_timing(
        self,
        audio_path: str | Path,
        setup_start: float,
        setup_end: float,
        punch_start: float,
        punch_end: float,
        joke_id: str = "",
    ) -> TimingAnalysis:
        """
        分析完整段子的時間節奏

        包含 Setup → 停頓 → Punchline → 停頓（等笑聲）

        Args:
            audio_path: 音訊路徑
            setup_start/end: Setup 時間範圍
            punch_start/end: Punchline 時間範圍
            joke_id: 段子 ID
        """
        logger.info(f"⏱️ 分析段子節奏: {joke_id}")

        # Setup 韻律
        setup_prosody = self.analyze_segment(audio_path, setup_start, setup_end)

        # Punchline 韻律
        punch_prosody = self.analyze_segment(audio_path, punch_start, punch_end)

        # Setup 結尾到 Punchline 開頭的停頓
        pre_punch_pause = max(0, punch_start - setup_end)

        # Punchline 結尾的停頓（通常是等笑聲）
        # 分析 Punchline 後 5 秒的停頓
        post_region = self.analyze_segment(
            audio_path, punch_end, min(punch_end + 5.0, punch_end + 10.0)
        )
        post_punch_pause = post_region.longest_pause

        # Setup vs Punchline 語速比
        speed_ratio = (
            setup_prosody.speech_rate / punch_prosody.speech_rate
            if punch_prosody.speech_rate > 0 else 0
        )

        # Punchline 音高下降
        f0_drop = setup_prosody.f0_mean - punch_prosody.f0_mean

        # 完整段子的停頓列表
        full_audio, sr = sf.read(str(audio_path),
            start=int(setup_start * self.sample_rate),
            stop=int(min(punch_end + 5.0, punch_end + 10.0) * self.sample_rate),
        )
        if full_audio.ndim > 1:
            full_audio = full_audio.mean(axis=1)
        all_pauses = self.detect_pauses(full_audio, sr)
        # 調整時間偏移
        for p in all_pauses:
            p.start += setup_start
            p.end += setup_start

        # 節奏品質分數
        timing_score = self._compute_timing_score(
            pre_punch_pause, post_punch_pause,
            speed_ratio, f0_drop, all_pauses,
        )

        return TimingAnalysis(
            joke_id=joke_id,
            setup_prosody=setup_prosody,
            punchline_prosody=punch_prosody,
            pre_punchline_pause=pre_punch_pause,
            post_punchline_pause=post_punch_pause,
            setup_to_punch_speed_ratio=speed_ratio,
            f0_drop_at_punch=f0_drop,
            timing_score=timing_score,
            pauses=all_pauses,
        )

    def _compute_timing_score(
        self,
        pre_pause: float,
        post_pause: float,
        speed_ratio: float,
        f0_drop: float,
        pauses: list[PauseEvent],
    ) -> float:
        """
        計算節奏品質分數

        好的脫口秀節奏特徵：
        1. Punchline 前有短暫停頓（0.3-1.5s comic beat）
        2. Punchline 後有長停頓（讓笑聲發酵，>1s）
        3. Setup 語速略快，Punchline 語速略慢（speed_ratio > 1）
        4. Punchline 處音高下降（self-assured delivery）
        """
        score = 0.0

        # 1. 前停頓（comic beat）: 理想在 0.3-1.5s
        if 0.3 <= pre_pause <= 1.5:
            score += 0.25
        elif 0.1 <= pre_pause <= 2.0:
            score += 0.15

        # 2. 後停頓: > 1s 表示演員有信心等觀眾笑
        if post_pause >= 1.0:
            score += 0.25
        elif post_pause >= 0.5:
            score += 0.15

        # 3. 語速對比: Setup 快 Punchline 慢
        if speed_ratio > 1.0:
            score += min(0.25, (speed_ratio - 1.0) * 0.5)

        # 4. 音高下降: 自信的 delivery
        if f0_drop > 10:
            score += min(0.25, f0_drop / 100)

        return float(np.clip(score, 0, 1))
