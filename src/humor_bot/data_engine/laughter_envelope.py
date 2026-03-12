"""
笑聲時序包絡線分析模組 — 進階音訊事件偵測 (SED)

功能：
1. 笑聲包絡線分析 (Laughter Envelope)
   - 從 YAMNet 偵測結果中提取笑聲的時序包絡線
   - 計算爆發點（Attack）、持續時間（Sustain）、衰減時間（Decay）
   - ADSR 模型（Attack-Decay-Sustain-Release）

2. 笑聲質量特徵
   - 爆發速度 (Attack Rate): 笑聲從無到峰值的速率
   - 峰值強度 (Peak Intensity): 梅爾頻譜的最大能量
   - 餘韻長度 (Decay Duration): 笑聲從峰值衰減的時間
   - 笑聲紋理 (Texture): 群體 vs 個人笑聲

3. 段子品質推論
   - 爆發型笑聲 (quick attack) → 意外的 Punchline
   - 漸進型笑聲 (slow build) → 層層遞進的段子
   - 長餘韻 → 高品質段子（觀眾持續回味）
   - 反復波動 → 段子有多個笑點（Rule of Three）

4. 節奏感訓練資料
   - 將包絡線特徵編碼為 TTS 控制參數
   - 告訴機器人「何時開始說下一句」

技術原理：
- 梅爾頻譜圖（Mel-spectrogram）→ 笑聲頻帶的能量曲線
- 笑聲通常分佈在 500Hz-4kHz 頻帶
- RMS 能量包絡 + 一階差分 → Attack/Decay 偵測
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


@dataclass
class LaughterEnvelope:
    """笑聲的 ADSR 包絡線"""
    event_start: float         # 笑聲事件起始時間（秒）
    event_end: float           # 笑聲事件結束時間（秒）
    total_duration: float      # 總持續時間

    # ADSR 時間點
    attack_start: float = 0.0  # 爆發起始
    attack_peak: float = 0.0   # 到達峰值的時間
    attack_duration: float = 0.0  # 爆發時間（attack_peak - attack_start）
    decay_start: float = 0.0   # 衰減起始（= attack_peak）
    decay_end: float = 0.0     # 衰減結束（= sustain level）
    sustain_level: float = 0.0 # 持續階段的平均能量
    release_start: float = 0.0 # 釋放起始
    release_end: float = 0.0   # 釋放結束

    # 量化特徵
    attack_rate: float = 0.0    # 爆發速率 (dB/s)
    peak_intensity_db: float = 0.0  # 峰值強度 (dB)
    decay_rate: float = 0.0     # 衰減速率 (dB/s)
    decay_duration: float = 0.0 # 餘韻長度（秒）
    sustain_duration: float = 0.0  # 持續時間

    # 笑聲結構
    num_bursts: int = 1         # 笑聲爆發次數（反復波動）
    burst_pattern: str = ""     # single / double / ripple / sustained
    texture: str = ""           # solo / group / crowd

    # 能量曲線（取樣）
    energy_curve: list[float] = field(default_factory=list)
    time_axis: list[float] = field(default_factory=list)


@dataclass
class LaughterQuality:
    """笑聲品質評估"""
    event_id: str
    envelope: LaughterEnvelope

    # 品質指標
    comedy_quality_score: float = 0.0  # 基於笑聲推斷的段子品質 (0-1)
    surprise_factor: float = 0.0       # 驚喜程度（attack 速度）
    resonance_factor: float = 0.0      # 共鳴程度（decay 長度）
    multi_hit_factor: float = 0.0      # 多笑點程度（burst 數量）

    # 段子類型推論
    inferred_joke_type: str = ""       # surprise / buildup / running / flat


class LaughterEnvelopeAnalyzer:
    """笑聲包絡線分析器"""

    # 笑聲頻帶 (Hz)
    LAUGHTER_FREQ_LOW = 500
    LAUGHTER_FREQ_HIGH = 4000

    def __init__(
        self,
        sample_rate: int = 16000,
        hop_length: int = 512,
        n_fft: int = 2048,
        n_mels: int = 64,
        burst_threshold_db: float = -30.0,
    ):
        """
        Args:
            sample_rate: 音訊取樣率
            hop_length: 梅爾頻譜跳步
            n_fft: FFT 視窗大小
            n_mels: 梅爾頻帶數
            burst_threshold_db: 笑聲爆發的最低分貝
        """
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.burst_threshold_db = burst_threshold_db

    def compute_mel_energy(
        self,
        audio: np.ndarray,
        sr: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        計算笑聲頻帶的梅爾頻譜能量曲線

        Returns:
            (energy_db, time_axis)
        """
        # 梅爾頻譜
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.LAUGHTER_FREQ_LOW,
            fmax=self.LAUGHTER_FREQ_HIGH,
        )

        # 轉為 dB
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)

        # 取整個笑聲頻帶的平均能量
        energy_db = np.mean(mel_db, axis=0)
        time_axis = librosa.times_like(energy_db, sr=sr, hop_length=self.hop_length)

        return energy_db, time_axis

    def analyze_event(
        self,
        audio_path: str | Path,
        event_start: float,
        event_end: float,
        margin_sec: float = 1.0,
    ) -> LaughterEnvelope:
        """
        分析單一笑聲事件的包絡線

        Args:
            audio_path: 音訊路徑
            event_start: 笑聲起始（秒）
            event_end: 笑聲結束（秒）
            margin_sec: 前後擴展的邊距

        Returns:
            LaughterEnvelope
        """
        # 讀取音訊（含邊距）
        start = max(0, event_start - margin_sec)
        audio, sr = sf.read(
            str(audio_path),
            start=int(start * self.sample_rate),
            stop=int((event_end + margin_sec) * self.sample_rate),
        )

        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        if len(audio) == 0:
            return LaughterEnvelope(
                event_start=event_start,
                event_end=event_end,
                total_duration=event_end - event_start,
            )

        # 計算能量曲線
        energy_db, time_axis = self.compute_mel_energy(audio, sr)
        # 調整時間偏移
        time_axis = time_axis + start

        # ADSR 分析
        envelope = self._extract_adsr(energy_db, time_axis, event_start, event_end)

        # 爆發次數
        envelope.num_bursts = self._count_bursts(energy_db)
        envelope.burst_pattern = self._classify_burst_pattern(envelope.num_bursts, envelope.total_duration)

        # 笑聲紋理（群體 vs 個人）
        envelope.texture = self._classify_texture(audio, sr)

        # 儲存能量曲線（降採樣到最多 100 點）
        step = max(1, len(energy_db) // 100)
        envelope.energy_curve = energy_db[::step].tolist()
        envelope.time_axis = time_axis[::step].tolist()

        return envelope

    def _extract_adsr(
        self,
        energy_db: np.ndarray,
        time_axis: np.ndarray,
        event_start: float,
        event_end: float,
    ) -> LaughterEnvelope:
        """從能量曲線提取 ADSR 參數"""
        envelope = LaughterEnvelope(
            event_start=event_start,
            event_end=event_end,
            total_duration=event_end - event_start,
        )

        if len(energy_db) < 3:
            return envelope

        # 找峰值
        peak_idx = int(np.argmax(energy_db))
        peak_db = float(energy_db[peak_idx])
        peak_time = float(time_axis[peak_idx])

        envelope.peak_intensity_db = peak_db
        envelope.attack_peak = peak_time

        # Attack: 從事件起始到峰值
        # 找到第一個超過 threshold 的點
        attack_indices = np.where(
            (time_axis >= event_start) & (time_axis <= peak_time) &
            (energy_db > self.burst_threshold_db)
        )[0]

        if len(attack_indices) > 0:
            attack_start_idx = attack_indices[0]
            envelope.attack_start = float(time_axis[attack_start_idx])
            envelope.attack_duration = peak_time - envelope.attack_start

            if envelope.attack_duration > 0:
                start_db = float(energy_db[attack_start_idx])
                envelope.attack_rate = (peak_db - start_db) / envelope.attack_duration
        else:
            envelope.attack_start = event_start
            envelope.attack_duration = peak_time - event_start

        # Decay: 從峰值到笑聲衰減到 -6dB
        decay_threshold = peak_db - 6.0
        decay_indices = np.where(
            (time_axis > peak_time) & (energy_db < decay_threshold)
        )[0]

        if len(decay_indices) > 0:
            decay_end_idx = decay_indices[0]
            envelope.decay_start = peak_time
            envelope.decay_end = float(time_axis[decay_end_idx])
            envelope.decay_duration = envelope.decay_end - peak_time

            if envelope.decay_duration > 0:
                envelope.decay_rate = 6.0 / envelope.decay_duration  # dB/s
        else:
            envelope.decay_duration = event_end - peak_time
            envelope.decay_start = peak_time
            envelope.decay_end = event_end

        # Sustain: 衰減結束到 release
        sustain_indices = np.where(
            (time_axis >= envelope.decay_end) & (time_axis <= event_end)
        )[0]

        if len(sustain_indices) > 0:
            envelope.sustain_level = float(np.mean(energy_db[sustain_indices]))
            envelope.sustain_duration = event_end - envelope.decay_end

        # Release
        envelope.release_start = event_end
        envelope.release_end = event_end + 0.5  # 假設 0.5s release

        return envelope

    def _count_bursts(self, energy_db: np.ndarray) -> int:
        """計算笑聲爆發次數（能量曲線的局部最大值）"""
        if len(energy_db) < 5:
            return 1

        # 用一階差分找局部峰值
        smoothed = np.convolve(energy_db, np.ones(3) / 3, mode="same")
        diff = np.diff(smoothed)

        # 正→負的轉折 = 峰值
        peaks = 0
        for i in range(len(diff) - 1):
            if diff[i] > 0 and diff[i + 1] < 0:
                if smoothed[i + 1] > self.burst_threshold_db:
                    peaks += 1

        return max(1, peaks)

    def _classify_burst_pattern(self, num_bursts: int, duration: float) -> str:
        """分類笑聲爆發模式"""
        if num_bursts == 1:
            return "single"
        elif num_bursts == 2:
            return "double"
        elif num_bursts >= 3 and duration < 4.0:
            return "ripple"  # 快速連續爆發
        else:
            return "sustained"  # 持續的笑聲

    def _classify_texture(self, audio: np.ndarray, sr: int) -> str:
        """
        分類笑聲紋理（個人 vs 群體）

        基於頻譜複雜度：群體笑聲的頻譜更「寬」且混亂
        """
        # 頻譜通量（變化率）
        spec = np.abs(librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length))
        flux = np.mean(np.diff(spec, axis=1) ** 2)

        # 頻譜平坦度（群體笑聲更平坦/白噪音化）
        flatness = float(np.mean(librosa.feature.spectral_flatness(y=audio)))

        if flatness > 0.1 and flux > 0.01:
            return "crowd"
        elif flatness > 0.05:
            return "group"
        else:
            return "solo"

    def assess_comedy_quality(
        self,
        envelope: LaughterEnvelope,
        event_id: str = "",
    ) -> LaughterQuality:
        """
        從笑聲包絡線推斷段子品質

        邏輯：
        - 快速爆發 → 高驚喜 (Punchline 意外性強)
        - 長餘韻 → 高共鳴 (段子品質高，觀眾持續回味)
        - 多次爆發 → 多笑點 (Rule of Three / Callback)
        """
        # 驚喜程度 = 爆發速度（越快越驚喜）
        if envelope.attack_duration > 0:
            surprise = float(np.clip(envelope.attack_rate / 100, 0, 1))
        else:
            surprise = 0.5

        # 共鳴程度 = 餘韻長度（越長越好）
        resonance = float(np.clip(envelope.decay_duration / 5.0, 0, 1))

        # 多笑點 = 爆發次數
        multi_hit = float(np.clip((envelope.num_bursts - 1) / 3.0, 0, 1))

        # 綜合品質分數
        quality_score = (
            surprise * 0.35 +
            resonance * 0.40 +
            multi_hit * 0.25
        )

        # 推論段子類型
        if surprise > 0.6 and envelope.attack_duration < 0.5:
            joke_type = "surprise"
        elif resonance > 0.5 and envelope.decay_duration > 2.0:
            joke_type = "buildup"
        elif multi_hit > 0.5:
            joke_type = "running"
        else:
            joke_type = "flat"

        return LaughterQuality(
            event_id=event_id,
            envelope=envelope,
            comedy_quality_score=float(np.clip(quality_score, 0, 1)),
            surprise_factor=surprise,
            resonance_factor=resonance,
            multi_hit_factor=multi_hit,
            inferred_joke_type=joke_type,
        )

    def analyze_batch(
        self,
        audio_path: str | Path,
        laughter_events: list[dict],
    ) -> list[LaughterQuality]:
        """
        批量分析笑聲事件

        Args:
            audio_path: 音訊路徑
            laughter_events: 笑聲偵測結果列表

        Returns:
            LaughterQuality 列表
        """
        logger.info(f"🔊 分析 {len(laughter_events)} 個笑聲事件的時序包絡線...")

        results = []
        for i, event in enumerate(laughter_events):
            start = event.get("start", 0)
            end = event.get("end", start + 1)
            event_id = event.get("id", f"event_{i:04d}")

            envelope = self.analyze_event(audio_path, start, end)
            quality = self.assess_comedy_quality(envelope, event_id)
            results.append(quality)

        # 統計
        avg_quality = np.mean([r.comedy_quality_score for r in results]) if results else 0
        type_counts = {}
        for r in results:
            type_counts[r.inferred_joke_type] = type_counts.get(r.inferred_joke_type, 0) + 1

        logger.info(
            f"✅ 包絡線分析完成 | 平均品質={avg_quality:.2f} | "
            f"類型分佈: {type_counts}"
        )

        return results

    def save_results(
        self,
        results: list[LaughterQuality],
        output_path: str | Path,
    ) -> Path:
        """儲存分析結果"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = []
        for r in results:
            entry = {
                "event_id": r.event_id,
                "comedy_quality_score": r.comedy_quality_score,
                "surprise_factor": r.surprise_factor,
                "resonance_factor": r.resonance_factor,
                "multi_hit_factor": r.multi_hit_factor,
                "inferred_joke_type": r.inferred_joke_type,
                "envelope": {
                    "attack_duration": r.envelope.attack_duration,
                    "attack_rate": r.envelope.attack_rate,
                    "peak_intensity_db": r.envelope.peak_intensity_db,
                    "decay_duration": r.envelope.decay_duration,
                    "decay_rate": r.envelope.decay_rate,
                    "num_bursts": r.envelope.num_bursts,
                    "burst_pattern": r.envelope.burst_pattern,
                    "texture": r.envelope.texture,
                    "total_duration": r.envelope.total_duration,
                },
            }
            data.append(entry)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"包絡線分析已儲存: {output_path}")
        return output_path
