"""
音訊分析模組

功能：
- 計算笑聲片段的 RMS 分貝值（笑聲強度）
- 計算笑聲持續時間與時域特徵
- 產生笑聲強度曲線圖
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


@dataclass
class AudioFeatures:
    """音訊特徵分析結果"""
    start: float                # 起始時間（秒）
    end: float                  # 結束時間（秒）
    duration: float             # 持續時間（秒）
    rms_db: float               # RMS 分貝值
    peak_db: float              # 峰值分貝值
    mean_energy: float          # 平均能量
    spectral_centroid: float    # 頻譜質心（Hz）
    zero_crossing_rate: float   # 過零率


class AudioAnalyzer:
    """音訊特徵分析器"""

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self._waveform_cache: dict[str, np.ndarray] = {}

    def load_audio(self, audio_path: str | Path) -> np.ndarray:
        """載入音訊檔案（帶快取）與尺度對齊"""
        key = str(audio_path)
        if key not in self._waveform_cache:
            waveform, sr = sf.read(str(audio_path), dtype="float32")
            if waveform.ndim > 1:
                waveform = np.mean(waveform, axis=1)
            if sr != self.sample_rate:
                waveform = librosa.resample(waveform, orig_sr=sr, target_sr=self.sample_rate)
                
            # 音量尺度對齊 (Volume Normalization) 解決每部影片音量不同的問題
            # 這裡使用峰值正規化 (Peak Normalization)，將最大震幅對齊到 1.0 (等同 0 dBFS)
            # 確保不會因為不同影片的收音音量差異，導致笑聲強度的評估失真
            max_val = np.max(np.abs(waveform))
            if max_val > 0:
                waveform = waveform / max_val
                
            self._waveform_cache[key] = waveform
        return self._waveform_cache[key]

    def analyze_segment(
        self,
        audio_path: str | Path,
        start: float,
        end: float,
    ) -> AudioFeatures:
        """
        分析音訊片段的特徵

        Args:
            audio_path: 音訊檔案路徑
            start: 起始時間（秒）
            end: 結束時間（秒）

        Returns:
            AudioFeatures 分析結果
        """
        waveform = self.load_audio(audio_path)
        start_sample = int(start * self.sample_rate)
        end_sample = int(end * self.sample_rate)

        # 邊界檢查
        start_sample = max(0, start_sample)
        end_sample = min(len(waveform), end_sample)

        segment = waveform[start_sample:end_sample]

        if len(segment) == 0:
            return AudioFeatures(
                start=start, end=end, duration=end - start,
                rms_db=-100, peak_db=-100, mean_energy=0,
                spectral_centroid=0, zero_crossing_rate=0,
            )

        # RMS 分貝值
        rms = np.sqrt(np.mean(segment ** 2))
        rms_db = float(20 * np.log10(rms + 1e-10))

        # 峰值分貝值
        peak = np.max(np.abs(segment))
        peak_db = float(20 * np.log10(peak + 1e-10))

        # 平均能量
        mean_energy = float(np.mean(segment ** 2))

        # 頻譜質心
        centroid = librosa.feature.spectral_centroid(
            y=segment, sr=self.sample_rate, n_fft=min(2048, len(segment))
        )
        spectral_centroid = float(np.mean(centroid))

        # 過零率
        zcr = librosa.feature.zero_crossing_rate(segment)
        zero_crossing_rate = float(np.mean(zcr))

        return AudioFeatures(
            start=start,
            end=end,
            duration=end - start,
            rms_db=rms_db,
            peak_db=peak_db,
            mean_energy=mean_energy,
            spectral_centroid=spectral_centroid,
            zero_crossing_rate=zero_crossing_rate,
        )

    def compute_intensity_curve(
        self,
        audio_path: str | Path,
        window_sec: float = 0.5,
        hop_sec: float = 0.25,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        計算整段音訊的強度曲線

        Args:
            audio_path: 音訊檔案路徑
            window_sec: 分析視窗大小（秒）
            hop_sec: 視窗步進大小（秒）

        Returns:
            (times, db_values) — 時間軸與分貝值陣列
        """
        waveform = self.load_audio(audio_path)
        window_samples = int(window_sec * self.sample_rate)
        hop_samples = int(hop_sec * self.sample_rate)

        times = []
        db_values = []

        for start_sample in range(0, len(waveform) - window_samples, hop_samples):
            segment = waveform[start_sample:start_sample + window_samples]
            rms = np.sqrt(np.mean(segment ** 2))
            db = float(20 * np.log10(rms + 1e-10))
            time = start_sample / self.sample_rate
            times.append(time)
            db_values.append(db)

        return np.array(times), np.array(db_values)

    def plot_intensity(
        self,
        audio_path: str | Path,
        laughter_events: list | None = None,
        output_path: str | Path | None = None,
        title: str = "Audio Intensity Curve",
    ):
        """
        產生笑聲強度曲線圖

        Args:
            audio_path: 音訊檔案路徑
            laughter_events: 笑聲事件列表（用於標記）
            output_path: 圖片輸出路徑
            title: 圖表標題
        """
        import matplotlib.pyplot as plt
        import matplotlib

        matplotlib.use("Agg")

        times, db_values = self.compute_intensity_curve(audio_path)

        fig, ax = plt.subplots(figsize=(16, 5))
        ax.plot(times, db_values, color="#2196F3", linewidth=0.8, alpha=0.8)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Intensity (dB)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        # 標記笑聲事件
        if laughter_events:
            for event in laughter_events:
                start = event.start if hasattr(event, "start") else event["start"]
                end = event.end if hasattr(event, "end") else event["end"]
                ax.axvspan(start, end, color="#FF9800", alpha=0.25, label="Laughter")

            # 去除重複 legend
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc="upper right")

        plt.tight_layout()

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
            logger.info(f"強度曲線圖已儲存: {output_path}")

        plt.close(fig)
        return fig
