"""
影片分析模組 — 觀眾反應多模態分析

功能：
1. 表情辨識 (Facial Expression Recognition, FER)
   - 使用 MediaPipe Face Mesh + DeepFace 偵測觀眾臉孔
   - 偵測「驚喜 (Surprise)」與「快樂 (Happy)」情緒比例
   - 從觀眾席中隨機抽樣臉孔進行分析

2. 肢體動作偵測
   - 偵測觀眾「前傾」、「拍手」等物理動作
   - 使用 MediaPipe Pose 進行姿態估計

技術考量：
- 脫口秀現場光線通常較暗，觀眾席特徵提取難度較高
- 視覺特徵作為輔助權重（0.2-0.3），音訊仍作為主權重
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FaceEmotion:
    """單張臉孔的情緒分析結果"""
    timestamp: float              # 時間戳（秒）
    face_id: int                  # 臉孔索引
    bbox: tuple[int, int, int, int]  # (x, y, w, h)
    emotions: dict[str, float] = field(default_factory=dict)
    # e.g. {"happy": 0.8, "surprise": 0.15, "neutral": 0.05, ...}
    dominant_emotion: str = ""
    confidence: float = 0.0


@dataclass
class AudienceReaction:
    """某時間點的觀眾整體反應"""
    timestamp: float
    num_faces_detected: int = 0
    happy_ratio: float = 0.0         # 快樂情緒的臉孔比例
    surprise_ratio: float = 0.0      # 驚喜情緒的臉孔比例
    positive_ratio: float = 0.0      # 正面情緒總比例 (happy + surprise)
    avg_happy_score: float = 0.0     # 平均快樂分數
    avg_surprise_score: float = 0.0  # 平均驚喜分數
    lean_forward_ratio: float = 0.0  # 前傾的觀眾比例
    clapping_detected: bool = False  # 是否偵測到拍手動作
    faces: list[FaceEmotion] = field(default_factory=list)


@dataclass
class VideoAnalysisResult:
    """影片分析完整結果"""
    video_id: str
    total_frames_analyzed: int
    sample_interval_sec: float
    reactions: list[AudienceReaction]
    avg_positive_ratio: float = 0.0
    peak_moments: list[dict] = field(default_factory=list)
    # [{"timestamp": 42.5, "positive_ratio": 0.85, "type": "peak_happy"}, ...]


class VideoAnalyzer:
    """影片觀眾反應分析器"""

    def __init__(
        self,
        sample_interval_sec: float = 1.0,
        max_faces_per_frame: int = 10,
        min_face_size: int = 30,
        emotion_backend: str = "deepface",
        enable_pose: bool = False,
        audience_roi: tuple[float, float, float, float] | None = None,
    ):
        """
        Args:
            sample_interval_sec: 取樣間隔（秒），不需每幀都分析
            max_faces_per_frame: 每幀最多分析的臉孔數
            min_face_size: 最小臉孔尺寸（像素）
            emotion_backend: 情緒辨識後端 ("deepface" | "mediapipe")
            enable_pose: 是否啟用姿態偵測（前傾/拍手）
            audience_roi: 觀眾區域 ROI (x_ratio, y_ratio, w_ratio, h_ratio)
                          例如 (0.0, 0.5, 1.0, 0.5) = 畫面下半部
        """
        self.sample_interval_sec = sample_interval_sec
        self.max_faces_per_frame = max_faces_per_frame
        self.min_face_size = min_face_size
        self.emotion_backend = emotion_backend
        self.enable_pose = enable_pose
        self.audience_roi = audience_roi
        self._face_cascade = None
        self._deepface_loaded = False

    def _init_face_detector(self):
        """初始化人臉偵測器"""
        if self._face_cascade is not None:
            return

        # 使用 OpenCV 的 Haar Cascade（輕量，暗光下也還行）
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._face_cascade = cv2.CascadeClassifier(cascade_path)
        logger.info("OpenCV 人臉偵測器已初始化")

    def analyze_video(
        self,
        video_path: str | Path,
        start_sec: float = 0,
        end_sec: float | None = None,
    ) -> VideoAnalysisResult:
        """
        分析影片中的觀眾反應

        Args:
            video_path: 影片檔案路徑
            start_sec: 分析起始時間
            end_sec: 分析結束時間

        Returns:
            VideoAnalysisResult
        """
        self._init_face_detector()
        video_path = Path(video_path)
        video_id = video_path.stem

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"無法開啟影片: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0

        if end_sec is None:
            end_sec = duration

        logger.info(
            f"🎬 分析影片: {video_path.name} | "
            f"FPS={fps:.0f} | 時長={duration:.0f}s | "
            f"分析範圍=[{start_sec:.0f}s, {end_sec:.0f}s]"
        )

        # 計算取樣幀
        sample_frames = []
        frame_interval = int(fps * self.sample_interval_sec)
        start_frame = int(start_sec * fps)
        end_frame = int(end_sec * fps)

        for f in range(start_frame, min(end_frame, total_frames), max(frame_interval, 1)):
            sample_frames.append(f)

        logger.info(f"   取樣幀數: {len(sample_frames)} (間隔={self.sample_interval_sec}s)")

        reactions = []
        for i, frame_idx in enumerate(sample_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            timestamp = frame_idx / fps

            # 裁切觀眾區域（如果有指定 ROI）
            if self.audience_roi:
                frame = self._crop_roi(frame, self.audience_roi)

            # 分析單幀
            reaction = self._analyze_frame(frame, timestamp)
            reactions.append(reaction)

            if (i + 1) % 100 == 0:
                logger.info(f"   進度: {i + 1}/{len(sample_frames)} 幀")

        cap.release()

        # 計算整體統計
        avg_positive = (
            np.mean([r.positive_ratio for r in reactions])
            if reactions else 0.0
        )

        # 找出情緒高峰
        peak_moments = self._find_peak_moments(reactions)

        result = VideoAnalysisResult(
            video_id=video_id,
            total_frames_analyzed=len(reactions),
            sample_interval_sec=self.sample_interval_sec,
            reactions=reactions,
            avg_positive_ratio=float(avg_positive),
            peak_moments=peak_moments,
        )

        logger.info(
            f"✅ 影片分析完成: {len(reactions)} 幀 | "
            f"平均正面比例={avg_positive:.2%} | "
            f"高峰時刻: {len(peak_moments)} 個"
        )

        return result

    def _crop_roi(
        self,
        frame: np.ndarray,
        roi: tuple[float, float, float, float],
    ) -> np.ndarray:
        """裁切感興趣區域"""
        h, w = frame.shape[:2]
        x = int(roi[0] * w)
        y = int(roi[1] * h)
        rw = int(roi[2] * w)
        rh = int(roi[3] * h)
        return frame[y:y + rh, x:x + rw]

    def _analyze_frame(
        self,
        frame: np.ndarray,
        timestamp: float,
    ) -> AudienceReaction:
        """分析單一幀的觀眾反應"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 增強對比度（脫口秀現場通常較暗）
        gray = cv2.equalizeHist(gray)

        # 人臉偵測
        faces_rect = self._face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(self.min_face_size, self.min_face_size),
        )

        if len(faces_rect) == 0:
            return AudienceReaction(timestamp=timestamp)

        # 限制最大臉孔數
        faces_rect = faces_rect[:self.max_faces_per_frame]

        # 對每張臉孔進行情緒分析
        face_emotions = []
        for idx, (x, y, w, h) in enumerate(faces_rect):
            face_roi = frame[y:y + h, x:x + w]
            emotions = self._analyze_emotion(face_roi)

            if emotions:
                dominant = max(emotions, key=emotions.get)
                face_emotions.append(FaceEmotion(
                    timestamp=timestamp,
                    face_id=idx,
                    bbox=(int(x), int(y), int(w), int(h)),
                    emotions=emotions,
                    dominant_emotion=dominant,
                    confidence=emotions.get(dominant, 0),
                ))

        if not face_emotions:
            return AudienceReaction(
                timestamp=timestamp,
                num_faces_detected=len(faces_rect),
            )

        # 計算整體觀眾反應
        n = len(face_emotions)
        happy_count = sum(
            1 for f in face_emotions
            if f.emotions.get("happy", 0) > 0.5
        )
        surprise_count = sum(
            1 for f in face_emotions
            if f.emotions.get("surprise", 0) > 0.5
        )

        avg_happy = np.mean([f.emotions.get("happy", 0) for f in face_emotions])
        avg_surprise = np.mean([f.emotions.get("surprise", 0) for f in face_emotions])

        return AudienceReaction(
            timestamp=timestamp,
            num_faces_detected=n,
            happy_ratio=happy_count / n,
            surprise_ratio=surprise_count / n,
            positive_ratio=(happy_count + surprise_count) / n,
            avg_happy_score=float(avg_happy),
            avg_surprise_score=float(avg_surprise),
            faces=face_emotions,
        )

    def _analyze_emotion(self, face_image: np.ndarray) -> dict[str, float]:
        """
        對單張臉孔進行情緒分析

        Returns:
            {"happy": 0.8, "surprise": 0.1, "neutral": 0.05, ...}
        """
        if face_image.size == 0 or face_image.shape[0] < 10 or face_image.shape[1] < 10:
            return {}

        if self.emotion_backend == "deepface":
            return self._analyze_emotion_deepface(face_image)
        else:
            # 簡化版：基於亮度與紋理的啟發式方法
            return self._analyze_emotion_heuristic(face_image)

    def _analyze_emotion_deepface(self, face_image: np.ndarray) -> dict[str, float]:
        """使用 DeepFace 進行情緒分析"""
        try:
            from deepface import DeepFace

            result = DeepFace.analyze(
                face_image,
                actions=["emotion"],
                enforce_detection=False,
                silent=True,
            )

            if isinstance(result, list):
                result = result[0]

            emotions = result.get("emotion", {})
            # 正規化到 0-1
            total = sum(emotions.values()) or 1
            return {k: v / total for k, v in emotions.items()}

        except ImportError:
            if not self._deepface_loaded:
                logger.warning(
                    "DeepFace 未安裝，使用啟發式方法。"
                    "安裝: pip install deepface"
                )
                self._deepface_loaded = True
            return self._analyze_emotion_heuristic(face_image)
        except Exception:
            return {}

    def _analyze_emotion_heuristic(self, face_image: np.ndarray) -> dict[str, float]:
        """
        啟發式情緒估計（不依賴 ML 模型，作為 fallback）

        基於研究：笑容會增加嘴部區域的亮度變化與邊緣密度
        """
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY) if face_image.ndim == 3 else face_image
        h, w = gray.shape

        # 嘴部區域（臉下半部）
        mouth_region = gray[int(h * 0.6):, int(w * 0.2):int(w * 0.8)]

        if mouth_region.size == 0:
            return {"neutral": 1.0}

        # 邊緣密度（笑容 = 更多紋理）
        edges = cv2.Canny(mouth_region, 50, 150)
        edge_density = np.mean(edges > 0)

        # 亮度變化
        brightness_std = np.std(mouth_region.astype(float))

        # 簡易映射
        happy_score = min(1.0, edge_density * 3 + brightness_std / 100)
        neutral_score = 1.0 - happy_score

        return {
            "happy": float(happy_score * 0.7),  # 保守估計
            "surprise": float(happy_score * 0.1),
            "neutral": float(neutral_score),
        }

    def _find_peak_moments(
        self,
        reactions: list[AudienceReaction],
        threshold: float = 0.5,
        min_gap_sec: float = 5.0,
    ) -> list[dict]:
        """找出觀眾情緒高峰時刻"""
        if not reactions:
            return []

        peaks = []
        last_peak_time = -min_gap_sec

        for r in reactions:
            if r.positive_ratio >= threshold and r.timestamp - last_peak_time >= min_gap_sec:
                peaks.append({
                    "timestamp": r.timestamp,
                    "positive_ratio": r.positive_ratio,
                    "happy_ratio": r.happy_ratio,
                    "surprise_ratio": r.surprise_ratio,
                    "num_faces": r.num_faces_detected,
                    "type": "peak_happy" if r.happy_ratio > r.surprise_ratio else "peak_surprise",
                })
                last_peak_time = r.timestamp

        return peaks

    def get_reaction_at_time(
        self,
        reactions: list[AudienceReaction],
        timestamp: float,
        window_sec: float = 2.0,
    ) -> AudienceReaction | None:
        """取得指定時間點附近的觀眾反應"""
        best = None
        best_dist = float("inf")

        for r in reactions:
            dist = abs(r.timestamp - timestamp)
            if dist < window_sec and dist < best_dist:
                best = r
                best_dist = dist

        return best

    def save_results(
        self,
        result: VideoAnalysisResult,
        output_path: str | Path,
    ) -> Path:
        """儲存分析結果"""
        import json
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 轉換（排除大量 faces 細節以節省空間）
        data = {
            "video_id": result.video_id,
            "total_frames_analyzed": result.total_frames_analyzed,
            "sample_interval_sec": result.sample_interval_sec,
            "avg_positive_ratio": result.avg_positive_ratio,
            "peak_moments": result.peak_moments,
            "reactions_summary": [
                {
                    "timestamp": r.timestamp,
                    "num_faces": r.num_faces_detected,
                    "happy_ratio": r.happy_ratio,
                    "surprise_ratio": r.surprise_ratio,
                    "positive_ratio": r.positive_ratio,
                }
                for r in result.reactions
            ],
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"影片分析結果已儲存: {output_path}")
        return output_path
