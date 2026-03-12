"""
表演者表情分析模組 — 多模態物理對齊

功能：
1. 臉部地標 (Face Landmarks)
   - 使用 MediaPipe Face Mesh 提取 468 個臉部特徵點
   - 追蹤眼睛、眉毛、嘴巴的動作
   - 標註演員在 Punchline 時的表情變化

2. 表情動作單元 (Action Units, AU)
   - AU12: 嘴角上揚（微笑）
   - AU2: 外眉上揚（驚喜表情）
   - AU4: 眉毛下壓（嚴肅/反轉前的表情）
   - AU25: 嘴巴張開（說話/驚訝）

3. 頭部姿態 (Head Pose)
   - 俯仰角 (Pitch)
   - 偏轉角 (Yaw)
   - 傾斜角 (Roll)
   - Punchline 時的頭部微傾（典型的 comedy delivery）

4. 表演者表情時間軸
   - 對齊表情變化與 Setup-Punchline 結構
   - 供機器人物理表現同步使用
   - 避免「恐怖谷效應」

技術考量：
- 脫口秀舞台通常有正面燈光，表演者的臉部辨識準確度 > 觀眾
- MediaPipe Face Mesh 即使在側光條件下也有不錯的表現
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FacialExpression:
    """單一時間點的臉部表情"""
    timestamp: float

    # Action Units (簡化版)
    au12_smile: float = 0.0         # 嘴角上揚程度 (0-1)
    au2_brow_raise: float = 0.0     # 眉毛上揚程度 (0-1)
    au4_brow_lower: float = 0.0     # 眉毛下壓程度 (0-1)
    au25_mouth_open: float = 0.0    # 嘴巴張開程度 (0-1)
    au45_blink: float = 0.0         # 眨眼 (0/1)

    # 頭部姿態
    head_pitch: float = 0.0         # 俯仰角 (degrees)
    head_yaw: float = 0.0           # 偏轉角 (degrees)
    head_roll: float = 0.0          # 傾斜角 (degrees)

    # 綜合指標
    expression_energy: float = 0.0  # 表情活力 (0-1)
    landmarks_confidence: float = 0.0


@dataclass
class PerformerTimeline:
    """表演者表情時間軸"""
    video_id: str
    total_frames: int
    expressions: list[FacialExpression]

    # 統計
    avg_smile: float = 0.0
    avg_brow_raise: float = 0.0
    avg_mouth_open: float = 0.0
    expression_peaks: list[dict] = field(default_factory=list)


@dataclass
class PunchlineExpression:
    """Punchline 時的表情分析"""
    joke_id: str
    # Punchline 前的表情（Setup 結尾 2 秒）
    pre_punch_smile: float = 0.0
    pre_punch_brow: float = 0.0
    pre_punch_energy: float = 0.0

    # Punchline 時的表情
    punch_smile: float = 0.0
    punch_brow: float = 0.0
    punch_energy: float = 0.0

    # 表情變化幅度
    smile_delta: float = 0.0        # Punchline vs Setup 的笑容變化
    brow_delta: float = 0.0         # 眉毛變化
    energy_delta: float = 0.0       # 整體活力變化

    # 頭部動作
    head_tilt_at_punch: float = 0.0  # Punchline 時的頭部傾斜

    # 表演品質分數
    delivery_score: float = 0.0      # 表演品質 (0-1)


class PerformerAnalyzer:
    """表演者表情分析器"""

    # MediaPipe Face Mesh 關鍵地標索引
    # 嘴角
    LEFT_MOUTH_CORNER = 61
    RIGHT_MOUTH_CORNER = 291
    UPPER_LIP = 13
    LOWER_LIP = 14

    # 眉毛
    LEFT_BROW_OUTER = 70
    LEFT_BROW_INNER = 107
    RIGHT_BROW_OUTER = 300
    RIGHT_BROW_INNER = 336

    # 眼睛
    LEFT_EYE_UPPER = 159
    LEFT_EYE_LOWER = 145
    RIGHT_EYE_UPPER = 386
    RIGHT_EYE_LOWER = 374

    # 鼻尖（頭部姿態參考）
    NOSE_TIP = 1
    FOREHEAD = 10
    CHIN = 152

    def __init__(
        self,
        sample_interval_sec: float = 0.5,
        performer_roi: tuple[float, float, float, float] | None = None,
    ):
        """
        Args:
            sample_interval_sec: 取樣間隔
            performer_roi: 表演者區域 ROI (x, y, w, h) 0-1 比例
                          通常在舞台中央，例如 (0.25, 0.1, 0.5, 0.8)
        """
        self.sample_interval_sec = sample_interval_sec
        self.performer_roi = performer_roi or (0.2, 0.05, 0.6, 0.9)
        self._face_mesh = None

    def _init_face_mesh(self):
        """初始化 MediaPipe Face Mesh"""
        if self._face_mesh is not None:
            return

        try:
            import mediapipe as mp
            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,       # 只追蹤表演者一張臉
                refine_landmarks=True,  # 精細化地標
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            logger.info("MediaPipe Face Mesh 已初始化")
        except ImportError:
            logger.warning(
                "MediaPipe 未安裝。安裝: pip install mediapipe"
            )

    def analyze_performer(
        self,
        video_path: str | Path,
        start_sec: float = 0,
        end_sec: float | None = None,
    ) -> PerformerTimeline:
        """
        分析表演者的臉部表情時間軸

        Args:
            video_path: 影片路徑
            start_sec: 起始秒數
            end_sec: 結束秒數
        """
        self._init_face_mesh()
        video_path = Path(video_path)

        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0

        if end_sec is None:
            end_sec = duration

        logger.info(f"🎭 分析表演者表情: {video_path.name}")

        frame_interval = max(1, int(fps * self.sample_interval_sec))
        start_frame = int(start_sec * fps)
        end_frame = int(end_sec * fps)

        expressions = []

        for frame_idx in range(start_frame, min(end_frame, total_frames), frame_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            timestamp = frame_idx / fps

            # 裁切表演者區域
            performer_frame = self._crop_roi(frame, self.performer_roi)

            # 提取表情
            expr = self._analyze_frame_expression(performer_frame, timestamp)
            if expr:
                expressions.append(expr)

        cap.release()

        # 計算統計
        timeline = PerformerTimeline(
            video_id=video_path.stem,
            total_frames=len(expressions),
            expressions=expressions,
        )

        if expressions:
            timeline.avg_smile = float(np.mean([e.au12_smile for e in expressions]))
            timeline.avg_brow_raise = float(np.mean([e.au2_brow_raise for e in expressions]))
            timeline.avg_mouth_open = float(np.mean([e.au25_mouth_open for e in expressions]))
            timeline.expression_peaks = self._find_expression_peaks(expressions)

        logger.info(
            f"✅ 表情分析完成: {len(expressions)} 幀 | "
            f"平均笑容={timeline.avg_smile:.2f} | "
            f"表情高峰: {len(timeline.expression_peaks)} 個"
        )

        return timeline

    def _crop_roi(self, frame: np.ndarray, roi: tuple) -> np.ndarray:
        """裁切 ROI"""
        h, w = frame.shape[:2]
        x = int(roi[0] * w)
        y = int(roi[1] * h)
        rw = int(roi[2] * w)
        rh = int(roi[3] * h)
        return frame[y:y + rh, x:x + rw]

    def _analyze_frame_expression(
        self, frame: np.ndarray, timestamp: float
    ) -> FacialExpression | None:
        """分析單幀的表情"""
        if self._face_mesh is None:
            return self._fallback_expression(frame, timestamp)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return None

        landmarks = results.multi_face_landmarks[0].landmark
        h, w = frame.shape[:2]

        # 轉換為像素座標
        pts = {
            i: (lm.x * w, lm.y * h, lm.z * w)
            for i, lm in enumerate(landmarks)
        }

        # 計算 Action Units
        au12 = self._compute_smile(pts)
        au2 = self._compute_brow_raise(pts)
        au4 = self._compute_brow_lower(pts)
        au25 = self._compute_mouth_open(pts)
        au45 = self._compute_blink(pts)

        # 頭部姿態
        pitch, yaw, roll = self._estimate_head_pose(pts)

        # 表情活力
        energy = (au12 + au2 + au25) / 3.0

        return FacialExpression(
            timestamp=timestamp,
            au12_smile=au12,
            au2_brow_raise=au2,
            au4_brow_lower=au4,
            au25_mouth_open=au25,
            au45_blink=au45,
            head_pitch=pitch,
            head_yaw=yaw,
            head_roll=roll,
            expression_energy=energy,
            landmarks_confidence=float(np.mean([lm.visibility for lm in landmarks[:10]])),
        )

    def _compute_smile(self, pts: dict) -> float:
        """計算笑容程度（AU12: 嘴角上揚）"""
        left_corner = pts.get(self.LEFT_MOUTH_CORNER, (0, 0, 0))
        right_corner = pts.get(self.RIGHT_MOUTH_CORNER, (0, 0, 0))
        upper_lip = pts.get(self.UPPER_LIP, (0, 0, 0))

        # 嘴角與上嘴唇中心的 Y 距離比
        mouth_width = abs(right_corner[0] - left_corner[0])
        if mouth_width < 1:
            return 0.0

        avg_corner_y = (left_corner[1] + right_corner[1]) / 2
        lip_y = upper_lip[1]

        # 嘴角比上嘴唇高 → 笑容
        smile_ratio = (lip_y - avg_corner_y) / mouth_width
        return float(np.clip(smile_ratio * 5, 0, 1))

    def _compute_brow_raise(self, pts: dict) -> float:
        """計算眉毛上揚（AU2）"""
        left_brow = pts.get(self.LEFT_BROW_OUTER, (0, 0, 0))
        left_eye = pts.get(self.LEFT_EYE_UPPER, (0, 0, 0))

        dist = abs(left_brow[1] - left_eye[1])
        return float(np.clip(dist / 30.0, 0, 1))

    def _compute_brow_lower(self, pts: dict) -> float:
        """計算眉毛下壓（AU4）"""
        left_inner = pts.get(self.LEFT_BROW_INNER, (0, 0, 0))
        right_inner = pts.get(self.RIGHT_BROW_INNER, (0, 0, 0))

        # 內眉距離越近 → 皺眉
        dist = abs(left_inner[0] - right_inner[0])
        return float(np.clip(1.0 - dist / 50.0, 0, 1))

    def _compute_mouth_open(self, pts: dict) -> float:
        """計算嘴巴張開程度（AU25）"""
        upper = pts.get(self.UPPER_LIP, (0, 0, 0))
        lower = pts.get(self.LOWER_LIP, (0, 0, 0))

        dist = abs(lower[1] - upper[1])
        return float(np.clip(dist / 30.0, 0, 1))

    def _compute_blink(self, pts: dict) -> float:
        """偵測眨眼（AU45）"""
        upper = pts.get(self.LEFT_EYE_UPPER, (0, 0, 0))
        lower = pts.get(self.LEFT_EYE_LOWER, (0, 0, 0))

        eye_dist = abs(lower[1] - upper[1])
        return 1.0 if eye_dist < 3.0 else 0.0

    def _estimate_head_pose(self, pts: dict) -> tuple[float, float, float]:
        """估計頭部姿態 (pitch, yaw, roll)"""
        nose = pts.get(self.NOSE_TIP, (0, 0, 0))
        forehead = pts.get(self.FOREHEAD, (0, 0, 0))
        chin = pts.get(self.CHIN, (0, 0, 0))

        # 簡化的姿態估計
        # Pitch: 鼻尖與額頭-下巴中線的偏移
        face_center_y = (forehead[1] + chin[1]) / 2
        pitch = (nose[1] - face_center_y) / max(abs(chin[1] - forehead[1]), 1) * 45

        # Yaw: 鼻尖的 z 軸偏移
        yaw = nose[2] * -100  # 簡化

        # Roll: 左右眉毛的 y 差
        left_brow = pts.get(self.LEFT_BROW_OUTER, (0, 0, 0))
        right_brow = pts.get(self.RIGHT_BROW_OUTER, (0, 0, 0))
        dx = right_brow[0] - left_brow[0]
        dy = right_brow[1] - left_brow[1]
        roll = float(np.degrees(np.arctan2(dy, max(dx, 1))))

        return float(pitch), float(yaw), float(roll)

    def _fallback_expression(self, frame: np.ndarray, timestamp: float) -> FacialExpression | None:
        """當 MediaPipe 不可用時的 fallback"""
        return FacialExpression(timestamp=timestamp)

    def _find_expression_peaks(
        self, expressions: list[FacialExpression], threshold: float = 0.5
    ) -> list[dict]:
        """找出表情高峰"""
        peaks = []
        for e in expressions:
            if e.expression_energy >= threshold:
                peaks.append({
                    "timestamp": e.timestamp,
                    "energy": e.expression_energy,
                    "smile": e.au12_smile,
                    "brow_raise": e.au2_brow_raise,
                })
        return peaks

    def analyze_punchline_delivery(
        self,
        timeline: PerformerTimeline,
        setup_end: float,
        punch_start: float,
        punch_end: float,
        joke_id: str = "",
    ) -> PunchlineExpression:
        """
        分析 Punchline 時的表演品質

        好的 delivery 特徵：
        1. Setup 結尾保持嚴肅/中性表情（不要預先露餡）
        2. Punchline 時有微妙的表情變化
        3. 頭部微傾（casual delivery）
        """
        # Setup 結尾 2 秒
        pre_exprs = [
            e for e in timeline.expressions
            if setup_end - 2.0 <= e.timestamp <= setup_end
        ]

        # Punchline 期間
        punch_exprs = [
            e for e in timeline.expressions
            if punch_start <= e.timestamp <= punch_end
        ]

        # 計算平均值
        pre_smile = float(np.mean([e.au12_smile for e in pre_exprs])) if pre_exprs else 0
        pre_brow = float(np.mean([e.au2_brow_raise for e in pre_exprs])) if pre_exprs else 0
        pre_energy = float(np.mean([e.expression_energy for e in pre_exprs])) if pre_exprs else 0

        punch_smile = float(np.mean([e.au12_smile for e in punch_exprs])) if punch_exprs else 0
        punch_brow = float(np.mean([e.au2_brow_raise for e in punch_exprs])) if punch_exprs else 0
        punch_energy = float(np.mean([e.expression_energy for e in punch_exprs])) if punch_exprs else 0

        head_tilt = float(np.mean([abs(e.head_roll) for e in punch_exprs])) if punch_exprs else 0

        # 表演品質分數
        delivery_score = self._compute_delivery_score(
            pre_smile, punch_smile, pre_energy, punch_energy, head_tilt
        )

        return PunchlineExpression(
            joke_id=joke_id,
            pre_punch_smile=pre_smile,
            pre_punch_brow=pre_brow,
            pre_punch_energy=pre_energy,
            punch_smile=punch_smile,
            punch_brow=punch_brow,
            punch_energy=punch_energy,
            smile_delta=punch_smile - pre_smile,
            brow_delta=punch_brow - pre_brow,
            energy_delta=punch_energy - pre_energy,
            head_tilt_at_punch=head_tilt,
            delivery_score=delivery_score,
        )

    def _compute_delivery_score(
        self,
        pre_smile: float, punch_smile: float,
        pre_energy: float, punch_energy: float,
        head_tilt: float,
    ) -> float:
        """
        計算表演品質分數

        好的 delivery：
        1. Setup 時不笑（不露餡）→ pre_smile 低
        2. Punchline 時表情微妙變化 → smile_delta 適中
        3. 整體活力提升 → energy_delta > 0
        4. 適度的頭部傾斜 → 5-15 度
        """
        score = 0.0

        # 1. Setup 時不露餡
        if pre_smile < 0.3:
            score += 0.3

        # 2. Punchline 表情變化
        smile_delta = punch_smile - pre_smile
        if 0.05 <= smile_delta <= 0.5:  # 微妙的笑 → 好的 delivery
            score += 0.3

        # 3. 活力提升
        if punch_energy > pre_energy:
            score += 0.2

        # 4. 適度頭部傾斜
        if 3 <= head_tilt <= 20:
            score += 0.2

        return float(np.clip(score, 0, 1))
