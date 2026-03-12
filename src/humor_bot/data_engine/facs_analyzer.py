"""
FACS 分析模組 — 臉部動作編碼系統 + 骨架姿勢偵測

功能：
1. 杜氏笑容偵測 (Duchenne Smile Detection)
   - AU6 (臉頰提升 / Cheek Raiser) + AU12 (嘴角提升 / Lip Corner Puller)
   - 同時觸發 = 真誠笑容 (Duchenne)
   - 只有 AU12 = 禮貌笑容 (Non-Duchenne)
   - 輸出笑容「真誠度 (Sincerity)」和「強度 (Intensity)」

2. 觀眾骨架與姿勢偵測 (Pose Estimation)
   - 肩膀震顫 (Shoulder Tremor) — 大笑時的高頻抖動
   - 身體前傾 (Lean Forward) — 興趣 / 專注
   - 身體後仰 (Lean Back) — 爆笑時的本能反應
   - 拍手偵測 (Clapping) — 雙手座標重疊

3. 综合觀眾分析
   - 將 FACS + Pose 結合為統一的反應分數
   - 比單純「happy/neutral」分類更精準
   - 可用於精煉 Humor Score

技術選用：
- MediaPipe Face Mesh: 468 個臉部特徵點（含眼周與嘴部精細地標）
- MediaPipe Pose: 33 個骨架關鍵點
- 不依賴 OpenFace（安裝繁瑣），改用地標幾何計算 AU 近似值

學術參考：
- Ekman & Friesen (1978): FACS Action Unit 定義
- Duchenne (1862): 真誠笑容需要眼輪匝肌（AU6）參與
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ── 資料結構 ────────────────────────────────────────────

@dataclass
class ActionUnits:
    """臉部動作單元"""
    # 核心 AU
    au6_cheek_raise: float = 0.0     # 臉頰提升（眼輪匝肌）(0-1)
    au12_lip_corner_pull: float = 0.0  # 嘴角上提（顴大肌）(0-1)
    au1_inner_brow_raise: float = 0.0  # 內眉上揚 (0-1)
    au2_outer_brow_raise: float = 0.0  # 外眉上揚 (0-1)
    au4_brow_lowerer: float = 0.0     # 眉毛下壓 (0-1)
    au25_lips_part: float = 0.0       # 嘴唇分開 (0-1)
    au26_jaw_drop: float = 0.0        # 下巴張開 (0-1)

    # 綜合指標
    is_duchenne: bool = False         # 是否為杜氏笑容
    smile_sincerity: float = 0.0      # 笑容真誠度 (0-1)
    smile_intensity: float = 0.0      # 笑容強度 (0-1)


@dataclass
class BodyPose:
    """身體姿勢特徵"""
    lean_angle: float = 0.0             # 軀幹傾斜角度（正=前傾, 負=後仰）
    lean_type: str = "neutral"          # forward / backward / neutral
    shoulder_tremor: float = 0.0        # 肩膀震顫強度 (std of y-position)
    is_shaking: bool = False            # 是否在震顫（大笑）
    is_clapping: bool = False           # 是否在拍手
    hand_distance: float = 0.0          # 雙手距離（用於拍手偵測）


@dataclass
class AudienceMemberReaction:
    """單一觀眾的反應"""
    timestamp: float
    face_id: int
    action_units: ActionUnits
    body_pose: BodyPose | None = None

    # 綜合反應分數
    reaction_score: float = 0.0       # 0-1 (結合 FACS + Pose)
    reaction_type: str = "neutral"    # genuine_laugh / polite_smile / engaged / neutral


@dataclass
class FACSAnalysisResult:
    """完整的 FACS 分析結果"""
    timestamp: float
    num_analyzed: int = 0

    # 群體統計
    duchenne_ratio: float = 0.0        # 展示杜氏笑容的比例
    avg_smile_sincerity: float = 0.0   # 平均真誠度
    avg_smile_intensity: float = 0.0   # 平均強度
    shaking_ratio: float = 0.0         # 肩膀震顫的比例
    leaning_forward_ratio: float = 0.0
    leaning_backward_ratio: float = 0.0
    clapping_ratio: float = 0.0

    # 個體明細
    members: list[AudienceMemberReaction] = field(default_factory=list)

    # 綜合反應分數
    composite_score: float = 0.0       # 所有模態的綜合分數


# ── MediaPipe 地標索引 ──────────────────────────────────

class _FaceLandmarks:
    """MediaPipe Face Mesh 468 點的關鍵索引"""
    # 左眼下方（AU6 臉頰提升指標）
    LEFT_CHEEK_UPPER = 111
    LEFT_CHEEK_LOWER = 117
    LEFT_EYE_OUTER_UPPER = 130
    LEFT_EYE_INNER_LOWER = 133
    # 右眼
    RIGHT_CHEEK_UPPER = 340
    RIGHT_CHEEK_LOWER = 346
    RIGHT_EYE_OUTER_UPPER = 359
    RIGHT_EYE_INNER_LOWER = 362

    # 眼睛上下緣（AU6 - 眼睛「擠壓」）
    LEFT_UPPER_EYELID = 159
    LEFT_LOWER_EYELID = 145
    RIGHT_UPPER_EYELID = 386
    RIGHT_LOWER_EYELID = 374

    # 嘴角（AU12）
    LEFT_MOUTH_CORNER = 61
    RIGHT_MOUTH_CORNER = 291
    UPPER_LIP_CENTER = 13
    LOWER_LIP_CENTER = 14

    # 嘴唇外緣
    UPPER_LIP_TOP = 0
    LOWER_LIP_BOTTOM = 17

    # 眉毛
    LEFT_BROW_INNER = 107
    LEFT_BROW_OUTER = 70
    RIGHT_BROW_INNER = 336
    RIGHT_BROW_OUTER = 300

    # 參考點
    NOSE_TIP = 1
    NOSE_BRIDGE = 6
    FOREHEAD = 10
    CHIN = 152


class _PoseLandmarks:
    """MediaPipe Pose 33 點的關鍵索引"""
    NOSE = 0
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14


# ── 核心分析器 ────────────────────────────────────────

class FACSAnalyzer:
    """
    FACS + 姿勢分析器

    結合 MediaPipe Face Mesh 的精細地標與 Pose 的骨架偵測，
    提供比基本情緒分類更精確的觀眾反應量化。
    """

    # 杜氏笑容判定閾值
    DUCHENNE_AU6_THRESHOLD = 0.3     # AU6 最低觸發值
    DUCHENNE_AU12_THRESHOLD = 0.3    # AU12 最低觸發值

    # 姿勢閾值
    LEAN_FORWARD_ANGLE = 10          # 前傾角度 (degrees)
    LEAN_BACKWARD_ANGLE = -10        # 後仰角度
    SHOULDER_TREMOR_THRESHOLD = 3.0  # 震顫閾值 (pixels std)
    CLAPPING_DISTANCE_THRESHOLD = 50  # 雙手距離閾值 (pixels)

    def __init__(
        self,
        enable_pose: bool = True,
        sample_interval_sec: float = 0.5,
        audience_roi: tuple[float, float, float, float] | None = None,
    ):
        self.enable_pose = enable_pose
        self.sample_interval_sec = sample_interval_sec
        self.audience_roi = audience_roi
        self._face_mesh = None
        self._pose = None
        self._shoulder_history: dict[int, list[float]] = {}

    def _init_models(self):
        """初始化 MediaPipe 模型"""
        if self._face_mesh is not None:
            return

        try:
            import mediapipe as mp

            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=10,
                refine_landmarks=True,
                min_detection_confidence=0.4,
                min_tracking_confidence=0.4,
            )

            if self.enable_pose:
                self._pose = mp.solutions.pose.Pose(
                    static_image_mode=False,
                    model_complexity=0,    # 輕量模型（觀眾席需要速度）
                    min_detection_confidence=0.3,
                    min_tracking_confidence=0.3,
                )

            logger.info("MediaPipe FACS + Pose 模型已初始化")

        except ImportError:
            logger.warning(
                "MediaPipe 未安裝。安裝: pip install mediapipe\n"
                "將使用 OpenCV 幾何 fallback"
            )

    def analyze_frame(
        self,
        frame: np.ndarray,
        timestamp: float,
    ) -> FACSAnalysisResult:
        """
        分析單幀的觀眾 FACS + 姿勢

        Args:
            frame: BGR 影像
            timestamp: 時間戳

        Returns:
            FACSAnalysisResult
        """
        self._init_models()

        # 裁切觀眾區域
        if self.audience_roi:
            h, w = frame.shape[:2]
            x = int(self.audience_roi[0] * w)
            y = int(self.audience_roi[1] * h)
            rw = int(self.audience_roi[2] * w)
            rh = int(self.audience_roi[3] * h)
            frame = frame[y:y + rh, x:x + rw]

        members = []

        # 1. FACS 分析（Face Mesh）
        if self._face_mesh is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_results = self._face_mesh.process(rgb)

            if face_results.multi_face_landmarks:
                h, w = frame.shape[:2]
                for face_id, face_lm in enumerate(face_results.multi_face_landmarks):
                    # 轉換為像素座標
                    pts = {}
                    for i, lm in enumerate(face_lm.landmark):
                        pts[i] = (lm.x * w, lm.y * h, lm.z * w)

                    # 計算 Action Units
                    aus = self._compute_action_units(pts)

                    # 姿勢分析
                    pose = None
                    if self.enable_pose and self._pose is not None:
                        pose = self._analyze_pose(frame, face_id, timestamp)

                    # 計算綜合分數
                    reaction_score, reaction_type = self._classify_reaction(aus, pose)

                    members.append(AudienceMemberReaction(
                        timestamp=timestamp,
                        face_id=face_id,
                        action_units=aus,
                        body_pose=pose,
                        reaction_score=reaction_score,
                        reaction_type=reaction_type,
                    ))

        # 群體統計
        result = self._aggregate_results(timestamp, members)
        return result

    def _compute_action_units(self, pts: dict) -> ActionUnits:
        """
        從 Face Mesh 地標計算 Action Units

        核心創新：使用幾何關係（距離比、角度）近似 FACS 編碼
        """
        FL = _FaceLandmarks

        # ── AU6: 臉頰提升（Cheek Raiser）──
        # 指標：下眼瞼上移 → 眼睛開合度降低
        left_eye_open = self._dist(pts, FL.LEFT_UPPER_EYELID, FL.LEFT_LOWER_EYELID)
        right_eye_open = self._dist(pts, FL.RIGHT_UPPER_EYELID, FL.RIGHT_LOWER_EYELID)
        face_height = self._dist(pts, FL.FOREHEAD, FL.CHIN)

        if face_height < 1:
            return ActionUnits()

        # 正常眼睛開合度 ≈ face_height × 0.04
        # AU6 啟動時眼睛被擠壓 → 開合度降低
        norm_eye_open = (left_eye_open + right_eye_open) / 2 / face_height
        # 基準 ~0.04, AU6 時 → ~0.02
        au6 = float(np.clip((0.045 - norm_eye_open) / 0.025, 0, 1))

        # 額外驗證：臉頰上部的上移距離
        left_cheek_lift = self._vertical_displacement(pts, FL.LEFT_CHEEK_UPPER, FL.LEFT_CHEEK_LOWER)
        right_cheek_lift = self._vertical_displacement(pts, FL.RIGHT_CHEEK_UPPER, FL.RIGHT_CHEEK_LOWER)
        cheek_factor = float(np.clip((left_cheek_lift + right_cheek_lift) / 2 / face_height * 20, 0, 1))
        au6 = (au6 * 0.6 + cheek_factor * 0.4)

        # ── AU12: 嘴角提升（Lip Corner Puller）──
        left_corner = pts.get(FL.LEFT_MOUTH_CORNER, (0, 0, 0))
        right_corner = pts.get(FL.RIGHT_MOUTH_CORNER, (0, 0, 0))
        upper_lip = pts.get(FL.UPPER_LIP_CENTER, (0, 0, 0))

        mouth_width = abs(right_corner[0] - left_corner[0])
        avg_corner_y = (left_corner[1] + right_corner[1]) / 2
        lip_y = upper_lip[1]

        # 嘴角比上嘴唇高 → 笑容
        au12 = float(np.clip((lip_y - avg_corner_y) / max(mouth_width, 1) * 5, 0, 1))

        # ── AU1: 內眉上揚 ──
        left_inner = pts.get(FL.LEFT_BROW_INNER, (0, 0, 0))
        left_eye_upper = pts.get(FL.LEFT_UPPER_EYELID, (0, 0, 0))
        brow_eye_dist = abs(left_inner[1] - left_eye_upper[1])
        au1 = float(np.clip(brow_eye_dist / face_height * 10, 0, 1))

        # ── AU2: 外眉上揚 ──
        left_outer = pts.get(FL.LEFT_BROW_OUTER, (0, 0, 0))
        au2 = float(np.clip(abs(left_outer[1] - left_eye_upper[1]) / face_height * 10, 0, 1))

        # ── AU4: 眉毛下壓 ──
        right_inner = pts.get(FL.RIGHT_BROW_INNER, (0, 0, 0))
        brow_dist = abs(left_inner[0] - right_inner[0])
        au4 = float(np.clip(1.0 - brow_dist / face_height * 3, 0, 1))

        # ── AU25: 嘴唇分開 ──
        upper_lip_pt = pts.get(FL.UPPER_LIP_CENTER, (0, 0, 0))
        lower_lip_pt = pts.get(FL.LOWER_LIP_CENTER, (0, 0, 0))
        lip_gap = abs(lower_lip_pt[1] - upper_lip_pt[1])
        au25 = float(np.clip(lip_gap / face_height * 10, 0, 1))

        # ── AU26: 下巴張開 ──
        jaw_drop = abs(pts.get(FL.CHIN, (0, 0, 0))[1] - lower_lip_pt[1])
        au26 = float(np.clip(jaw_drop / face_height * 5, 0, 1))

        # ── 杜氏笑容判定 ──
        is_duchenne = (au6 >= self.DUCHENNE_AU6_THRESHOLD and
                       au12 >= self.DUCHENNE_AU12_THRESHOLD)

        # 真誠度 = AU6 和 AU12 的調和平均值
        if au6 + au12 > 0:
            sincerity = 2 * au6 * au12 / (au6 + au12) if is_duchenne else au12 * 0.3
        else:
            sincerity = 0.0

        # 強度 = AU12 × (1 + AU6 加成) × 嘴巴張開加成
        intensity = au12 * (1 + au6 * 0.5) * (1 + au25 * 0.3)
        intensity = float(np.clip(intensity, 0, 1))

        return ActionUnits(
            au6_cheek_raise=au6,
            au12_lip_corner_pull=au12,
            au1_inner_brow_raise=au1,
            au2_outer_brow_raise=au2,
            au4_brow_lowerer=au4,
            au25_lips_part=au25,
            au26_jaw_drop=au26,
            is_duchenne=is_duchenne,
            smile_sincerity=float(sincerity),
            smile_intensity=float(intensity),
        )

    def _analyze_pose(self, frame: np.ndarray, face_id: int, timestamp: float) -> BodyPose:
        """分析身體姿勢（前傾/後仰/震顫/拍手）"""
        if self._pose is None:
            return BodyPose()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._pose.process(rgb)

        if not results.pose_landmarks:
            return BodyPose()

        h, w = frame.shape[:2]
        lm = results.pose_landmarks.landmark
        PL = _PoseLandmarks

        # 軀幹傾斜角（肩膀中心 vs 臀部中心）
        shoulder_mid_y = (lm[PL.LEFT_SHOULDER].y + lm[PL.RIGHT_SHOULDER].y) / 2 * h
        hip_mid_y = (lm[PL.LEFT_HIP].y + lm[PL.RIGHT_HIP].y) / 2 * h
        shoulder_mid_x = (lm[PL.LEFT_SHOULDER].x + lm[PL.RIGHT_SHOULDER].x) / 2 * w
        hip_mid_x = (lm[PL.LEFT_HIP].x + lm[PL.RIGHT_HIP].x) / 2 * w

        # 計算傾斜角度
        dx = shoulder_mid_x - hip_mid_x
        dy = shoulder_mid_y - hip_mid_y
        lean_angle = float(np.degrees(np.arctan2(dx, -dy)))

        if lean_angle > self.LEAN_FORWARD_ANGLE:
            lean_type = "forward"
        elif lean_angle < self.LEAN_BACKWARD_ANGLE:
            lean_type = "backward"
        else:
            lean_type = "neutral"

        # 肩膀震顫（追蹤 y 座標的高頻變化）
        shoulder_y = (lm[PL.LEFT_SHOULDER].y + lm[PL.RIGHT_SHOULDER].y) / 2

        if face_id not in self._shoulder_history:
            self._shoulder_history[face_id] = []
        self._shoulder_history[face_id].append(shoulder_y * h)

        # 保持最近 10 幀的歷史
        if len(self._shoulder_history[face_id]) > 10:
            self._shoulder_history[face_id] = self._shoulder_history[face_id][-10:]

        tremor = 0.0
        is_shaking = False
        if len(self._shoulder_history[face_id]) >= 5:
            tremor = float(np.std(self._shoulder_history[face_id][-5:]))
            is_shaking = tremor > self.SHOULDER_TREMOR_THRESHOLD

        # 拍手偵測（雙手腕距離）
        left_wrist = (lm[PL.LEFT_WRIST].x * w, lm[PL.LEFT_WRIST].y * h)
        right_wrist = (lm[PL.RIGHT_WRIST].x * w, lm[PL.RIGHT_WRIST].y * h)
        hand_dist = float(np.sqrt(
            (left_wrist[0] - right_wrist[0]) ** 2 +
            (left_wrist[1] - right_wrist[1]) ** 2
        ))
        is_clapping = hand_dist < self.CLAPPING_DISTANCE_THRESHOLD

        return BodyPose(
            lean_angle=lean_angle,
            lean_type=lean_type,
            shoulder_tremor=tremor,
            is_shaking=is_shaking,
            is_clapping=is_clapping,
            hand_distance=hand_dist,
        )

    def _classify_reaction(
        self,
        aus: ActionUnits,
        pose: BodyPose | None,
    ) -> tuple[float, str]:
        """
        分類觀眾反應

        Returns:
            (reaction_score, reaction_type)
        """
        score = 0.0

        # FACS 貢獻（70%）
        if aus.is_duchenne:
            score += 0.4 * aus.smile_sincerity + 0.30 * aus.smile_intensity
            reaction_type = "genuine_laugh"
        elif aus.au12_lip_corner_pull > 0.2:
            score += 0.15 * aus.au12_lip_corner_pull
            reaction_type = "polite_smile"
        else:
            reaction_type = "neutral"

        # 嘴巴大開（驚喜）
        if aus.au25_lips_part > 0.5 or aus.au26_jaw_drop > 0.3:
            score += 0.1

        # 姿勢貢獻（30%）
        if pose:
            if pose.is_shaking:
                score += 0.15
                reaction_type = "genuine_laugh"
            if pose.lean_type == "backward":
                score += 0.10  # 爆笑後仰
            elif pose.lean_type == "forward":
                score += 0.05  # 前傾（興趣）
                if reaction_type == "neutral":
                    reaction_type = "engaged"
            if pose.is_clapping:
                score += 0.10

        return float(np.clip(score, 0, 1)), reaction_type

    def _aggregate_results(
        self,
        timestamp: float,
        members: list[AudienceMemberReaction],
    ) -> FACSAnalysisResult:
        """聚合個體結果為群體統計"""
        result = FACSAnalysisResult(
            timestamp=timestamp,
            num_analyzed=len(members),
            members=members,
        )

        if not members:
            return result

        n = len(members)
        result.duchenne_ratio = sum(1 for m in members if m.action_units.is_duchenne) / n
        result.avg_smile_sincerity = float(np.mean([m.action_units.smile_sincerity for m in members]))
        result.avg_smile_intensity = float(np.mean([m.action_units.smile_intensity for m in members]))

        poses = [m for m in members if m.body_pose is not None]
        if poses:
            result.shaking_ratio = sum(1 for m in poses if m.body_pose.is_shaking) / len(poses)
            result.leaning_forward_ratio = sum(1 for m in poses if m.body_pose.lean_type == "forward") / len(poses)
            result.leaning_backward_ratio = sum(1 for m in poses if m.body_pose.lean_type == "backward") / len(poses)
            result.clapping_ratio = sum(1 for m in poses if m.body_pose.is_clapping) / len(poses)

        # 綜合分數 = FACS(70%) + Pose(30%)
        facs_score = (result.duchenne_ratio * 0.5 + result.avg_smile_intensity * 0.3 +
                      result.avg_smile_sincerity * 0.2)
        pose_score = (result.shaking_ratio * 0.4 + result.leaning_backward_ratio * 0.3 +
                      result.clapping_ratio * 0.3)
        result.composite_score = facs_score * 0.7 + pose_score * 0.3

        return result

    # ── 幾何工具 ─────────────────────────────────────

    @staticmethod
    def _dist(pts: dict, idx1: int, idx2: int) -> float:
        """兩點歐式距離"""
        p1 = pts.get(idx1, (0, 0, 0))
        p2 = pts.get(idx2, (0, 0, 0))
        return float(np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2))

    @staticmethod
    def _vertical_displacement(pts: dict, idx1: int, idx2: int) -> float:
        """兩點垂直位移"""
        p1 = pts.get(idx1, (0, 0, 0))
        p2 = pts.get(idx2, (0, 0, 0))
        return abs(p1[1] - p2[1])
