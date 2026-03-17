"""
自動標註管線 — 弱監督學習 (Weak Supervision) 結合多模態特徵

功能：
完整的自動標註流水線，結合音訊與視覺特徵，
產生候選笑點集合供專家微調。

流程：
1. 音畫分離：MoviePy / ffmpeg 分離音軌與影像
2. 音訊標註：YAMNet 笑聲機率曲線 + 分貝分析
3. 文本辨識：Whisper Large-v3 帶時間戳字幕
4. 影像分析：觀眾表情/動作偵測（輔助權重 0.2-0.3）
5. 初步打分：Humor Score (0.0 - 1.0)
6. 輸出候選集：評分 > 閾值的片段 → CSV/JSON

權重公式：
    HumorScore = α × audio_score + β × video_score

    其中:
    - audio_score = w1 × laughter_prob + w2 × norm_dB + w3 × norm_duration
    - video_score = positive_emotion_ratio
    - α = 0.75（音訊主權重）
    - β = 0.25（視覺輔助權重）
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class HumorCandidate:
    """自動標註的幽默候選片段"""
    id: str
    video_id: str

    # 文本
    setup_text: str
    punchline_text: str
    full_text: str

    # 時間
    setup_start: float
    setup_end: float
    punchline_start: float
    punchline_end: float
    laughter_start: float

    # 自動標註分數
    humor_score: float              # 最終綜合分數 (0-1)
    audio_score: float              # 音訊分數
    video_score: float              # 視覺分數
    laughter_confidence: float      # YAMNet 笑聲信心
    laughter_db: float              # 笑聲強度分貝
    laughter_duration: float        # 笑聲持續時間

    # 視覺特徵
    audience_happy_ratio: float = 0.0
    audience_surprise_ratio: float = 0.0
    audience_positive_ratio: float = 0.0

    # 幽默底層理論數值特徵
    incongruity_score: float = 0.0      # 1 - cos(v_setup, v_punch)
    violation_score: float = 0.0        # 內容危險/倒霉程度 (Text Negative Valence)
    safety_score: float = 0.0           # 語氣/表情安全/輕鬆程度
    bvt_score: float = 0.0              # Violation x Safety

    # 分類標籤
    auto_label: str = ""            # 自動標籤 (funny / neutral / not_funny)
    humor_technique: str = ""       # 幽默技巧分析（BVT、PIJ-Q、錯配等）
    tags: list[str] = field(default_factory=list)


@dataclass
class AnnotationStats:
    """標註統計"""
    total_candidates: int = 0
    funny_count: int = 0          # humor_score >= 0.7
    neutral_count: int = 0        # 0.4 <= humor_score < 0.7
    not_funny_count: int = 0      # humor_score < 0.4
    expert_reviewed_count: int = 0
    avg_humor_score: float = 0.0
    avg_audio_score: float = 0.0
    avg_video_score: float = 0.0


class AutoAnnotationPipeline:
    """自動標註管線"""

    # 綜合評分權重
    AUDIO_WEIGHT = 0.75   # α: 音訊主權重
    VIDEO_WEIGHT = 0.25   # β: 視覺輔助權重

    # 音訊子權重
    LAUGHTER_PROB_WEIGHT = 0.4     # w1: YAMNet 笑聲機率
    LAUGHTER_DB_WEIGHT = 0.35      # w2: 笑聲分貝
    LAUGHTER_DUR_WEIGHT = 0.25     # w3: 笑聲持續時間

    # 自動標籤閾值
    FUNNY_THRESHOLD = 0.7
    NEUTRAL_THRESHOLD = 0.4

    def __init__(
        self,
        audio_weight: float = 0.75,
        video_weight: float = 0.25,
        funny_threshold: float = 0.7,
        enable_video: bool = True,
        enable_technique_analysis: bool = True,
    ):
        """
        Args:
            audio_weight: 音訊分數權重 (α)
            video_weight: 視覺分數權重 (β)
            funny_threshold: "好笑" 閾值
            enable_video: 是否啟用視覺分析
            enable_technique_analysis: 是否開啟幽默技巧特徵分析
        """
        self.audio_weight = audio_weight
        self.video_weight = video_weight if enable_video else 0.0
        self.funny_threshold = funny_threshold
        self.enable_video = enable_video
        self.enable_technique_analysis = enable_technique_analysis

        # 初始化 NLP 模型用於計算語義漂移與 Valence
        self.encoder = None
        self.sentiment_analyzer = None
        if enable_technique_analysis:
            try:
                from sentence_transformers import SentenceTransformer
                # 計算 Incongruity，1 - cos(v_setup, v_punch)
                self.encoder = SentenceTransformer("BAAI/bge-m3")
                from transformers import pipeline
                # SOTA 2024: 零樣本跨語言 NLI 分類，精準捕捉 Violation (冒犯/禁忌) 與 Benign (安全/無害)
                self.bvt_analyzer = pipeline(
                    "zero-shot-classification", 
                    model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli", 
                )
                logger.info("已成功載入語義漂移與 BVT 情緒評估模型！")
            except ImportError:
                logger.warning("未安裝 sentence-transformers 或 transformers，將跳過數學層面的 BVT 特徵量化。")

        # 如果不啟用視覺，音訊權重要調整到 1.0
        if not enable_video:
            self.audio_weight = 1.0

    def run(
        self,
        video_id: str,
        aligned_jokes: list[dict],
        laughter_events: list[dict],
        audio_features: list[dict],
        video_reactions: list[dict] | None = None,
    ) -> list[HumorCandidate]:
        """
        執行自動標註管線

        Args:
            video_id: 影片 ID
            aligned_jokes: SetupPunchlineAligner 的輸出
            laughter_events: LaughterDetector 的輸出
            audio_features: AudioAnalyzer 的輸出
            video_reactions: VideoAnalyzer 的輸出（如有影片）

        Returns:
            HumorCandidate 列表
        """
        logger.info(f"🏷️ 自動標註管線: {video_id} ({len(aligned_jokes)} 個段子)")

        candidates = []

        for i, joke in enumerate(aligned_jokes):
            # 將 dataclass 轉為 dict（相容 SetupPunchline 與純 dict 兩種輸入）
            if not isinstance(joke, dict):
                joke = asdict(joke)
            # 1. 計算音訊分數
            audio_score = self._compute_audio_score(joke, laughter_events, audio_features)

            # 2. 計算視覺分數（如果有影片資料）
            video_score = 0.0
            audience_metrics = {}
            if video_reactions and self.enable_video:
                video_score, audience_metrics = self._compute_video_score(
                    joke, video_reactions
                )

            # 3. 綜合分數
            humor_score = (
                self.audio_weight * audio_score
                + self.video_weight * video_score
            )
            humor_score = float(np.clip(humor_score, 0, 1))

            # 4. 自動標籤
            if humor_score >= self.FUNNY_THRESHOLD:
                auto_label = "funny"
            elif humor_score >= self.NEUTRAL_THRESHOLD:
                auto_label = "neutral"
            else:
                auto_label = "not_funny"
                
            # 5. 分析幽默技巧與數學底層邏輯（BVT / Semantic Drift）
            technique = ""
            incongruity_score = 0.0
            violation_score = 0.0
            safety_score = 0.0
            bvt_score = 0.0

            if self.enable_technique_analysis:
                # A. 不協調度量 (Semantic Drift): 1 - cos(v_setup, v_punch)
                if self.encoder and joke.get("setup_text") and joke.get("punchline_text"):
                    v_s = self.encoder.encode(joke["setup_text"])
                    v_p = self.encoder.encode(joke["punchline_text"])
                    cos_sim = np.dot(v_s, v_p) / (np.linalg.norm(v_s) * np.linalg.norm(v_p) + 1e-10)
                    incongruity_score = float(1.0 - cos_sim)

                # B. 安全-危險平衡 (Benign Violation Balance) 與 悲劇內核 (Tragedy)
                # Violation: 取決於文字內容的「危險/禁忌/冒犯」程度
                # Tragedy: 從根源上判斷是否建立在痛苦或剝奪之上
                if getattr(self, "bvt_analyzer", None) and joke.get("full_text"):
                    try:
                        candidate_labels = [
                            "安全無害 (Safe/Benign)", 
                            "冒犯禁忌 (Violation/Threat)",
                            "悲劇痛苦 (Tragedy/Misery)"
                        ]
                        res = self.bvt_analyzer(joke["full_text"][:512], candidate_labels)
                        scores_dict = dict(zip(res["labels"], res["scores"]))
                        violation_score = scores_dict.get("冒犯禁忌 (Violation/Threat)", 0.0)
                        tragedy_score = scores_dict.get("悲劇痛苦 (Tragedy/Misery)", 0.0)
                        # 將文字安全度也融入 Safety 中
                        txt_safety = scores_dict.get("安全無害 (Safe/Benign)", 0.0)
                    except Exception:
                        txt_safety = 0.5
                        tragedy_score = 0.0
                else:
                    tragedy_score = 0.0
                    txt_safety = 0.5
                
                # Safety: 取決於演員的輕鬆語氣、自嘲態度（結合視覺/聽覺正面特徵，再加上文本 Zero-shot 特徵）
                safety_score = max(0.3, video_score)
                if audience_metrics.get("positive_ratio"):
                    safety_score = (safety_score + float(audience_metrics["positive_ratio"])) / 2.0
                # 將新導入的 mDeBERTa 文字安全分數加權混入
                safety_score = (safety_score + txt_safety) / 2.0
                
                # Humor(t) = Violation(t) x Safety(t)
                bvt_score = violation_score * safety_score

                # 利用 LLM 整合成最終一句話的分析
                if auto_label == "funny":
                    technique = self._analyze_humor_technique(
                        setup=joke.get("setup_text", ""),
                        punchline=joke.get("punchline_text", ""),
                        incongruity=incongruity_score,
                        bvt=bvt_score,
                        tragedy=tragedy_score,
                        video_desc=f"{video_score:.2f} (大於0.5表示演員表情誇張/生動)",
                        audio_desc=f"{audio_score:.2f} (大於0.5表示聲音情緒高昂/起伏大)"
                    )

            candidate = HumorCandidate(
                id=joke.get("id", f"{video_id}_{i:04d}"),
                video_id=video_id,
                setup_text=joke.get("setup_text", ""),
                punchline_text=joke.get("punchline_text", ""),
                full_text=joke.get("full_text", ""),
                setup_start=joke.get("setup_start", 0),
                setup_end=joke.get("setup_end", 0),
                punchline_start=joke.get("punchline_start", 0),
                punchline_end=joke.get("punchline_end", 0),
                laughter_start=joke.get("laughter_start", 0),
                humor_score=humor_score,
                audio_score=audio_score,
                video_score=video_score,
                laughter_confidence=joke.get("laughter_confidence", 0),
                laughter_db=joke.get("laughter_db", 0),
                laughter_duration=joke.get("laughter_duration", 0),
                audience_happy_ratio=audience_metrics.get("happy_ratio", 0),
                audience_surprise_ratio=audience_metrics.get("surprise_ratio", 0),
                audience_positive_ratio=audience_metrics.get("positive_ratio", 0),
                incongruity_score=incongruity_score,
                violation_score=violation_score,
                safety_score=safety_score,
                bvt_score=bvt_score,
                auto_label=auto_label,
                humor_technique=technique,
            )

            candidates.append(candidate)

        # 按 humor_score 排序
        candidates.sort(key=lambda c: c.humor_score, reverse=True)

        stats = self._compute_stats(candidates)
        logger.info(
            f"✅ 自動標註完成: "
            f"😂 funny={stats.funny_count} | "
            f"😐 neutral={stats.neutral_count} | "
            f"😑 not_funny={stats.not_funny_count} | "
            f"平均分={stats.avg_humor_score:.3f}"
        )

        return candidates

    def _compute_audio_score(
        self,
        joke: dict,
        laughter_events: list[dict],
        audio_features: list[dict],
    ) -> float:
        """
        計算音訊分數

        audio_score = w1 × laughter_prob + w2 × norm_dB + w3 × norm_duration
        """
        # 笑聲信心分數
        laughter_prob = joke.get("laughter_confidence", 0)

        # 分貝正規化（-60dB ~ 0dB → 0 ~ 1）
        laughter_db = joke.get("laughter_db", -60)
        norm_db = float(np.clip((laughter_db + 60) / 60, 0, 1))

        # 持續時間正規化（0 ~ 15秒 → 0 ~ 1）
        duration = joke.get("laughter_duration", 0)
        norm_duration = float(np.clip(duration / 15.0, 0, 1))

        audio_score = (
            self.LAUGHTER_PROB_WEIGHT * laughter_prob
            + self.LAUGHTER_DB_WEIGHT * norm_db
            + self.LAUGHTER_DUR_WEIGHT * norm_duration
        )

        return float(np.clip(audio_score, 0, 1))

    def _analyze_humor_technique(
        self, 
        setup: str, 
        punchline: str, 
        incongruity: float = 0.0, 
        bvt: float = 0.0,
        tragedy: float = 0.0,
        video_desc: str = "",
        audio_desc: str = ""
    ) -> str:
        """
        結合安全-危險理論 (BVT)、悲劇核心與特定濾鏡，分析脫口秀的幽默技巧。
        讓 LLM 作為具備多模態推論能力的分析師，同時給定音訊與視覺的量化特徵進行判斷。
        """
        if not setup.strip() and not punchline.strip():
            return ""
            
        prompt = f"""請以一句話簡潔分析以下脫口秀笑話使用了哪些幽默邏輯與技巧。
請參考以下教材理論：
1. 喜劇源於悲劇 (Comedy = Tragedy + Time)
2. 安全-危險理論 (Benign Violation Theory)
3. Setup-Punchline (建構安全認知 -> 拋出危險/不一致 -> Why Problem)

目前測得多模態與數學特徵：
- 語義漂移 (Incongruity Score): {incongruity:.2f} (大於0.5表示文字邏輯打破預期)
- 幽默指數 (BVT Score): {bvt:.2f} (Violation x Safety)
- 悲劇指數 (Tragedy Base): {tragedy:.2f} (衡量 Setup 建立的痛苦或困境程度)
- 視覺情緒特徵: {video_desc}
- 聽覺情緒特徵: {audio_desc}

Setup: {setup}
Punchline: {punchline}

技術分析（50字以內）："""

        try:
            from openai import OpenAI
            client = OpenAI()
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=100,
            )
            return response.choices[0].message.content.strip()
        except ImportError:
            return "分析失敗（缺少 openai 套件）"
        except Exception:
            # 為了避免 API 錯誤中斷整個 pipeline
            return "API 分析錯誤"

    def _compute_video_score(
        self,
        joke: dict,
        video_reactions: list[dict],
    ) -> tuple[float, dict]:
        """
        計算視覺分數

        取笑聲時間點前後 2 秒內的觀眾反應
        """
        laughter_time = joke.get("laughter_start", 0)
        window = 2.0

        # 找笑聲時間附近的觀眾反應
        relevant = [
            r for r in video_reactions
            if abs(r.get("timestamp", 0) - laughter_time) <= window
        ]

        if not relevant:
            return 0.0, {}

        avg_positive = np.mean([r.get("positive_ratio", 0) for r in relevant])
        avg_happy = np.mean([r.get("happy_ratio", 0) for r in relevant])
        avg_surprise = np.mean([r.get("surprise_ratio", 0) for r in relevant])

        metrics = {
            "positive_ratio": float(avg_positive),
            "happy_ratio": float(avg_happy),
            "surprise_ratio": float(avg_surprise),
        }

        return float(avg_positive), metrics

    def _compute_stats(self, candidates: list[HumorCandidate]) -> AnnotationStats:
        """計算標註統計"""
        if not candidates:
            return AnnotationStats()

        return AnnotationStats(
            total_candidates=len(candidates),
            funny_count=sum(1 for c in candidates if c.auto_label == "funny"),
            neutral_count=sum(1 for c in candidates if c.auto_label == "neutral"),
            not_funny_count=sum(1 for c in candidates if c.auto_label == "not_funny"),
            expert_reviewed_count=0,  # 由 CLI 工具另行統計，HumorCandidate 本身不含此欄位
            avg_humor_score=float(np.mean([c.humor_score for c in candidates])),
            avg_audio_score=float(np.mean([c.audio_score for c in candidates])),
            avg_video_score=float(np.mean([c.video_score for c in candidates])),
        )

    # ── 輸出格式 ────────────────────────────────────────────

    def save_candidates_json(
        self,
        candidates: list[HumorCandidate],
        output_path: str | Path,
        min_score: float = 0.0,
    ) -> Path:
        """儲存候選集為 JSON"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        filtered = [c for c in candidates if c.humor_score >= min_score]
        data = [asdict(c) for c in filtered]

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"候選集 JSON 已儲存: {output_path} ({len(filtered)} 筆)")
        return output_path

    def save_candidates_csv(
        self,
        candidates: list[HumorCandidate],
        output_path: str | Path,
        min_score: float = 0.0,
    ) -> Path:
        """
        儲存候選集為 CSV（方便專家瀏覽與標註）

        包含專家標註欄位供填寫
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        filtered = [c for c in candidates if c.humor_score >= min_score]

        fieldnames = [
            "id", "video_id", "humor_score", "auto_label", "humor_technique",
            "setup_text", "punchline_text",
            "incongruity_score", "violation_score", "safety_score", "bvt_score",
            "laughter_confidence", "laughter_db", "laughter_duration",
            "audio_score", "video_score",
            "audience_happy_ratio", "audience_positive_ratio",
        ]

        with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for c in filtered:
                row = asdict(c)
                writer.writerow({k: row.get(k, "") for k in fieldnames})

        logger.info(f"候選集 CSV 已儲存: {output_path} ({len(filtered)} 筆)")
        return output_path

