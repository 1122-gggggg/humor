"""
文本後處理模組 — 標點復原與說話者分離

功能：
1. 標點復原 (Punctuation Restoration)
   - ASR 輸出通常缺乏標點符號
   - 使用 CT-Transformer 或 LLM 自動修復標點與語氣停頓
   - 對脫口秀的節奏感（timing）至關重要

2. 說話者分離 (Speaker Diarization)
   - 使用 Pyannote-audio 區分多位講者
   - 適用於訪談型脫口秀（如 賀瓏 vs 藍恩）
   - 避免機器人學到混亂的對話邏輯

3. 文本清理
   - 去除 ASR 幻覺（hallucination）
   - 合併斷裂的句子
   - 正規化中英混合文本
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class DiarizedSegment:
    """帶說話者標籤的逐字稿片段"""
    start: float
    end: float
    text: str
    speaker: str              # 說話者標籤 (SPEAKER_00, SPEAKER_01, ...)
    speaker_name: str = ""    # 說話者名稱（人工指定）
    words: list[dict] = field(default_factory=list)


@dataclass
class ProcessedTranscript:
    """完整處理後的逐字稿"""
    video_id: str
    segments: list[DiarizedSegment]
    num_speakers: int
    speaker_map: dict[str, str]  # {SPEAKER_00: "藍恩", SPEAKER_01: "賀瓏"}
    ground_truth_events: list[dict] = field(default_factory=list)  # (Laughter) 等官方標記
    processing_info: dict = field(default_factory=dict)


class TextProcessor:
    """文本後處理器"""

    def __init__(
        self,
        enable_diarization: bool = True,
        enable_punctuation: bool = True,
        diarization_model: str = "pyannote/speaker-diarization-3.1",
        punctuation_method: str = "llm",  # "llm" | "ct_transformer" | "rule"
        llm_backend: str = "openai",
        llm_model: str = "gpt-4o",
        hf_token: str = "",
    ):
        """
        Args:
            enable_diarization: 是否啟用說話者分離
            enable_punctuation: 是否啟用標點復原
            diarization_model: Pyannote 模型名稱
            punctuation_method: 標點復原方法
            llm_backend: LLM API 後端
            llm_model: LLM 模型名稱
            hf_token: HuggingFace token（Pyannote 需要）
        """
        self.enable_diarization = enable_diarization
        self.enable_punctuation = enable_punctuation
        self.diarization_model = diarization_model
        self.punctuation_method = punctuation_method
        self.llm_backend = llm_backend
        self.llm_model = llm_model
        self.hf_token = hf_token
        self._diarization_pipeline = None

    # ── 說話者分離 (Speaker Diarization) ────────────────────

    def _init_diarization(self):
        """初始化 Pyannote 說話者分離管線"""
        if self._diarization_pipeline is not None:
            return

        try:
            from pyannote.audio import Pipeline

            logger.info(f"載入說話者分離模型: {self.diarization_model}")
            self._diarization_pipeline = Pipeline.from_pretrained(
                self.diarization_model,
                use_auth_token=self.hf_token,
            )

            # 如果有 GPU，使用 GPU
            import torch
            if torch.cuda.is_available():
                self._diarization_pipeline.to(torch.device("cuda"))
                logger.info("說話者分離模型已移至 GPU")

        except ImportError:
            logger.warning(
                "pyannote.audio 未安裝。說話者分離功能不可用。\n"
                "安裝: pip install pyannote.audio"
            )
            self.enable_diarization = False
        except Exception as e:
            logger.warning(f"說話者分離模型載入失敗: {e}")
            self.enable_diarization = False

    def diarize(
        self,
        audio_path: str | Path,
        num_speakers: int | None = None,
        min_speakers: int = 1,
        max_speakers: int = 5,
    ) -> list[dict]:
        """
        對音訊進行說話者分離

        Args:
            audio_path: 音訊檔案路徑
            num_speakers: 已知的說話者數量（可選）
            min_speakers: 最少說話者
            max_speakers: 最多說話者

        Returns:
            [{"start": float, "end": float, "speaker": str}, ...]
        """
        self._init_diarization()

        if self._diarization_pipeline is None:
            logger.warning("說話者分離不可用，所有片段標記為同一說話者")
            return []

        logger.info(f"🔊 執行說話者分離: {audio_path}")

        diarization_params = {}
        if num_speakers is not None:
            diarization_params["num_speakers"] = num_speakers
        else:
            diarization_params["min_speakers"] = min_speakers
            diarization_params["max_speakers"] = max_speakers

        diarization = self._diarization_pipeline(
            str(audio_path),
            **diarization_params,
        )

        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker,
            })

        # 統計
        speakers = set(s["speaker"] for s in segments)
        logger.info(f"✅ 偵測到 {len(speakers)} 位說話者, {len(segments)} 個語段")

        return segments

    def assign_speakers(
        self,
        transcript_segments: list[dict],
        diarization_segments: list[dict],
    ) -> list[dict]:
        """
        將說話者標籤指派給逐字稿片段

        使用時間重疊度來配對逐字稿與 diarization 結果

        Args:
            transcript_segments: 逐字稿片段
            diarization_segments: 說話者分離片段

        Returns:
            帶 speaker 欄位的逐字稿片段
        """
        if not diarization_segments:
            for seg in transcript_segments:
                seg["speaker"] = "SPEAKER_00"
            return transcript_segments

        for seg in transcript_segments:
            seg_start = seg.get("start", 0)
            seg_end = seg.get("end", 0)

            # 找重疊最多的 diarization segment
            best_speaker = "SPEAKER_00"
            best_overlap = 0

            for diar_seg in diarization_segments:
                overlap_start = max(seg_start, diar_seg["start"])
                overlap_end = min(seg_end, diar_seg["end"])
                overlap = max(0, overlap_end - overlap_start)

                if overlap > best_overlap:
                    best_overlap = overlap
                    best_speaker = diar_seg["speaker"]

            seg["speaker"] = best_speaker

        return transcript_segments

    # ── 標點復原 (Punctuation Restoration) ──────────────────

    def restore_punctuation(self, text: str) -> str:
        """
        復原文本的標點符號

        Args:
            text: 無標點的文本

        Returns:
            加上標點的文本
        """
        if not text.strip():
            return text

        if self.punctuation_method == "llm":
            return self._restore_punctuation_llm(text)
        elif self.punctuation_method == "ct_transformer":
            return self._restore_punctuation_ct(text)
        elif self.punctuation_method == "rule":
            return self._restore_punctuation_rules(text)
        else:
            return text

    def _restore_punctuation_llm(self, text: str) -> str:
        """使用 LLM 復原標點"""
        prompt = f"""請為以下脫口秀逐字稿加上適當的標點符號。

規則：
1. 保持原始用字不變，只加標點
2. 使用繁體中文標點（，。！？、：「」）
3. 注意脫口秀的節奏感，語氣停頓處用逗號
4. 反問句用問號，強調語氣用驚嘆號
5. 引用/模仿他人說話時用「」括起來

原文：
{text}

加標點後的文本："""

        try:
            result = self._call_llm(prompt)
            # 驗證：確保沒有大幅改變原文
            if self._text_similarity(text, result) > 0.7:
                return result
            else:
                logger.warning("LLM 標點復原結果偏差過大，使用規則復原")
                return self._restore_punctuation_rules(text)
        except Exception as e:
            logger.warning(f"LLM 標點復原失敗: {e}")
            return self._restore_punctuation_rules(text)

    def _restore_punctuation_ct(self, text: str) -> str:
        """使用 CT-Transformer 復原標點（輕量方案）"""
        try:
            from modelscope.pipelines import pipeline as ms_pipeline

            punc_pipeline = ms_pipeline(
                task="punctuation",
                model="damo/punc_ct-transformer_cn-en-common-vocab471067-large",
            )
            result = punc_pipeline(text_in=text)
            return result.get("text", text)

        except ImportError:
            logger.warning("modelscope 未安裝，fallback 到規則復原")
            return self._restore_punctuation_rules(text)
        except Exception as e:
            logger.warning(f"CT-Transformer 標點復原失敗: {e}")
            return self._restore_punctuation_rules(text)

    def _restore_punctuation_rules(self, text: str) -> str:
        """
        基於規則的標點復原（最穩定的 fallback）

        適用於中文+英文夾雜的脫口秀文本
        """
        # 句尾加句號
        if text and text[-1] not in "。！？，、；：":
            text += "。"

        # 常見的語氣詞後加逗號
        filler_words = [
            "就是", "然後", "所以", "對啊", "真的", "其實",
            "可是", "但是", "不過", "就是說", "你知道嗎",
            "對不對", "是不是", "就", "那個",
        ]
        for word in filler_words:
            text = re.sub(
                rf'({word})(?!\s*[，。！？])',
                r'\1，',
                text,
            )

        # 問句結尾
        question_patterns = [
            r'(嗎|呢|吧|啊|嘛)(?=[^，。！？])',
            r'(什麼|怎麼|為什麼|哪裡|誰|幾|多少).*?(?=[^？]$)',
        ]
        for pattern in question_patterns:
            text = re.sub(pattern, r'\g<0>？', text, count=1)

        # 清理多餘標點
        text = re.sub(r'[，。！？]{2,}', lambda m: m.group(0)[0], text)

        return text

    # ── Ground Truth 提取 ─────────────────────────────────────

    def extract_ground_truth_events(self, segments: list[dict]) -> list[dict]:
        """
        從逐字稿中提取官方標註的觀眾反應（如 TED 的 (Laughter)）作為 Ground Truth
        
        這對 AI 來說是完美的監督信號，因為這是人類聽寫出來的絕對好笑點。
        """
        gt_events = []
        # 涵蓋中英文的笑聲與掌聲標記 (忽略大小寫的正規表達式會在迴圈套用)
        gt_patterns = [
            r'\[laughter\]', r'\(laughter\)', r'\[laugh\]', r'\(laughs\)',
            r'\[applause\]', r'\(applause\)',
            r'\[笑聲\]', r'\(笑聲\)', r'（笑）', r'\[掌聲\]', r'\(掌聲\)', r'（掌聲）'
        ]
        
        for seg in segments:
            text = seg.get("text", "").lower()
            for pattern in gt_patterns:
                if re.search(pattern, text):
                    duration = seg.get("end", 0) - seg.get("start", 0)
                    gt_events.append({
                        "start": seg.get("start", 0),
                        "end": seg.get("end", 0),
                        "duration": max(1.0, duration), # 基本設為1秒
                        "confidence": 1.0,  # 來自官方逐字稿，信心度為 100% 黃金標準
                        "event_class": "GroundTruth_Laughter" if "laugh" in pattern or "笑" in pattern else "GroundTruth_Applause",
                        "is_ground_truth": True
                    })
                    break # 一句話通常只需擷取一種最顯著的標記
                    
        if gt_events:
            logger.info(f"🎯 成功從逐字稿中提取了 {len(gt_events)} 筆官方 (Laughter)/掌聲 Ground Truth 標記！")
            
        return gt_events

    # ── 文本清理 ────────────────────────────────────────────

    def clean_transcript(self, segments: list[dict]) -> list[dict]:
        """
        清理逐字稿

        1. 去除 ASR 幻覺
        2. 合併斷裂的句子
        3. 正規化中英混合文本
        """
        cleaned = []

        for seg in segments:
            text = seg.get("text", "")

            # 去除常見 ASR 幻覺
            text = self._remove_hallucinations(text)

            # 正規化
            text = self._normalize_text(text)

            if text.strip():
                seg = {**seg, "text": text}
                cleaned.append(seg)

        # 合併過短的連續片段
        merged = self._merge_short_segments(cleaned)

        return merged

    def _remove_hallucinations(self, text: str) -> str:
        """去除 Whisper 常見幻覺，以及已經提取為 Ground Truth 的標記"""
        # 由於 (Laughter) 等標記已經透過 extract_ground_truth_events 取出，
        # 在送給 JokeWriter 訓練前，我們應該把這些雜訊文本清掉，避免影響語意。
        hallucination_patterns = [
            r'((.{2,10})\2{3,})',        # 短語重複 3 次以上
            r'(感謝.*?觀看)',             # YouTube 結尾語
            r'(請訂閱.*?頻道)',           # YouTube 結尾語
            r'^\[音樂\]$',               # 音樂標記
            r'^\[掌聲\]$',               # 掌聲標記
            r'^\(掌聲\)$',               
            r'^\[笑聲\]$',               # 笑聲標記
            r'^\(Laughter\)$',           # TED 笑聲標記 (區分大小寫)
            r'^\(laughter\)$',           
            r'字幕.*?社區',              # YouTube 字幕社區標記
            # 清除非首尾但夾雜在句子中的 (Laughter)
            r'\s*\[(laughter|laugh|applause|笑聲|掌聲)\]\s*',
            r'\s*\((laughter|laugh|applause|笑聲|掌聲)\)\s*',
            r'\s*（(笑|掌聲)）\s*'
        ]
        for pattern in hallucination_patterns:
            text = re.sub(pattern, ' ', text, flags=re.IGNORECASE)

        return text.strip()

    def _normalize_text(self, text: str) -> str:
        """正規化文本"""
        # 全形英文 → 半形
        text = text.translate(str.maketrans(
            'ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ'
            'ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ'
            '０１２３４５６７８９',
            'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            'abcdefghijklmnopqrstuvwxyz'
            '0123456789',
        ))

        # 多餘空白
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def _merge_short_segments(
        self,
        segments: list[dict],
        min_length: int = 3,
        max_gap: float = 1.0,
    ) -> list[dict]:
        """合併過短的連續片段"""
        if not segments:
            return []

        merged = [segments[0].copy()]

        for seg in segments[1:]:
            prev = merged[-1]
            gap = seg.get("start", 0) - prev.get("end", 0)
            same_speaker = seg.get("speaker", "") == prev.get("speaker", "")

            # 條件：同一說話者 + 間距小 + 前段太短
            if same_speaker and gap < max_gap and len(prev.get("text", "")) < min_length:
                merged[-1]["text"] = prev["text"] + seg["text"]
                merged[-1]["end"] = seg.get("end", prev.get("end", 0))
                # 合併 words
                prev_words = prev.get("words", [])
                seg_words = seg.get("words", [])
                merged[-1]["words"] = prev_words + seg_words
            else:
                merged.append(seg.copy())

        return merged

    # ── 完整處理流程 ────────────────────────────────────────

    def process(
        self,
        audio_path: str | Path,
        transcript_segments: list[dict],
        video_id: str = "",
        num_speakers: int | None = None,
        speaker_names: dict[str, str] | None = None,
    ) -> ProcessedTranscript:
        """
        完整文本後處理流程

        1. 說話者分離（如果啟用）
        2. 標點復原（如果啟用）
        3. 文本清理

        Args:
            audio_path: 音訊路徑
            transcript_segments: 原始逐字稿片段
            video_id: 影片 ID
            num_speakers: 已知的說話者數量
            speaker_names: 說話者名稱映射 {SPEAKER_00: "藍恩"}

        Returns:
            ProcessedTranscript 完整處理結果
        """
        processing_info = {}

        # 0. 提取官方 Ground Truth 標記 (必須在清理幻覺之前執行)
        gt_events = self.extract_ground_truth_events(transcript_segments)
        processing_info["ground_truth_count"] = len(gt_events)

        # 1. 說話者分離
        if self.enable_diarization:
            logger.info("📢 執行說話者分離...")
            diar_segments = self.diarize(
                audio_path,
                num_speakers=num_speakers,
            )
            transcript_segments = self.assign_speakers(
                transcript_segments, diar_segments
            )
            processing_info["diarization"] = True
            processing_info["num_diar_segments"] = len(diar_segments)
        else:
            for seg in transcript_segments:
                seg.setdefault("speaker", "SPEAKER_00")
            processing_info["diarization"] = False

        # 2. 文本清理
        logger.info("🧹 清理文本...")
        transcript_segments = self.clean_transcript(transcript_segments)
        processing_info["cleaned_segments"] = len(transcript_segments)

        # 3. 標點復原
        if self.enable_punctuation:
            logger.info("✏️ 復原標點...")
            for seg in transcript_segments:
                seg["text"] = self.restore_punctuation(seg["text"])
            processing_info["punctuation"] = self.punctuation_method

        # 建立說話者映射
        speakers = sorted(set(s.get("speaker", "SPEAKER_00") for s in transcript_segments))
        speaker_map = speaker_names or {s: s for s in speakers}

        # 轉換為 DiarizedSegment
        diarized = [
            DiarizedSegment(
                start=s.get("start", 0),
                end=s.get("end", 0),
                text=s.get("text", ""),
                speaker=s.get("speaker", "SPEAKER_00"),
                speaker_name=speaker_map.get(s.get("speaker", ""), ""),
                words=s.get("words", []),
            )
            for s in transcript_segments
        ]

        return ProcessedTranscript(
            video_id=video_id,
            segments=diarized,
            num_speakers=len(speakers),
            speaker_map=speaker_map,
            ground_truth_events=gt_events,
            processing_info=processing_info,
        )

    # ── 工具方法 ────────────────────────────────────────────

    def _text_similarity(self, text1: str, text2: str) -> float:
        """簡單的文本相似度（字元級 Jaccard）"""
        chars1 = set(re.sub(r'[^\w]', '', text1))
        chars2 = set(re.sub(r'[^\w]', '', text2))
        if not chars1 and not chars2:
            return 1.0
        intersection = chars1 & chars2
        union = chars1 | chars2
        return len(intersection) / max(len(union), 1)

    def _call_llm(self, prompt: str) -> str:
        """呼叫 LLM API"""
        if self.llm_backend == "openai":
            from openai import OpenAI
            client = OpenAI()
            response = client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=2000,
            )
            return response.choices[0].message.content.strip()
        else:
            raise ValueError(f"未支援的 LLM 後端: {self.llm_backend}")

    def save_processed(
        self,
        result: ProcessedTranscript,
        output_path: str | Path,
    ) -> Path:
        """儲存處理結果"""
        from dataclasses import asdict
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = asdict(result)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"處理結果已儲存: {output_path}")
        return output_path
