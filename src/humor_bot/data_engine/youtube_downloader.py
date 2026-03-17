"""
YouTube 影片下載與音訊提取模組

功能：
- 從 YouTube URL 下載影片並分離音軌（WAV 16kHz mono）
- 字幕策略：人工字幕（zh-Hant）> YouTube 自動字幕 > Whisper ASR
- 使用 faster-whisper + VAD 預過濾產生逐字稿（含逐字時間戳記）
- 支援批次處理（讀取 URL 清單檔案）

字幕優先級：
1. 人工繁體中文字幕 (zh-Hant / zh-TW) — Ground Truth
2. 人工中文字幕 (zh)
3. 人工英文字幕 (en)
4. YouTube 自動生成字幕 (auto-captions)
5. Whisper Large-v3 ASR 轉錄（搭配 VAD 過濾）
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path

import yt_dlp
from faster_whisper import WhisperModel
from tqdm import tqdm  # 匯入進度條工具

logger = logging.getLogger(__name__)


class SubtitleSource(str, Enum):
    """字幕來源類型"""
    MANUAL_ZH_HANT = "manual_zh_hant"    # 人工繁體中文（最高優先）
    MANUAL_ZH = "manual_zh"              # 人工中文
    MANUAL_EN = "manual_en"              # 人工英文
    AUTO_CAPTION = "auto_caption"         # YouTube 自動字幕
    WHISPER_ASR = "whisper_asr"          # Whisper ASR 轉錄
    NONE = "none"                         # 無字幕


@dataclass
class TranscriptSegment:
    """逐字稿片段"""
    start: float          # 起始時間（秒）
    end: float            # 結束時間（秒）
    text: str             # 文字內容
    speaker: str = ""     # 說話者標籤（由 diarization 填入）
    words: list[dict] = field(default_factory=list)  # 逐字時間戳記


@dataclass
class DownloadResult:
    """下載結果"""
    video_id: str
    title: str
    audio_path: Path
    transcript_path: Path | None = None
    subtitle_path: Path | None = None
    subtitle_source: SubtitleSource = SubtitleSource.NONE
    duration: float = 0.0
    has_manual_subs: bool = False
    comments_path: Path | None = None
    speaker_persona: str = ""


class YouTubeDownloader:
    """YouTube 影片下載與音訊提取器"""

    # 字幕語言搜尋優先順序
    MANUAL_SUB_LANGS = ["zh-TW", "zh-Hant", "zh", "en"]
    AUTO_SUB_LANGS = ["zh-TW", "zh-Hant", "zh", "en"]

    def __init__(
        self,
        output_dir: str | Path = "data/raw",
        audio_sample_rate: int = 16000,
        whisper_model_size: str = "large-v3",
        whisper_language: str | None = None,
        whisper_beam_size: int = 5,
        prefer_manual_subs: bool = True,
        force_whisper: bool = False,
    ):
        """
        Args:
            output_dir: 輸出根目錄
            audio_sample_rate: 音訊取樣率（YAMNet 需要 16kHz）
            whisper_model_size: Whisper 模型大小 (large-v3 推薦)
            whisper_language: 主要語言 (None 代表自動偵測多國語言)
            whisper_beam_size: Beam search 寬度
            prefer_manual_subs: 是否優先使用人工字幕
            force_whisper: 是否強制使用 Whisper（忽略所有現有字幕）
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.audio_sample_rate = audio_sample_rate
        self.whisper_model_size = whisper_model_size
        self.whisper_language = whisper_language
        self.whisper_beam_size = whisper_beam_size
        self.prefer_manual_subs = prefer_manual_subs
        self.force_whisper = force_whisper
        self._whisper_model: WhisperModel | None = None

    @property
    def whisper_model(self) -> WhisperModel:
        """延遲載入 Whisper 模型"""
        if self._whisper_model is None:
            logger.info(f"載入 Whisper 模型: {self.whisper_model_size}")
            self._whisper_model = WhisperModel(
                self.whisper_model_size,
                device="cuda",
                compute_type="float16",
            )
        return self._whisper_model

    @staticmethod
    def _find_ffmpeg() -> str | None:
        """自動尋找 ffmpeg 路徑（優先用 imageio_ffmpeg 內建版本）"""
        try:
            import imageio_ffmpeg
            return str(Path(imageio_ffmpeg.get_ffmpeg_exe()).parent)
        except Exception:
            pass
        return None

    def _get_ydl_opts(self, video_id: str) -> dict:
        """產生 yt-dlp 下載選項"""
        audio_dir = self.output_dir / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)

        opts = {
            "format": "bestaudio/best",
            "outtmpl": str(audio_dir / f"{video_id}.%(ext)s"),
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "wav",
                },
            ],
            "postprocessor_args": [
                "-ar", str(self.audio_sample_rate),
                "-ac", "1",  # mono
            ],
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitleslangs": ["zh-TW", "zh-Hant", "zh", "en"],
            "subtitlesformat": "json3",
            "getcomments": True,
            "quiet": True,
            "no_warnings": True,
        }

        # 自動填入 ffmpeg 路徑
        ffmpeg_dir = self._find_ffmpeg()
        if ffmpeg_dir:
            opts["ffmpeg_location"] = ffmpeg_dir

        return opts

    def _extract_video_id(self, url: str) -> str:
        """從 URL 提取 video ID"""
        import re
        patterns = [
            r'(?:v=|/v/|youtu\.be/)([a-zA-Z0-9_-]{11})',
            r'(?:embed/)([a-zA-Z0-9_-]{11})',
            r'^([a-zA-Z0-9_-]{11})$',
        ]
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        raise ValueError(f"無法從 URL 提取 video ID: {url}")

    def _detect_subtitle_source(
        self,
        video_id: str,
        ydl_info: dict | None = None,
    ) -> tuple[Path | None, SubtitleSource]:
        """偵測最佳字幕來源"""
        subtitle_dir = self.output_dir / "audio"

        if ydl_info:
            manual_subs = ydl_info.get("subtitles", {})
            auto_subs = ydl_info.get("automatic_captions", {})
        else:
            manual_subs = {}
            auto_subs = {}

        source_mapping = {
            "zh-TW": SubtitleSource.MANUAL_ZH_HANT,
            "zh-Hant": SubtitleSource.MANUAL_ZH_HANT,
            "zh": SubtitleSource.MANUAL_ZH,
            "en": SubtitleSource.MANUAL_EN,
        }

        for lang in self.MANUAL_SUB_LANGS:
            if lang in manual_subs:
                candidate = subtitle_dir / f"{video_id}.{lang}.json3"
                if candidate.exists():
                    source = source_mapping.get(lang, SubtitleSource.MANUAL_ZH)
                    logger.info(f"✅ 找到人工字幕: {lang} ({source.value})")
                    return candidate, source

        for lang in self.AUTO_SUB_LANGS:
            if lang in auto_subs:
                candidate = subtitle_dir / f"{video_id}.{lang}.json3"
                if candidate.exists():
                    logger.info(f"⚠️ 使用自動字幕: {lang}")
                    return candidate, SubtitleSource.AUTO_CAPTION

        for lang in self.MANUAL_SUB_LANGS + self.AUTO_SUB_LANGS:
            candidate = subtitle_dir / f"{video_id}.{lang}.json3"
            if candidate.exists():
                logger.info(f"📁 找到字幕檔案: {candidate.name}")
                return candidate, SubtitleSource.AUTO_CAPTION

        return None, SubtitleSource.NONE

    def download_audio(self, url: str) -> DownloadResult:
        """下載 YouTube 影片的音軌"""
        video_id = self._extract_video_id(url)
        audio_dir = self.output_dir / "audio"
        audio_path = audio_dir / f"{video_id}.wav"

        if audio_path.exists():
            logger.info(f"音訊檔案已存在，跳過下載: {audio_path}")
            sub_path, sub_source = self._detect_subtitle_source(video_id)
            return DownloadResult(
                video_id=video_id,
                title="(cached)",
                audio_path=audio_path,
                subtitle_path=sub_path,
                subtitle_source=sub_source,
                has_manual_subs=sub_source in (
                    SubtitleSource.MANUAL_ZH_HANT,
                    SubtitleSource.MANUAL_ZH,
                    SubtitleSource.MANUAL_EN,
                ),
            )

        logger.info(f"下載 YouTube 影片音軌: {url}")
        opts = self._get_ydl_opts(video_id)

        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
            title = info.get("title", "unknown")
            duration = info.get("duration", 0.0)
            
            comments = info.get("comments", [])
            comments_path = None
            speaker_persona = ""
            if comments:
                comments_dir = self.output_dir / "comments"
                comments_dir.mkdir(parents=True, exist_ok=True)
                comments_path = comments_dir / f"{video_id}.json"
                text_comments = [c.get("text", "") for c in comments[:100]]
                with open(comments_path, "w", encoding="utf-8") as f:
                    json.dump(text_comments, f, ensure_ascii=False, indent=2)
                
                speaker_persona = self._analyze_persona(title, text_comments)

        sub_path, sub_source = self._detect_subtitle_source(video_id, info)

        result = DownloadResult(
            video_id=video_id,
            title=title,
            audio_path=audio_path,
            subtitle_path=sub_path,
            subtitle_source=sub_source,
            duration=duration,
            has_manual_subs=sub_source in (
                SubtitleSource.MANUAL_ZH_HANT,
                SubtitleSource.MANUAL_ZH,
                SubtitleSource.MANUAL_EN,
            ),
            comments_path=comments_path,
            speaker_persona=speaker_persona,
        )
        logger.info(
            f"下載完成: {title} ({duration:.0f}s) | "
            f"字幕: {sub_source.value} | 人設: {speaker_persona}"
        )
        return result

    def _analyze_persona(self, title: str, comments: list[str]) -> str:
        """分析單口喜劇人設"""
        if not comments:
            return "General (無特定人設)"
            
        comments_text = "\n- ".join(comments[:30])
        prompt = f"""請根據以下脫口秀影片標題與 YouTube 觀眾留言草稿，以『極簡的三個形容詞』總結這位脫口秀演員主要的舞台人設 (Persona)。
影片標題：{title}
觀眾留言：
- {comments_text}
總結人設（格式：形容詞1, 形容詞2, 形容詞3）："""

        try:
            from openai import OpenAI
            client = OpenAI()
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=30,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"人設分析失敗: {e}")
            return "General (無法判斷)"

    def transcribe(
        self,
        audio_path: str | Path,
        use_subtitle: Path | None = None,
        subtitle_source: SubtitleSource = SubtitleSource.NONE,
    ) -> list[TranscriptSegment]:
        """產生逐字稿"""
        if self.force_whisper:
            logger.info("強制 Whisper 模式：忽略所有現有字幕")
            return self._whisper_transcribe(audio_path)

        if use_subtitle and subtitle_source in (
            SubtitleSource.MANUAL_ZH_HANT,
            SubtitleSource.MANUAL_ZH,
            SubtitleSource.MANUAL_EN,
        ):
            logger.info(f"🏆 使用人工字幕（Ground Truth）: {use_subtitle}")
            return self._parse_json3_subtitle(use_subtitle)

        if use_subtitle and subtitle_source == SubtitleSource.AUTO_CAPTION:
            logger.info(f"⚠️ 使用自動字幕（品質可能不佳）: {use_subtitle}")
            segments = self._parse_json3_subtitle(use_subtitle)
            if self._is_subtitle_quality_poor(segments):
                logger.warning("自動字幕品質太差，改用 Whisper ASR")
                return self._whisper_transcribe(audio_path)
            return segments

        logger.info("無可用字幕，使用 Whisper ASR")
        return self._whisper_transcribe(audio_path)

    def _is_subtitle_quality_poor(self, segments: list[TranscriptSegment]) -> bool:
        """判斷字幕品質"""
        if len(segments) < 10:
            return True
        avg_len = sum(len(s.text) for s in segments) / max(len(segments), 1)
        if avg_len < 2:
            return True
        empty_ratio = sum(1 for s in segments if len(s.text.strip()) == 0) / max(len(segments), 1)
        if empty_ratio > 0.3:
            return True
        return False

    def _whisper_transcribe(self, audio_path: str | Path) -> list[TranscriptSegment]:
        """使用 Whisper 進行 ASR 轉錄並顯示進度"""
        logger.info(f"🎙️ Whisper ASR 轉錄: {audio_path}")
        logger.info(f"   模型: {self.whisper_model_size} | VAD 啟用 | 逐字時間戳記啟用")

        segments_gen, info = self.whisper_model.transcribe(
            str(audio_path),
            language=self.whisper_language,
            beam_size=self.whisper_beam_size,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=500,
                speech_pad_ms=300,
                min_speech_duration_ms=250,
            ),
        )

        segments = []
        # 加入 tqdm 進度條，根據音訊秒數更新
        with tqdm(total=round(info.duration, 2), unit="sec", desc="🎙️ Whisper 轉錄中") as pbar:
            for seg in segments_gen:
                words = []
                if seg.words:
                    words = [
                        {
                            "word": w.word,
                            "start": w.start,
                            "end": w.end,
                            "probability": w.probability,
                        }
                        for w in seg.words
                    ]
                segments.append(TranscriptSegment(
                    start=seg.start,
                    end=seg.end,
                    text=seg.text.strip(),
                    words=words,
                ))
                # 更新進度條至當前片段結束時間
                pbar.update(seg.end - pbar.n)

        logger.info(f"✅ Whisper 轉錄完成: {len(segments)} 個片段")
        return segments

    def _parse_json3_subtitle(self, subtitle_path: str | Path) -> list[TranscriptSegment]:
        """解析 YouTube json3 字幕格式"""
        with open(subtitle_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        segments = []
        events = data.get("events", [])

        for event in events:
            if "segs" not in event:
                continue

            start_ms = event.get("tStartMs", 0)
            duration_ms = event.get("dDurationMs", 0)
            start = start_ms / 1000.0
            end = (start_ms + duration_ms) / 1000.0

            text_parts = []
            words = []
            base_offset = start_ms

            for seg in event["segs"]:
                utf8 = seg.get("utf8", "")
                if utf8.strip():
                    text_parts.append(utf8)
                    seg_offset = seg.get("tOffsetMs", 0)
                    word_start = (base_offset + seg_offset) / 1000.0
                    words.append({
                        "word": utf8.strip(),
                        "start": word_start,
                        "end": word_start + 0.5,
                        "probability": 1.0,
                    })

            text = "".join(text_parts).strip()
            if text:
                segments.append(TranscriptSegment(
                    start=start,
                    end=end,
                    text=text,
                    words=words,
                ))
        return segments

    def save_transcript(
        self,
        segments: list[TranscriptSegment],
        output_path: str | Path,
        metadata: dict | None = None,
    ) -> Path:
        """儲存逐字稿為 JSON"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "metadata": metadata or {},
            "segments": [asdict(seg) for seg in segments],
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"逐字稿已儲存: {output_path}")
        return output_path

    def process_url(self, url: str) -> DownloadResult:
        """完整處理單一 URL"""
        result = self.download_audio(url)
        segments = self.transcribe(
            result.audio_path,
            use_subtitle=result.subtitle_path,
            subtitle_source=result.subtitle_source,
        )

        transcript_dir = self.output_dir / "transcripts"
        transcript_path = transcript_dir / f"{result.video_id}_transcript.json"
        self.save_transcript(
            segments,
            transcript_path,
            metadata={
                "video_id": result.video_id,
                "title": result.title,
                "subtitle_source": result.subtitle_source.value,
                "has_manual_subs": result.has_manual_subs,
                "speaker_persona": result.speaker_persona,
                "whisper_model": self.whisper_model_size if result.subtitle_source == SubtitleSource.WHISPER_ASR else None,
                "segment_count": len(segments),
            },
        )
        result.transcript_path = transcript_path
        return result

    def process_url_list(self, url_list_path: str | Path) -> list[DownloadResult]:
        """批次處理 URL 清單"""
        url_list_path = Path(url_list_path)
        urls = [
            line.strip()
            for line in url_list_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]

        logger.info(f"批次處理 {len(urls)} 個 URL")
        results = []
        stats = {"manual": 0, "auto": 0, "whisper": 0, "failed": 0}

        for i, url in enumerate(urls, 1):
            logger.info(f"[{i}/{len(urls)}] 處理: {url}")
            try:
                result = self.process_url(url)
                results.append(result)

                if result.has_manual_subs:
                    stats["manual"] += 1
                elif result.subtitle_source == SubtitleSource.AUTO_CAPTION:
                    stats["auto"] += 1
                else:
                    stats["whisper"] += 1
            except Exception as e:
                logger.error(f"處理失敗: {url} — {e}")
                stats["failed"] += 1
        return results