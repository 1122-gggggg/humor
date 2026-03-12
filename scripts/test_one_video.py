import os
import sys
from pathlib import Path

# 加入 src 到環境變數
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import yt_dlp
import json
import dataclasses
import time
import shutil
import re

from humor_bot.data_engine.youtube_downloader import YouTubeDownloader
from humor_bot.data_engine.laughter_detector import LaughterDetector
from humor_bot.data_engine.audio_analyzer import AudioAnalyzer
from humor_bot.data_engine.alignment import SetupPunchlineAligner
from humor_bot.data_engine.auto_annotator import AutoAnnotationPipeline

def download_audio_only(url: str, output_dir: Path):
    """測試用：僅下載 wav (分析聲音) 以及字幕，完全不抓影片省空間"""
    print(f"📥 正在下載音軌: {url}")
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': str(output_dir / '%(id)s.%(ext)s'),
        'writesubtitles': True,
        'writeautomaticsub': True,
        'subtitleslangs': ['zh-TW', 'zh-Hant', 'zh', 'en'],
        'subtitlesformat': 'json3',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
        }],
        'postprocessor_args': ['-ar', '16000', '-ac', '1'],
        'quiet': True,
        'no_warnings': True,
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        vid_id = info['id']
        
    audio_path = output_dir / f"{vid_id}.wav"
    
    subtitle_path = None
    for lang in ['zh-TW', 'zh-Hant', 'zh', 'en']:
        p = output_dir / f"{vid_id}.{lang}.json3"
        if p.exists():
            subtitle_path = p
            break
            
    print(f"✅ 音軌及字幕下載完成: {info.get('title')}")
    return vid_id, audio_path, subtitle_path

def main():
    # 測試用的 URL，預設抓 08comedy 頻道的短單口喜劇影片 
    # (您也可以換成其他的 youtube url)
    TEST_URL = "https://www.youtube.com/watch?v=kYJq30T25O4"  # 範例影片
    if len(sys.argv) > 1:
        TEST_URL = sys.argv[1]

    # 設定目錄
    work_dir = Path("data/raw/test_run")
    work_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 初始化模型
    print("🤖 載入 AI 模型中...")
    whisper_downloader = YouTubeDownloader(output_dir=work_dir, whisper_model_size="medium") # 本機測試先用 medium
    detector = LaughterDetector(confidence_threshold=0.7)
    audio_analyzer = AudioAnalyzer()
    aligner = SetupPunchlineAligner(min_laughter_confidence=0.7)
    
    # 強制關閉視覺分析
    enable_video = False
    annotator = AutoAnnotationPipeline(enable_video=enable_video, enable_technique_analysis=False)

    print("=" * 60)
    # 2. 下載音軌
    vid_id, audio_path, subtitle_path = download_audio_only(TEST_URL, work_dir)

    try:
        # 3. 語音轉錄
        print("🎙️ 進行語音轉錄...")
        if subtitle_path:
            segments = whisper_downloader._parse_json3_subtitle(subtitle_path)
        else:
            segments = whisper_downloader._whisper_transcribe(audio_path)
            
        transcript_data = {
            "metadata": {"video_id": vid_id},
            "segments": [dataclasses.asdict(s) for s in segments]
        }
        
        # 4. 笑聲偵測
        print("😂 偵測笑聲中...")
        laughter_events = detector.detect(audio_path)
        laughter_dicts = [
            {'start': e.start, 'end': e.end, 'duration': e.duration,
             'confidence': e.confidence, 'event_class': e.event_class}
            for e in laughter_events
        ]
        
        # 5. 音訊分析
        print("📊 分析音訊特徵...")
        audio_features = []
        for e in laughter_events:
            feat = audio_analyzer.analyze_segment(audio_path, e.start, e.end)
            audio_features.append({'start': feat.start, 'end': feat.end, 'rms_db': feat.rms_db})
            
        # 6. 此處已省略影像分析
        video_reactions = None
            
        # 7. 對齊
        print("📎 對齊 Setup 與 Punchline...")
        aligned_jokes = aligner.align(vid_id, transcript_data, laughter_dicts, audio_features)
        
        # 8. 評分與標註
        print("🏷️ 計算幽默分數 (Humor Score)...")
        candidates = annotator.run(
            video_id=vid_id,
            aligned_jokes=aligned_jokes,
            laughter_events=laughter_dicts,
            audio_features=audio_features,
            video_reactions=video_reactions
        )
        
        print(f"\n🎉 測試完成！提取出 {len(candidates)} 個潛在段子：")
        for i, c in enumerate(candidates[:5], 1):
            print(f"  [{i}] 分數: {c.humor_score:.2f} | 標籤: {c.auto_label}")
            print(f"      Setup: {c.setup_text[:30]}...")
            print(f"      Punchline: {c.punchline_text[:30]}...")
            print("-" * 40)
            
        # 9. 儲存結果
        out_file = work_dir / f"{vid_id}_test_result.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump([dataclasses.asdict(c) for c in candidates], f, ensure_ascii=False, indent=2)
        print(f"\n💾 完整結果儲存於: {out_file}")

    finally:
        # 清理暫存的音軌檔
        print("\n🧹 清除暫存的音檔與字幕...")
        for p in [audio_path, subtitle_path]:
            if p and p.exists():
                p.unlink()

if __name__ == "__main__":
    main()
