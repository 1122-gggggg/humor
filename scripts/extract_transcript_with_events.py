import os
import sys
import json
import dataclasses
from pathlib import Path

# 加入 src 到環境變數
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import yt_dlp
from humor_bot.data_engine.youtube_downloader import YouTubeDownloader
from humor_bot.data_engine.laughter_detector import LaughterDetector
from humor_bot.data_engine.audio_analyzer import AudioAnalyzer

URL = "https://www.youtube.com/playlist?list=PLIjpwRtLsLaoj1bdTJgMCjaRQ3-g1VQwI"
MAX_VIDEOS = 1  # 示範先抓 1 部讓您檢查

WORK_DIR = Path("data/review_transcript")

def get_videos(url: str, max_count: int):
    print(f"🔍 解析 URL: {url} ...")
    ydl_opts = {
        'quiet': True,
        'extract_flat': 'in_playlist',
    }
    videos = []
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        if 'entries' in info:
            entries = info.get('entries', [])
            for entry in entries:
                if entry.get('url'):
                    videos.append({"title": entry.get('title', ''), "url": entry.get('url'), "id": entry.get('id')})
                if len(videos) >= max_count:
                    break
        else:
            videos.append({"title": info.get('title', ''), "url": info.get('original_url', url), "id": info.get('id')})
            
    return videos[:max_count]

def download_audio_and_subs(url: str, output_dir: Path):
    print(f"📥 正在下載音軌與字幕: {url}")
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': str(output_dir / '%(id)s.%(ext)s'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
        }],
        'postprocessor_args': ['-ar', '16000', '-ac', '1'],
        'quiet': True,
        'no_warnings': True,
        'writesubtitles': True,         # 只去抓取「人工上傳的 CC 字幕」
        'writeautomaticsub': False,     # 關閉爛品質的自動字幕，若沒 CC，我們寧願用我們自己的 SOTA Whisper 來聽
        'subtitleslangs': ['zh-TW', 'zh-Hant', 'zh', 'en'],
        'subtitlesformat': 'json3',
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        vid_id = info['id']
        title = info.get('title', 'Unknown Title')
        
    audio_path = output_dir / f"{vid_id}.wav"
    
    subtitle_path = None
    for lang in ['zh-TW', 'zh-Hant', 'zh', 'en']:
        p = output_dir / f"{vid_id}.{lang}.json3"
        if p.exists():
            subtitle_path = p
            break
            
    return vid_id, title, audio_path, subtitle_path

def format_time(seconds: float) -> str:
    """轉換秒數為 MM:SS 格式"""
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"

def main():
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    out_txt_path = WORK_DIR / "transcript_with_events.txt"
    
    videos = get_videos(URL, MAX_VIDEOS)
    if not videos:
        print("沒有找到影片喔")
        return
        
    print("🤖 載入 AI 模型中...")
    whisper_downloader = YouTubeDownloader(output_dir=WORK_DIR, whisper_model_size="large-v3")
    # ✅ 大幅降低笑聲偵測門檻：0.15 (原0.4)，並讓持續時間降至 0.3 秒，捕捉觀眾微弱的回饋
    detector = LaughterDetector(confidence_threshold=0.15, min_duration_sec=0.3)
    analyzer = AudioAnalyzer()
    
    with open(out_txt_path, "w", encoding="utf-8") as f_out:
        f_out.write("=============== 脫口秀逐字稿與事件人工核對表 ===============\n")
        f_out.write("說明：此表按時間軸合併了「演員講話文字」以及「觀眾笑聲/掌聲」\n\n")

        for i, v in enumerate(videos, 1):
            print(f"\n========================================================")
            print(f"🎞️ [{i}/{len(videos)}] 處理影片: {v['title']}")
            f_out.write(f"\n影片標題: {v['title']}\n")
            f_out.write(f"影片連結: {v['url']}\n")
            f_out.write("-" * 60 + "\n")
            
            try:
                vid_id, title, audio_path, subtitle_path = download_audio_and_subs(v["url"], WORK_DIR)
                
                print("🎙️ 解析逐字稿 (語音轉文字) ...")
                if subtitle_path:
                    segments = whisper_downloader._parse_json3_subtitle(subtitle_path)
                else:
                    segments = whisper_downloader._whisper_transcribe(audio_path)
                    
                print("🎧 偵測觀眾笑聲與掌聲...")
                events = detector.detect(audio_path)
                
                # 將所有事件與逐字稿合併到同一個時間軸清單 (timeline) 中
                timeline = []
                
                # 加入講話片段
                for s in segments:
                    timeline.append({
                        "type": "speech",
                        "start": s.start,
                        "end": s.end,
                        "text": s.text.strip()
                    })
                    
                # 加入環境音效片段 (笑聲、掌聲)
                for e in events:
                    # 量測這個時間段的分貝數
                    feat = analyzer.analyze_segment(audio_path, e.start, e.end)
                    volume_db = int(feat.rms_db)
                    
                    if volume_db >= -15:
                        vol_desc = "震耳欲聾"
                    elif volume_db >= -22:
                        vol_desc = "大聲"
                    elif volume_db >= -30:
                        vol_desc = "中等"
                    else:
                        vol_desc = "微小"

                    evt_type = e.event_class
                    if "Laugh" in evt_type or "Giggle" in evt_type or "Chuckle" in evt_type:
                        label = "😂 笑聲"
                    elif "Clap" in evt_type or "Crowd" in evt_type:
                        label = "👏 掌聲/群眾"
                    else:
                        label = f"🔊 {evt_type}"
                        
                    timeline.append({
                        "type": "event",
                        "start": e.start,
                        "end": e.end,
                        "label": label,
                        "confidence": e.confidence,
                        "volume_db": volume_db,
                        "vol_desc": vol_desc
                    })
                    
                # 按照開始時間排序
                timeline.sort(key=lambda x: x["start"])
                
                # 寫入結果
                for item in timeline:
                    t_start = format_time(item["start"])
                    t_end = format_time(item["end"])
                    time_mark = f"[{t_start} - {t_end}]"
                    
                    if item["type"] == "speech":
                        line = f"{time_mark} 演員: {item['text']}\n"
                        f_out.write(line)
                    else:
                        line = f"{time_mark} ---> {item['label']} (音量: {item['volume_db']}dB {item['vol_desc']}, AI信心: {item['confidence']:.2f}) <--- \n"
                        f_out.write(line)

                print("✅ 逐字稿與音效合併完成！")
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"❌ 影片處理失敗: {e}")
                f_out.write(f"處理失敗: {e}\n")
            finally:
                if 'audio_path' in locals() and audio_path and audio_path.exists():
                    audio_path.unlink()
                if 'subtitle_path' in locals() and subtitle_path and subtitle_path.exists():
                    subtitle_path.unlink()
                
    print(f"\n\n📝 詳盡審查報告已產出！請打開供人工核驗： {out_txt_path.absolute()}")

if __name__ == "__main__":
    main()
