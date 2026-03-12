import os
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import yt_dlp
from humor_bot.data_engine.laughter_detector import LaughterDetector

CHANNEL_URL = "https://www.youtube.com/playlist?list=PLIjpwRtLsLaoj1bdTJgMCjaRQ3-g1VQwI"
MAX_VIDEOS = 5 # 先抓 5 部示範

OUTPUT_DIR = Path("data/laughter_only")

def get_channel_videos():
    print(f"🔍 尋找播放清單 {CHANNEL_URL} 中的影片...")
    ydl_opts = {
        'quiet': True,
        'extract_flat': 'in_playlist',
    }
    videos_to_process = []
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(CHANNEL_URL, download=False)
        entries = info.get('entries', [])
        
        for entry in entries:
            title = entry.get('title', '')
            url = entry.get('url', '')
            if url:
                videos_to_process.append({"title": title, "url": url, "id": entry.get('id')})
                if len(videos_to_process) >= MAX_VIDEOS:
                    break
    return videos_to_process

def download_audio_only(url: str, output_dir: Path):
    print(f"📥 正在下載音軌: {url}")
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
        'writesubtitles': False, # 連字幕都不抓
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        vid_id = info['id']
        
    audio_path = output_dir / f"{vid_id}.wav"
    return vid_id, audio_path

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    videos = get_channel_videos()
    
    # 初始化笑聲偵測器 (設定信心閥值 0.6)
    print("🤖 載入 YAMNet 笑聲偵測模型...")
    detector = LaughterDetector(confidence_threshold=0.6)

    for i, v in enumerate(videos, 1):
        print(f"\n========================================================")
        print(f"🎞️ [{i}/{len(videos)}] 處理影片: {v['title']}")
        
        try:
            # 1. 下載音檔
            vid_id, audio_path = download_audio_only(v["url"], OUTPUT_DIR)
            
            # 2. 偵測笑聲
            print("😂 開始掃描音軌找出笑聲時間點...")
            laughter_events = detector.detect(audio_path)
            
            # 3. 整理結果並儲存 JSON
            laughter_dicts = [
                {
                    'start_time': round(e.start, 2), 
                    'end_time': round(e.end, 2), 
                    'duration': round(e.duration, 2),
                    'confidence': round(e.confidence, 3), 
                    'type': e.event_class
                }
                for e in laughter_events
            ]
            
            out_file = OUTPUT_DIR / f"{vid_id}_laughter.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(laughter_dicts, f, ensure_ascii=False, indent=2)
                
            print(f"✅ 成功找出 {len(laughter_events)} 次笑聲！時間點已儲存於: {out_file}")
            
            # 也可以選擇在這裡印出前幾個笑聲
            for j, event in enumerate(laughter_dicts[:3]):
                print(f"   [笑聲 {j+1}] 從 {event['start_time']}秒 到 {event['end_time']}秒 (信心度: {event['confidence']})")
            if len(laughter_dicts) > 3:
                print("   ...")
            
        except Exception as e:
            print(f"❌ 影片處理失敗: {e}")
            
        finally:
            # 處理完立刻刪除音檔，不佔空間
            print("🧹 清理用完的音訊檔...")
            if 'audio_path' in locals() and audio_path and audio_path.exists():
                audio_path.unlink()

if __name__ == "__main__":
    main()
