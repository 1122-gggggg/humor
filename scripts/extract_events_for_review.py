import os
import sys
from pathlib import Path

# 加入 src 到環境變數
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import yt_dlp
from humor_bot.data_engine.laughter_detector import LaughterDetector
from humor_bot.data_engine.audio_analyzer import AudioAnalyzer

# 使用使用者提供的播放清單 (或是可以改為特定單支影片)
URL = "https://www.youtube.com/playlist?list=PLIjpwRtLsLaoj1bdTJgMCjaRQ3-g1VQwI"
MAX_VIDEOS = 1  # 示範先抓 1 部讓您檢查

WORK_DIR = Path("data/review_audio")

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
        'writesubtitles': False,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        vid_id = info['id']
        title = info.get('title', 'Unknown Title')
        
    audio_path = output_dir / f"{vid_id}.wav"
    return vid_id, title, audio_path

def main():
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    out_txt_path = WORK_DIR / "events_review.txt"
    
    videos = get_videos(URL, MAX_VIDEOS)
    if not videos:
        print("沒有找到影片喔")
        return
        
    print("🤖 載入 AI 模型中...")
    # YAMNet 會抓的包含: Laughter, Clapping (掌聲), Crowd 等
    # 我們把門檻調低一點，讓您不會漏掉
    detector = LaughterDetector(confidence_threshold=0.4, min_duration_sec=0.5)
    analyzer = AudioAnalyzer()
    
    # 準備寫入文字檔
    with open(out_txt_path, "w", encoding="utf-8") as f_out:
        f_out.write("=============== 脫口秀音效事件人工檢查表 ===============\n\n")

        for i, v in enumerate(videos, 1):
            print(f"\n========================================================")
            print(f"🎞️ [{i}/{len(videos)}] 處理影片: {v['title']}")
            f_out.write(f"影片: {v['title']} (URL: {v['url']})\n")
            f_out.write("-" * 60 + "\n")
            
            audio_path = None
            try:
                vid_id, title, audio_path = download_audio_only(v["url"], WORK_DIR)
                
                print("🎧 偵測笑聲與掌聲...")
                events = detector.detect(audio_path)
                
                if not events:
                    print("⚠️ 這部影片沒有偵測到任何笑聲或掌聲")
                    f_out.write("無偵測到任何事件\n\n")
                    continue
                
                f_out.write(f"共偵測到 {len(events)} 個事件：\n")
                
                # 逐一分析每個事件的大小聲
                for idx, e in enumerate(events, 1):
                    # 把 AI 認定的類別稍微轉成中文方便看
                    evt_type = e.event_class
                    if "Laugh" in evt_type or "Giggle" in evt_type or "Chuckle" in evt_type:
                        label = "😂 笑聲"
                    elif "Clap" in evt_type or "Crowd" in evt_type:
                        label = "👏 掌聲/群眾"
                    else:
                        label = f"🔊 {evt_type}"

                    # 量測這個時間段的真實分貝大小 (RMS dB)
                    feat = analyzer.analyze_segment(audio_path, e.start, e.end)
                    volume_db = int(feat.rms_db)
                    
                    # -30dB 以下通常算很小聲，0dB 是最大聲
                    if volume_db >= -15:
                        vol_desc = "震耳欲聾"
                    elif volume_db >= -22:
                        vol_desc = "大聲"
                    elif volume_db >= -30:
                        vol_desc = "中等"
                    else:
                        vol_desc = "微小"

                    # 寫入文字檔
                    line = (
                        f"[{idx:02d}] {e.start:6.2f} 秒 ~ {e.end:6.2f} 秒 (長 {e.duration:4.2f}s) | "
                        f"{label:<8} | AI信心度: {e.confidence:.2f} | 音量: {volume_db:3d} dB ({vol_desc})"
                    )
                    f_out.write(line + "\n")
                    
                    # 同時印在前台給您看前幾筆
                    if idx <= 5:
                        print("  " + line)
                
                if len(events) > 5:
                    print(f"  ... 還有 {len(events)-5} 個事件，請見 {out_txt_path.name}")
                
            except Exception as e:
                print(f"❌ 影片處理失敗: {e}")
                f_out.write(f"處理失敗: {e}\n")
            finally:
                f_out.write("\n")
                if audio_path and audio_path.exists():
                    audio_path.unlink()
                
    print(f"\n\n✅ 報告已產出！請打開這個檔案檢查： {out_txt_path.absolute()}")

if __name__ == "__main__":
    main()
