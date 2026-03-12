import os
import sys
import json
import dataclasses
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import yt_dlp
from humor_bot.data_engine.youtube_downloader import YouTubeDownloader
from humor_bot.data_engine.laughter_detector import LaughterDetector
from humor_bot.data_engine.audio_analyzer import AudioAnalyzer
from humor_bot.data_engine.alignment import SetupPunchlineAligner
from humor_bot.data_engine.auto_annotator import AutoAnnotationPipeline
from humor_bot.training.reward_model import RewardModelTrainer

CHANNEL_URL = "https://www.youtube.com/playlist?list=PLIjpwRtLsLaoj1bdTJgMCjaRQ3-g1VQwI"
MAX_VIDEOS = 20

PROCESSED_DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models/reward_model")
PROCESSED_IDS_FILE = PROCESSED_DATA_DIR / "processed_ids.json"
CANDIDATES_FILE = PROCESSED_DATA_DIR / "candidates.json"

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
            
            # 播放清單內的影片直接加入
            if url:
                videos_to_process.append({"title": title, "url": url, "id": entry.get('id')})
                if len(videos_to_process) >= MAX_VIDEOS * 3:
                    break
    return videos_to_process

def download_audio_only(url: str, output_dir: Path):
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

def load_json(path: Path, default=None):
    if not default:
        default = []
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default

def save_json(data, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def main():
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    processed_ids = load_json(PROCESSED_IDS_FILE)
    all_candidates = load_json(CANDIDATES_FILE)

    videos = get_channel_videos()
    unprocessed_videos = [v for v in videos if v["id"] not in processed_ids][:MAX_VIDEOS]

    if not unprocessed_videos:
        print("🤷 找不到新的未處理影片，將直接進入訓練階段。")
    else:
        print(f"🎯 找到 {len(unprocessed_videos)} 部未處理的喜劇影片，準備開始串流處理 (Replay 機制)...")
        # 1. 初始化資料處理模型
        print("🤖 載入 AI 處理模型...")
        whisper_downloader = YouTubeDownloader(output_dir=PROCESSED_DATA_DIR, whisper_model_size="medium")
        detector = LaughterDetector(confidence_threshold=0.6) # 調低閾值，容忍更多笑聲
        audio_analyzer = AudioAnalyzer()
        aligner = SetupPunchlineAligner(min_laughter_confidence=0.6)
        annotator = AutoAnnotationPipeline(enable_video=False, enable_technique_analysis=False)

        for i, v in enumerate(unprocessed_videos, 1):
            print(f"\n========================================================")
            print(f"🎞️ [{i}/{len(unprocessed_videos)}] 處理影片: {v['title']}")
            
            try:
                vid_id, audio_path, subtitle_path = download_audio_only(v["url"], PROCESSED_DATA_DIR)
                
                print("🎙️ 進行語音轉錄...")
                if subtitle_path:
                    segments = whisper_downloader._parse_json3_subtitle(subtitle_path)
                else:
                    segments = whisper_downloader._whisper_transcribe(audio_path)
                    
                transcript_data = {
                    "metadata": {"video_id": vid_id},
                    "segments": [dataclasses.asdict(s) for s in segments]
                }
                
                print("😂 偵測笑聲中...")
                laughter_events = detector.detect(audio_path)
                laughter_dicts = [
                    {'start': e.start, 'end': e.end, 'duration': e.duration,
                    'confidence': e.confidence, 'event_class': e.event_class}
                    for e in laughter_events
                ]
                
                print("📊 分析音訊特徵...")
                audio_features = []
                for e in laughter_events:
                    feat = audio_analyzer.analyze_segment(audio_path, e.start, e.end)
                    audio_features.append({'start': feat.start, 'end': feat.end, 'rms_db': feat.rms_db})
                    
                print("📎 對齊 Setup 與 Punchline...")
                aligned_jokes = aligner.align(vid_id, transcript_data, laughter_dicts, audio_features)
                
                print("🏷️ 計算幽默分數 (Humor Score)...")
                candidates = annotator.run(
                    video_id=vid_id,
                    aligned_jokes=aligned_jokes,
                    laughter_events=laughter_dicts,
                    audio_features=audio_features,
                    video_reactions=None
                )
                
                valid_candidates = [dataclasses.asdict(c) for c in candidates if c.humor_score >= 0.2]
                all_candidates.extend(valid_candidates)
                save_json(all_candidates, CANDIDATES_FILE) # 每處理完一部就保存，斷點續傳
                
                processed_ids.append(vid_id)
                save_json(processed_ids, PROCESSED_IDS_FILE)
                
                print(f"🎉 影片處理成功！本影片萃取出了 {len(valid_candidates)} 個段子。目前總累積: {len(all_candidates)} 個段子")
                
            except Exception as e:
                print(f"❌ 影片處理失敗: {e}")
                
            finally:
                # 每部影片處理完立刻刪除音軌，實踐串流節省空間
                print("🧹 清除暫存的音檔與字幕...")
                if 'audio_path' in locals() and audio_path and audio_path.exists():
                    audio_path.unlink()
                if 'subtitle_path' in locals() and subtitle_path and subtitle_path.exists():
                    subtitle_path.unlink()

    # ────────────────────────────────────────────────────────
    # 訓練階段 (方法 A: 混合重現訓練 Joint Training / Replay)
    # ────────────────────────────────────────────────────────
    print("\n\n========================================================")
    print("🧠 開始訓練 Reward Model (Continual Learning)")
    print(f"總共讀取了 {len(all_candidates)} 個段子資料。")
    
    if len(all_candidates) < 10:
        print("⚠️ 累積段子數太少，取消訓練。請抓取更多影片。")
        sys.exit(0)

    # 將抽出資料轉換成偏好對
    pairs = RewardModelTrainer.build_preference_pairs(all_candidates, max_pairs=3000)
    
    if len(pairs) < 5:
        print("⚠️ 無法構成足夠的 Preference Pairs，可能是笑點分數差異不夠，或資料太少。")
        sys.exit(0)

    # 初始化 Trainer
    print("🤖 初始化 Reward Model Trainer...")
    trainer = RewardModelTrainer(
        base_model="roberta-base", # 考慮到您的設備，roberta-base 速度會稍微快點
        batch_size=4,
        num_epochs=5,
        device="cpu" # 您的環境如果遇到 CUDA OOM，可以確保在 CPU 或 MPS 上跑。這裡預設給它偵測，或先設為可以跑
    )
    
    import torch
    trainer.device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint_path = MODEL_DIR / "final_model.pt"
    
    if checkpoint_path.exists():
        print(f"🔄 發現已存在的舊模型 ({checkpoint_path})，啟動【混合接續訓練】(Continual Learning)")
        trainer.load_checkpoint(checkpoint_path)
        # 用較小的學習率 Fine-tune 接續訓練，避免災難性遺忘
        trainer.learning_rate = 5e-6 
    else:
        print(f"⭐ 找不到舊模型，啟動【全新訓練】")
        trainer.learning_rate = 2e-5

    print(f"🏃‍♂️ 使用 Learning Rate: {trainer.learning_rate}")
    trainer.train(pairs, output_dir=MODEL_DIR)

if __name__ == "__main__":
    main()
