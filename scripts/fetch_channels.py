import os
import argparse
from typing import List

def fetch_channel_videos(channel_url: str, max_videos: int) -> List[str]:
    """
    使用 yt-dlp 抓取頻道播放清單中的影片 URL
    """
    import yt_dlp
    
    # yt-dlp 參數設定：只擷取影片 URL，不下載實體檔案
    ydl_opts = {
        'extract_flat': True,       # 只提取清單，不深入下載
        'quiet': True,              # 不印出雜訊
        'playlistend': max_videos,  # 每個設定的頻道最多抓幾支
    }
    
    urls = []
    print(f"📡 正在搜尋頻道內容: {channel_url}...")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(channel_url, download=False)
            if 'entries' in info:
                # 這是播放清單或頻道影片列表
                for entry in info['entries']:
                    if entry and entry.get('url'):
                        url = entry['url']
                        # 有時 extract_flat 拿到的是相對路徑或不含網域的 ID
                        if not url.startswith('http'):
                            url = f"https://www.youtube.com/watch?v={entry.get('id', url)}"
                        urls.append(url)
            else:
                # 單一影片 (fallback)
                urls.append(info.get('webpage_url', channel_url))
        except Exception as e:
            print(f"❌ 擷取失敗 ({channel_url}): {e}")
            
    return urls

def main():
    parser = argparse.ArgumentParser(description="自動化爬取高品質脫口秀頻道 URL")
    parser.add_argument("--channels", type=str, required=True, 
                        help="頻道網址清單 (多個請用逗號分隔)，例如: 'https://www.youtube.com/@STRNetwork,https://www.youtube.com/@ComedyCentral'")
    parser.add_argument("--max_videos", type=int, default=100,
                        help="每個頻道最多抓取的影片數量 (預設: 100)")
    parser.add_argument("--output", type=str, default="data/raw/high_quality_urls.txt",
                        help="輸出的 url_list.txt 路徑")
    
    args = parser.parse_args()
    
    channel_list = [c.strip() for c in args.channels.split(",")]
    all_urls = []
    
    print("\n🚀 [Standup4AI] 啟動大規模高品質段子爬蟲管線...")
    print(f"🎯 目標頻道數: {len(channel_list)}")
    print(f"🎯 每頻道最大抓取量: {args.max_videos} 支影片\n")
    
    for channel in channel_list:
        # 如果使用者只輸入名稱，也可以擴增自動從 API 搜尋的功能 (這版先讀取正確 URL)
        urls = fetch_channel_videos(channel, args.max_videos)
        print(f"   => 成功截獲 {len(urls)} 支影片 URL")
        all_urls.extend(urls)
        
    # 去除重複的 URL
    all_urls = list(set(all_urls))
    
    print(f"\n✅ 爬取完畢！共收集到 {len(all_urls)} 支高品質脫口秀影片。")
    
    # 寫入檔案
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for u in all_urls:
            f.write(f"{u}\n")
            
    print(f"💾 已儲存至: {args.output}")
    print(f"\n💡 下一步：將此清單丟入端到端多模態處裡管線：")
    print(f"   python src/humor_bot/cli.py process_pipeline {args.output} --cleanup")

if __name__ == "__main__":
    main()
