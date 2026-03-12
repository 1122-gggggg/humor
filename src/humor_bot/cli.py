"""
CLI 入口 — 幽默脫口秀 AI 機器人

使用 Typer 建立命令行介面。
"""

from __future__ import annotations

import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.logging import RichHandler

app = typer.Typer(
    name="humor-bot",
    help="幽默脫口秀 AI 機器人 — 多模態幽默偵測與生成系統",
    no_args_is_help=True,
)

console = Console()


def setup_logging(verbose: bool = False):
    """設定 logging"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, show_path=False)],
    )


# ── 資料引擎命令 ──────────────────────────────────────────

@app.command()
def download(
    url: str = typer.Option(None, "--url", "-u", help="單一 YouTube URL"),
    url_list: str = typer.Option(None, "--url-list", "-l", help="URL 清單檔案路徑"),
    output_dir: str = typer.Option("data/raw", "--output", "-o", help="輸出目錄"),
    whisper_model: str = typer.Option("large-v3", "--whisper-model", help="Whisper 模型大小"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """下載 YouTube 影片音軌並產生逐字稿"""
    setup_logging(verbose)
    from humor_bot.data_engine.youtube_downloader import YouTubeDownloader

    downloader = YouTubeDownloader(
        output_dir=output_dir,
        whisper_model_size=whisper_model,
    )

    if url:
        result = downloader.process_url(url)
        console.print(f"✅ 完成: {result.title}")
        console.print(f"   音訊: {result.audio_path}")
        console.print(f"   逐字稿: {result.transcript_path}")
    elif url_list:
        results = downloader.process_url_list(url_list)
        console.print(f"✅ 批次完成: {len(results)} 個影片")
    else:
        console.print("[red]請提供 --url 或 --url-list 參數[/red]")
        raise typer.Exit(1)


@app.command()
def detect_laughter(
    audio_path: str = typer.Argument(..., help="音訊檔案路徑"),
    output: str = typer.Option(None, "--output", "-o", help="輸出 JSON 路徑"),
    threshold: float = typer.Option(0.8, "--threshold", "-t", help="信心閾值"),
    min_duration: float = typer.Option(0.5, "--min-duration", help="最短持續時間（秒）"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """偵測音訊中的笑聲與掌聲"""
    setup_logging(verbose)
    from humor_bot.data_engine.laughter_detector import LaughterDetector

    detector = LaughterDetector(
        confidence_threshold=threshold,
        min_duration_sec=min_duration,
    )

    events = detector.detect(audio_path)
    console.print(f"🎤 偵測到 {len(events)} 個笑聲/掌聲事件")

    for e in events:
        console.print(
            f"  [{e.start:.1f}s - {e.end:.1f}s] "
            f"{e.event_class} (conf={e.confidence:.2f}, dur={e.duration:.1f}s)"
        )

    if output:
        detector.to_json(events, output)
        console.print(f"💾 已儲存: {output}")


@app.command()
def align(
    video_id: str = typer.Argument(..., help="影片 ID"),
    transcript: str = typer.Option(..., "--transcript", "-t", help="逐字稿 JSON 路徑"),
    laughter: str = typer.Option(..., "--laughter", "-l", help="笑聲偵測 JSON 路徑"),
    output: str = typer.Option("data/processed/aligned.json", "--output", "-o"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """對齊 Setup-Punchline 結構"""
    setup_logging(verbose)
    import json
    from humor_bot.data_engine.alignment import SetupPunchlineAligner

    with open(transcript, "r", encoding="utf-8") as f:
        transcript_data = json.load(f)
    with open(laughter, "r", encoding="utf-8") as f:
        laughter_data = json.load(f)

    aligner = SetupPunchlineAligner(min_laughter_confidence=0.8)
    results = aligner.align(video_id, transcript_data, laughter_data)

    console.print(f"📝 對齊結果: {len(results)} 個 Setup-Punchline 段子")
    for sp in results[:5]:
        console.print(f"\n  [bold]Setup:[/bold] {sp.setup_text[:80]}...")
        console.print(f"  [bold]Punchline:[/bold] {sp.punchline_text[:80]}...")
        console.print(f"  Humor Score: {sp.humor_score:.2f}")

    aligner.save_dataset(results, output)
    console.print(f"💾 已儲存: {output}")


@app.command()
def process_pipeline(
    url_list: str = typer.Argument(..., help="YouTube URL 清單檔案"),
    output_dir: str = typer.Option("data/processed", "--output", "-o"),
    whisper_model: str = typer.Option("large-v3", "--whisper-model"),
    threshold: float = typer.Option(0.8, "--threshold"),
    cleanup: bool = typer.Option(True, "--cleanup/--keep-raw", help="處理完畢後立即刪除原始影音檔，節省硬碟空間"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """端到端資料處理管線：下載 → 笑聲偵測 → 對齊"""
    setup_logging(verbose)
    import json
    from humor_bot.data_engine.youtube_downloader import YouTubeDownloader
    from humor_bot.data_engine.laughter_detector import LaughterDetector
    from humor_bot.data_engine.audio_analyzer import AudioAnalyzer
    from humor_bot.data_engine.alignment import SetupPunchlineAligner

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 初始化模組
    downloader = YouTubeDownloader(whisper_model_size=whisper_model)
    detector = LaughterDetector(confidence_threshold=threshold)
    analyzer = AudioAnalyzer()
    aligner = SetupPunchlineAligner(min_laughter_confidence=threshold)

    # 下載
    console.print("[bold blue]Phase 1: 下載影片...[/bold blue]")
    download_results = downloader.process_url_list(url_list)

    all_aligned = []

    for result in download_results:
        console.print(f"\n[bold]處理: {result.video_id}[/bold]")

        # 笑聲偵測
        console.print("  🎤 偵測笑聲...")
        events = detector.detect(result.audio_path)
        console.print(f"  → {len(events)} 個事件")

        # 音訊分析
        console.print("  📊 分析音訊特徵...")
        audio_features = []
        for e in events:
            feat = analyzer.analyze_segment(result.audio_path, e.start, e.end)
            audio_features.append({
                "start": feat.start, "end": feat.end, "rms_db": feat.rms_db,
            })

        # 載入逐字稿
        if result.transcript_path and result.transcript_path.exists():
            with open(result.transcript_path, "r", encoding="utf-8") as f:
                transcript = json.load(f)
        else:
            console.print("  [yellow]⚠ 無逐字稿，跳過對齊[/yellow]")
            continue

        # 笑聲事件轉 dict
        laughter_dicts = [
            {
                "start": e.start, "end": e.end, "duration": e.duration,
                "confidence": e.confidence, "event_class": e.event_class,
            }
            for e in events
        ]

        # 對齊
        console.print("  📎 對齊 Setup-Punchline...")
        aligned = aligner.align(
            result.video_id, transcript, laughter_dicts, audio_features
        )
        all_aligned.extend(aligned)
        console.print(f"  → {len(aligned)} 個段子")

        # 產生視覺化
        plot_path = output_path / f"{result.video_id}_intensity.png"
        analyzer.plot_intensity(
            result.audio_path,
            laughter_events=events,
            output_path=plot_path,
            title=f"Laughter Intensity: {result.video_id}",
        )
        
        # 雲端串流/節省空間策略：即存即刪 (Stream & Delete)
        if cleanup:
            console.print("  🧹 正在清理原始音軌與暫存檔 (Stream & Delete)...")
            try:
                if result.audio_path and result.audio_path.exists():
                    result.audio_path.unlink()
                if result.subtitle_path and result.subtitle_path.exists():
                    result.subtitle_path.unlink()
            except Exception as e:
                console.print(f"  [red]清理失敗: {e}[/red]")

    # 儲存完整資料集
    final_output = output_path / "dataset.json"
    aligner.save_dataset(all_aligned, final_output)
    console.print(f"\n✅ [bold green]管線完成！[/bold green]")
    console.print(f"   總計: {len(all_aligned)} 個 Setup-Punchline 段子")
    console.print(f"   儲存: {final_output}")


@app.command()
def generate(
    topic: str = typer.Argument(..., help="段子主題/情境"),
    adapter_path: str = typer.Option("", "--adapter", "-a", help="LoRA adapter 路徑"),
    num: int = typer.Option(3, "--num", "-n", help="生成數量"),
    temperature: float = typer.Option(0.8, "--temperature", "-t"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """生成脫口秀段子"""
    setup_logging(verbose)
    from humor_bot.models.joke_writer import JokeWriter, JokeWriterConfig

    config = JokeWriterConfig()
    writer = JokeWriter(config)

    if adapter_path:
        writer.load_adapter(adapter_path)

    console.print(f"🎭 為主題「{topic}」生成 {num} 個段子...\n")

    jokes = writer.generate(
        context=topic,
        num_return_sequences=num,
        temperature=temperature,
    )

    for i, joke in enumerate(jokes, 1):
        console.print(f"[bold]段子 {i}:[/bold]")
        console.print(joke)
        console.print()


@app.command()
def evaluate(
    input_file: str = typer.Argument(..., help="段子 JSON 檔案路徑"),
    output: str = typer.Option("data/evaluation/scores.json", "--output", "-o"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """使用 LLM-as-a-Judge 評分段子"""
    setup_logging(verbose)
    import json
    from humor_bot.evaluation.judge import HumorJudge

    with open(input_file, "r", encoding="utf-8") as f:
        jokes = json.load(f)

    judge = HumorJudge()
    results = judge.batch_judge(jokes)

    # 統計
    avg_score = sum(r.total_score for r in results) / len(results)
    console.print(f"\n📊 評分結果:")
    console.print(f"   平均分: {avg_score:.2f}/10")
    console.print(f"   總計: {len(results)} 個段子")

    judge.save_results(results, output)
    console.print(f"💾 已儲存: {output}")


# ── 自動標註命令 ──────────────────────────────────────────

@app.command()
def annotate_auto(
    dataset: str = typer.Argument(..., help="對齊資料集 JSON 路徑"),
    laughter: str = typer.Option(..., "--laughter", "-l", help="笑聲偵測 JSON 路徑"),
    video_path: str = typer.Option(None, "--video", help="原始影片路徑（啟用視覺分析）"),
    output: str = typer.Option("data/processed/candidates.json", "--output", "-o"),
    csv_output: str = typer.Option(None, "--csv", help="同時輸出 CSV（供專家標註）"),
    min_score: float = typer.Option(0.3, "--min-score", help="最低候選分數"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """自動標註管線：結合音訊 + 視覺特徵計算 Humor Score"""
    setup_logging(verbose)
    import json
    from humor_bot.data_engine.auto_annotator import AutoAnnotationPipeline

    with open(dataset, "r", encoding="utf-8") as f:
        aligned_jokes = json.load(f)
    with open(laughter, "r", encoding="utf-8") as f:
        laughter_events = json.load(f)

    # 視覺分析（如有影片）
    video_reactions = None
    if video_path:
        console.print("🎬 分析影片觀眾反應...")
        from humor_bot.data_engine.video_analyzer import VideoAnalyzer
        analyzer = VideoAnalyzer(
            sample_interval_sec=1.0,
            audience_roi=(0.0, 0.5, 1.0, 0.5),  # 畫面下半部
        )
        result = analyzer.analyze_video(video_path)
        video_reactions = [
            {"timestamp": r.timestamp, "positive_ratio": r.positive_ratio,
             "happy_ratio": r.happy_ratio, "surprise_ratio": r.surprise_ratio}
            for r in result.reactions
        ]

    # 執行自動標註
    pipeline = AutoAnnotationPipeline(enable_video=video_path is not None)
    candidates = pipeline.run(
        video_id=Path(dataset).stem,
        aligned_jokes=aligned_jokes,
        laughter_events=laughter_events,
        audio_features=[],
        video_reactions=video_reactions,
    )

    # 輸出
    pipeline.save_candidates_json(candidates, output, min_score=min_score)
    console.print(f"💾 候選集: {output}")

    if csv_output:
        pipeline.save_candidates_csv(candidates, csv_output, min_score=min_score)
        console.print(f"📄 CSV: {csv_output}")




@app.command()
def analyze_video(
    video_path: str = typer.Argument(..., help="影片檔案路徑"),
    output: str = typer.Option(None, "--output", "-o", help="分析結果 JSON 路徑"),
    interval: float = typer.Option(1.0, "--interval", help="取樣間隔（秒）"),
    roi: str = typer.Option(None, "--roi", help="觀眾 ROI (x,y,w,h) 0-1 比例"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """分析影片中的觀眾反應（表情辨識 + 肢體動作）"""
    setup_logging(verbose)
    from humor_bot.data_engine.video_analyzer import VideoAnalyzer

    audience_roi = None
    if roi:
        parts = [float(x) for x in roi.split(",")]
        if len(parts) == 4:
            audience_roi = tuple(parts)

    analyzer = VideoAnalyzer(
        sample_interval_sec=interval,
        audience_roi=audience_roi,
    )

    result = analyzer.analyze_video(video_path)

    console.print(f"\n📊 觀眾反應分析:")
    console.print(f"   分析幀數: {result.total_frames_analyzed}")
    console.print(f"   平均正面比例: {result.avg_positive_ratio:.1%}")
    console.print(f"   高峰時刻: {len(result.peak_moments)} 個")

    for peak in result.peak_moments[:10]:
        console.print(
            f"   ⭐ {peak['timestamp']:.1f}s — "
            f"😄={peak['happy_ratio']:.0%} 😮={peak['surprise_ratio']:.0%}"
        )

    if output:
        analyzer.save_results(result, output)
        console.print(f"💾 已儲存: {output}")


# ── 韻律、負面樣本、表演分析、新聞爬取 ──────────────────

@app.command()
def analyze_prosody(
    audio_path: str = typer.Argument(..., help="音訊檔案路徑"),
    aligned: str = typer.Option(None, "--aligned", "-a", help="對齊資料集 JSON（分析每段 timing）"),
    output: str = typer.Option(None, "--output", "-o"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """分析演員韻律（F0、語速、停頓 — Comedy is Timing）"""
    setup_logging(verbose)
    from humor_bot.data_engine.prosody_analyzer import ProsodyAnalyzer

    analyzer = ProsodyAnalyzer()

    if aligned:
        import json
        with open(aligned, "r", encoding="utf-8") as f:
            jokes = json.load(f)

        console.print(f"⏱️ 分析 {len(jokes)} 段段子的節奏...")
        results = []
        for joke in jokes:
            timing = analyzer.analyze_joke_timing(
                audio_path,
                setup_start=joke.get("setup_start", 0),
                setup_end=joke.get("setup_end", 0),
                punch_start=joke.get("punch_start", 0),
                punch_end=joke.get("punch_end", 0),
                joke_id=joke.get("id", ""),
            )
            results.append({
                "id": timing.joke_id,
                "timing_score": timing.timing_score,
                "pre_pause": timing.pre_punchline_pause,
                "post_pause": timing.post_punchline_pause,
                "speed_ratio": timing.setup_to_punch_speed_ratio,
                "f0_drop": timing.f0_drop_at_punch,
            })
            console.print(
                f"  [{timing.joke_id}] timing={timing.timing_score:.2f} "
                f"pre_pause={timing.pre_punchline_pause:.2f}s "
                f"f0_drop={timing.f0_drop_at_punch:.0f}Hz"
            )

        if output:
            with open(output, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            console.print(f"💾 已儲存: {output}")
    else:
        # 全段分析
        features = analyzer.analyze_segment(audio_path, 0, 30)
        console.print(f"\n⏱️ 韻律分析:")
        console.print(f"   F0 平均: {features.f0_mean:.1f} Hz")
        console.print(f"   語速: {features.speech_rate:.1f} syl/s")
        console.print(f"   停頓數: {features.pause_count}")
        console.print(f"   停頓比例: {features.pause_ratio:.1%}")


@app.command()
def detect_bombing(
    transcript: str = typer.Argument(..., help="逐字稿 JSON 路徑"),
    laughter: str = typer.Option(..., "--laughter", "-l", help="笑聲偵測 JSON 路徑"),
    positive: str = typer.Option(None, "--positive", "-p", help="正面候選集（建構對比對）"),
    output: str = typer.Option("data/processed/bombing.json", "--output", "-o"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """偵測冷場段落（負面樣本收集，解決倖存者偏差）"""
    setup_logging(verbose)
    import json
    from humor_bot.data_engine.negative_collector import NegativeSampleCollector

    with open(transcript, "r", encoding="utf-8") as f:
        segments = json.load(f)
    with open(laughter, "r", encoding="utf-8") as f:
        laughter_events = json.load(f)

    collector = NegativeSampleCollector()
    bombings = collector.detect_bombing(segments, laughter_events, [])

    console.print(f"\n🥶 冷場段落: {len(bombings)} 個")
    for b in bombings[:10]:
        console.print(f"  [{b.bombing_type}] {b.text[:60]}... (dB={b.actual_laughter_db:.0f})")

    collector.save_bombings(bombings, output)

    # 建構對比對
    if positive:
        with open(positive, "r", encoding="utf-8") as f:
            pos_data = json.load(f)
        pairs = collector.build_contrast_pairs(pos_data, bombings)
        pairs_path = Path(output).with_name("contrast_pairs.json")
        collector.save_contrast_pairs(pairs, pairs_path)
        console.print(f"🔗 對比對: {len(pairs)} 組 → {pairs_path}")


@app.command()
def analyze_performer(
    video_path: str = typer.Argument(..., help="影片檔案路徑"),
    aligned: str = typer.Option(None, "--aligned", "-a", help="對齊資料集（分析 delivery）"),
    output: str = typer.Option(None, "--output", "-o"),
    interval: float = typer.Option(0.5, "--interval", help="取樣間隔（秒）"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """分析表演者臉部表情（多模態物理對齊）"""
    setup_logging(verbose)
    import json
    from humor_bot.data_engine.performer_analyzer import PerformerAnalyzer

    analyzer = PerformerAnalyzer(
        sample_interval_sec=interval,
        performer_roi=(0.2, 0.05, 0.6, 0.9),  # 舞台中央
    )

    timeline = analyzer.analyze_performer(video_path)

    console.print(f"\n🎭 表演者表情分析:")
    console.print(f"   分析幀數: {timeline.total_frames}")
    console.print(f"   平均笑容: {timeline.avg_smile:.2f}")
    console.print(f"   表情高峰: {len(timeline.expression_peaks)} 個")

    if aligned:
        with open(aligned, "r", encoding="utf-8") as f:
            jokes = json.load(f)

        deliveries = []
        for joke in jokes:
            d = analyzer.analyze_punchline_delivery(
                timeline,
                setup_end=joke.get("setup_end", 0),
                punch_start=joke.get("punch_start", 0),
                punch_end=joke.get("punch_end", 0),
                joke_id=joke.get("id", ""),
            )
            deliveries.append({
                "id": d.joke_id,
                "delivery_score": d.delivery_score,
                "smile_delta": d.smile_delta,
                "head_tilt": d.head_tilt_at_punch,
            })
            console.print(f"  [{d.joke_id}] delivery={d.delivery_score:.2f}")

        if output:
            with open(output, "w", encoding="utf-8") as f:
                json.dump(deliveries, f, ensure_ascii=False, indent=2)

    if output and not aligned:
        with open(output, "w", encoding="utf-8") as f:
            json.dump({
                "avg_smile": timeline.avg_smile,
                "peaks": timeline.expression_peaks[:20],
            }, f, ensure_ascii=False, indent=2)

    if output:
        console.print(f"💾 已儲存: {output}")


@app.command()
def fetch_news(
    output: str = typer.Option("data/rag/news.json", "--output", "-o"),
    max_items: int = typer.Option(20, "--max", "-n", help="每來源最大抓取數"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """抓取即時新聞素材（RAG 即時語境，解決 Context Gap）"""
    setup_logging(verbose)
    from humor_bot.data_engine.news_crawler import NewsCrawler

    crawler = NewsCrawler(max_items_per_source=max_items)
    items = crawler.fetch_rss()

    console.print(f"\n📰 新聞素材: {len(items)} 條")
    # 按幽默潛力顯示前 10
    for item in items[:10]:
        console.print(
            f"  [{item.category}] {item.title[:50]}  "
            f"(humor={item.humor_potential:.1f}, fresh={item.freshness_score:.1f})"
        )

    crawler.save_items(items, output)
    console.print(f"\n💾 已儲存: {output}")
    console.print("   下一步: 將素材寫入 ChromaDB → humor-bot rag-ingest " + output)


@app.command()
def analyze_facs(
    video_path: str = typer.Argument(..., help="影片檔案路徑"),
    output: str = typer.Option(None, "--output", "-o"),
    interval: float = typer.Option(1.0, "--interval", help="取樣間隔（秒）"),
    enable_pose: bool = typer.Option(True, "--pose/--no-pose", help="啟用骨架偵測"),
    roi: str = typer.Option(None, "--roi", help="觀眾 ROI (x,y,w,h) 0-1 比例"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """FACS 杜氏笑容與骨架姿勢分析（AU6+AU12 真誠度量化）"""
    setup_logging(verbose)
    import json
    from humor_bot.data_engine.facs_analyzer import FACSAnalyzer

    audience_roi = None
    if roi:
        parts = [float(x) for x in roi.split(",")]
        if len(parts) == 4:
            audience_roi = tuple(parts)

    analyzer = FACSAnalyzer(
        enable_pose=enable_pose,
        sample_interval_sec=interval,
        audience_roi=audience_roi,
    )

    import cv2
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, int(fps * interval))

    results = []
    for idx in range(0, total_frames, frame_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        result = analyzer.analyze_frame(frame, idx / fps)
        results.append(result)

    cap.release()

    console.print(f"\n😊 FACS 分析: {len(results)} 幀")
    duchenne_frames = [r for r in results if r.duchenne_ratio > 0]
    if results:
        avg_sincerity = sum(r.avg_smile_sincerity for r in results) / len(results)
        avg_duchenne = sum(r.duchenne_ratio for r in results) / len(results)
        console.print(f"   杜氏笑容比例: {avg_duchenne:.1%}")
        console.print(f"   平均真誠度: {avg_sincerity:.2f}")
        console.print(f"   肩膀震顫幀: {sum(1 for r in results if r.shaking_ratio > 0)}")

    if output:
        data = [{
            "t": r.timestamp, "duchenne": r.duchenne_ratio,
            "sincerity": r.avg_smile_sincerity, "intensity": r.avg_smile_intensity,
            "shaking": r.shaking_ratio, "composite": r.composite_score,
        } for r in results]
        with open(output, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        console.print(f"💾 已儲存: {output}")


@app.command()
def analyze_envelope(
    audio_path: str = typer.Argument(..., help="音訊檔案路徑"),
    laughter: str = typer.Option(..., "--laughter", "-l", help="笑聲偵測 JSON 路徑"),
    output: str = typer.Option("data/analysis/envelopes.json", "--output", "-o"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """笑聲時序包絡線分析（ADSR + 爆發點 + 餘韻長度）"""
    setup_logging(verbose)
    import json
    from humor_bot.data_engine.laughter_envelope import LaughterEnvelopeAnalyzer

    with open(laughter, "r", encoding="utf-8") as f:
        events = json.load(f)

    analyzer = LaughterEnvelopeAnalyzer()
    results = analyzer.analyze_batch(audio_path, events)

    console.print(f"\n🔊 笑聲包絡線分析: {len(results)} 個事件")
    for r in results[:10]:
        console.print(
            f"  [{r.event_id}] quality={r.comedy_quality_score:.2f} "
            f"type={r.inferred_joke_type} "
            f"attack={r.envelope.attack_duration:.2f}s "
            f"decay={r.envelope.decay_duration:.2f}s "
            f"bursts={r.envelope.num_bursts} ({r.envelope.burst_pattern})"
        )

    analyzer.save_results(results, output)
    console.print(f"💾 已儲存: {output}")


@app.command()
def annotate_web(
    video: str = typer.Argument(..., help="影片檔案路徑"),
    transcript: str = typer.Argument(..., help="逐字稿 JSON 路徑"),
    laughter: str = typer.Option(None, "--laughter", "-l", help="笑聲偵測 JSON"),
    output_dir: str = typer.Option("data/annotations", "--output-dir", "-o"),
    port: int = typer.Option(8501, "--port", "-p"),
):
    """啟動 Web 標註介面（影片截幀 + 逐字稿 + 互動標註）"""
    from humor_bot.annotator.server import create_app

    console.print(f"\n🚀 啟動標註介面...")
    console.print(f"   影片: {video}")
    console.print(f"   逐字稿: {transcript}")
    console.print(f"   👉 http://localhost:{port}\n")

    create_app(
        video_path=video,
        transcript_path=transcript,
        laughter_path=laughter,
        output_dir=output_dir,
        port=port,
    )


@app.command()
def coach(
    setup: str = typer.Argument(..., help="段子的鋪陳 (Setup)"),
    punchline: str = typer.Argument(..., help="段子的笑點 (Punchline)"),
    persona: str = typer.Option("一般觀察者", "--persona", "-p", help="演員的舞台人設 (Persona)"),
    joke_type: str = typer.Option("一般", "--type", "-t", help="段子類型 (如: 短篇故事型, One-liner)"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """
    StandUp Coach 脫口秀教練系統 (互動式)
    
    輸入你自己寫的幽默段子，讓蘊含 2024-2025 SOTA 理論的 AI 教練
    為你鐵口直斷會不會 work、預測爆點以及點出冷場風險！
    """
    setup_logging(verbose)
    from humor_bot.evaluation.coach import StandupCoach
    
    console.print(f"\n[bold blue]🎭 演員人設:[/bold blue] {persona}")
    console.print(f"[bold]   Setup:[/bold] {setup}")
    console.print(f"[bold]   Punchline:[/bold] {punchline}\n")
    
    # 初始化教練模型 (載入 BERT 特徵擷取)
    ai_coach = StandupCoach()
    metrics = ai_coach._compute_metrics(setup, punchline)
    
    console.print(f"📊 [bold]底層特徵解構:[/bold]")
    console.print(f"   語意反轉 (Incongruity): {metrics['incongruity']:.2f}")
    console.print(f"   威脅/禁忌指數 (Violation): {metrics['violation']:.2f}")
    console.print(f"   安全/自嘲指數 (Safety): {metrics['safety']:.2f}")
    console.print(f"   (BVT 理論合規度: {metrics['bvt_product']:.2f})\n")
    
    with console.status("[bold green]🧠 教練正在結合學術理論為你分析段子..."):
        critique = ai_coach.critique(setup, punchline, persona, joke_type)
        
    console.print(f"[bold magenta]🤖 AI 教練的診斷報告:[/bold magenta]")
    console.print(f"{critique}\n")


if __name__ == "__main__":
    app()
