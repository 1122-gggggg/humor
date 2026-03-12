"""
標註介面後端 — Flask 伺服器

功能：
1. 自動從影片擷取每段逐字稿對應的畫面
2. 載入逐字稿 + 笑聲偵測結果
3. 提供 REST API 給前端標註界面
4. 即時儲存標註進度
5. 匯出 SFT / RM / Gold Set 格式
"""

from __future__ import annotations

import json
import logging
import os
import base64
from pathlib import Path
from dataclasses import dataclass, asdict, field

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SegmentAnnotation:
    """單段標註結果"""
    segment_id: int
    text: str
    start: float
    end: float
    humor_score: int = 0          # 0-5 (0=未標)
    humor_type: str = ""          # sarcasm/self_deprecation/pun/misdirection/...
    is_punchline: bool = False
    is_setup: bool = False
    offensive_level: int = 0      # 0-5
    notes: str = ""
    annotated: bool = False
    # 自動特徵（唯讀）
    laughter_confidence: float = 0.0
    laughter_db: float = -60.0


def create_app(
    video_path: str,
    transcript_path: str,
    laughter_path: str | None = None,
    output_dir: str = "data/annotations",
    port: int = 8501,
):
    """
    建立並啟動標註介面

    Args:
        video_path: 影片檔案路徑
        transcript_path: 逐字稿 JSON 路徑
        laughter_path: 笑聲偵測 JSON 路徑（可選）
        output_dir: 標註結果輸出目錄
        port: 伺服器埠號
    """
    from flask import Flask, jsonify, request, send_file, Response

    app = Flask(__name__, static_folder=None)
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 載入資料
    with open(transcript_path, "r", encoding="utf-8") as f:
        raw_segments = json.load(f)

    laughter_events = []
    if laughter_path and Path(laughter_path).exists():
        with open(laughter_path, "r", encoding="utf-8") as f:
            laughter_events = json.load(f)

    # 建立標註資料
    annotations: list[dict] = []
    progress_file = output_dir / f"{video_path.stem}_progress.json"

    # 嘗試載入先前進度
    if progress_file.exists():
        with open(progress_file, "r", encoding="utf-8") as f:
            annotations = json.load(f)
        logger.info(f"載入先前進度: {len(annotations)} 段")
    else:
        for i, seg in enumerate(raw_segments):
            # 找出對應的笑聲事件
            seg_end = seg.get("end", seg.get("start", 0) + 3)
            matching_laughter = [
                l for l in laughter_events
                if abs(l.get("start", 0) - seg_end) < 5
            ]
            best_confidence = max(
                (l.get("confidence", 0) for l in matching_laughter), default=0
            )
            best_db = max(
                (l.get("rms_db", -60) for l in matching_laughter), default=-60
            )

            annotations.append({
                "segment_id": i,
                "text": seg.get("text", ""),
                "start": seg.get("start", 0),
                "end": seg.get("end", 0),
                "humor_score": 0,
                "humor_type": "",
                "is_punchline": False,
                "is_setup": False,
                "offensive_level": 0,
                "notes": "",
                "annotated": False,
                "laughter_confidence": best_confidence,
                "laughter_db": best_db,
            })

    # 影片資訊
    cap = cv2.VideoCapture(str(video_path))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    video_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = video_total_frames / video_fps if video_fps > 0 else 0
    cap.release()

    # ── API 路由 ──────────────────────────────────────

    @app.route("/")
    def index():
        html_path = Path(__file__).parent / "templates" / "annotator.html"
        return send_file(str(html_path))

    @app.route("/api/info")
    def get_info():
        annotated_count = sum(1 for a in annotations if a["annotated"])
        return jsonify({
            "video_name": video_path.name,
            "total_segments": len(annotations),
            "annotated_count": annotated_count,
            "video_duration": video_duration,
            "video_fps": video_fps,
        })

    @app.route("/api/segments")
    def get_segments():
        return jsonify(annotations)

    @app.route("/api/segment/<int:seg_id>")
    def get_segment(seg_id):
        if 0 <= seg_id < len(annotations):
            return jsonify(annotations[seg_id])
        return jsonify({"error": "Invalid segment ID"}), 404

    @app.route("/api/segment/<int:seg_id>", methods=["PUT"])
    def update_segment(seg_id):
        if 0 <= seg_id < len(annotations):
            data = request.get_json()
            annotations[seg_id].update(data)
            annotations[seg_id]["annotated"] = True
            # 自動儲存
            _save_progress()
            return jsonify({"status": "ok", "segment": annotations[seg_id]})
        return jsonify({"error": "Invalid segment ID"}), 404

    @app.route("/api/frame/<int:seg_id>")
    def get_frame(seg_id):
        """擷取該段對應的影片畫面（base64 JPEG）"""
        if seg_id < 0 or seg_id >= len(annotations):
            return jsonify({"error": "Invalid"}), 404

        seg = annotations[seg_id]
        # 取中間時間點的畫面
        mid_time = (seg["start"] + seg["end"]) / 2

        cap = cv2.VideoCapture(str(video_path))
        frame_idx = int(mid_time * video_fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            return jsonify({"error": "Frame not found"}), 404

        # 縮放到合理大小
        h, w = frame.shape[:2]
        max_w = 640
        if w > max_w:
            scale = max_w / w
            frame = cv2.resize(frame, (max_w, int(h * scale)))

        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        img_b64 = base64.b64encode(buffer).decode("utf-8")

        return jsonify({"image": img_b64, "timestamp": mid_time})

    @app.route("/api/save", methods=["POST"])
    def save_all():
        _save_progress()
        return jsonify({"status": "saved", "path": str(progress_file)})

    @app.route("/api/export/sft", methods=["POST"])
    def export_sft():
        """匯出 SFT 訓練資料（鋪陳-笑點對話對）"""
        sft_data = []
        for i, a in enumerate(annotations):
            if not a["annotated"] or a["humor_score"] < 3:
                continue
            if a["is_punchline"] and i > 0:
                setup_text = annotations[i - 1]["text"]
                punch_text = a["text"]
                sft_data.append({
                    "instruction": "請寫一段脫口秀段子。",
                    "input": f"鋪陳: {setup_text}",
                    "output": f"笑點: {punch_text}",
                    "humor_score": a["humor_score"],
                    "humor_type": a["humor_type"],
                })

        out = output_dir / f"{video_path.stem}_sft.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(sft_data, f, ensure_ascii=False, indent=2)

        return jsonify({"status": "ok", "count": len(sft_data), "path": str(out)})

    @app.route("/api/export/rm", methods=["POST"])
    def export_rm():
        """匯出 Reward Model 成對比較資料"""
        # 按分數排序，取成對
        scored = [a for a in annotations if a["annotated"] and a["humor_score"] > 0]
        scored.sort(key=lambda x: x["humor_score"], reverse=True)

        pairs = []
        for i in range(len(scored)):
            for j in range(i + 1, min(i + 5, len(scored))):
                if scored[i]["humor_score"] > scored[j]["humor_score"]:
                    pairs.append({
                        "chosen": scored[i]["text"],
                        "rejected": scored[j]["text"],
                        "chosen_score": scored[i]["humor_score"],
                        "rejected_score": scored[j]["humor_score"],
                    })

        out = output_dir / f"{video_path.stem}_rm_pairs.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(pairs, f, ensure_ascii=False, indent=2)

        return jsonify({"status": "ok", "count": len(pairs), "path": str(out)})

    @app.route("/api/export/gold", methods=["POST"])
    def export_gold():
        """匯出 Gold Standard 測試集"""
        gold = [a for a in annotations if a["annotated"]]
        out = output_dir / f"{video_path.stem}_gold.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(gold, f, ensure_ascii=False, indent=2)

        return jsonify({"status": "ok", "count": len(gold), "path": str(out)})

    def _save_progress():
        with open(progress_file, "w", encoding="utf-8") as f:
            json.dump(annotations, f, ensure_ascii=False, indent=2)

    # 啟動
    print(f"\n🎬 標註介面已啟動!")
    print(f"   影片: {video_path.name}")
    print(f"   逐字稿: {len(annotations)} 段")
    print(f"   已標註: {sum(1 for a in annotations if a['annotated'])} 段")
    print(f"\n   👉 開啟瀏覽器: http://localhost:{port}")
    print(f"   按 Ctrl+C 停止\n")

    app.run(host="0.0.0.0", port=port, debug=False)
