# 幽默脫口秀 AI 機器人 🎭

> 多模態幽默偵測與生成系統 — 從 YouTube 脫口秀影片中自動學習幽默。

## 系統架構

```
YouTube 影片 → 音訊、視覺提取 → YAMNet 笑聲偵測 → Setup-Punchline 與 T/A/V 特徵對齊
                                                        ↓
Multimodal Humor Classifier (T, A, V Fusion) ← 結構化多語言/多模態資料集
                        ↓
                    段子生成 ← JokeWriter (Llama3+LoRA) 
                        ↓
                Expert Reward Model (RoBERTa-Large) → PPO/DPO 強化學習
```

## 快速開始

### 1. 安裝

```bash
# 建立虛擬環境
python -m venv .venv
.venv\Scripts\activate  # Windows

# 安裝專案
pip install -e ".[dev]"
```

### 2. 處理 YouTube 影片

```bash
# 單一影片
humor-bot download --url "https://youtube.com/watch?v=VIDEO_ID"

# 批次處理（URL 清單）
humor-bot download --url-list urls.txt

# 端到端管線
humor-bot process-pipeline urls.txt --output data/processed
```

### 3. 笑聲偵測

```bash
humor-bot detect-laughter data/raw/audio/VIDEO_ID.wav --threshold 0.8
```

### 4. 生成段子

```bash
humor-bot generate "台灣的交通" --num 3 --temperature 0.8
```

### 5. 評估段子

```bash
humor-bot evaluate data/processed/dataset.json
```

## 專案結構

```
src/humor_bot/
├── data_engine/          # Phase 1：數據引擎
│   ├── youtube_downloader.py   # YouTube 下載 + Whisper 轉錄
│   ├── laughter_detector.py    # YAMNet 笑聲偵測
│   ├── audio_analyzer.py       # 音訊特徵分析
│   ├── alignment.py            # Setup-Punchline 對齊
│   └── safety_labeler.py       # Safety-Humor 標註
├── models/               # Phase 2：模型架構
│   ├── multimodal_classifier.py # 多模態幽默序列分類 (StandUp4AI 架構)
│   ├── joke_writer.py          # Llama 3 + LoRA SFT
│   ├── script_extractor.py     # GTVH 衝突腳本提取
│   └── rag_retriever.py        # RAG 新聞素材庫
├── training/             # Phase 3：訓練策略
│   ├── reward_model.py         # RoBERTa-Large 獎勵模型
│   ├── ppo_trainer.py          # PPO 強化學習
│   └── dpo_trainer.py          # DPO 直接偏好優化
└── evaluation/           # Phase 4：評估系統
    └── judge.py                # LLM-as-a-Judge
```

## 核心模型

| 層級 | 模型 | 用途 |
|------|------|------|
| 音訊感知 | YAMNet | 笑聲爆發點/掌聲偵測 (黃金標準資料集構建) |
| 多模態融合 | Multimodal Classifier | 處理 (T, A, V) 特徵與序列分類 (加權交叉熵損失) |
| 語言生成 | Llama 3-8B + LoRA | JokeWriter SFT (跨語言結構幽默轉移) |
| 安全過濾 | Llama Guard 3-8B | 良性衝突判定 |
| 獎勵模型 | RoBERTa-Large | 幽默分數預測 R(x,y) |
| 語音轉文字 | Faster-Whisper (large-v3) | 中英逐字稿 |

## License

MIT
