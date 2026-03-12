"""
GTVH 衝突腳本提取器 (Script Opposition Extractor)

基於 General Theory of Verbal Humor (GTVH) 的六大知識資源 (Knowledge Resources)：
1. Script Opposition (SO) — 腳本對立
2. Logical Mechanism (LM) — 邏輯機制
3. Situation (SI) — 情境
4. Target (TA) — 對象
5. Narrative Strategy (NS) — 敘事策略
6. Language (LA) — 語言

使用 LLM 進行 zero-shot 結構化提取。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ScriptOpposition:
    """腳本對立結構"""
    script_a: str              # 腳本 A（表面意義/預期）
    script_b: str              # 腳本 B（翻轉意義/實際）
    opposition_type: str       # 對立類型 (real/unreal, actual/non-actual, normal/abnormal...)
    overlap_point: str         # 兩個腳本的重疊點（歧義/雙關觸發點）


@dataclass
class GTVHAnalysis:
    """GTVH 完整分析結果"""
    id: str
    text: str

    # 六大知識資源
    script_opposition: ScriptOpposition | None = None
    logical_mechanism: str = ""       # 幽默的邏輯機制 (exaggeration, analogy, reversal...)
    situation: str = ""               # 情境描述
    target: str = ""                  # 對象
    narrative_strategy: str = ""      # 敘事策略 (simple narrative, dialogue, riddle...)
    language: str = ""                # 語言手法 (pun, irony, register shift...)

    # 衍生特徵
    incongruity_score: float = 0.0    # 不協調程度 (0-1)
    resolution_score: float = 0.0     # 解決清晰度 (0-1)
    surprise_score: float = 0.0       # 驚喜程度 (0-1)

    # 原始 LLM 回應
    raw_response: str = ""
    tags: list[str] = field(default_factory=list)


GTVH_EXTRACTION_PROMPT = """你是幽默理論專家，精通 General Theory of Verbal Humor (GTVH)。
請分析以下脫口秀段子的幽默結構。

## 段子內容
{text}

## 請提供以下 GTVH 分析（JSON 格式）

```json
{{
  "script_opposition": {{
    "script_a": "表面意義/觀眾的預期是什麼",
    "script_b": "翻轉後的實際意義是什麼",
    "opposition_type": "對立類型，例如: real/unreal, expected/unexpected, normal/abnormal, literal/figurative",
    "overlap_point": "兩個腳本的重疊點，也就是觸發翻轉的關鍵詞或句子"
  }},
  "logical_mechanism": "幽默的邏輯機制，例如: exaggeration, analogy, reversal, false_analogy, juxtaposition",
  "situation": "段子描述的情境",
  "target": "被幽默的對象（自己/某群體/某現象）",
  "narrative_strategy": "敘事策略，例如: simple_narrative, dialogue, expository, riddle",
  "language": "語言手法，例如: pun, irony, sarcasm, register_shift, understatement, hyperbole",
  "incongruity_score": 0.0到1.0的數字,
  "resolution_score": 0.0到1.0的數字,
  "surprise_score": 0.0到1.0的數字
}}
```

請嚴格只回覆 JSON，不要回覆其他文字。"""


class ScriptExtractor:
    """GTVH 衝突腳本提取器"""

    def __init__(
        self,
        llm_backend: str = "openai",
        model_name: str = "gpt-4o",
        temperature: float = 0.3,
    ):
        self.llm_backend = llm_backend
        self.model_name = model_name
        self.temperature = temperature

    def analyze(self, joke_id: str, text: str) -> GTVHAnalysis:
        """
        對一段笑話進行 GTVH 分析

        Args:
            joke_id: 笑話 ID
            text: 笑話全文

        Returns:
            GTVHAnalysis 分析結果
        """
        prompt = GTVH_EXTRACTION_PROMPT.format(text=text)
        raw_response = self._call_llm(prompt)

        try:
            # 提取 JSON
            import re
            json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = {}
        except json.JSONDecodeError:
            logger.warning(f"JSON 解析失敗: {joke_id}")
            data = {}

        # 解析 Script Opposition
        so_data = data.get("script_opposition", {})
        script_opposition = None
        if so_data and so_data.get("script_a"):
            script_opposition = ScriptOpposition(
                script_a=so_data.get("script_a", ""),
                script_b=so_data.get("script_b", ""),
                opposition_type=so_data.get("opposition_type", ""),
                overlap_point=so_data.get("overlap_point", ""),
            )

        return GTVHAnalysis(
            id=joke_id,
            text=text,
            script_opposition=script_opposition,
            logical_mechanism=data.get("logical_mechanism", ""),
            situation=data.get("situation", ""),
            target=data.get("target", ""),
            narrative_strategy=data.get("narrative_strategy", ""),
            language=data.get("language", ""),
            incongruity_score=float(data.get("incongruity_score", 0)),
            resolution_score=float(data.get("resolution_score", 0)),
            surprise_score=float(data.get("surprise_score", 0)),
            raw_response=raw_response,
        )

    def batch_analyze(self, jokes: list[dict]) -> list[GTVHAnalysis]:
        """
        批次分析

        Args:
            jokes: [{"id": str, "text": str}, ...]
        """
        results = []
        for i, joke in enumerate(jokes):
            logger.info(f"GTVH 分析: [{i + 1}/{len(jokes)}] {joke['id']}")
            analysis = self.analyze(joke["id"], joke["text"])
            results.append(analysis)
        return results

    def _call_llm(self, prompt: str) -> str:
        """呼叫 LLM"""
        if self.llm_backend == "openai":
            from openai import OpenAI

            client = OpenAI()
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=1000,
            )
            return response.choices[0].message.content.strip()

        elif self.llm_backend == "local":
            # 本地模型推論（透過 vLLM 或 transformers）
            raise NotImplementedError("本地模型推論尚未實作")

        else:
            raise ValueError(f"未支援的 LLM 後端: {self.llm_backend}")

    def save_analyses(self, analyses: list[GTVHAnalysis], output_path: str | Path) -> Path:
        """儲存分析結果"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = []
        for a in analyses:
            d = asdict(a)
            # ScriptOpposition 已在 asdict 中自動序列化
            data.append(d)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"GTVH 分析已儲存: {output_path} ({len(analyses)} 筆)")
        return output_path
