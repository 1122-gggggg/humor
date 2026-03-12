"""
Safety-Humor 權衡特徵標註模組

功能：
- 使用 Llama Guard 進行安全性分類
- 建立 Safety-Humor 權衡特徵向量
- 區分「純粹的惡意冒犯」與「脫口秀式的諷刺（良性衝突）」
- 提供 CLI 互動式人工複核介面
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict, field
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class HumorType(str, Enum):
    """幽默類型分類"""
    SELF_DEPRECATION = "self_deprecation"   # 自嘲
    SATIRE = "satire"                       # 諷刺
    ABSURDIST = "absurdist"                 # 荒謬
    WORDPLAY = "wordplay"                   # 語言遊戲
    OBSERVATIONAL = "observational"         # 觀察式
    ANECDOTAL = "anecdotal"                 # 故事型
    CALLBACK = "callback"                   # 回呼
    DARK = "dark"                           # 黑色幽默
    PHYSICAL = "physical"                   # 肢體/模仿
    OTHER = "other"


class TargetGroup(str, Enum):
    """冒犯對象分類"""
    SELF = "self"                 # 自嘲（最安全）
    UNIVERSAL = "universal"       # 普世主題
    POLITICAL = "political"       # 政治人物/政策
    SOCIAL = "social"             # 社會現象
    CELEBRITY = "celebrity"       # 名人/公眾人物
    CULTURAL = "cultural"         # 文化差異
    DEMOGRAPHIC = "demographic"   # 特定族群（高風險）
    NONE = "none"                 # 無特定對象


@dataclass
class SafetyHumorLabel:
    """Safety-Humor 權衡特徵向量"""
    id: str                              # 對應 SetupPunchline 的 id

    # 良性衝突量表 (Benign Violation Theory)
    benign_violation_score: float = 0.0  # 0=完全無衝突, 1=強烈但良性衝突
    is_benign: bool = True               # 是否為良性（可接受的）

    # 冒犯性維度
    offensive_level: int = 0             # 0-5 等級 (0=無冒犯, 5=嚴重冒犯)
    target_group: str = TargetGroup.NONE
    humor_type: str = HumorType.OTHER

    # Llama Guard 安全分類
    llama_guard_safe: bool = True        # Llama Guard 判定是否安全
    llama_guard_category: str = ""       # 觸發的安全類別

    # 人工複核
    human_reviewed: bool = False
    human_override: bool | None = None   # 人工覆寫安全判定
    reviewer_notes: str = ""

    # 複合特徵
    safety_humor_tradeoff: float = 0.0   # Safety-Humor 權衡分數
    tags: list[str] = field(default_factory=list)


class SafetyLabeler:
    """Safety-Humor 標註器"""

    def __init__(
        self,
        llama_guard_model: str = "meta-llama/Llama-Guard-3-8B",
        device: str = "cuda",
        use_api: bool = False,
        api_backend: str = "openai",
        api_model: str = "gpt-4o",
    ):
        """
        Args:
            llama_guard_model: Llama Guard 模型名稱
            device: 運算裝置
            use_api: 是否使用 API 代替本地模型
            api_backend: API 後端 (openai / anthropic)
            api_model: API 模型名稱
        """
        self.llama_guard_model = llama_guard_model
        self.device = device
        self.use_api = use_api
        self.api_backend = api_backend
        self.api_model = api_model
        self._model = None
        self._tokenizer = None

    def _load_model(self):
        """載入 Llama Guard 模型（本地模式）"""
        if self._model is not None or self.use_api:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        logger.info(f"載入 Llama Guard: {self.llama_guard_model}")
        self._tokenizer = AutoTokenizer.from_pretrained(self.llama_guard_model)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.llama_guard_model,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
        )

    def label(self, joke_id: str, text: str) -> SafetyHumorLabel:
        """
        對一段笑話進行 Safety-Humor 標註

        Args:
            joke_id: 笑話 ID
            text: 笑話全文

        Returns:
            SafetyHumorLabel 標注結果
        """
        # 1. 安全性分類（Llama Guard 或 API）
        safety_result = self._classify_safety(text)

        # 2. 幽默類型分類
        humor_type = self._classify_humor_type(text)

        # 3. 冒犯對象分類
        target_group = self._classify_target(text)

        # 4. 良性衝突評分
        bv_score = self._score_benign_violation(text)

        # 5. 冒犯性等級
        offensive_level = self._score_offensiveness(text, safety_result)

        # 6. 計算 Safety-Humor 權衡分數
        tradeoff = self._compute_tradeoff(bv_score, offensive_level)

        return SafetyHumorLabel(
            id=joke_id,
            benign_violation_score=bv_score,
            is_benign=offensive_level <= 2,
            offensive_level=offensive_level,
            target_group=target_group,
            humor_type=humor_type,
            llama_guard_safe=safety_result["safe"],
            llama_guard_category=safety_result.get("category", ""),
            safety_humor_tradeoff=tradeoff,
        )

    def _classify_safety(self, text: str) -> dict:
        """使用 Llama Guard 分類安全性"""
        if self.use_api:
            return self._classify_safety_api(text)

        self._load_model()

        prompt = f"""[INST] Task: Check if there is unsafe content in the following comedy/stand-up text.
Evaluate this as COMEDY CONTENT - satire and mild social commentary are generally acceptable.

<comedy_text>
{text}
</comedy_text>

Provide your safety assessment. Output 'safe' or 'unsafe' followed by the category if unsafe. [/INST]"""

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self.device)
        import torch
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.1,
            )

        response = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.split("[/INST]")[-1].strip().lower()

        is_safe = "safe" in response and "unsafe" not in response
        category = ""
        if not is_safe:
            parts = response.split("\n")
            if len(parts) > 1:
                category = parts[1].strip()

        return {"safe": is_safe, "category": category}

    def _classify_safety_api(self, text: str) -> dict:
        """使用 API 分類安全性"""
        prompt = f"""你是一位脫口秀安全審核專家。請評估以下脫口秀段子的安全性。

脫口秀的諷刺和社會觀察是可以接受的。
只有真正涉及仇恨、歧視、對特定群體有害的內容才應標為不安全。

段子內容：
{text}

請回覆 JSON 格式：
{{"safe": true/false, "category": "觸發的類別或空字串", "reasoning": "簡短理由"}}"""

        return self._call_llm_api(prompt, parse_json=True, fallback={"safe": True, "category": ""})

    def _classify_humor_type(self, text: str) -> str:
        """分類幽默類型"""
        prompt = f"""分析以下脫口秀段子的幽默類型，從以下選項中選擇最符合的：
- self_deprecation (自嘲)
- satire (諷刺)
- absurdist (荒謬)
- wordplay (語言遊戲)
- observational (觀察式)
- anecdotal (故事型)
- callback (回呼)
- dark (黑色幽默)
- physical (肢體/模仿)
- other

段子：{text}

只回覆類型名稱，不要其他文字。"""

        result = self._call_llm_api(prompt, parse_json=False)
        result = result.strip().lower()

        valid_types = {t.value for t in HumorType}
        if result in valid_types:
            return result
        return HumorType.OTHER

    def _classify_target(self, text: str) -> str:
        """分類冒犯對象"""
        prompt = f"""分析以下脫口秀段子的幽默對象（被調侃的對象），從以下選項中選擇：
- self (自嘲)
- universal (普世主題)
- political (政治人物/政策)
- social (社會現象)
- celebrity (名人/公眾人物)
- cultural (文化差異)
- demographic (特定族群)
- none (無特定對象)

段子：{text}

只回覆對象名稱。"""

        result = self._call_llm_api(prompt, parse_json=False)
        result = result.strip().lower()

        valid_targets = {t.value for t in TargetGroup}
        if result in valid_targets:
            return result
        return TargetGroup.NONE

    def _score_benign_violation(self, text: str) -> float:
        """評估良性衝突分數"""
        prompt = f"""根據 Benign Violation Theory，評估以下脫口秀段子的「良性衝突」程度。

良性衝突 (Benign Violation) 是指一個潛在的威脅/冒犯，但被包裝成安全/可接受的形式。
- 分數 0.0 = 完全無衝突（太無聊）
- 分數 0.5 = 適中的良性衝突（最佳甜蜜點）
- 分數 1.0 = 極強衝突但仍屬良性

段子：{text}

只回覆一個 0.0 到 1.0 之間的數字。"""

        result = self._call_llm_api(prompt, parse_json=False)
        try:
            return float(result.strip())
        except ValueError:
            return 0.3

    def _score_offensiveness(self, text: str, safety_result: dict) -> int:
        """評估冒犯性等級 (0-5)"""
        prompt = f"""評估以下脫口秀段子的冒犯性等級 (0-5)：
0 = 完全無冒犯（適合所有人）
1 = 極輕微（一般社交場合可接受）
2 = 輕微（脫口秀場景可接受）
3 = 中等（成人限定）
4 = 嚴重（可能引起爭議）
5 = 極度冒犯（不應公開表演）

段子：{text}

只回覆一個 0 到 5 之間的整數。"""

        result = self._call_llm_api(prompt, parse_json=False)
        try:
            level = int(result.strip())
            return max(0, min(5, level))
        except ValueError:
            return 2 if not safety_result.get("safe", True) else 1

    def _compute_tradeoff(self, bv_score: float, offensive_level: int) -> float:
        """
        計算 Safety-Humor 權衡分數

        理想的脫口秀段子應該有高 benign_violation 但低 offensive_level。
        tradeoff = benign_violation - penalty(offensive_level)
        """
        # 冒犯性懲罰：指數衰減
        import math
        penalty = 1 - math.exp(-0.5 * offensive_level)

        tradeoff = bv_score * (1 - penalty)
        return round(tradeoff, 3)

    def _call_llm_api(self, prompt: str, parse_json: bool = False, fallback=None):
        """呼叫 LLM API"""
        try:
            if self.api_backend == "openai":
                from openai import OpenAI
                client = OpenAI()
                response = client.chat.completions.create(
                    model=self.api_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=200,
                )
                text = response.choices[0].message.content.strip()
            else:
                raise ValueError(f"未支援的 API 後端: {self.api_backend}")

            if parse_json:
                # 嘗試提取 JSON
                import re
                json_match = re.search(r'\{.*\}', text, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
                return fallback or {}

            return text

        except Exception as e:
            logger.error(f"LLM API 呼叫失敗: {e}")
            if parse_json:
                return fallback or {}
            return ""

    def batch_label(self, jokes: list[dict]) -> list[SafetyHumorLabel]:
        """
        批次標註

        Args:
            jokes: [{"id": str, "text": str}, ...]

        Returns:
            SafetyHumorLabel 列表
        """
        results = []
        for i, joke in enumerate(jokes):
            logger.info(f"標註進度: [{i + 1}/{len(jokes)}] {joke['id']}")
            label = self.label(joke["id"], joke["text"])
            results.append(label)
        return results

    def save_labels(self, labels: list[SafetyHumorLabel], output_path: str | Path) -> Path:
        """儲存標註結果"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = [asdict(label) for label in labels]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"標註結果已儲存: {output_path} ({len(labels)} 筆)")
        return output_path

    @staticmethod
    def load_labels(path: str | Path) -> list[SafetyHumorLabel]:
        """載入標註結果"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [SafetyHumorLabel(**d) for d in data]
