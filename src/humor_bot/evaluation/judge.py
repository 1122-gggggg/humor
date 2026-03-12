"""
LLM-as-a-Judge 多維度評分系統

功能：
- Persona Fidelity (30%): 幽默風格一致性
- Humor Mechanics (25%): 翻轉（The Twist）的驚喜程度
- Safety Sensitivity (25%): 良性衝突控制
- Language Quality (20%): 語言流暢度

協調子評分器並彙整加權總分。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class JudgeScore:
    """單一維度評分"""
    dimension: str           # 評分維度名稱
    score: float             # 分數 (0-10)
    weight: float            # 權重
    weighted_score: float    # 加權分數
    reasoning: str           # 評分理由


@dataclass
class JudgeResult:
    """完整評分結果"""
    id: str
    text: str
    scores: list[JudgeScore]
    total_score: float       # 加權總分 (0-10)
    overall_feedback: str    # 整體回饋
    raw_responses: dict = field(default_factory=dict)


# ── 評分 Prompt 模板 ──────────────────────────────────────────

PERSONA_FIDELITY_PROMPT = """你是脫口秀風格評審專家。請評估以下段子的 **風格一致性 (Persona Fidelity)**。

評估標準：
1. 用語風格是否一致（口語化、節奏感）
2. 主題選擇是否符合設定的 persona
3. 幽默方法是否有辨識度（自嘲/諷刺/觀察式等）
4. 整體調性是否連貫

段子：
{text}

{persona_description}

請回覆 JSON:
{{"score": 0到10的分數, "reasoning": "詳細評分理由"}}"""


HUMOR_MECHANICS_PROMPT = """你是幽默結構分析專家。請評估以下段子的 **幽默機制 (Humor Mechanics)**。

評估標準：
1. **Setup 鋪陳**：是否有效建立觀眾的預期？
2. **Punchline 翻轉**：The Twist 的驚喜程度如何？是否出人意料？
3. **結構完整性**：Setup 到 Punchline 的過渡是否自然？
4. **Callback 使用**：是否有效使用回呼/伏筆？
5. **節奏感 (Timing)**：段子的節奏是否得當？

段子：
{text}

請回覆 JSON:
{{"score": 0到10的分數, "reasoning": "詳細評分理由"}}"""


SAFETY_SENSITIVITY_PROMPT = """你是脫口秀安全審核專家。請評估以下段子的 **安全敏感度 (Safety Sensitivity)**。

評估標準：
1. 是否屬於「良性衝突 (Benign Violation)」的範圍？
2. 冒犯性內容是否有被巧妙包裝？
3. 是否存在對特定族群的歧視或仇恨？
4. 在公開表演場合中是否適當？

注意：脫口秀本來就會挑戰邊界，適當的社會諷刺不應被過度懲罰。
高分 = 安全但不無趣，低分 = 具有傷害性或太過保守無聊。

段子：
{text}

請回覆 JSON:
{{"score": 0到10的分數, "reasoning": "詳細評分理由"}}"""


LANGUAGE_QUALITY_PROMPT = """你是語言品質評審。請評估以下段子的 **語言品質 (Language Quality)**。

評估標準：
1. 語法正確性
2. 用詞精準度
3. 語感自然度（是否像真人說話）
4. 繁體中文的慣用表達
5. 整體流暢度

段子：
{text}

請回覆 JSON:
{{"score": 0到10的分數, "reasoning": "詳細評分理由"}}"""


class HumorJudge:
    """LLM-as-a-Judge 多維度評分器"""

    # 默認權重配置
    DEFAULT_WEIGHTS = {
        "persona_fidelity": 0.30,
        "humor_mechanics": 0.25,
        "safety_sensitivity": 0.25,
        "language_quality": 0.20,
    }

    def __init__(
        self,
        llm_backend: str = "openai",
        model_name: str = "gpt-4o",
        temperature: float = 0.1,
        weights: dict[str, float] | None = None,
        persona_description: str = "",
    ):
        self.llm_backend = llm_backend
        self.model_name = model_name
        self.temperature = temperature
        self.weights = weights or self.DEFAULT_WEIGHTS
        self.persona_description = persona_description or "目標風格：台灣脫口秀風格，融合自嘲與社會觀察。"

    def judge(self, joke_id: str, text: str) -> JudgeResult:
        """
        對段子進行多維度評分

        Args:
            joke_id: 段子 ID
            text: 段子全文

        Returns:
            JudgeResult 多維度評分結果
        """
        scores = []
        raw_responses = {}

        # 1. Persona Fidelity
        pf_score = self._score_dimension(
            "persona_fidelity",
            PERSONA_FIDELITY_PROMPT.format(
                text=text,
                persona_description=f"Persona 描述：{self.persona_description}",
            ),
        )
        scores.append(pf_score)
        raw_responses["persona_fidelity"] = pf_score.reasoning

        # 2. Humor Mechanics
        hm_score = self._score_dimension(
            "humor_mechanics",
            HUMOR_MECHANICS_PROMPT.format(text=text),
        )
        scores.append(hm_score)
        raw_responses["humor_mechanics"] = hm_score.reasoning

        # 3. Safety Sensitivity
        ss_score = self._score_dimension(
            "safety_sensitivity",
            SAFETY_SENSITIVITY_PROMPT.format(text=text),
        )
        scores.append(ss_score)
        raw_responses["safety_sensitivity"] = ss_score.reasoning

        # 4. Language Quality
        lq_score = self._score_dimension(
            "language_quality",
            LANGUAGE_QUALITY_PROMPT.format(text=text),
        )
        scores.append(lq_score)
        raw_responses["language_quality"] = lq_score.reasoning

        # 加權總分
        total_score = sum(s.weighted_score for s in scores)

        # 整體回饋
        overall = self._generate_overall_feedback(text, scores, total_score)

        return JudgeResult(
            id=joke_id,
            text=text,
            scores=scores,
            total_score=round(total_score, 2),
            overall_feedback=overall,
            raw_responses=raw_responses,
        )

    def _score_dimension(self, dimension: str, prompt: str) -> JudgeScore:
        """對單一維度評分"""
        weight = self.weights.get(dimension, 0.25)

        try:
            response = self._call_llm(prompt)

            # 解析 JSON
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                score = float(data.get("score", 5))
                score = max(0, min(10, score))
                reasoning = data.get("reasoning", "")
            else:
                score = 5.0
                reasoning = response
        except Exception as e:
            logger.warning(f"評分失敗 ({dimension}): {e}")
            score = 5.0
            reasoning = f"評分過程出錯: {str(e)}"

        return JudgeScore(
            dimension=dimension,
            score=score,
            weight=weight,
            weighted_score=round(score * weight, 2),
            reasoning=reasoning,
        )

    def _generate_overall_feedback(
        self,
        text: str,
        scores: list[JudgeScore],
        total_score: float,
    ) -> str:
        """生成整體回饋"""
        score_summary = "\n".join([
            f"- {s.dimension}: {s.score}/10 ({s.reasoning[:100]})"
            for s in scores
        ])

        prompt = f"""基於以下段子的多維度評分結果，請給出簡要的整體回饋與改進建議。

段子：{text}

評分結果（加權總分: {total_score:.1f}/10）：
{score_summary}

請用 2-3 句話概述這段笑話的優缺點和改進方向。"""

        try:
            return self._call_llm(prompt)
        except Exception:
            return f"加權總分: {total_score:.1f}/10"

    def batch_judge(self, jokes: list[dict]) -> list[JudgeResult]:
        """批次評分"""
        results = []
        for i, joke in enumerate(jokes):
            logger.info(f"評分進度: [{i + 1}/{len(jokes)}] {joke.get('id', i)}")
            result = self.judge(joke.get("id", str(i)), joke["text"])
            results.append(result)
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
                max_tokens=500,
            )
            return response.choices[0].message.content.strip()
        else:
            raise ValueError(f"未支援的 LLM 後端: {self.llm_backend}")

    def save_results(self, results: list[JudgeResult], output_path: str | Path) -> Path:
        """儲存評分結果"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = []
        for r in results:
            d = asdict(r)
            data.append(d)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"評分結果已儲存: {output_path} ({len(results)} 筆)")
        return output_path
