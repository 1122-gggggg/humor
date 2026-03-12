import logging
import json
from pathlib import Path

logger = logging.getLogger("humor_bot.coach")

class StandupCoach:
    """
    脫口秀教練系統 (Stand-up Coach System)
    
    結合 BVT 理論、語意漂移演算法 (Semantic Drift) 與 LLM 認知能力，
    幫助脫口秀演員測試自己寫的段子是否具備「幽默的物理量與張力」，
    並預測可能會遇到的冷場點與改進方向。
    """
    
    def __init__(self):
        logger.info("初始化 AI 脫口秀教練...")
        try:
            from sentence_transformers import SentenceTransformer
            from transformers import pipeline
            # 2024 跨語系最強 SOTA Embedding: BGE-M3 (支援100+語言, 8192長度)
            self.encoder = SentenceTransformer("BAAI/bge-m3")
            # 2024 最強多語系 NLI 零樣本分類: mDeBERTa-v3 
            self.bvt_analyzer = pipeline(
                "zero-shot-classification", 
                model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
            )
        except Exception as e:
            logger.warning(f"本地 NLP 依賴載入失敗 (語義漂移與BVT指數可能無法精確計算): {e}")
            self.encoder = None
            self.bvt_analyzer = None

        # 初始化 RAG (檢索增強生成 - 外掛知識庫)
        self.chroma_collection = None
        try:
            import chromadb
            client = chromadb.PersistentClient(path="data/chroma_db")
            self.chroma_collection = client.get_collection(name="comedy_rules")
            logger.info("✅ 成功連線實戰知識庫 (RAG)！")
        except Exception:
            logger.info("⚠️ 尚未載入外部知識庫。若有 PDF 教材，請執行 scripts/build_knowledge_base.py")
            
    def _compute_metrics(self, setup: str, punchline: str) -> dict:
        """計算底層特徵"""
        if not self.encoder or not getattr(self, "bvt_analyzer", None):
            return {"incongruity": 0.0, "violation": 0.0, "safety": 0.0}
            
        import numpy as np
        
        # 1. 計算 Incongruity (語意漂移)
        v_setup = self.encoder.encode(setup)
        v_punch = self.encoder.encode(punchline)
        v_setup = v_setup / (np.linalg.norm(v_setup) + 1e-8)
        v_punch = v_punch / (np.linalg.norm(v_punch) + 1e-8)
        
        cosine_sim = np.dot(v_setup, v_punch)
        incongruity = float(max(0.0, min(1.0, 1.0 - cosine_sim)))
        
        # 2. 計算 Violation & Safety (BVT) - Zero Shot 逼近理論原意
        # 引入核心心法：喜劇源於悲劇 (Tragedy = 痛苦、失敗、不堪)
        full_text = f"{setup} {punchline}"
        candidate_labels = [
            "安全無害 (Safe/Benign)", 
            "冒犯禁忌 (Violation/Threat)",
            "悲劇痛苦 (Tragedy/Misery)"
        ]
        res = self.bvt_analyzer(full_text[:512], candidate_labels)
        
        # 解析 Zero-shot 回傳分數
        scores_dict = dict(zip(res["labels"], res["scores"]))
        safety_score = scores_dict.get("安全無害 (Safe/Benign)", 0.0)
        violation_score = scores_dict.get("冒犯禁忌 (Violation/Threat)", 0.0)
        tragedy_score = scores_dict.get("悲劇痛苦 (Tragedy/Misery)", 0.0)
        
        return {
            "incongruity": incongruity,
            "violation": violation_score,
            "safety": safety_score,
            "tragedy": tragedy_score,
            "bvt_product": violation_score * safety_score
        }

    def critique(self, setup: str, punchline: str, persona: str = "General", joke_type: str = "一般") -> str:
        """綜合診斷與教練回饋"""
        metrics = self._compute_metrics(setup, punchline)
        incongruity = metrics["incongruity"]
        violation = metrics["violation"]
        safety = metrics["safety"]
        tragedy = metrics["tragedy"]
        
        # 針對段子類別，注入編劇理論的約束條件
        type_instruction = ""
        if "故事" in joke_type:
            type_instruction = """
[特定喜劇類型結構：短篇故事型 (Short Story)]
這是一個短篇故事型的段子，請教練特別採用以下「誤會堆疊與放棄」的三階段編劇架構來嚴格檢視：
1. 觸發 (Trigger/產生誤會)：Setup 是否在最短時間內用 5W1H 交代完背景？是否成功留下一個可能引發誤會或錯置的破綻？(這階段可以塞微小的角色設定笑點)
2. 衝突 (Conflict/誤會擴大)：段子核心是否建立了一個「不能被順利解決的困境」？這個誤會是否開始擴大(吵架或荒謬的合作)？
3. 解決 (Resolution/放棄解釋)：Punchline 是否成功透過「翻出另一個層次的邏輯」或是「乾脆放棄解釋的絕望感」來收尾？(絕不能用常理去順解)

[短篇故事型實戰範例 (Few-Shot Examples)]
- 案例一【共享機車】：
  觸發：上班趕時間騎民宅前的共享機車，前面跑出一個阿嬤。(角色形象笑點)
  衝突：阿嬤誤會我是偷車賊，我快遲到無法跟老人家解釋。
  解決：(放棄解釋) 我只好丟下一句「阿嬤對不起啦！」直接騎走。 (荒謬感拉滿)

- 案例二【D罩杯】：
  觸發：在國中教書，有一天走廊兩位女學生衝過來。
  衝突：A女大喊「老師你看B女有D罩杯！」我想說這不該跟我說吧！而且...
  解決：(翻出新邏輯) 妳以為我看不出來嗎。

- 案例三【賣小孩】：
  觸發：以前小孩不乖父母都說「把你賣掉」。有天路上遇到一對父母和小孩。
  衝突：媽媽罵小孩「我要把你賣掉！」我上前問「多少賣？」媽媽嚇到拉著小孩走，但爸爸留在原地，狐疑地盯著我... (空拍)
  解決：(塞笑點在人物形象上) 他想賣！

- 案例四【當流浪漢】：
  觸發：我在家門口被當成流浪漢。
  衝突：跟鄰居和鄰居小孩產生互動，鄰居以為我要傷害他們或要飯，無法解釋清楚自己的身分。
  解決：(順水推舟/放棄解釋) 乾脆偽裝成流浪漢對小孩大喊：「弟弟我要的是錢！」 (衝突極大化)
"""
        elif joke_type != "一般":
            type_instruction = f"\n[段子風格要求] {joke_type}\n請根據這種類型的本質發揮標準教練視角（如 One-liner 講求極致高效反轉；生活觀察型講求高度觀眾共鳴）。\n"
        
        prompt = f"""你現在是一位頂級的脫口秀技術教練（如：Jerry Corley 結合 AI 科學家）。
你的任務是利用學術理論來評估演員寫的新段子，並告訴他上台時「會不會 work」、用到了哪些「底層邏輯」，以及「為什麼可能會冷場 (Bombing)」。

[核心心法提醒]
「喜劇源於悲劇」(Comedy = Tragedy + Time) 是一切的大原則。如果一個段子的 Setup 沒有建立在痛苦、困境、失敗或剝奪感之上，就很難產生真正的喜劇張力。
{type_instruction}

[外掛教材檢索 (RAG)]
以下是從您的喜劇社課教材庫中，由向量引擎比對出最相關的「2條講義節錄」。
這可能是演員目前碰到的盲點，或是他們正在嘗試使用的技巧：
---
{self._retrieve_knowledge(setup + " " + punchline)}
---

[演員設定]
人設 (Persona): {persona}

[輸入段子]
Setup (鋪陳): {setup}
Punchline (笑點): {punchline}

[AI 系統計算出的底層特徵]
- 語意漂移 (Incongruity / 預期反轉): {incongruity:.2f} (0.0=無聊可預測, 接近1.0=極度荒謬/反轉)
- 理論張力 (Violation 威脅風險): {violation:.2f} (大於0.6表示觸及敏感或禁忌)
- 悲劇指數 (Tragedy/Misery 痛點): {tragedy:.2f} (大於0.5表示具備足夠的悲劇建構)
- 安全閥值 (Safety 安全感): {safety:.2f} (大於0.6表示語境是安全的/自嘲的)

請用繁體中文給出以下幾個段落的具體分析：
1. 💡 現場預測：這個段子在現場能引來大笑、會心一笑、還是可能冷場？為什麼？
2. 🎭 悲劇與運作原理：這個段子是否成功建立在「悲劇/痛苦」的基底上？它的反轉標籤是什麼（例如：誤導/反轉、觀察式、地獄梗）？
3. 📉 不響的風險 (Bombing Risk)：這個段子最大的漏洞是什麼？（例如：沒有真正的悲劇內核導致只是在說教、或者預期反轉太弱、不符合該演員的人設等）。
4. 🛠️ 教練建議：(給出2種修改建議)
   a. 文本修改：如何重寫一句 Punchline 讓 Incongruity 衝得更高？
   b. 多模態表演 (TFN張量)：在舞台上，演員應該搭配怎樣的語氣或表情（加強反差）？"""

        try:
            from openai import OpenAI
            client = OpenAI()
            logger.info("思考中... 正在調用教練模型...")
            response = client.chat.completions.create(
                model="gpt-4o",  # 使用更強的模型來做指導
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=800,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"教練模型連線失敗: {e}")
            return "無法連線至 AI 教練，請檢查網路或 API KEY。"

    def _retrieve_knowledge(self, query: str, top_k: int = 2) -> str:
        """透過 RAG 從 ChromaDB 檢索相關教材"""
        if not self.chroma_collection:
            return "(目前系統尚未匯入任何社課教材。)"
            
        try:
            results = self.chroma_collection.query(
                query_texts=[query],
                n_results=top_k
            )
            
            docs = results.get('documents', [[]])[0]
            metas = results.get('metadatas', [[]])[0]
            
            if not docs:
                return "(資料庫中找不到相關教材。)"
                
            snippets = []
            for d, m in zip(docs, metas):
                src = m.get('source', '未知來源')
                snippets.append(f"《{src}》: {d.strip()}")
                
            return "\n\n".join(snippets)
        except Exception as e:
            logger.warning(f"RAG 檢索失敗: {e}")
            return "(知識庫擷取異常。)"
