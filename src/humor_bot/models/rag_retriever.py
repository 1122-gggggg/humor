"""
RAG 素材庫 — 檢索增強生成

功能：
- 使用 ChromaDB 儲存新聞/社會事件素材
- 支援從 RSS 或新聞 API 更新素材
- 結合語意相似度 + 時效性權重進行檢索
- 為 JokeWriter 提供最新素材上下文
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Material:
    """素材資料"""
    id: str
    title: str
    content: str
    source: str                  # 來源 (rss, api, manual)
    category: str                # 分類 (political, social, entertainment, tech...)
    published_at: str            # 發布時間 ISO 格式
    url: str = ""
    tags: list[str] | None = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []


@dataclass
class RetrievalResult:
    """檢索結果"""
    material: Material
    semantic_score: float        # 語意相似度分數
    freshness_score: float       # 時效性分數
    combined_score: float        # 綜合分數


class RAGRetriever:
    """RAG 素材庫檢索器"""

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        collection_name: str = "humor_materials",
        persist_directory: str = "data/chromadb",
        top_k: int = 5,
        freshness_weight: float = 0.3,
    ):
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.top_k = top_k
        self.freshness_weight = freshness_weight

        self._client = None
        self._collection = None
        self._embedder = None

    def _init_db(self):
        """初始化 ChromaDB"""
        if self._client is not None:
            return

        import chromadb
        from chromadb.config import Settings

        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"ChromaDB 已初始化: {self.collection_name} "
            f"({self._collection.count()} 筆素材)"
        )

    def _init_embedder(self):
        """初始化句子嵌入模型"""
        if self._embedder is not None:
            return

        from sentence_transformers import SentenceTransformer
        self._embedder = SentenceTransformer(self.embedding_model)
        logger.info(f"嵌入模型已載入: {self.embedding_model}")

    def add_material(self, material: Material):
        """新增單一素材"""
        self._init_db()
        self._init_embedder()

        # 組合文本用於嵌入
        combined_text = f"{material.title}\n{material.content}"
        embedding = self._embedder.encode(combined_text).tolist()

        self._collection.upsert(
            ids=[material.id],
            embeddings=[embedding],
            documents=[combined_text],
            metadatas=[{
                "title": material.title,
                "source": material.source,
                "category": material.category,
                "published_at": material.published_at,
                "url": material.url,
                "tags": json.dumps(material.tags or [], ensure_ascii=False),
            }],
        )

    def add_materials(self, materials: list[Material]):
        """批次新增素材"""
        self._init_db()
        self._init_embedder()

        if not materials:
            return

        combined_texts = [f"{m.title}\n{m.content}" for m in materials]
        embeddings = self._embedder.encode(combined_texts).tolist()

        self._collection.upsert(
            ids=[m.id for m in materials],
            embeddings=embeddings,
            documents=combined_texts,
            metadatas=[{
                "title": m.title,
                "source": m.source,
                "category": m.category,
                "published_at": m.published_at,
                "url": m.url,
                "tags": json.dumps(m.tags or [], ensure_ascii=False),
            } for m in materials],
        )
        logger.info(f"已新增 {len(materials)} 筆素材")

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        category_filter: str | None = None,
    ) -> list[RetrievalResult]:
        """
        檢索相關素材

        Args:
            query: 查詢文本（主題/情境）
            top_k: 返回數量
            category_filter: 分類過濾

        Returns:
            RetrievalResult 列表，按綜合分數降序排列
        """
        self._init_db()
        self._init_embedder()

        top_k = top_k or self.top_k

        # 構建過濾條件
        where_filter = None
        if category_filter:
            where_filter = {"category": category_filter}

        # 嵌入查詢
        query_embedding = self._embedder.encode(query).tolist()

        # ChromaDB 查詢（取更多結果用於重排序）
        fetch_k = min(top_k * 3, self._collection.count() or top_k)
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=fetch_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        if not results["ids"][0]:
            return []

        # 構建結果並計算綜合分數
        retrieval_results = []
        now = datetime.now()

        for i, doc_id in enumerate(results["ids"][0]):
            metadata = results["metadatas"][0][i]
            distance = results["distances"][0][i]

            # 語意分數（cosine distance → similarity）
            semantic_score = 1.0 - distance

            # 時效性分數
            freshness_score = self._compute_freshness(
                metadata.get("published_at", ""), now
            )

            # 綜合分數 = (1-w) * semantic + w * freshness
            combined_score = (
                (1 - self.freshness_weight) * semantic_score
                + self.freshness_weight * freshness_score
            )

            material = Material(
                id=doc_id,
                title=metadata.get("title", ""),
                content=results["documents"][0][i],
                source=metadata.get("source", ""),
                category=metadata.get("category", ""),
                published_at=metadata.get("published_at", ""),
                url=metadata.get("url", ""),
                tags=json.loads(metadata.get("tags", "[]")),
            )

            retrieval_results.append(RetrievalResult(
                material=material,
                semantic_score=semantic_score,
                freshness_score=freshness_score,
                combined_score=combined_score,
            ))

        # 按綜合分數降序排列
        retrieval_results.sort(key=lambda r: r.combined_score, reverse=True)
        return retrieval_results[:top_k]

    def _compute_freshness(self, published_at: str, now: datetime) -> float:
        """計算時效性分數（指數衰減）"""
        if not published_at:
            return 0.5  # 無日期的給中等分數

        try:
            pub_date = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
            pub_date = pub_date.replace(tzinfo=None)
            days_old = (now - pub_date).days
        except (ValueError, TypeError):
            return 0.5

        import math
        # 指數衰減：半衰期 7 天
        freshness = math.exp(-0.1 * days_old)
        return max(0.0, min(1.0, freshness))

    def format_context(self, results: list[RetrievalResult], max_chars: int = 2000) -> str:
        """
        將檢索結果格式化為 LLM 上下文

        Args:
            results: 檢索結果列表
            max_chars: 最大字數限制

        Returns:
            格式化的上下文字串
        """
        if not results:
            return ""

        context_parts = ["## 相關素材\n"]
        char_count = 0

        for i, r in enumerate(results, 1):
            entry = (
                f"### {i}. {r.material.title}\n"
                f"- 來源: {r.material.source} | "
                f"時間: {r.material.published_at[:10] if r.material.published_at else '未知'}\n"
                f"- 內容: {r.material.content[:300]}\n\n"
            )

            if char_count + len(entry) > max_chars:
                break

            context_parts.append(entry)
            char_count += len(entry)

        return "".join(context_parts)

    def get_stats(self) -> dict:
        """取得素材庫統計"""
        self._init_db()
        count = self._collection.count()
        return {
            "collection_name": self.collection_name,
            "total_materials": count,
            "persist_directory": self.persist_directory,
        }
