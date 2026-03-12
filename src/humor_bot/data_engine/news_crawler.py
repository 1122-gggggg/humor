"""
即時新聞爬取模組 — 解決語境斷層 (Context Gap)

功能：
1. 新聞爬取
   - RSS Feed 聚合（各大中文新聞源）
   - 社群熱點偵測（PTT、Dcard 熱門文章）
   - Google Trends 即時趨勢

2. 素材處理
   - 自動提取「可諷刺化」的新聞要素
   - 標註新聞的情緒調性（正/負/爭議）
   - 計算時效性分數（越新越有價值）

3. RAG 整合
   - 將時事素材自動寫入 ChromaDB
   - 按主題分類存儲
   - 提供 JokeWriter 即時語境

核心洞見：
- 幽默往往是「悲劇 + 時間」
- AI 必須知道「當下的悲劇」才能進行翻轉
- 過時的梗 = 不好笑 → 時效性是關鍵權重
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class NewsItem:
    """新聞素材"""
    id: str
    title: str
    summary: str
    source: str                   # 來源（自由時報、聯合新聞、PTT...）
    url: str = ""
    published: str = ""           # ISO 8601 時間
    category: str = ""            # 政治/社會/娛樂/科技/國際
    sentiment: str = ""           # positive / negative / controversial / neutral
    satirizable_elements: list[str] = field(default_factory=list)
    freshness_score: float = 1.0  # 時效性 (0-1, 越新越高)
    humor_potential: float = 0.0  # 幽默潛力 (0-1)
    tags: list[str] = field(default_factory=list)


class NewsCrawler:
    """新聞爬取器"""

    # 中文新聞 RSS 來源
    DEFAULT_RSS_FEEDS = {
        "自由時報": "https://news.ltn.com.tw/rss/all.xml",
        "中央社": "https://www.cna.com.tw/rss/aall.xml",
        "聯合新聞": "https://udn.com/rssfeed/news/2/6644",
        "ETtoday": "https://feeds.feedburner.com/ettoday/focus",
    }

    # 時效性衰減
    FRESHNESS_HALFLIFE_DAYS = 3  # 3 天後時效性減半

    def __init__(
        self,
        rss_feeds: dict[str, str] | None = None,
        max_items_per_source: int = 20,
        freshness_halflife_days: int = 3,
    ):
        """
        Args:
            rss_feeds: RSS 來源 {名稱: URL}
            max_items_per_source: 每個來源最多抓取條數
            freshness_halflife_days: 時效性半衰期（天）
        """
        self.rss_feeds = rss_feeds or self.DEFAULT_RSS_FEEDS
        self.max_items_per_source = max_items_per_source
        self.freshness_halflife_days = freshness_halflife_days

    def fetch_rss(self) -> list[NewsItem]:
        """
        從 RSS 來源抓取新聞

        Returns:
            NewsItem 列表
        """
        try:
            import feedparser
        except ImportError:
            logger.warning("feedparser 未安裝。安裝: pip install feedparser")
            return []

        all_items = []

        for source_name, feed_url in self.rss_feeds.items():
            try:
                logger.info(f"📰 抓取 RSS: {source_name}")
                feed = feedparser.parse(feed_url)

                for entry in feed.entries[:self.max_items_per_source]:
                    # 時間解析
                    published = ""
                    if hasattr(entry, "published"):
                        published = entry.published
                    elif hasattr(entry, "updated"):
                        published = entry.updated

                    # 摘要
                    summary = ""
                    if hasattr(entry, "summary"):
                        summary = re.sub(r'<[^>]+>', '', entry.summary)
                    elif hasattr(entry, "description"):
                        summary = re.sub(r'<[^>]+>', '', entry.description)

                    item = NewsItem(
                        id=f"{source_name}_{hash(entry.get('link', '')) % 100000:05d}",
                        title=entry.get("title", ""),
                        summary=summary[:500],
                        source=source_name,
                        url=entry.get("link", ""),
                        published=published,
                    )

                    # 計算時效性
                    item.freshness_score = self._compute_freshness(published)

                    # 分類
                    item.category = self._classify_category(item.title, item.summary)

                    # 情緒分析
                    item.sentiment = self._classify_sentiment(item.title, item.summary)

                    # 幽默潛力
                    item.humor_potential = self._assess_humor_potential(item)

                    # 提取可諷刺化元素
                    item.satirizable_elements = self._extract_satirizable(item)

                    all_items.append(item)

            except Exception as e:
                logger.warning(f"RSS 抓取失敗: {source_name} — {e}")

        # 按時效性 × 幽默潛力排序
        all_items.sort(
            key=lambda x: x.freshness_score * x.humor_potential,
            reverse=True,
        )

        logger.info(f"📰 共抓取 {len(all_items)} 條新聞")
        return all_items

    def _compute_freshness(self, published_str: str) -> float:
        """計算時效性分數（指數衰減）"""
        if not published_str:
            return 0.5  # 無時間 → 中等

        try:
            from email.utils import parsedate_to_datetime
            pub_time = parsedate_to_datetime(published_str)
            now = datetime.now(pub_time.tzinfo) if pub_time.tzinfo else datetime.now()
            age_days = (now - pub_time).total_seconds() / 86400
            # 指數衰減: freshness = 2^(-age/halflife)
            return float(2 ** (-age_days / self.freshness_halflife_days))
        except Exception:
            return 0.5

    def _classify_category(self, title: str, summary: str) -> str:
        """簡單的新聞分類"""
        text = title + summary
        categories = {
            "政治": ["立法院", "總統", "行政院", "選舉", "民進黨", "國民黨", "柯文哲", "政策"],
            "社會": ["警方", "車禍", "詐騙", "法院", "火災", "意外", "犯罪"],
            "娛樂": ["藝人", "演唱會", "電影", "金曲", "網紅", "直播"],
            "科技": ["AI", "手機", "台積電", "科技", "iPhone", "半導體"],
            "國際": ["美國", "中國", "日本", "戰爭", "G7", "聯合國"],
            "生活": ["天氣", "美食", "旅遊", "房價", "交通", "物價"],
        }

        for cat, keywords in categories.items():
            if any(kw in text for kw in keywords):
                return cat

        return "other"

    def _classify_sentiment(self, title: str, summary: str) -> str:
        """情緒分類"""
        text = title + summary

        negative_words = ["死亡", "車禍", "詐騙", "暴力", "崩盤", "悲劇", "災害"]
        controversial_words = ["爭議", "批評", "炎上", "怒批", "荒謬", "離譜", "扯"]
        positive_words = ["喜訊", "突破", "奪冠", "好消息", "感動"]

        neg_count = sum(1 for w in negative_words if w in text)
        con_count = sum(1 for w in controversial_words if w in text)
        pos_count = sum(1 for w in positive_words if w in text)

        if con_count > 0:
            return "controversial"
        if neg_count > pos_count:
            return "negative"
        if pos_count > neg_count:
            return "positive"
        return "neutral"

    def _assess_humor_potential(self, item: NewsItem) -> float:
        """
        評估新聞的幽默潛力

        高潛力指標：
        - 爭議性事件（最適合諷刺）
        - 政治/社會類（觀眾共鳴度高）
        - 荒謬性事件（天然的笑點）
        - 名人/公眾人物相關
        """
        score = 0.3  # 基準

        if item.sentiment == "controversial":
            score += 0.3
        elif item.sentiment == "negative":
            score += 0.1

        if item.category in ("政治", "社會"):
            score += 0.2
        elif item.category == "娛樂":
            score += 0.1

        # 荒謬性關鍵詞
        absurd_words = ["離譜", "荒謬", "扯", "不可思議", "傻眼", "瞎"]
        if any(w in item.title + item.summary for w in absurd_words):
            score += 0.2

        return min(score, 1.0)

    def _extract_satirizable(self, item: NewsItem) -> list[str]:
        """提取可諷刺化的元素"""
        elements = []
        text = item.title + " " + item.summary

        # 數字（政策金額、統計等）
        numbers = re.findall(r'\d+[萬億兆%]', text)
        if numbers:
            elements.append(f"數字: {', '.join(numbers[:3])}")

        # 矛盾語句
        contradictions = ["但是", "然而", "不過", "卻"]
        for c in contradictions:
            if c in text:
                elements.append("存在轉折/矛盾")
                break

        # 引述
        quotes = re.findall(r'「([^」]+)」', text)
        if quotes:
            elements.append(f"引述: {quotes[0][:30]}")

        return elements

    def save_items(self, items: list[NewsItem], output_path: str | Path) -> Path:
        """儲存新聞素材"""
        from dataclasses import asdict
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = [asdict(item) for item in items]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"新聞素材已儲存: {output_path} ({len(items)} 筆)")
        return output_path
