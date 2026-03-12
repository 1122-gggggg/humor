"""數據引擎 — 多模態幽默偵測、標註、分析完整管線"""

try:
    import sys
    # 解決 tensorflow-hub 在高版本缺少 pkg_resources 的問題
    if 'pkg_resources' not in sys.modules:
        import types
        dummy_pkg = types.ModuleType('pkg_resources')
        dummy_pkg.parse_version = lambda v: (999, 999, 999) # 永遠通過版本檢查
        sys.modules['pkg_resources'] = dummy_pkg

    import tensorflow as tf
    # 解決高版本 TF 遺失 register_load_context_function 的問題
    if hasattr(tf, '__internal__') and not hasattr(tf.__internal__, 'register_load_context_function'):
        tf.__internal__.register_load_context_function = lambda *args, **kwargs: None
        
    if hasattr(tf, 'compat') and hasattr(tf.compat, 'v2') and hasattr(tf.compat.v2, '__internal__'):
        if not hasattr(tf.compat.v2.__internal__, 'register_load_context_function'):
            tf.compat.v2.__internal__.register_load_context_function = lambda *args, **kwargs: None
except Exception:
    pass


from humor_bot.data_engine.youtube_downloader import YouTubeDownloader, SubtitleSource
from humor_bot.data_engine.laughter_detector import LaughterDetector
from humor_bot.data_engine.audio_analyzer import AudioAnalyzer
from humor_bot.data_engine.alignment import SetupPunchlineAligner
from humor_bot.data_engine.safety_labeler import SafetyLabeler
from humor_bot.data_engine.text_processor import TextProcessor
from humor_bot.data_engine.video_analyzer import VideoAnalyzer
from humor_bot.data_engine.auto_annotator import AutoAnnotationPipeline
from humor_bot.data_engine.prosody_analyzer import ProsodyAnalyzer
from humor_bot.data_engine.negative_collector import NegativeSampleCollector
from humor_bot.data_engine.performer_analyzer import PerformerAnalyzer
from humor_bot.data_engine.news_crawler import NewsCrawler
from humor_bot.data_engine.facs_analyzer import FACSAnalyzer
from humor_bot.data_engine.laughter_envelope import LaughterEnvelopeAnalyzer

__all__ = [
    "YouTubeDownloader",
    "SubtitleSource",
    "LaughterDetector",
    "AudioAnalyzer",
    "SetupPunchlineAligner",
    "SafetyLabeler",
    "TextProcessor",
    "VideoAnalyzer",
    "AutoAnnotationPipeline",
    "ProsodyAnalyzer",
    "NegativeSampleCollector",
    "PerformerAnalyzer",
    "NewsCrawler",
    "FACSAnalyzer",
    "LaughterEnvelopeAnalyzer",
]
