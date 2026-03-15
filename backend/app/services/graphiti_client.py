"""
Graphiti 客户端单例
使用 Gemini LLM + Gemini Embedder + Kuzu 本地图数据库
"""

import asyncio
import os
from typing import Optional

from graphiti_core import Graphiti
from graphiti_core.llm_client.gemini_client import GeminiClient
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.embedder.gemini import GeminiEmbedder, GeminiEmbedderConfig

from ..config import Config
from ..utils.logger import get_logger

logger = get_logger('mirofish.graphiti_client')

# ── 配置默认值 ──────────────────────────────────────────────────────────────
_DEFAULT_LLM_MODEL = 'gemini-2.0-flash'
_DEFAULT_DB_PATH = './data/graphiti_db'
_DEFAULT_EMBEDDING_MODEL = 'gemini-embedding-001'
_EMBEDDING_DIM = 3072

# ── 单例状态 ─────────────────────────────────────────────────────────────────
_graphiti_instance: Optional[Graphiti] = None
_init_lock: asyncio.Lock = asyncio.Lock()


def _get_llm_model() -> str:
    return getattr(Config, 'GRAPHITI_LLM_MODEL', None) or _DEFAULT_LLM_MODEL


def _get_db_path() -> str:
    return getattr(Config, 'GRAPHITI_DB_PATH', None) or _DEFAULT_DB_PATH


async def get_graphiti() -> Graphiti:
    """
    返回已初始化的 Graphiti 单例（懒加载）。

    首次调用时会：
    1. 创建 KuzuDriver（本地图数据库）
    2. 创建 GeminiClient（LLM）
    3. 创建 GeminiEmbedder（向量嵌入）
    4. 实例化并初始化 Graphiti

    后续调用直接返回缓存的实例。
    """
    global _graphiti_instance

    if _graphiti_instance is not None:
        return _graphiti_instance

    async with _init_lock:
        # 双重检查，防止并发初始化
        if _graphiti_instance is not None:
            return _graphiti_instance

        llm_model = _get_llm_model()
        db_path = _get_db_path()

        logger.info(
            f"初始化 Graphiti 客户端: llm_model={llm_model}, db_path={db_path}"
        )

        # 确保数据库目录存在
        os.makedirs(db_path, exist_ok=True)

        # ── KuzuDriver ─────────────────────────────────────────────────────
        # graphiti_core 使用 KuzuDriver 封装本地 Kuzu 图数据库。
        # 根据 graphiti_core 的 API，传入数据库目录路径即可。
        try:
            from graphiti_core.driver.kuzu import KuzuDriver
            driver = KuzuDriver(db_path)
        except ImportError:
            # 如果 graphiti_core 尚未提供 KuzuDriver，使用 URI 方式
            # TODO: 当 graphiti_core 稳定后验证 KuzuDriver 导入路径
            logger.warning(
                "KuzuDriver 导入失败，将尝试通过 URI 'kuzu://<path>' 方式初始化"
            )
            driver = None

        # ── GeminiClient (LLM) ─────────────────────────────────────────────
        gemini_api_key = os.environ.get('GEMINI_API_KEY') or os.environ.get('LLM_API_KEY')
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY 或 LLM_API_KEY 未配置")

        llm_config = LLMConfig(model=llm_model, api_key=gemini_api_key)
        llm_client = GeminiClient(config=llm_config)

        # ── GeminiEmbedder ─────────────────────────────────────────────────
        embedder_config = GeminiEmbedderConfig(
            embedding_model=_DEFAULT_EMBEDDING_MODEL,
            api_key=gemini_api_key,
            embedding_dim=_EMBEDDING_DIM,
        )
        embedder = GeminiEmbedder(config=embedder_config)

        # ── Graphiti 实例化 ────────────────────────────────────────────────
        if driver is not None:
            graphiti = Graphiti(
                driver=driver,
                llm_client=llm_client,
                embedder=embedder,
            )
        else:
            # 回退：使用 URI 字符串（取决于 graphiti_core 版本支持情况）
            # TODO: 验证此 URI 方案是否被 graphiti_core 支持
            graphiti = Graphiti(
                uri=f"kuzu://{os.path.abspath(db_path)}",
                llm_client=llm_client,
                embedder=embedder,
            )

        # 初始化图结构（创建必要的 schema / 索引）
        await graphiti.build_indices_and_constraints()

        _graphiti_instance = graphiti
        logger.info("Graphiti 客户端初始化完成")
        return _graphiti_instance


async def close_graphiti():
    """
    关闭 Graphiti 客户端，释放数据库连接。
    应在应用关闭时调用。
    """
    global _graphiti_instance

    if _graphiti_instance is None:
        return

    async with _init_lock:
        if _graphiti_instance is None:
            return

        try:
            await _graphiti_instance.close()
            logger.info("Graphiti 客户端已关闭")
        except Exception as e:
            logger.error(f"关闭 Graphiti 客户端时发生错误: {e}")
        finally:
            _graphiti_instance = None
