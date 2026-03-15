"""
Graphiti client singleton
Uses Gemini LLM + Gemini Embedder + Kuzu local graph database
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

_DEFAULT_LLM_MODEL = 'gemini-2.5-flash-lite'
_DEFAULT_DB_PATH = './data/graphiti_db'
_DEFAULT_EMBEDDING_MODEL = 'gemini-embedding-001'
_EMBEDDING_DIM = 3072

_graphiti_instance: Optional[Graphiti] = None
_init_lock: asyncio.Lock = asyncio.Lock()


def _get_llm_model() -> str:
    return getattr(Config, 'GRAPHITI_LLM_MODEL', None) or _DEFAULT_LLM_MODEL


def _get_db_path() -> str:
    return getattr(Config, 'GRAPHITI_DB_PATH', None) or _DEFAULT_DB_PATH


async def get_graphiti() -> Graphiti:
    """
    Returns the initialized Graphiti singleton (lazy loading).

    On first call:
    1. Creates KuzuDriver (local graph database)
    2. Creates GeminiClient (LLM)
    3. Creates GeminiEmbedder (vector embeddings)
    4. Instantiates and initializes Graphiti

    Subsequent calls return the cached instance.
    """
    global _graphiti_instance

    if _graphiti_instance is not None:
        return _graphiti_instance

    async with _init_lock:
        # Double-check to prevent concurrent initialization
        if _graphiti_instance is not None:
            return _graphiti_instance

        llm_model = _get_llm_model()
        db_path = _get_db_path()

        logger.info(f"Initializing Graphiti client: llm_model={llm_model}, db_path={db_path}")

        os.makedirs(db_path, exist_ok=True)

        try:
            from graphiti_core.driver.kuzu import KuzuDriver
            driver = KuzuDriver(db_path)
        except ImportError:
            logger.warning("KuzuDriver import failed, falling back to URI mode")
            driver = None

        gemini_api_key = os.environ.get('GEMINI_API_KEY') or os.environ.get('LLM_API_KEY')
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY or LLM_API_KEY not configured")

        llm_config = LLMConfig(model=llm_model, api_key=gemini_api_key)
        llm_client = GeminiClient(config=llm_config)

        embedder_config = GeminiEmbedderConfig(
            embedding_model=_DEFAULT_EMBEDDING_MODEL,
            api_key=gemini_api_key,
            embedding_dim=_EMBEDDING_DIM,
        )
        embedder = GeminiEmbedder(config=embedder_config)

        if driver is not None:
            graphiti = Graphiti(
                driver=driver,
                llm_client=llm_client,
                embedder=embedder,
            )
        else:
            graphiti = Graphiti(
                uri=f"kuzu://{os.path.abspath(db_path)}",
                llm_client=llm_client,
                embedder=embedder,
            )

        await graphiti.build_indices_and_constraints()

        _graphiti_instance = graphiti
        logger.info("Graphiti client initialized successfully")
        return _graphiti_instance


async def close_graphiti():
    """
    Closes the Graphiti client and releases the database connection.
    Should be called when the application shuts down.
    """
    global _graphiti_instance

    if _graphiti_instance is None:
        return

    async with _init_lock:
        if _graphiti_instance is None:
            return

        try:
            await _graphiti_instance.close()
            logger.info("Graphiti client closed")
        except Exception as e:
            logger.error(f"Error closing Graphiti client: {e}")
        finally:
            _graphiti_instance = None
