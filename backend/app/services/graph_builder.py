"""
图谱构建服务
使用 Graphiti + Kuzu + Gemini 构建知识图谱

替换原基于 Zep Cloud 的 GraphBuilderService。

主要变化：
- create_graph()：不调用外部 API，直接生成本地 UUID 作为 group_id
- set_ontology()：Graphiti 自动提取实体，无法像 Zep 那样定义显式 ontology schema。
  本方法将 ontology 存入内存 dict，供后续 add_episode 参考（当前为 no-op，附带日志）。
- add_text_batches()：使用 graphiti.add_episode() 逐块写入，返回 episode name 列表
  （非 UUID，因为 Graphiti add_episode 不返回唯一 ID）。
- _wait_for_episodes()：已移除 —— Graphiti 的 add_episode() 是同步等待的（内部 await），
  写入完成即代表处理完成，无需轮询 processed 状态。
- get_graph_data()：通过 graphiti.search(query="", group_ids=[group_id]) 获取所有边，
  再从边推断节点集合。
- delete_graph()：通过逐条搜索并删除实现（Graphiti 无 bulk delete group API）。
"""

import asyncio
import uuid
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from graphiti_core.nodes import EpisodeType

from .graphiti_client import get_graphiti
from ..models.task import TaskManager, TaskStatus
from .text_processor import TextProcessor
from ..utils.logger import get_logger

logger = get_logger('mirofish.graph_builder')


# ────────────────────────────────────────────────────────────────
# 数据类
# ────────────────────────────────────────────────────────────────

class GraphInfo:
    """图谱信息"""

    def __init__(
        self,
        graph_id: str,
        node_count: int,
        edge_count: int,
        entity_types: List[str],
    ):
        self.graph_id = graph_id
        self.node_count = node_count
        self.edge_count = edge_count
        self.entity_types = entity_types

    def to_dict(self) -> Dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "entity_types": self.entity_types,
        }


# ────────────────────────────────────────────────────────────────
# 辅助函数
# ────────────────────────────────────────────────────────────────

def _run_async(coro):
    """在同步（线程）上下文中运行异步协程"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


# ────────────────────────────────────────────────────────────────
# 主服务类
# ────────────────────────────────────────────────────────────────

class GraphBuilderService:
    """
    图谱构建服务
    负责调用 Graphiti API 构建知识图谱

    公共接口（与原 Zep 版本保持兼容）：
    - build_graph_async(text, ontology, graph_name, chunk_size, chunk_overlap, batch_size) -> str
    - create_graph(name) -> str
    - set_ontology(group_id, ontology)
    - add_text_batches(group_id, chunks, batch_size, progress_callback) -> List[str]
    - delete_graph(group_id)
    - get_graph_data(group_id) -> Dict
    """

    def __init__(self):
        self.task_manager = TaskManager()
        # 存储 ontology 定义（Graphiti 不接受显式 schema，仅保存供参考/日志）
        self._ontology_store: Dict[str, Dict[str, Any]] = {}

    # ──────────────────────────────────────────────────────────────
    # 公共接口：异步构建
    # ──────────────────────────────────────────────────────────────

    def build_graph_async(
        self,
        text: str,
        ontology: Dict[str, Any],
        graph_name: str = "MiroFish Graph",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        batch_size: int = 3,
    ) -> str:
        """
        异步构建图谱（在后台线程中执行）

        Args:
            text: 输入文本
            ontology: 本体定义（来自接口1的输出）
            graph_name: 图谱名称
            chunk_size: 文本块大小
            chunk_overlap: 块重叠大小
            batch_size: 每批发送的块数量

        Returns:
            task_id: 任务 ID，可通过 TaskManager 查询进度
        """
        task_id = self.task_manager.create_task(
            task_type="graph_build",
            metadata={
                "graph_name": graph_name,
                "chunk_size": chunk_size,
                "text_length": len(text),
            },
        )

        thread = threading.Thread(
            target=self._build_graph_worker,
            args=(task_id, text, ontology, graph_name, chunk_size, chunk_overlap, batch_size),
        )
        thread.daemon = True
        thread.start()

        return task_id

    def _build_graph_worker(
        self,
        task_id: str,
        text: str,
        ontology: Dict[str, Any],
        graph_name: str,
        chunk_size: int,
        chunk_overlap: int,
        batch_size: int,
    ):
        """图谱构建工作线程（在后台线程中运行）"""
        try:
            self.task_manager.update_task(
                task_id,
                status=TaskStatus.PROCESSING,
                progress=5,
                message="开始构建图谱...",
            )

            # 1. 创建图谱（生成本地 group_id）
            group_id = self.create_graph(graph_name)
            self.task_manager.update_task(
                task_id,
                progress=10,
                message=f"图谱已创建: {group_id}",
            )

            # 2. 设置本体（Graphiti 中为 no-op，仅记录）
            self.set_ontology(group_id, ontology)
            self.task_manager.update_task(
                task_id,
                progress=15,
                message="本体已记录（Graphiti 自动提取实体）",
            )

            # 3. 文本分块
            chunks = TextProcessor.split_text(text, chunk_size, chunk_overlap)
            total_chunks = len(chunks)
            self.task_manager.update_task(
                task_id,
                progress=20,
                message=f"文本已分割为 {total_chunks} 个块",
            )

            # 4. 分批写入 Graphiti
            # Graphiti add_episode 内部是 await（同步等待处理完成），
            # 无需额外的 _wait_for_episodes 步骤。
            episode_names = self.add_text_batches(
                group_id,
                chunks,
                batch_size,
                lambda msg, prog: self.task_manager.update_task(
                    task_id,
                    progress=20 + int(prog * 0.7),  # 20-90%
                    message=msg,
                ),
            )

            # 5. 获取图谱信息
            self.task_manager.update_task(
                task_id,
                progress=90,
                message="获取图谱信息...",
            )
            graph_info = self._get_graph_info(group_id)

            # 完成
            self.task_manager.complete_task(task_id, {
                "graph_id": group_id,
                "graph_info": graph_info.to_dict(),
                "chunks_processed": total_chunks,
                "episodes_added": len(episode_names),
            })

        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            logger.error(f"图谱构建失败: {error_msg}")
            self.task_manager.fail_task(task_id, error_msg)

    # ──────────────────────────────────────────────────────────────
    # 公共接口：图谱 CRUD
    # ──────────────────────────────────────────────────────────────

    def create_graph(self, name: str) -> str:
        """
        创建图谱

        与 Zep 不同，Graphiti 不需要事先注册 group。
        直接生成一个本地 UUID 作为 group_id，供后续 add_episode 使用。

        Args:
            name: 图谱名称（仅用于日志，不传给 Graphiti）

        Returns:
            group_id: 形如 "mirofish_<16位hex>"
        """
        group_id = f"mirofish_{uuid.uuid4().hex[:16]}"
        logger.info(f"创建图谱 group_id={group_id}, name={name}")
        return group_id

    def set_ontology(self, group_id: str, ontology: Dict[str, Any]):
        """
        设置图谱本体

        Graphiti 通过 LLM 自动从文本中提取实体和关系，不接受显式 ontology schema。
        本方法将 ontology 存储到内存 dict 供调试参考，不调用任何 Graphiti API。

        如需让 Graphiti 提取特定类型实体，可在 add_episode 时传入 entity_types 参数
        （当前实现使用默认的自动提取模式）。

        Args:
            group_id: 图谱 ID
            ontology: 本体定义（entity_types, edge_types 等）
        """
        self._ontology_store[group_id] = ontology
        entity_count = len(ontology.get("entity_types", []))
        edge_count = len(ontology.get("edge_types", []))
        logger.info(
            f"本体已记录（Graphiti 自动提取）: group_id={group_id}, "
            f"entity_types={entity_count}, edge_types={edge_count}"
        )

    def add_text_batches(
        self,
        group_id: str,
        chunks: List[str],
        batch_size: int = 3,
        progress_callback: Optional[Callable] = None,
    ) -> List[str]:
        """
        分批添加文本到 Graphiti，返回所有 episode name 列表

        每个 chunk 调用一次 graphiti.add_episode()。
        Graphiti 的 add_episode() 是异步的，内部会等待 LLM 提取完成，
        因此写入即处理完成，无需额外轮询。

        Args:
            group_id: 图谱 ID（Graphiti group_id）
            chunks: 文本块列表
            batch_size: 每批处理的块数（用于进度回调节奏，实际仍逐块写入）
            progress_callback: 可选进度回调 (message, progress_0_to_1)

        Returns:
            episode_names: 每个 episode 的名称列表（格式："chunk_<index>"）
        """
        episode_names = []
        total_chunks = len(chunks)

        async def _add_all():
            graphiti = await get_graphiti()
            for i, chunk in enumerate(chunks):
                episode_name = f"chunk_{i}"
                batch_num = i // batch_size + 1
                total_batches = (total_chunks + batch_size - 1) // batch_size

                if progress_callback and i % batch_size == 0:
                    progress = (i + 1) / total_chunks
                    progress_callback(
                        f"写入第 {batch_num}/{total_batches} 批数据 (块 {i+1}/{total_chunks})...",
                        progress,
                    )

                try:
                    await graphiti.add_episode(
                        name=episode_name,
                        episode_body=chunk,
                        source=EpisodeType.text,
                        group_id=group_id,
                        reference_time=datetime.now(tz=timezone.utc),
                    )
                    episode_names.append(episode_name)
                    logger.debug(f"Episode 写入成功: {episode_name} (group={group_id})")
                except Exception as e:
                    logger.error(f"Episode 写入失败 ({episode_name}): {e}")
                    if progress_callback:
                        progress_callback(f"块 {i+1} 写入失败: {str(e)}", (i + 1) / total_chunks)
                    raise

            return episode_names

        result = _run_async(_add_all())

        if progress_callback:
            progress_callback(
                f"所有 {len(result)} 个文本块写入完成",
                1.0,
            )

        return result

    def delete_graph(self, group_id: str):
        """
        删除图谱

        Graphiti 无 bulk delete by group API。
        本方法通过搜索该 group 的所有边，然后逐条删除实现。

        注意：Graphiti driver 的删除 API 根据版本可能有所不同，
        此处尝试调用 graphiti.driver.execute_query() 或 graphiti.delete_episode() 等。
        如果 Graphiti 提供了 invalidate_all()，优先使用。

        Args:
            group_id: 图谱 ID
        """
        logger.info(f"删除图谱 group_id={group_id}...")

        async def _delete():
            graphiti = await get_graphiti()

            # 优先尝试 invalidate_all（部分版本支持）
            if hasattr(graphiti, 'invalidate_all'):
                try:
                    await graphiti.invalidate_all(group_id=group_id)
                    logger.info(f"图谱 {group_id} 已通过 invalidate_all 清除")
                    return
                except Exception as e:
                    logger.warning(f"invalidate_all 失败，改用逐条删除: {e}")

            # 降级：搜索并逐条删除 edge
            try:
                edges = await graphiti.search(
                    query="",
                    group_ids=[group_id],
                    num_results=9999,
                )
                deleted_count = 0
                for edge in edges:
                    edge_uuid = str(getattr(edge, 'uuid', None) or "")
                    if not edge_uuid:
                        continue
                    try:
                        # 尝试 delete_edge / remove_edge / driver 直接查询
                        if hasattr(graphiti, 'delete_edge'):
                            await graphiti.delete_edge(uuid=edge_uuid)
                        elif hasattr(graphiti, 'driver'):
                            await graphiti.driver.execute_query(
                                "MATCH ()-[e {uuid: $uuid}]-() DELETE e",
                                {"uuid": edge_uuid},
                            )
                        deleted_count += 1
                    except Exception as edge_err:
                        logger.debug(f"删除边 {edge_uuid} 失败: {edge_err}")

                logger.info(f"图谱 {group_id} 已删除 {deleted_count} 条边")
            except Exception as e:
                logger.error(f"删除图谱 {group_id} 失败: {e}")
                raise

        _run_async(_delete())
        # 清理 ontology 存储
        self._ontology_store.pop(group_id, None)

    def get_graph_data(self, group_id: str) -> Dict[str, Any]:
        """
        获取完整图谱数据（包含详细信息）

        通过 graphiti.search(query="") 获取所有 EntityEdge，
        再从边推断节点集合。

        注意：与 Zep 不同，Graphiti 的节点无独立的 uuid/labels/summary/attributes，
        本方法仅返回从边推断的节点名称集合。

        Args:
            group_id: 图谱 ID

        Returns:
            包含 nodes 和 edges 的字典
        """
        logger.info(f"获取图谱数据: group_id={group_id}...")

        async def _fetch():
            graphiti = await get_graphiti()
            return await graphiti.search(
                query="",
                group_ids=[group_id],
                num_results=9999,
            )

        try:
            raw_edges = _run_async(_fetch())
        except Exception as e:
            logger.error(f"获取图谱数据失败: {e}")
            raw_edges = []

        # 构建节点映射（从边推断）
        node_names: Dict[str, str] = {}  # name -> uuid(空)

        edges_data = []
        for edge in raw_edges:
            src_name = str(getattr(edge, 'source_node_name', '') or "")
            tgt_name = str(getattr(edge, 'target_node_name', '') or "")

            # 节点 uuid 在 Graphiti 中可从 source_node_uuid / source_node_id 取
            src_uuid = str(
                getattr(edge, 'source_node_uuid', None)
                or getattr(edge, 'source_node_id', None)
                or src_name
                or ""
            )
            tgt_uuid = str(
                getattr(edge, 'target_node_uuid', None)
                or getattr(edge, 'target_node_id', None)
                or tgt_name
                or ""
            )

            if src_name:
                node_names[src_name] = src_uuid
            if tgt_name:
                node_names[tgt_name] = tgt_uuid

            def _dt_str(val) -> Optional[str]:
                if val is None:
                    return None
                if isinstance(val, datetime):
                    return val.isoformat()
                return str(val)

            edge_uuid = str(getattr(edge, 'uuid', '') or "")
            edge_name = str(getattr(edge, 'name', '') or "")
            fact = str(getattr(edge, 'fact', '') or "")
            created_at = _dt_str(getattr(edge, 'created_at', None))
            valid_at = _dt_str(getattr(edge, 'valid_at', None))
            invalid_at = _dt_str(getattr(edge, 'invalid_at', None))
            # expired_at 是 Zep 特有字段，Graphiti 无此概念
            expired_at = _dt_str(getattr(edge, 'expired_at', None))

            edges_data.append({
                "uuid": edge_uuid,
                "name": edge_name,
                "fact": fact,
                "fact_type": edge_name,  # 兼容字段
                "source_node_uuid": src_uuid,
                "target_node_uuid": tgt_uuid,
                "source_node_name": src_name,
                "target_node_name": tgt_name,
                "attributes": {},        # Graphiti 不在 edge 上挂载任意属性
                "created_at": created_at,
                "valid_at": valid_at,
                "invalid_at": invalid_at,
                "expired_at": expired_at,
                "episodes": [],          # Graphiti 无 episode 关联字段
            })

        # 构建 nodes_data
        nodes_data = []
        for name, node_uuid in node_names.items():
            nodes_data.append({
                "uuid": node_uuid,
                "name": name,
                "labels": [],       # Graphiti 不提供节点 labels
                "summary": "",      # Graphiti 不提供节点 summary
                "attributes": {},
                "created_at": None,
            })

        return {
            "graph_id": group_id,
            "nodes": nodes_data,
            "edges": edges_data,
            "node_count": len(nodes_data),
            "edge_count": len(edges_data),
        }

    # ──────────────────────────────────────────────────────────────
    # 内部工具
    # ──────────────────────────────────────────────────────────────

    def _get_graph_info(self, group_id: str) -> GraphInfo:
        """获取图谱简要信息"""
        data = self.get_graph_data(group_id)
        return GraphInfo(
            graph_id=group_id,
            node_count=data["node_count"],
            edge_count=data["edge_count"],
            # Graphiti 无 labels，entity_types 为空列表
            entity_types=[],
        )
