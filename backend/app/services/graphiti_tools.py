"""
Graphiti 检索工具服务
封装图谱搜索、节点读取、边查询等工具，供 Report Agent 使用

替换 zep_tools.py 的 ZepToolsService，改用 Graphiti + Kuzu + Gemini。

核心检索工具：
1. InsightForge（深度洞察检索）- 最强大的混合检索，自动生成子问题并多维度检索
2. PanoramaSearch（广度搜索）- 获取全貌
   注意：Graphiti 不区分"过期/失效"边的概念，所有 search() 结果均为当前知识图谱
   中存储的 EntityEdge。本实现使用多个子查询模拟广度，并在 edges 结构中保留
   valid_at / invalid_at 字段（Graphiti EntityEdge 原生支持）。
3. QuickSearch（简单搜索）- 快速检索

API 差异说明（与 Zep 相比）：
- Graphiti 没有独立的 graph_id 概念，使用 group_id（字符串）来隔离多租户数据。
  本模块对外仍使用参数名 graph_id 以保持接口兼容，内部映射为 group_id。
- Graphiti 没有分页获取所有节点/边的 API，get_all_nodes/get_all_edges 通过
  search(query="", num_results=9999) 近似实现。
- Graphiti 的 EntityEdge 包含 .fact、.name、.source_node_name、.target_node_name
  等字段，直接可用。
- Graphiti 不提供 get_node_detail(uuid) 这样的单节点查询 API；实体信息从
  search() 返回的 EntityEdge 中的 source/target node name 推断。
"""

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .graphiti_client import get_graphiti
from ..utils.logger import get_logger
from ..utils.llm_client import LLMClient

logger = get_logger('mirofish.graphiti_tools')


# ────────────────────────────────────────────────────────────────
# 数据类定义（与 zep_tools.py 保持相同的公共接口）
# ────────────────────────────────────────────────────────────────

@dataclass
class SearchResult:
    """搜索结果"""
    facts: List[str]
    edges: List[Dict[str, Any]]
    nodes: List[Dict[str, Any]]
    query: str
    total_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "facts": self.facts,
            "edges": self.edges,
            "nodes": self.nodes,
            "query": self.query,
            "total_count": self.total_count,
        }

    def to_text(self) -> str:
        """转换为文本格式，供 LLM 理解"""
        text_parts = [f"搜索查询: {self.query}", f"找到 {self.total_count} 条相关信息"]
        if self.facts:
            text_parts.append("\n### 相关事实:")
            for i, fact in enumerate(self.facts, 1):
                text_parts.append(f"{i}. {fact}")
        return "\n".join(text_parts)


@dataclass
class NodeInfo:
    """节点信息（从 EntityEdge 推断，不含 uuid）"""
    uuid: str      # Graphiti 无直接节点 uuid API，置空或使用 name 作为标识
    name: str
    labels: List[str]
    summary: str
    attributes: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uuid": self.uuid,
            "name": self.name,
            "labels": self.labels,
            "summary": self.summary,
            "attributes": self.attributes,
        }

    def to_text(self) -> str:
        entity_type = next((l for l in self.labels if l not in ["Entity", "Node"]), "未知类型")
        return f"实体: {self.name} (类型: {entity_type})\n摘要: {self.summary}"


@dataclass
class EdgeInfo:
    """边信息"""
    uuid: str
    name: str
    fact: str
    source_node_uuid: str
    target_node_uuid: str
    source_node_name: Optional[str] = None
    target_node_name: Optional[str] = None
    created_at: Optional[str] = None
    valid_at: Optional[str] = None
    invalid_at: Optional[str] = None
    expired_at: Optional[str] = None  # Graphiti 无此字段，保留兼容

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uuid": self.uuid,
            "name": self.name,
            "fact": self.fact,
            "source_node_uuid": self.source_node_uuid,
            "target_node_uuid": self.target_node_uuid,
            "source_node_name": self.source_node_name,
            "target_node_name": self.target_node_name,
            "created_at": self.created_at,
            "valid_at": self.valid_at,
            "invalid_at": self.invalid_at,
            "expired_at": self.expired_at,
        }

    def to_text(self, include_temporal: bool = False) -> str:
        source = self.source_node_name or (self.source_node_uuid[:8] if self.source_node_uuid else "?")
        target = self.target_node_name or (self.target_node_uuid[:8] if self.target_node_uuid else "?")
        base_text = f"关系: {source} --[{self.name}]--> {target}\n事实: {self.fact}"
        if include_temporal and self.valid_at:
            invalid_at = self.invalid_at or "至今"
            base_text += f"\n时效: {self.valid_at} - {invalid_at}"
        return base_text

    @property
    def is_expired(self) -> bool:
        return self.expired_at is not None

    @property
    def is_invalid(self) -> bool:
        return self.invalid_at is not None


@dataclass
class InsightForgeResult:
    """深度洞察检索结果 (InsightForge)"""
    query: str
    simulation_requirement: str
    sub_queries: List[str]

    semantic_facts: List[str] = field(default_factory=list)
    entity_insights: List[Dict[str, Any]] = field(default_factory=list)
    relationship_chains: List[str] = field(default_factory=list)

    total_facts: int = 0
    total_entities: int = 0
    total_relationships: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "simulation_requirement": self.simulation_requirement,
            "sub_queries": self.sub_queries,
            "semantic_facts": self.semantic_facts,
            "entity_insights": self.entity_insights,
            "relationship_chains": self.relationship_chains,
            "total_facts": self.total_facts,
            "total_entities": self.total_entities,
            "total_relationships": self.total_relationships,
        }

    def to_text(self) -> str:
        text_parts = [
            f"## 未来预测深度分析",
            f"分析问题: {self.query}",
            f"预测场景: {self.simulation_requirement}",
            f"\n### 预测数据统计",
            f"- 相关预测事实: {self.total_facts}条",
            f"- 涉及实体: {self.total_entities}个",
            f"- 关系链: {self.total_relationships}条",
        ]
        if self.sub_queries:
            text_parts.append(f"\n### 分析的子问题")
            for i, sq in enumerate(self.sub_queries, 1):
                text_parts.append(f"{i}. {sq}")
        if self.semantic_facts:
            text_parts.append(f"\n### 【关键事实】(请在报告中引用这些原文)")
            for i, fact in enumerate(self.semantic_facts, 1):
                text_parts.append(f'{i}. "{fact}"')
        if self.entity_insights:
            text_parts.append(f"\n### 【核心实体】")
            for entity in self.entity_insights:
                text_parts.append(f"- **{entity.get('name', '未知')}** ({entity.get('type', '实体')})")
                if entity.get('summary'):
                    text_parts.append(f"  摘要: \"{entity.get('summary')}\"")
                if entity.get('related_facts'):
                    text_parts.append(f"  相关事实: {len(entity.get('related_facts', []))}条")
        if self.relationship_chains:
            text_parts.append(f"\n### 【关系链】")
            for chain in self.relationship_chains:
                text_parts.append(f"- {chain}")
        return "\n".join(text_parts)


@dataclass
class PanoramaResult:
    """广度搜索结果 (Panorama)"""
    query: str

    all_nodes: List[NodeInfo] = field(default_factory=list)
    all_edges: List[EdgeInfo] = field(default_factory=list)
    active_facts: List[str] = field(default_factory=list)
    # 注意：Graphiti 不区分过期/失效边，historical_facts 通过 invalid_at 字段判断
    historical_facts: List[str] = field(default_factory=list)

    total_nodes: int = 0
    total_edges: int = 0
    active_count: int = 0
    historical_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "all_nodes": [n.to_dict() for n in self.all_nodes],
            "all_edges": [e.to_dict() for e in self.all_edges],
            "active_facts": self.active_facts,
            "historical_facts": self.historical_facts,
            "total_nodes": self.total_nodes,
            "total_edges": self.total_edges,
            "active_count": self.active_count,
            "historical_count": self.historical_count,
        }

    def to_text(self) -> str:
        text_parts = [
            f"## 广度搜索结果（未来全景视图）",
            f"查询: {self.query}",
            f"\n### 统计信息",
            f"- 总节点数: {self.total_nodes}",
            f"- 总边数: {self.total_edges}",
            f"- 当前有效事实: {self.active_count}条",
            f"- 历史/过期事实: {self.historical_count}条",
        ]
        if self.active_facts:
            text_parts.append(f"\n### 【当前有效事实】(模拟结果原文)")
            for i, fact in enumerate(self.active_facts, 1):
                text_parts.append(f'{i}. "{fact}"')
        if self.historical_facts:
            text_parts.append(f"\n### 【历史/过期事实】(演变过程记录)")
            for i, fact in enumerate(self.historical_facts, 1):
                text_parts.append(f'{i}. "{fact}"')
        if self.all_nodes:
            text_parts.append(f"\n### 【涉及实体】")
            for node in self.all_nodes:
                entity_type = next((l for l in node.labels if l not in ["Entity", "Node"]), "实体")
                text_parts.append(f"- **{node.name}** ({entity_type})")
        return "\n".join(text_parts)


@dataclass
class AgentInterview:
    """单个 Agent 的采访结果"""
    agent_name: str
    agent_role: str
    agent_bio: str
    question: str
    response: str
    key_quotes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "agent_role": self.agent_role,
            "agent_bio": self.agent_bio,
            "question": self.question,
            "response": self.response,
            "key_quotes": self.key_quotes,
        }

    def to_text(self) -> str:
        text = f"**{self.agent_name}** ({self.agent_role})\n"
        text += f"_简介: {self.agent_bio}_\n\n"
        text += f"**Q:** {self.question}\n\n"
        text += f"**A:** {self.response}\n"
        if self.key_quotes:
            text += "\n**关键引言:**\n"
            for quote in self.key_quotes:
                clean_quote = quote.replace('\u201c', '').replace('\u201d', '').replace('"', '')
                clean_quote = clean_quote.replace('\u300c', '').replace('\u300d', '')
                clean_quote = clean_quote.strip()
                while clean_quote and clean_quote[0] in '，,；;：:、。！？\n\r\t ':
                    clean_quote = clean_quote[1:]
                skip = False
                for d in '123456789':
                    if f'\u95ee\u9898{d}' in clean_quote:
                        skip = True
                        break
                if skip:
                    continue
                if len(clean_quote) > 150:
                    dot_pos = clean_quote.find('\u3002', 80)
                    if dot_pos > 0:
                        clean_quote = clean_quote[:dot_pos + 1]
                    else:
                        clean_quote = clean_quote[:147] + "..."
                if clean_quote and len(clean_quote) >= 10:
                    text += f'> "{clean_quote}"\n'
        return text


@dataclass
class InterviewResult:
    """采访结果 (Interview)"""
    interview_topic: str
    interview_questions: List[str]

    selected_agents: List[Dict[str, Any]] = field(default_factory=list)
    interviews: List[AgentInterview] = field(default_factory=list)

    selection_reasoning: str = ""
    summary: str = ""

    total_agents: int = 0
    interviewed_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "interview_topic": self.interview_topic,
            "interview_questions": self.interview_questions,
            "selected_agents": self.selected_agents,
            "interviews": [i.to_dict() for i in self.interviews],
            "selection_reasoning": self.selection_reasoning,
            "summary": self.summary,
            "total_agents": self.total_agents,
            "interviewed_count": self.interviewed_count,
        }

    def to_text(self) -> str:
        text_parts = [
            "## 深度采访报告",
            f"**采访主题:** {self.interview_topic}",
            f"**采访人数:** {self.interviewed_count} / {self.total_agents} 位模拟Agent",
            "\n### 采访对象选择理由",
            self.selection_reasoning or "（自动选择）",
            "\n---",
            "\n### 采访实录",
        ]
        if self.interviews:
            for i, interview in enumerate(self.interviews, 1):
                text_parts.append(f"\n#### 采访 #{i}: {interview.agent_name}")
                text_parts.append(interview.to_text())
                text_parts.append("\n---")
        else:
            text_parts.append("（无采访记录）\n\n---")
        text_parts.append("\n### 采访摘要与核心观点")
        text_parts.append(self.summary or "（无摘要）")
        return "\n".join(text_parts)


# ────────────────────────────────────────────────────────────────
# 辅助函数：将 EntityEdge 转换为内部数据类
# ────────────────────────────────────────────────────────────────

def _edge_to_edge_info(edge) -> EdgeInfo:
    """将 graphiti_core EntityEdge 对象转换为 EdgeInfo"""
    uuid = str(getattr(edge, 'uuid', None) or "")
    name = str(getattr(edge, 'name', None) or "")
    fact = str(getattr(edge, 'fact', None) or "")
    source_node_name = str(getattr(edge, 'source_node_name', None) or "")
    target_node_name = str(getattr(edge, 'target_node_name', None) or "")

    # Graphiti 的 EntityEdge 使用 source_node_uuid / target_node_uuid
    # 也可能叫 source_node_id / target_node_id，兼容两种写法
    source_node_uuid = str(
        getattr(edge, 'source_node_uuid', None)
        or getattr(edge, 'source_node_id', None)
        or source_node_name  # 降级：用名字替代 uuid
        or ""
    )
    target_node_uuid = str(
        getattr(edge, 'target_node_uuid', None)
        or getattr(edge, 'target_node_id', None)
        or target_node_name
        or ""
    )

    # 时间字段（Graphiti EntityEdge 原生支持）
    def _dt_str(val) -> Optional[str]:
        if val is None:
            return None
        if isinstance(val, datetime):
            return val.isoformat()
        return str(val)

    created_at = _dt_str(getattr(edge, 'created_at', None))
    valid_at = _dt_str(getattr(edge, 'valid_at', None))
    invalid_at = _dt_str(getattr(edge, 'invalid_at', None))
    # expired_at 是 Zep 特有的，Graphiti 无此字段
    expired_at = _dt_str(getattr(edge, 'expired_at', None))

    return EdgeInfo(
        uuid=uuid,
        name=name,
        fact=fact,
        source_node_uuid=source_node_uuid,
        target_node_uuid=target_node_uuid,
        source_node_name=source_node_name or None,
        target_node_name=target_node_name or None,
        created_at=created_at,
        valid_at=valid_at,
        invalid_at=invalid_at,
        expired_at=expired_at,
    )


def _run_async(coro):
    """在同步上下文中运行异步协程（兼容 Flask 等非 async 框架）"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # 已有事件循环（如在 Jupyter 或 async 框架中）
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

class GraphitiToolsService:
    """
    Graphiti 检索工具服务（对应原 ZepToolsService）

    所有 graph_id 参数在内部映射为 Graphiti 的 group_id，
    以保持与原 ZepToolsService 接口的完全兼容。

    核心检索工具：
    1. insight_forge       - 深度洞察检索
    2. panorama_search     - 广度搜索
    3. quick_search        - 快速检索
    4. interview_agents    - 深度采访（调用 OASIS 真实 API）

    基础工具：
    - search_graph         - 图谱语义搜索
    - get_all_nodes        - 获取图谱所有节点（近似实现，通过 edges 推断）
    - get_all_edges        - 获取图谱所有边
    - get_entities_by_type - 按类型获取实体
    - get_entity_summary   - 获取实体的关系摘要
    - get_graph_statistics - 获取图谱统计信息
    - get_simulation_context - 获取模拟上下文
    """

    MAX_RETRIES = 3
    RETRY_DELAY = 2.0

    def __init__(self, llm_client: Optional[LLMClient] = None):
        # Graphiti 通过 graphiti_client.get_graphiti() 懒加载，无需在构造时初始化
        self._llm_client = llm_client
        logger.info("GraphitiToolsService 初始化完成")

    @property
    def llm(self) -> LLMClient:
        """延迟初始化 LLM 客户端"""
        if self._llm_client is None:
            self._llm_client = LLMClient()
        return self._llm_client

    # ──────────────────────────────────────────────────────────────
    # 内部辅助方法
    # ──────────────────────────────────────────────────────────────

    async def _search_async(
        self,
        group_id: str,
        query: str,
        num_results: int = 10,
    ) -> List:
        """异步调用 graphiti.search()，返回 EntityEdge 列表"""
        graphiti = await get_graphiti()
        try:
            results = await graphiti.search(
                query=query,
                group_ids=[group_id],
                num_results=num_results,
            )
            return results if results else []
        except Exception as e:
            logger.warning(f"Graphiti search 失败 (query={query[:40]}...): {e}")
            return []

    def _search(self, group_id: str, query: str, num_results: int = 10) -> List:
        """同步包装 _search_async"""
        return _run_async(self._search_async(group_id, query, num_results))

    # ──────────────────────────────────────────────────────────────
    # 基础工具
    # ──────────────────────────────────────────────────────────────

    def search_graph(
        self,
        graph_id: str,
        query: str,
        limit: int = 10,
        scope: str = "edges",  # 保留参数，Graphiti 默认搜索 edges
    ) -> SearchResult:
        """
        图谱语义搜索

        Graphiti 的 search() 返回 EntityEdge 列表，每条 edge 包含 .fact 字段。
        scope 参数保留兼容性，Graphiti 不区分 edges/nodes 搜索。

        Args:
            graph_id: 图谱 ID（内部映射为 group_id）
            query: 搜索查询
            limit: 返回结果数量
            scope: 搜索范围（兼容参数，Graphiti 仅搜索 edges）

        Returns:
            SearchResult: 搜索结果
        """
        group_id = graph_id
        logger.info(f"图谱搜索: group_id={group_id}, query={query[:50]}...")

        raw_edges = self._search(group_id, query, num_results=limit)

        facts = []
        edges = []
        nodes = []
        seen_facts = set()
        seen_nodes = set()

        for edge in raw_edges:
            fact = str(getattr(edge, 'fact', None) or "")
            if fact and fact not in seen_facts:
                facts.append(fact)
                seen_facts.add(fact)

            edge_info = _edge_to_edge_info(edge)
            edges.append({
                "uuid": edge_info.uuid,
                "name": edge_info.name,
                "fact": edge_info.fact,
                "source_node_uuid": edge_info.source_node_uuid,
                "target_node_uuid": edge_info.target_node_uuid,
                "source_node_name": edge_info.source_node_name,
                "target_node_name": edge_info.target_node_name,
            })

            # 从 edge 推断节点（Graphiti 无独立节点 API）
            for node_name in [edge_info.source_node_name, edge_info.target_node_name]:
                if node_name and node_name not in seen_nodes:
                    seen_nodes.add(node_name)
                    nodes.append({
                        "uuid": "",
                        "name": node_name,
                        "labels": [],
                        "summary": "",
                    })

        logger.info(f"搜索完成: 找到 {len(facts)} 条相关事实")
        return SearchResult(
            facts=facts,
            edges=edges,
            nodes=nodes,
            query=query,
            total_count=len(facts),
        )

    def get_all_edges(
        self,
        graph_id: str,
        include_temporal: bool = True,
    ) -> List[EdgeInfo]:
        """
        获取图谱的所有边

        注意：Graphiti 没有枚举所有边的专用 API。
        本方法通过 search(query="") 近似实现，返回所有存储的 EntityEdge。
        如果 num_results 上限不足，可能未能返回全部边。

        Args:
            graph_id: 图谱 ID（内部映射为 group_id）
            include_temporal: 保留兼容参数，始终包含时间信息

        Returns:
            边列表
        """
        group_id = graph_id
        logger.info(f"获取图谱 {group_id} 的所有边（通过空查询近似）...")

        raw_edges = self._search(group_id, query="", num_results=9999)

        result = []
        for edge in raw_edges:
            result.append(_edge_to_edge_info(edge))

        logger.info(f"获取到 {len(result)} 条边")
        return result

    def get_all_nodes(self, graph_id: str) -> List[NodeInfo]:
        """
        获取图谱的所有节点（从边中推断）

        Graphiti 不提供独立的节点列表 API。
        本方法通过获取所有边，然后从 source_node_name / target_node_name 去重得到节点集合。

        Args:
            graph_id: 图谱 ID（内部映射为 group_id）

        Returns:
            节点列表（uuid 为空，name 为节点名）
        """
        logger.info(f"获取图谱 {graph_id} 的所有节点（从边推断）...")

        all_edges = self.get_all_edges(graph_id)

        seen = set()
        result = []
        for edge in all_edges:
            for name in [edge.source_node_name, edge.target_node_name]:
                if name and name not in seen:
                    seen.add(name)
                    result.append(NodeInfo(
                        uuid="",
                        name=name,
                        labels=[],
                        summary="",
                        attributes={},
                    ))

        logger.info(f"推断到 {len(result)} 个节点")
        return result

    def get_entities_by_type(
        self,
        graph_id: str,
        entity_type: str,
    ) -> List[NodeInfo]:
        """
        按类型获取实体

        注意：Graphiti 不在 EntityEdge 中携带实体的类型标签（labels）。
        本方法通过搜索 entity_type 关键词，将结果中出现的节点名视为该类型实体。

        Args:
            graph_id: 图谱 ID
            entity_type: 实体类型关键词

        Returns:
            符合类型的实体列表
        """
        logger.info(f"获取类型为 {entity_type} 的实体（通过关键词搜索近似）...")

        raw_edges = self._search(graph_id, query=entity_type, num_results=100)

        seen = set()
        result = []
        for edge in raw_edges:
            for name in [
                str(getattr(edge, 'source_node_name', '') or ""),
                str(getattr(edge, 'target_node_name', '') or ""),
            ]:
                if name and name not in seen:
                    seen.add(name)
                    result.append(NodeInfo(
                        uuid="",
                        name=name,
                        labels=[entity_type],
                        summary="",
                        attributes={},
                    ))

        logger.info(f"找到 {len(result)} 个与 {entity_type} 相关的实体")
        return result

    def get_entity_summary(
        self,
        graph_id: str,
        entity_name: str,
    ) -> Dict[str, Any]:
        """
        获取指定实体的关系摘要

        Args:
            graph_id: 图谱 ID
            entity_name: 实体名称

        Returns:
            实体摘要信息
        """
        logger.info(f"获取实体 {entity_name} 的关系摘要...")

        search_result = self.search_graph(graph_id=graph_id, query=entity_name, limit=20)

        # 从 edges 中找与实体直接相关的边
        related_edges = []
        for edge_dict in search_result.edges:
            src = edge_dict.get('source_node_name', '') or ""
            tgt = edge_dict.get('target_node_name', '') or ""
            if entity_name.lower() in src.lower() or entity_name.lower() in tgt.lower():
                related_edges.append(edge_dict)

        entity_node = None
        for node_dict in search_result.nodes:
            if (node_dict.get('name') or "").lower() == entity_name.lower():
                entity_node = node_dict
                break

        return {
            "entity_name": entity_name,
            "entity_info": entity_node,
            "related_facts": search_result.facts,
            "related_edges": related_edges,
            "total_relations": len(related_edges),
        }

    def get_graph_statistics(self, graph_id: str) -> Dict[str, Any]:
        """
        获取图谱的统计信息

        注意：Graphiti 无节点类型标签，entity_types 和 relation_types
        从边的 name 字段统计关系类型分布。

        Args:
            graph_id: 图谱 ID

        Returns:
            统计信息
        """
        logger.info(f"获取图谱 {graph_id} 的统计信息...")

        all_edges = self.get_all_edges(graph_id)

        # 从边推断节点集合
        node_names = set()
        relation_types: Dict[str, int] = {}

        for edge in all_edges:
            if edge.source_node_name:
                node_names.add(edge.source_node_name)
            if edge.target_node_name:
                node_names.add(edge.target_node_name)
            if edge.name:
                relation_types[edge.name] = relation_types.get(edge.name, 0) + 1

        return {
            "graph_id": graph_id,
            "total_nodes": len(node_names),
            "total_edges": len(all_edges),
            # Graphiti 无类型标签，entity_types 为空（保留字段兼容）
            "entity_types": {},
            "relation_types": relation_types,
        }

    def get_simulation_context(
        self,
        graph_id: str,
        simulation_requirement: str,
        limit: int = 30,
    ) -> Dict[str, Any]:
        """
        获取模拟相关的上下文信息

        Args:
            graph_id: 图谱 ID
            simulation_requirement: 模拟需求描述
            limit: 每类信息的数量限制

        Returns:
            模拟上下文信息
        """
        logger.info(f"获取模拟上下文: {simulation_requirement[:50]}...")

        search_result = self.search_graph(
            graph_id=graph_id,
            query=simulation_requirement,
            limit=limit,
        )
        stats = self.get_graph_statistics(graph_id)
        all_nodes = self.get_all_nodes(graph_id)

        # Graphiti 无类型标签，所有节点均视为有意义实体
        entities = []
        for node in all_nodes:
            entities.append({
                "name": node.name,
                "type": "Entity",
                "summary": node.summary,
            })

        return {
            "simulation_requirement": simulation_requirement,
            "related_facts": search_result.facts,
            "graph_statistics": stats,
            "entities": entities[:limit],
            "total_entities": len(entities),
        }

    # ──────────────────────────────────────────────────────────────
    # 核心检索工具
    # ──────────────────────────────────────────────────────────────

    def insight_forge(
        self,
        graph_id: str,
        query: str,
        simulation_requirement: str,
        report_context: str = "",
        max_sub_queries: int = 5,
    ) -> InsightForgeResult:
        """
        【InsightForge - 深度洞察检索】

        1. 使用 LLM 将问题分解为多个子问题
        2. 对每个子问题进行 Graphiti 语义搜索
        3. 从边中提取相关实体并汇总
        4. 构建关系链
        5. 整合所有结果，生成深度洞察

        Args:
            graph_id: 图谱 ID（对应 group_id）
            query: 用户问题
            simulation_requirement: 模拟需求描述
            report_context: 报告上下文（可选）
            max_sub_queries: 最大子问题数量

        Returns:
            InsightForgeResult: 深度洞察检索结果
        """
        logger.info(f"InsightForge 深度洞察检索: {query[:50]}...")

        result = InsightForgeResult(
            query=query,
            simulation_requirement=simulation_requirement,
            sub_queries=[],
        )

        # Step 1: 使用 LLM 生成子问题
        sub_queries = self._generate_sub_queries(
            query=query,
            simulation_requirement=simulation_requirement,
            report_context=report_context,
            max_queries=max_sub_queries,
        )
        result.sub_queries = sub_queries
        logger.info(f"生成 {len(sub_queries)} 个子问题")

        # Step 2: 对每个子问题进行语义搜索
        all_facts: List[str] = []
        all_edge_dicts: List[Dict[str, Any]] = []
        seen_facts: set = set()
        # 节点名 -> NodeInfo 映射（用于关系链构建）
        node_map: Dict[str, NodeInfo] = {}

        for sub_query in sub_queries:
            raw_edges = self._search(graph_id, sub_query, num_results=15)
            for edge in raw_edges:
                fact = str(getattr(edge, 'fact', '') or "")
                if fact and fact not in seen_facts:
                    all_facts.append(fact)
                    seen_facts.add(fact)
                edge_info = _edge_to_edge_info(edge)
                all_edge_dicts.append({
                    "uuid": edge_info.uuid,
                    "name": edge_info.name,
                    "fact": edge_info.fact,
                    "source_node_uuid": edge_info.source_node_uuid,
                    "target_node_uuid": edge_info.target_node_uuid,
                    "source_node_name": edge_info.source_node_name,
                    "target_node_name": edge_info.target_node_name,
                })
                # 更新 node_map
                for name in [edge_info.source_node_name, edge_info.target_node_name]:
                    if name and name not in node_map:
                        node_map[name] = NodeInfo(uuid="", name=name, labels=[], summary="", attributes={})

        # 对原始问题也进行搜索
        main_raw = self._search(graph_id, query, num_results=20)
        for edge in main_raw:
            fact = str(getattr(edge, 'fact', '') or "")
            if fact and fact not in seen_facts:
                all_facts.append(fact)
                seen_facts.add(fact)
            edge_info = _edge_to_edge_info(edge)
            all_edge_dicts.append({
                "uuid": edge_info.uuid,
                "name": edge_info.name,
                "fact": edge_info.fact,
                "source_node_uuid": edge_info.source_node_uuid,
                "target_node_uuid": edge_info.target_node_uuid,
                "source_node_name": edge_info.source_node_name,
                "target_node_name": edge_info.target_node_name,
            })
            for name in [edge_info.source_node_name, edge_info.target_node_name]:
                if name and name not in node_map:
                    node_map[name] = NodeInfo(uuid="", name=name, labels=[], summary="", attributes={})

        result.semantic_facts = all_facts
        result.total_facts = len(all_facts)

        # Step 3: 汇总实体洞察
        entity_insights = []
        for node_name, node_info in node_map.items():
            related_facts = [f for f in all_facts if node_name.lower() in f.lower()]
            entity_insights.append({
                "uuid": node_info.uuid,
                "name": node_info.name,
                "type": "Entity",
                "summary": node_info.summary,
                "related_facts": related_facts,
            })

        result.entity_insights = entity_insights
        result.total_entities = len(entity_insights)

        # Step 4: 构建关系链
        relationship_chains = []
        seen_chains: set = set()
        for edge_data in all_edge_dicts:
            src = edge_data.get('source_node_name') or ""
            tgt = edge_data.get('target_node_name') or ""
            rel = edge_data.get('name') or ""
            if not src or not tgt:
                continue
            chain = f"{src} --[{rel}]--> {tgt}"
            if chain not in seen_chains:
                seen_chains.add(chain)
                relationship_chains.append(chain)

        result.relationship_chains = relationship_chains
        result.total_relationships = len(relationship_chains)

        logger.info(
            f"InsightForge 完成: {result.total_facts}条事实, "
            f"{result.total_entities}个实体, {result.total_relationships}条关系"
        )
        return result

    def _generate_sub_queries(
        self,
        query: str,
        simulation_requirement: str,
        report_context: str = "",
        max_queries: int = 5,
    ) -> List[str]:
        """使用 LLM 将复杂问题分解为多个可以独立检索的子问题"""
        system_prompt = (
            "你是一个专业的问题分析专家。你的任务是将一个复杂问题分解为多个可以在模拟世界中独立观察的子问题。\n\n"
            "要求：\n"
            "1. 每个子问题应该足够具体，可以在模拟世界中找到相关的Agent行为或事件\n"
            "2. 子问题应该覆盖原问题的不同维度（如：谁、什么、为什么、怎么样、何时、何地）\n"
            "3. 子问题应该与模拟场景相关\n"
            '4. 返回JSON格式：{"sub_queries": ["子问题1", "子问题2", ...]}'
        )
        user_prompt = (
            f"模拟需求背景：\n{simulation_requirement}\n\n"
            + (f"报告上下文：{report_context[:500]}\n\n" if report_context else "")
            + f"请将以下问题分解为{max_queries}个子问题：\n{query}\n\n返回JSON格式的子问题列表。"
        )

        try:
            response = self.llm.chat_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
            )
            sub_queries = response.get("sub_queries", [])
            return [str(sq) for sq in sub_queries[:max_queries]]
        except Exception as e:
            logger.warning(f"生成子问题失败: {str(e)}，使用默认子问题")
            return [
                query,
                f"{query} 的主要参与者",
                f"{query} 的原因和影响",
                f"{query} 的发展过程",
            ][:max_queries]

    def panorama_search(
        self,
        graph_id: str,
        query: str,
        include_expired: bool = True,
        limit: int = 50,
    ) -> PanoramaResult:
        """
        【PanoramaSearch - 广度搜索】

        获取全貌视图，包括所有相关内容：
        1. 获取所有相关边（通过空查询 + 关键词查询）
        2. 从边推断节点集合
        3. 按 valid_at / invalid_at 区分当前有效与历史信息

        注意：与 Zep 不同，Graphiti 不单独追踪"过期"(expired)边。
        本方法以 invalid_at 不为 None 作为"历史事实"的判断标准。
        include_expired 参数保留兼容性，控制是否输出历史事实。

        Args:
            graph_id: 图谱 ID
            query: 搜索查询（用于相关性排序）
            include_expired: 是否包含历史/失效内容（默认 True）
            limit: 返回结果数量限制

        Returns:
            PanoramaResult: 广度搜索结果
        """
        logger.info(f"PanoramaSearch 广度搜索: {query[:50]}...")

        result = PanoramaResult(query=query)

        # 使用两轮搜索：空查询（全量）+ 关键词查询（相关性）
        all_raw_edges = self._search(graph_id, query="", num_results=9999)
        if query.strip():
            keyword_raw = self._search(graph_id, query=query, num_results=limit)
            # 合并，按 uuid 去重
            seen_uuids = set()
            merged = []
            for edge in keyword_raw + all_raw_edges:
                uid = str(getattr(edge, 'uuid', id(edge)))
                if uid not in seen_uuids:
                    seen_uuids.add(uid)
                    merged.append(edge)
            all_raw_edges = merged

        # 转换并构建 node_map
        all_edges: List[EdgeInfo] = []
        node_map: Dict[str, NodeInfo] = {}

        for raw_edge in all_raw_edges:
            edge_info = _edge_to_edge_info(raw_edge)
            all_edges.append(edge_info)
            for name in [edge_info.source_node_name, edge_info.target_node_name]:
                if name and name not in node_map:
                    node_map[name] = NodeInfo(uuid="", name=name, labels=[], summary="", attributes={})

        result.all_edges = all_edges
        result.all_nodes = list(node_map.values())
        result.total_edges = len(all_edges)
        result.total_nodes = len(result.all_nodes)

        # 分类事实
        active_facts: List[str] = []
        historical_facts: List[str] = []

        for edge in all_edges:
            if not edge.fact:
                continue
            # Graphiti: invalid_at 不为 None 视为历史/失效
            is_historical = edge.invalid_at is not None or edge.expired_at is not None
            if is_historical:
                valid_at = edge.valid_at or "未知"
                invalid_at = edge.invalid_at or edge.expired_at or "未知"
                historical_facts.append(f"[{valid_at} - {invalid_at}] {edge.fact}")
            else:
                active_facts.append(edge.fact)

        # 相关性排序
        query_lower = query.lower()
        keywords = [
            w.strip() for w in query_lower.replace(',', ' ').replace('，', ' ').split()
            if len(w.strip()) > 1
        ]

        def relevance_score(fact: str) -> int:
            fl = fact.lower()
            score = 100 if query_lower in fl else 0
            for kw in keywords:
                if kw in fl:
                    score += 10
            return score

        active_facts.sort(key=relevance_score, reverse=True)
        historical_facts.sort(key=relevance_score, reverse=True)

        result.active_facts = active_facts[:limit]
        result.historical_facts = historical_facts[:limit] if include_expired else []
        result.active_count = len(active_facts)
        result.historical_count = len(historical_facts)

        logger.info(
            f"PanoramaSearch 完成: {result.active_count}条有效, {result.historical_count}条历史"
        )
        return result

    def quick_search(
        self,
        graph_id: str,
        query: str,
        limit: int = 10,
    ) -> SearchResult:
        """
        【QuickSearch - 简单搜索】

        快速、轻量级的检索工具：
        直接调用 Graphiti 语义搜索，返回最相关的结果。

        Args:
            graph_id: 图谱 ID
            query: 搜索查询
            limit: 返回结果数量

        Returns:
            SearchResult: 搜索结果
        """
        logger.info(f"QuickSearch 简单搜索: {query[:50]}...")
        result = self.search_graph(graph_id=graph_id, query=query, limit=limit, scope="edges")
        logger.info(f"QuickSearch 完成: {result.total_count}条结果")
        return result

    # ──────────────────────────────────────────────────────────────
    # 采访工具（与 ZepToolsService 完全相同的逻辑，仅替换 graph_id 参数名）
    # ──────────────────────────────────────────────────────────────

    def interview_agents(
        self,
        simulation_id: str,
        interview_requirement: str,
        simulation_requirement: str = "",
        max_agents: int = 5,
        custom_questions: List[str] = None,
    ) -> InterviewResult:
        """
        【InterviewAgents - 深度采访】

        调用真实的 OASIS 采访 API，采访模拟中正在运行的 Agent。
        逻辑与原 ZepToolsService.interview_agents 完全相同，不依赖 Graphiti。

        Args:
            simulation_id: 模拟 ID
            interview_requirement: 采访需求描述
            simulation_requirement: 模拟需求背景（可选）
            max_agents: 最多采访的 Agent 数量
            custom_questions: 自定义采访问题（可选）

        Returns:
            InterviewResult: 采访结果
        """
        from .simulation_runner import SimulationRunner

        logger.info(f"InterviewAgents 深度采访（真实API）: {interview_requirement[:50]}...")

        result = InterviewResult(
            interview_topic=interview_requirement,
            interview_questions=custom_questions or [],
        )

        # Step 1: 读取人设文件
        profiles = self._load_agent_profiles(simulation_id)
        if not profiles:
            logger.warning(f"未找到模拟 {simulation_id} 的人设文件")
            result.summary = "未找到可采访的Agent人设文件"
            return result

        result.total_agents = len(profiles)
        logger.info(f"加载到 {len(profiles)} 个Agent人设")

        # Step 2: 使用 LLM 选择要采访的 Agent
        selected_agents, selected_indices, selection_reasoning = self._select_agents_for_interview(
            profiles=profiles,
            interview_requirement=interview_requirement,
            simulation_requirement=simulation_requirement,
            max_agents=max_agents,
        )
        result.selected_agents = selected_agents
        result.selection_reasoning = selection_reasoning
        logger.info(f"选择了 {len(selected_agents)} 个Agent进行采访: {selected_indices}")

        # Step 3: 生成采访问题
        if not result.interview_questions:
            result.interview_questions = self._generate_interview_questions(
                interview_requirement=interview_requirement,
                simulation_requirement=simulation_requirement,
                selected_agents=selected_agents,
            )
            logger.info(f"生成了 {len(result.interview_questions)} 个采访问题")

        combined_prompt = "\n".join(
            [f"{i+1}. {q}" for i, q in enumerate(result.interview_questions)]
        )

        INTERVIEW_PROMPT_PREFIX = (
            "你正在接受一次采访。请结合你的人设、所有的过往记忆与行动，"
            "以纯文本方式直接回答以下问题。\n"
            "回复要求：\n"
            "1. 直接用自然语言回答，不要调用任何工具\n"
            "2. 不要返回JSON格式或工具调用格式\n"
            "3. 不要使用Markdown标题（如#、##、###）\n"
            "4. 按问题编号逐一回答，每个回答以「问题X：」开头（X为问题编号）\n"
            "5. 每个问题的回答之间用空行分隔\n"
            "6. 回答要有实质内容，每个问题至少回答2-3句话\n\n"
        )
        optimized_prompt = f"{INTERVIEW_PROMPT_PREFIX}{combined_prompt}"

        # Step 4: 调用真实的采访 API
        try:
            interviews_request = []
            for agent_idx in selected_indices:
                interviews_request.append({
                    "agent_id": agent_idx,
                    "prompt": optimized_prompt,
                })

            logger.info(f"调用批量采访API（双平台）: {len(interviews_request)} 个Agent")

            api_result = SimulationRunner.interview_agents_batch(
                simulation_id=simulation_id,
                interviews=interviews_request,
                platform=None,
                timeout=180.0,
            )

            logger.info(
                f"采访API返回: {api_result.get('interviews_count', 0)} 个结果, "
                f"success={api_result.get('success')}"
            )

            if not api_result.get("success", False):
                error_msg = api_result.get("error", "未知错误")
                logger.warning(f"采访API返回失败: {error_msg}")
                result.summary = f"采访API调用失败：{error_msg}。请检查OASIS模拟环境状态。"
                return result

            # Step 5: 解析 API 返回结果
            api_data = api_result.get("result", {})
            results_dict = api_data.get("results", {}) if isinstance(api_data, dict) else {}

            for i, agent_idx in enumerate(selected_indices):
                agent = selected_agents[i]
                agent_name = agent.get("realname", agent.get("username", f"Agent_{agent_idx}"))
                agent_role = agent.get("profession", "未知")
                agent_bio = agent.get("bio", "")

                twitter_result = results_dict.get(f"twitter_{agent_idx}", {})
                reddit_result = results_dict.get(f"reddit_{agent_idx}", {})

                twitter_response = self._clean_tool_call_response(
                    twitter_result.get("response", "")
                )
                reddit_response = self._clean_tool_call_response(
                    reddit_result.get("response", "")
                )

                twitter_text = twitter_response if twitter_response else "（该平台未获得回复）"
                reddit_text = reddit_response if reddit_response else "（该平台未获得回复）"
                response_text = f"【Twitter平台回答】\n{twitter_text}\n\n【Reddit平台回答】\n{reddit_text}"

                combined_responses = f"{twitter_response} {reddit_response}"
                clean_text = re.sub(r'#{1,6}\s+', '', combined_responses)
                clean_text = re.sub(r'\{[^}]*tool_name[^}]*\}', '', clean_text)
                clean_text = re.sub(r'[*_`|>~\-]{2,}', '', clean_text)
                clean_text = re.sub(r'问题\d+[：:]\s*', '', clean_text)
                clean_text = re.sub(r'【[^】]+】', '', clean_text)

                sentences = re.split(r'[。！？]', clean_text)
                meaningful = [
                    s.strip() for s in sentences
                    if 20 <= len(s.strip()) <= 150
                    and not re.match(r'^[\s\W，,；;：:、]+', s.strip())
                    and not s.strip().startswith(('{', '问题'))
                ]
                meaningful.sort(key=len, reverse=True)
                key_quotes = [s + "。" for s in meaningful[:3]]

                if not key_quotes:
                    paired = re.findall(r'\u201c([^\u201c\u201d]{15,100})\u201d', clean_text)
                    paired += re.findall(r'\u300c([^\u300c\u300d]{15,100})\u300d', clean_text)
                    key_quotes = [q for q in paired if not re.match(r'^[，,；;：:、]', q)][:3]

                interview = AgentInterview(
                    agent_name=agent_name,
                    agent_role=agent_role,
                    agent_bio=agent_bio[:1000],
                    question=combined_prompt,
                    response=response_text,
                    key_quotes=key_quotes[:5],
                )
                result.interviews.append(interview)

            result.interviewed_count = len(result.interviews)

        except ValueError as e:
            logger.warning(f"采访API调用失败（环境未运行？）: {e}")
            result.summary = f"采访失败：{str(e)}。模拟环境可能已关闭，请确保OASIS环境正在运行。"
            return result
        except Exception as e:
            logger.error(f"采访API调用异常: {e}")
            import traceback
            logger.error(traceback.format_exc())
            result.summary = f"采访过程发生错误：{str(e)}"
            return result

        # Step 6: 生成采访摘要
        if result.interviews:
            result.summary = self._generate_interview_summary(
                interviews=result.interviews,
                interview_requirement=interview_requirement,
            )

        logger.info(f"InterviewAgents 完成: 采访了 {result.interviewed_count} 个Agent（双平台）")
        return result

    @staticmethod
    def _clean_tool_call_response(response: str) -> str:
        """清理 Agent 回复中的 JSON 工具调用包裹，提取实际内容"""
        if not response or not response.strip().startswith('{'):
            return response
        text = response.strip()
        if 'tool_name' not in text[:80]:
            return response
        try:
            data = json.loads(text)
            if isinstance(data, dict) and 'arguments' in data:
                for key in ('content', 'text', 'body', 'message', 'reply'):
                    if key in data['arguments']:
                        return str(data['arguments'][key])
        except (json.JSONDecodeError, KeyError, TypeError):
            match = re.search(r'"content"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
            if match:
                return match.group(1).replace('\\n', '\n').replace('\\"', '"')
        return response

    def _load_agent_profiles(self, simulation_id: str) -> List[Dict[str, Any]]:
        """加载模拟的 Agent 人设文件"""
        import os
        import csv

        sim_dir = os.path.join(
            os.path.dirname(__file__),
            f'../../uploads/simulations/{simulation_id}',
        )
        profiles = []

        reddit_profile_path = os.path.join(sim_dir, "reddit_profiles.json")
        if os.path.exists(reddit_profile_path):
            try:
                with open(reddit_profile_path, 'r', encoding='utf-8') as f:
                    profiles = json.load(f)
                logger.info(f"从 reddit_profiles.json 加载了 {len(profiles)} 个人设")
                return profiles
            except Exception as e:
                logger.warning(f"读取 reddit_profiles.json 失败: {e}")

        twitter_profile_path = os.path.join(sim_dir, "twitter_profiles.csv")
        if os.path.exists(twitter_profile_path):
            try:
                with open(twitter_profile_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        profiles.append({
                            "realname": row.get("name", ""),
                            "username": row.get("username", ""),
                            "bio": row.get("description", ""),
                            "persona": row.get("user_char", ""),
                            "profession": "未知",
                        })
                logger.info(f"从 twitter_profiles.csv 加载了 {len(profiles)} 个人设")
                return profiles
            except Exception as e:
                logger.warning(f"读取 twitter_profiles.csv 失败: {e}")

        return profiles

    def _select_agents_for_interview(
        self,
        profiles: List[Dict[str, Any]],
        interview_requirement: str,
        simulation_requirement: str,
        max_agents: int,
    ) -> tuple:
        """使用 LLM 选择要采访的 Agent，返回 (selected_agents, selected_indices, reasoning)"""
        agent_summaries = []
        for i, profile in enumerate(profiles):
            agent_summaries.append({
                "index": i,
                "name": profile.get("realname", profile.get("username", f"Agent_{i}")),
                "profession": profile.get("profession", "未知"),
                "bio": profile.get("bio", "")[:200],
                "interested_topics": profile.get("interested_topics", []),
            })

        system_prompt = (
            "你是一个专业的采访策划专家。你的任务是根据采访需求，从模拟Agent列表中选择最适合采访的对象。\n\n"
            "选择标准：\n"
            "1. Agent的身份/职业与采访主题相关\n"
            "2. Agent可能持有独特或有价值的观点\n"
            "3. 选择多样化的视角（如：支持方、反对方、中立方、专业人士等）\n"
            "4. 优先选择与事件直接相关的角色\n\n"
            "返回JSON格式：\n"
            '{\n    "selected_indices": [选中Agent的索引列表],\n    "reasoning": "选择理由说明"\n}'
        )
        user_prompt = (
            f"采访需求：\n{interview_requirement}\n\n"
            f"模拟背景：\n{simulation_requirement if simulation_requirement else '未提供'}\n\n"
            f"可选择的Agent列表（共{len(agent_summaries)}个）：\n"
            + json.dumps(agent_summaries, ensure_ascii=False, indent=2)
            + f"\n\n请选择最多{max_agents}个最适合采访的Agent，并说明选择理由。"
        )

        try:
            response = self.llm.chat_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
            )
            selected_indices = response.get("selected_indices", [])[:max_agents]
            reasoning = response.get("reasoning", "基于相关性自动选择")
            selected_agents = []
            valid_indices = []
            for idx in selected_indices:
                if 0 <= idx < len(profiles):
                    selected_agents.append(profiles[idx])
                    valid_indices.append(idx)
            return selected_agents, valid_indices, reasoning
        except Exception as e:
            logger.warning(f"LLM选择Agent失败，使用默认选择: {e}")
            selected = profiles[:max_agents]
            indices = list(range(min(max_agents, len(profiles))))
            return selected, indices, "使用默认选择策略"

    def _generate_interview_questions(
        self,
        interview_requirement: str,
        simulation_requirement: str,
        selected_agents: List[Dict[str, Any]],
    ) -> List[str]:
        """使用 LLM 生成采访问题"""
        agent_roles = [a.get("profession", "未知") for a in selected_agents]

        system_prompt = (
            "你是一个专业的记者/采访者。根据采访需求，生成3-5个深度采访问题。\n\n"
            "问题要求：\n"
            "1. 开放性问题，鼓励详细回答\n"
            "2. 针对不同角色可能有不同答案\n"
            "3. 涵盖事实、观点、感受等多个维度\n"
            "4. 语言自然，像真实采访一样\n"
            "5. 每个问题控制在50字以内，简洁明了\n"
            "6. 直接提问，不要包含背景说明或前缀\n\n"
            '返回JSON格式：{"questions": ["问题1", "问题2", ...]}'
        )
        user_prompt = (
            f"采访需求：{interview_requirement}\n\n"
            f"模拟背景：{simulation_requirement if simulation_requirement else '未提供'}\n\n"
            f"采访对象角色：{', '.join(agent_roles)}\n\n"
            "请生成3-5个采访问题。"
        )

        try:
            response = self.llm.chat_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.5,
            )
            return response.get("questions", [f"关于{interview_requirement}，您有什么看法？"])
        except Exception as e:
            logger.warning(f"生成采访问题失败: {e}")
            return [
                f"关于{interview_requirement}，您的观点是什么？",
                "这件事对您或您所代表的群体有什么影响？",
                "您认为应该如何解决或改进这个问题？",
            ]

    def _generate_interview_summary(
        self,
        interviews: List[AgentInterview],
        interview_requirement: str,
    ) -> str:
        """生成采访摘要"""
        if not interviews:
            return "未完成任何采访"

        interview_texts = []
        for interview in interviews:
            interview_texts.append(
                f"【{interview.agent_name}（{interview.agent_role}）】\n{interview.response[:500]}"
            )

        system_prompt = (
            "你是一个专业的新闻编辑。请根据多位受访者的回答，生成一份采访摘要。\n\n"
            "摘要要求：\n"
            "1. 提炼各方主要观点\n"
            "2. 指出观点的共识和分歧\n"
            "3. 突出有价值的引言\n"
            "4. 客观中立，不偏袒任何一方\n"
            "5. 控制在1000字内\n\n"
            "格式约束（必须遵守）：\n"
            "- 使用纯文本段落，用空行分隔不同部分\n"
            "- 不要使用Markdown标题（如#、##、###）\n"
            "- 不要使用分割线（如---、***）\n"
            "- 引用受访者原话时使用中文引号「」\n"
            "- 可以使用**加粗**标记关键词，但不要使用其他Markdown语法"
        )
        user_prompt = (
            f"采访主题：{interview_requirement}\n\n"
            f"采访内容：\n{''.join(interview_texts)}\n\n"
            "请生成采访摘要。"
        )

        try:
            summary = self.llm.chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=800,
            )
            return summary
        except Exception as e:
            logger.warning(f"生成采访摘要失败: {e}")
            return f"共采访了{len(interviews)}位受访者，包括：" + "、".join(
                [i.agent_name for i in interviews]
            )
