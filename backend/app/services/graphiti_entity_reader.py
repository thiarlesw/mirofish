"""
Graphiti 实体读取与过滤服务
从 Graphiti/Kuzu 图谱中读取节点，筛选出符合预定义实体类型的节点

替代原 zep_entity_reader.py，保持相同的公共接口。
"""

import asyncio
from typing import Dict, Any, List, Optional, Set, TypeVar

from dataclasses import dataclass, field

from .graphiti_client import get_graphiti
from ..utils.logger import get_logger

logger = get_logger('mirofish.graphiti_entity_reader')

T = TypeVar('T')


# ── 数据结构（与 zep_entity_reader.py 保持完全一致）────────────────────────

@dataclass
class EntityNode:
    """实体节点数据结构"""
    uuid: str
    name: str
    labels: List[str]
    summary: str
    attributes: Dict[str, Any]
    # 相关的边信息
    related_edges: List[Dict[str, Any]] = field(default_factory=list)
    # 相关的其他节点信息
    related_nodes: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uuid": self.uuid,
            "name": self.name,
            "labels": self.labels,
            "summary": self.summary,
            "attributes": self.attributes,
            "related_edges": self.related_edges,
            "related_nodes": self.related_nodes,
        }

    def get_entity_type(self) -> Optional[str]:
        """获取实体类型（排除默认的 Entity 标签）"""
        for label in self.labels:
            if label not in ["Entity", "Node"]:
                return label
        return None


@dataclass
class FilteredEntities:
    """过滤后的实体集合"""
    entities: List[EntityNode]
    entity_types: Set[str]
    total_count: int
    filtered_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entities": [e.to_dict() for e in self.entities],
            "entity_types": list(self.entity_types),
            "total_count": self.total_count,
            "filtered_count": self.filtered_count,
        }


# ── 辅助函数：将 graphiti_core 对象转为字典 ──────────────────────────────────

def _node_result_to_dict(node) -> Dict[str, Any]:
    """
    将 graphiti_core 搜索结果中的节点对象转换为标准字典格式。

    graphiti_core 的 search() 返回 SearchResult 列表，每个对象大致包含：
      .uuid / .node_id  — 节点唯一标识
      .name             — 节点名称
      .labels           — 节点标签列表（部分版本使用 .entity_type）
      .summary          — 摘要
      .attributes       — 扩展属性字典

    因为 graphiti_core 的 API 仍在演进，这里做防御性访问。
    """
    uuid = (
        getattr(node, 'uuid', None)
        or getattr(node, 'node_id', None)
        or getattr(node, 'id', '')
        or ''
    )
    name = getattr(node, 'name', '') or ''
    summary = getattr(node, 'summary', '') or getattr(node, 'fact', '') or ''

    # labels：优先取 .labels 列表，退而取 .entity_type 字符串
    raw_labels = getattr(node, 'labels', None)
    if raw_labels and isinstance(raw_labels, list):
        labels = raw_labels
    else:
        entity_type = getattr(node, 'entity_type', None) or getattr(node, 'type', None)
        labels = [entity_type] if entity_type else ["Entity"]

    attributes = getattr(node, 'attributes', None) or {}

    return {
        "uuid": str(uuid),
        "name": name,
        "labels": labels,
        "summary": summary,
        "attributes": attributes,
    }


def _edge_result_to_dict(edge) -> Dict[str, Any]:
    """
    将 graphiti_core 边对象转换为标准字典格式。

    graphiti_core 边对象（EpisodicEdge / EntityEdge）大致包含：
      .uuid             — 边唯一标识
      .name / .relation — 关系名称
      .fact             — 事实描述
      .source_node_uuid — 源节点 UUID
      .target_node_uuid — 目标节点 UUID
      .attributes       — 扩展属性
    """
    uuid = (
        getattr(edge, 'uuid', None)
        or getattr(edge, 'edge_id', None)
        or getattr(edge, 'id', '')
        or ''
    )
    name = (
        getattr(edge, 'name', None)
        or getattr(edge, 'relation', None)
        or ''
    )
    fact = getattr(edge, 'fact', '') or ''
    source_node_uuid = (
        getattr(edge, 'source_node_uuid', None)
        or getattr(edge, 'source_uuid', None)
        or ''
    )
    target_node_uuid = (
        getattr(edge, 'target_node_uuid', None)
        or getattr(edge, 'target_uuid', None)
        or ''
    )
    attributes = getattr(edge, 'attributes', None) or {}

    return {
        "uuid": str(uuid),
        "name": name,
        "fact": fact,
        "source_node_uuid": str(source_node_uuid),
        "target_node_uuid": str(target_node_uuid),
        "attributes": attributes,
    }


# ── 主类 ─────────────────────────────────────────────────────────────────────

class GraphitiEntityReader:
    """
    Graphiti 实体读取与过滤服务（替代 ZepEntityReader）

    主要功能：
    1. 从 Graphiti 图谱读取指定 group_id 的所有节点
    2. 筛选出符合预定义实体类型的节点
    3. 获取每个实体的相关边和关联节点信息

    所有方法均为 async。
    """

    def __init__(self):
        """
        无需传入 api_key；通过 get_graphiti() 单例获取客户端。
        """
        pass

    async def get_all_nodes(self, group_id: str) -> List[Dict[str, Any]]:
        """
        获取指定 group_id 的所有节点。

        Args:
            group_id: 图谱分组 ID（对应原 Zep 的 graph_id）

        Returns:
            节点字典列表
        """
        logger.info(f"获取 group_id={group_id} 的所有节点...")

        graphiti = await get_graphiti()

        nodes_data: List[Dict[str, Any]] = []

        try:
            # graphiti_core 的 search() 支持 group_ids 过滤。
            # 使用空字符串查询以获取所有节点；num_results 设置较大值。
            # TODO: 如果 graphiti_core 提供专用的 list_nodes() 接口，优先使用它。
            results = await graphiti.search(
                query='',
                group_ids=[group_id],
                num_results=10000,
            )

            for result in results:
                # search() 可能返回节点或边的混合结果，仅保留节点
                # graphiti_core SearchResult 通常有 .node 属性
                node_obj = getattr(result, 'node', result)
                nodes_data.append(_node_result_to_dict(node_obj))

        except Exception as e:
            logger.error(f"查询 Graphiti 节点失败 (group_id={group_id}): {e}")
            # 回退：尝试通过底层 driver 直接查询
            try:
                nodes_data = await self._query_nodes_via_driver(graphiti, group_id)
            except Exception as e2:
                logger.error(f"通过 driver 查询节点也失败: {e2}")

        logger.info(f"共获取 {len(nodes_data)} 个节点 (group_id={group_id})")
        return nodes_data

    async def _query_nodes_via_driver(
        self, graphiti, group_id: str
    ) -> List[Dict[str, Any]]:
        """
        通过 graphiti_core 底层 driver 直接执行 Kuzu 查询获取节点。

        TODO: 根据实际 graphiti_core 版本调整查询语法。
        """
        # Kuzu 的 Cypher 查询示例
        # TODO: 验证 graphiti_core KuzuDriver 的 execute() 方法签名
        cypher = (
            "MATCH (n:Entity) WHERE n.group_id = $group_id "
            "RETURN n.uuid AS uuid, n.name AS name, n.summary AS summary, "
            "labels(n) AS labels LIMIT 10000"
        )
        try:
            rows = await graphiti.driver.execute_query(
                cypher, parameters={"group_id": group_id}
            )
            return [
                {
                    "uuid": str(row.get("uuid", "")),
                    "name": row.get("name", ""),
                    "labels": row.get("labels", ["Entity"]),
                    "summary": row.get("summary", ""),
                    "attributes": {},
                }
                for row in rows
            ]
        except Exception:
            return []

    async def get_all_edges(self, group_id: str) -> List[Dict[str, Any]]:
        """
        获取指定 group_id 的所有边。

        Args:
            group_id: 图谱分组 ID

        Returns:
            边字典列表
        """
        logger.info(f"获取 group_id={group_id} 的所有边...")

        graphiti = await get_graphiti()
        edges_data: List[Dict[str, Any]] = []

        try:
            # TODO: 如果 graphiti_core 提供专用的 get_edges() 接口，优先使用它。
            # 当前通过底层 driver 查询所有 EntityEdge。
            edges_data = await self._query_edges_via_driver(graphiti, group_id)

        except Exception as e:
            logger.error(f"查询 Graphiti 边失败 (group_id={group_id}): {e}")

        logger.info(f"共获取 {len(edges_data)} 条边 (group_id={group_id})")
        return edges_data

    async def _query_edges_via_driver(
        self, graphiti, group_id: str
    ) -> List[Dict[str, Any]]:
        """
        通过底层 driver 查询边数据。

        TODO: 根据实际 graphiti_core schema 调整查询。
        """
        cypher = (
            "MATCH (s:Entity)-[r:RELATES_TO]->(t:Entity) "
            "WHERE r.group_id = $group_id "
            "RETURN r.uuid AS uuid, r.name AS name, r.fact AS fact, "
            "s.uuid AS source_node_uuid, t.uuid AS target_node_uuid LIMIT 10000"
        )
        try:
            rows = await graphiti.driver.execute_query(
                cypher, parameters={"group_id": group_id}
            )
            return [
                {
                    "uuid": str(row.get("uuid", "")),
                    "name": row.get("name", ""),
                    "fact": row.get("fact", ""),
                    "source_node_uuid": str(row.get("source_node_uuid", "")),
                    "target_node_uuid": str(row.get("target_node_uuid", "")),
                    "attributes": {},
                }
                for row in rows
            ]
        except Exception as e:
            logger.warning(f"driver 边查询失败: {e}")
            return []

    async def filter_defined_entities(
        self,
        group_id: str,
        defined_entity_types: Optional[List[str]] = None,
        enrich_with_edges: bool = True,
    ) -> FilteredEntities:
        """
        筛选出符合预定义实体类型的节点。

        筛选逻辑（与原 ZepEntityReader 完全一致）：
        - 节点的 labels 只有 "Entity" / "Node" → 跳过
        - 节点包含除 "Entity" / "Node" 之外的自定义标签 → 保留
        - 若 defined_entity_types 非空，则仅保留匹配的类型

        Args:
            group_id: 图谱分组 ID
            defined_entity_types: 预定义实体类型列表（可选）
            enrich_with_edges: 是否获取每个实体的相关边信息

        Returns:
            FilteredEntities
        """
        logger.info(f"开始筛选 group_id={group_id} 的实体...")

        all_nodes = await self.get_all_nodes(group_id)
        total_count = len(all_nodes)

        all_edges = await self.get_all_edges(group_id) if enrich_with_edges else []

        # 构建 UUID → 节点数据的映射
        node_map = {n["uuid"]: n for n in all_nodes}

        filtered_entities: List[EntityNode] = []
        entity_types_found: Set[str] = set()

        for node in all_nodes:
            labels = node.get("labels", [])

            # 筛选：Labels 必须包含除 "Entity" 和 "Node" 之外的标签
            custom_labels = [lb for lb in labels if lb not in ("Entity", "Node")]

            if not custom_labels:
                continue

            if defined_entity_types:
                matching_labels = [lb for lb in custom_labels if lb in defined_entity_types]
                if not matching_labels:
                    continue
                entity_type = matching_labels[0]
            else:
                entity_type = custom_labels[0]

            entity_types_found.add(entity_type)

            entity = EntityNode(
                uuid=node["uuid"],
                name=node["name"],
                labels=labels,
                summary=node["summary"],
                attributes=node["attributes"],
            )

            if enrich_with_edges:
                related_edges: List[Dict[str, Any]] = []
                related_node_uuids: Set[str] = set()

                for edge in all_edges:
                    if edge["source_node_uuid"] == node["uuid"]:
                        related_edges.append({
                            "direction": "outgoing",
                            "edge_name": edge["name"],
                            "fact": edge["fact"],
                            "target_node_uuid": edge["target_node_uuid"],
                        })
                        related_node_uuids.add(edge["target_node_uuid"])
                    elif edge["target_node_uuid"] == node["uuid"]:
                        related_edges.append({
                            "direction": "incoming",
                            "edge_name": edge["name"],
                            "fact": edge["fact"],
                            "source_node_uuid": edge["source_node_uuid"],
                        })
                        related_node_uuids.add(edge["source_node_uuid"])

                entity.related_edges = related_edges

                related_nodes: List[Dict[str, Any]] = []
                for related_uuid in related_node_uuids:
                    if related_uuid in node_map:
                        rn = node_map[related_uuid]
                        related_nodes.append({
                            "uuid": rn["uuid"],
                            "name": rn["name"],
                            "labels": rn["labels"],
                            "summary": rn.get("summary", ""),
                        })
                entity.related_nodes = related_nodes

            filtered_entities.append(entity)

        logger.info(
            f"筛选完成: 总节点 {total_count}, 符合条件 {len(filtered_entities)}, "
            f"实体类型: {entity_types_found}"
        )

        return FilteredEntities(
            entities=filtered_entities,
            entity_types=entity_types_found,
            total_count=total_count,
            filtered_count=len(filtered_entities),
        )

    async def get_entity_with_context(
        self,
        group_id: str,
        entity_name: str,
    ) -> Optional[EntityNode]:
        """
        按名称搜索单个实体及其完整上下文（边和关联节点）。

        原 ZepEntityReader 使用 UUID 查询；Graphiti 以语义搜索为主，
        因此这里改为按名称搜索，取最匹配的第一条结果。

        Args:
            group_id: 图谱分组 ID
            entity_name: 实体名称（用于语义搜索）

        Returns:
            EntityNode 或 None
        """
        logger.info(f"搜索实体: group_id={group_id}, entity_name={entity_name}")

        graphiti = await get_graphiti()

        try:
            results = await graphiti.search(
                query=entity_name,
                group_ids=[group_id],
                num_results=5,
            )

            if not results:
                logger.info(f"未找到实体: {entity_name}")
                return None

            # 取最相关的第一条
            node_obj = getattr(results[0], 'node', results[0])
            node = _node_result_to_dict(node_obj)
            entity_uuid = node["uuid"]

        except Exception as e:
            logger.error(f"搜索实体 '{entity_name}' 失败: {e}")
            return None

        # 获取该实体的所有边和节点
        try:
            all_nodes = await self.get_all_nodes(group_id)
            all_edges = await self.get_all_edges(group_id)
        except Exception as e:
            logger.error(f"获取实体上下文失败: {e}")
            return None

        node_map = {n["uuid"]: n for n in all_nodes}

        related_edges: List[Dict[str, Any]] = []
        related_node_uuids: Set[str] = set()

        for edge in all_edges:
            if edge["source_node_uuid"] == entity_uuid:
                related_edges.append({
                    "direction": "outgoing",
                    "edge_name": edge["name"],
                    "fact": edge["fact"],
                    "target_node_uuid": edge["target_node_uuid"],
                })
                related_node_uuids.add(edge["target_node_uuid"])
            elif edge["target_node_uuid"] == entity_uuid:
                related_edges.append({
                    "direction": "incoming",
                    "edge_name": edge["name"],
                    "fact": edge["fact"],
                    "source_node_uuid": edge["source_node_uuid"],
                })
                related_node_uuids.add(edge["source_node_uuid"])

        related_nodes: List[Dict[str, Any]] = []
        for related_uuid in related_node_uuids:
            if related_uuid in node_map:
                rn = node_map[related_uuid]
                related_nodes.append({
                    "uuid": rn["uuid"],
                    "name": rn["name"],
                    "labels": rn["labels"],
                    "summary": rn.get("summary", ""),
                })

        return EntityNode(
            uuid=node["uuid"],
            name=node["name"],
            labels=node["labels"],
            summary=node["summary"],
            attributes=node["attributes"],
            related_edges=related_edges,
            related_nodes=related_nodes,
        )

    async def get_entities_by_type(
        self,
        group_id: str,
        entity_type: str,
        enrich_with_edges: bool = True,
    ) -> List[EntityNode]:
        """
        获取指定类型的所有实体。

        Args:
            group_id: 图谱分组 ID
            entity_type: 实体类型（如 "Student", "PublicFigure" 等）
            enrich_with_edges: 是否获取相关边信息

        Returns:
            实体列表
        """
        result = await self.filter_defined_entities(
            group_id=group_id,
            defined_entity_types=[entity_type],
            enrich_with_edges=enrich_with_edges,
        )
        return result.entities
