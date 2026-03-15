"""
Graphiti entity reading and filtering service.
Reads nodes from the Graphiti/Kuzu knowledge graph and filters out those that match
predefined entity types.

Replaces the original zep_entity_reader.py, maintaining the same public interface.
"""

import asyncio
from typing import Dict, Any, List, Optional, Set, TypeVar

from dataclasses import dataclass, field

from .graphiti_client import get_graphiti
from ..utils.logger import get_logger

logger = get_logger('mirofish.graphiti_entity_reader')

T = TypeVar('T')


# ── Data structures (fully identical to zep_entity_reader.py) ────────────────────────

@dataclass
class EntityNode:
    """Entity node data structure"""
    uuid: str
    name: str
    labels: List[str]
    summary: str
    attributes: Dict[str, Any]
    # Related edge information
    related_edges: List[Dict[str, Any]] = field(default_factory=list)
    # Related node information
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
        """Get entity type (excluding default 'Entity' and 'Node' labels)"""
        for label in self.labels:
            if label not in ["Entity", "Node"]:
                return label
        return None


@dataclass
class FilteredEntities:
    """Filtered entity collection"""
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


# ── Helper functions: convert graphiti_core objects to dicts ──────────────────────────────────

def _node_result_to_dict(node) -> Dict[str, Any]:
    """
    Convert a node object from a graphiti_core search result to the standard dictionary format.

    graphiti_core's search() returns a list of SearchResult objects, each roughly containing:
      .uuid / .node_id  — unique node identifier
      .name             — node name
      .labels           — list of node labels (some versions use .entity_type)
      .summary          — summary text
      .attributes       — extended attributes dict

    Defensive access is used here because graphiti_core's API is still evolving.
    """
    uuid = (
        getattr(node, 'uuid', None)
        or getattr(node, 'node_id', None)
        or getattr(node, 'id', '')
        or ''
    )
    name = getattr(node, 'name', '') or ''
    summary = getattr(node, 'summary', '') or getattr(node, 'fact', '') or ''

    # labels: prefer .labels list, fall back to .entity_type string
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
    Convert a graphiti_core edge object to the standard dictionary format.

    graphiti_core edge objects (EpisodicEdge / EntityEdge) roughly contain:
      .uuid             — unique edge identifier
      .name / .relation — relationship name
      .fact             — fact description
      .source_node_uuid — source node UUID
      .target_node_uuid — target node UUID
      .attributes       — extended attributes
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


# ── Main class ─────────────────────────────────────────────────────────────────────

class GraphitiEntityReader:
    """
    Graphiti entity reading and filtering service (replaces ZepEntityReader).

    Core functionality:
    1. Read all nodes for a given group_id from the Graphiti graph
    2. Filter out nodes that match predefined entity types
    3. Retrieve related edges and connected node information for each entity

    All methods are async.
    """

    def __init__(self):
        """
        No api_key needed; the client is obtained via the get_graphiti() singleton.
        """
        pass

    async def get_all_nodes(self, group_id: str) -> List[Dict[str, Any]]:
        """
        Get all nodes for a given group_id.

        Args:
            group_id: Graph group ID (corresponds to the original Zep graph_id)

        Returns:
            List of node dictionaries
        """
        logger.info(f"Fetching all nodes for group_id={group_id}...")

        graphiti = await get_graphiti()

        nodes_data: List[Dict[str, Any]] = []

        try:
            # graphiti_core's search() supports group_ids filtering.
            # Use an empty string query to retrieve all nodes; num_results is set large.
            # TODO: If graphiti_core provides a dedicated list_nodes() interface, prefer that.
            results = await graphiti.search(
                query='',
                group_ids=[group_id],
                num_results=10000,
            )

            for result in results:
                # search() may return a mix of nodes and edges; keep only nodes.
                # graphiti_core SearchResult typically has a .node attribute
                node_obj = getattr(result, 'node', result)
                nodes_data.append(_node_result_to_dict(node_obj))

        except Exception as e:
            logger.error(f"Failed to query Graphiti nodes (group_id={group_id}): {e}")
            # Fallback: try querying via the underlying driver directly
            try:
                nodes_data = await self._query_nodes_via_driver(graphiti, group_id)
            except Exception as e2:
                logger.error(f"Driver-based node query also failed: {e2}")

        logger.info(f"Retrieved {len(nodes_data)} nodes total (group_id={group_id})")
        return nodes_data

    async def _query_nodes_via_driver(
        self, graphiti, group_id: str
    ) -> List[Dict[str, Any]]:
        """
        Query nodes directly via the graphiti_core underlying driver (Kuzu query).

        TODO: Adjust query syntax based on the actual graphiti_core version.
        """
        # Kuzu Cypher query example
        # TODO: Verify the execute() method signature for the graphiti_core KuzuDriver
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
        Get all edges for a given group_id.

        Args:
            group_id: Graph group ID

        Returns:
            List of edge dictionaries
        """
        logger.info(f"Fetching all edges for group_id={group_id}...")

        graphiti = await get_graphiti()
        edges_data: List[Dict[str, Any]] = []

        try:
            # TODO: If graphiti_core provides a dedicated get_edges() interface, prefer that.
            # Currently querying all EntityEdges via the underlying driver.
            edges_data = await self._query_edges_via_driver(graphiti, group_id)

        except Exception as e:
            logger.error(f"Failed to query Graphiti edges (group_id={group_id}): {e}")

        logger.info(f"Retrieved {len(edges_data)} edges total (group_id={group_id})")
        return edges_data

    async def _query_edges_via_driver(
        self, graphiti, group_id: str
    ) -> List[Dict[str, Any]]:
        """
        Query edge data via the underlying driver.

        TODO: Adjust the query based on the actual graphiti_core schema.
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
            logger.warning(f"Driver edge query failed: {e}")
            return []

    async def filter_defined_entities(
        self,
        group_id: str,
        defined_entity_types: Optional[List[str]] = None,
        enrich_with_edges: bool = True,
    ) -> FilteredEntities:
        """
        Filter out nodes that match predefined entity types.

        Filtering logic (identical to the original ZepEntityReader):
        - Nodes whose labels contain only "Entity" / "Node" are skipped
        - Nodes that contain custom labels beyond "Entity" / "Node" are kept
        - If defined_entity_types is non-empty, only matching types are kept

        Args:
            group_id: Graph group ID
            defined_entity_types: List of predefined entity types (optional)
            enrich_with_edges: Whether to retrieve related edge information for each entity

        Returns:
            FilteredEntities
        """
        logger.info(f"Starting entity filtering for group_id={group_id}...")

        all_nodes = await self.get_all_nodes(group_id)
        total_count = len(all_nodes)

        all_edges = await self.get_all_edges(group_id) if enrich_with_edges else []

        # Build UUID → node data mapping
        node_map = {n["uuid"]: n for n in all_nodes}

        filtered_entities: List[EntityNode] = []
        entity_types_found: Set[str] = set()

        for node in all_nodes:
            labels = node.get("labels", [])

            # Filter: labels must contain at least one label other than "Entity" and "Node"
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
            f"Filtering complete: total nodes {total_count}, matching {len(filtered_entities)}, "
            f"entity types: {entity_types_found}"
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
        Search for a single entity by name and retrieve its full context (edges and connected nodes).

        The original ZepEntityReader used UUID-based lookup; Graphiti relies on semantic search,
        so this method searches by name and takes the best-matching first result.

        Args:
            group_id: Graph group ID
            entity_name: Entity name (used for semantic search)

        Returns:
            EntityNode or None
        """
        logger.info(f"Searching for entity: group_id={group_id}, entity_name={entity_name}")

        graphiti = await get_graphiti()

        try:
            results = await graphiti.search(
                query=entity_name,
                group_ids=[group_id],
                num_results=5,
            )

            if not results:
                logger.info(f"Entity not found: {entity_name}")
                return None

            # Take the most relevant first result
            node_obj = getattr(results[0], 'node', results[0])
            node = _node_result_to_dict(node_obj)
            entity_uuid = node["uuid"]

        except Exception as e:
            logger.error(f"Failed to search for entity '{entity_name}': {e}")
            return None

        # Retrieve all edges and nodes for this entity
        try:
            all_nodes = await self.get_all_nodes(group_id)
            all_edges = await self.get_all_edges(group_id)
        except Exception as e:
            logger.error(f"Failed to retrieve entity context: {e}")
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
        Get all entities of a specific type.

        Args:
            group_id: Graph group ID
            entity_type: Entity type (e.g., "Student", "PublicFigure")
            enrich_with_edges: Whether to retrieve related edge information

        Returns:
            List of entities
        """
        result = await self.filter_defined_entities(
            group_id=group_id,
            defined_entity_types=[entity_type],
            enrich_with_edges=enrich_with_edges,
        )
        return result.entities
