"""
Graphiti retrieval tools service.
Wraps graph search, node reading, edge queries, and other tools for use by the Report Agent.

Replaces the ZepToolsService from zep_tools.py, now using Graphiti + Kuzu + Gemini.

Core retrieval tools:
1. InsightForge (deep insight retrieval) - The most powerful hybrid retrieval: automatically
   generates sub-questions and searches across multiple dimensions.
2. PanoramaSearch (breadth search) - Gets the full picture.
   Note: Graphiti does not distinguish "expired/invalid" edges. All search() results are
   EntityEdges currently stored in the knowledge graph. This implementation uses multiple
   sub-queries to simulate breadth, and preserves valid_at / invalid_at fields in the edges
   structure (natively supported by Graphiti EntityEdge).
3. QuickSearch (simple search) - Fast retrieval.

API differences (compared to Zep):
- Graphiti has no separate graph_id concept; it uses group_id (string) to isolate multi-tenant
  data. This module still accepts graph_id as a parameter name for interface compatibility,
  and maps it internally to group_id.
- Graphiti has no paginated API for fetching all nodes/edges. get_all_nodes/get_all_edges are
  approximated via search(query="", num_results=9999).
- Graphiti's EntityEdge includes .fact, .name, .source_node_name, .target_node_name fields,
  which are directly usable.
- Graphiti does not provide a single-node query API like get_node_detail(uuid); entity
  information is inferred from the source/target node names in search() EntityEdge results.
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
# Data class definitions (maintaining the same public interface as zep_tools.py)
# ────────────────────────────────────────────────────────────────

@dataclass
class SearchResult:
    """Search result"""
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
        """Convert to text format for LLM consumption"""
        text_parts = [f"Search query: {self.query}", f"Found {self.total_count} relevant items"]
        if self.facts:
            text_parts.append("\n### Related facts:")
            for i, fact in enumerate(self.facts, 1):
                text_parts.append(f"{i}. {fact}")
        return "\n".join(text_parts)


@dataclass
class NodeInfo:
    """Node information (inferred from EntityEdge, no uuid)"""
    uuid: str      # Graphiti has no direct node uuid API; leave empty or use name as identifier
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
        entity_type = next((l for l in self.labels if l not in ["Entity", "Node"]), "Unknown type")
        return f"Entity: {self.name} (type: {entity_type})\nSummary: {self.summary}"


@dataclass
class EdgeInfo:
    """Edge information"""
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
    expired_at: Optional[str] = None  # Zep-specific field; retained for compatibility

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
        base_text = f"Relationship: {source} --[{self.name}]--> {target}\nFact: {self.fact}"
        if include_temporal and self.valid_at:
            invalid_at = self.invalid_at or "present"
            base_text += f"\nValidity: {self.valid_at} - {invalid_at}"
        return base_text

    @property
    def is_expired(self) -> bool:
        return self.expired_at is not None

    @property
    def is_invalid(self) -> bool:
        return self.invalid_at is not None


@dataclass
class InsightForgeResult:
    """Deep insight retrieval result (InsightForge)"""
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
            f"## Deep Future Prediction Analysis",
            f"Analysis question: {self.query}",
            f"Prediction scenario: {self.simulation_requirement}",
            f"\n### Prediction data statistics",
            f"- Relevant predicted facts: {self.total_facts}",
            f"- Entities involved: {self.total_entities}",
            f"- Relationship chains: {self.total_relationships}",
        ]
        if self.sub_queries:
            text_parts.append(f"\n### Sub-questions analyzed")
            for i, sq in enumerate(self.sub_queries, 1):
                text_parts.append(f"{i}. {sq}")
        if self.semantic_facts:
            text_parts.append(f"\n### [Key facts] (cite these verbatim in the report)")
            for i, fact in enumerate(self.semantic_facts, 1):
                text_parts.append(f'{i}. "{fact}"')
        if self.entity_insights:
            text_parts.append(f"\n### [Core entities]")
            for entity in self.entity_insights:
                text_parts.append(f"- **{entity.get('name', 'Unknown')}** ({entity.get('type', 'Entity')})")
                if entity.get('summary'):
                    text_parts.append(f"  Summary: \"{entity.get('summary')}\"")
                if entity.get('related_facts'):
                    text_parts.append(f"  Related facts: {len(entity.get('related_facts', []))}")
        if self.relationship_chains:
            text_parts.append(f"\n### [Relationship chains]")
            for chain in self.relationship_chains:
                text_parts.append(f"- {chain}")
        return "\n".join(text_parts)


@dataclass
class PanoramaResult:
    """Breadth search result (Panorama)"""
    query: str

    all_nodes: List[NodeInfo] = field(default_factory=list)
    all_edges: List[EdgeInfo] = field(default_factory=list)
    active_facts: List[str] = field(default_factory=list)
    # Note: Graphiti does not distinguish expired/invalid edges;
    # historical_facts are determined by the invalid_at field
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
            f"## Breadth Search Results (Future panoramic view)",
            f"Query: {self.query}",
            f"\n### Statistics",
            f"- Total nodes: {self.total_nodes}",
            f"- Total edges: {self.total_edges}",
            f"- Currently active facts: {self.active_count}",
            f"- Historical/expired facts: {self.historical_count}",
        ]
        if self.active_facts:
            text_parts.append(f"\n### [Currently active facts] (raw simulation results)")
            for i, fact in enumerate(self.active_facts, 1):
                text_parts.append(f'{i}. "{fact}"')
        if self.historical_facts:
            text_parts.append(f"\n### [Historical/expired facts] (evolution process records)")
            for i, fact in enumerate(self.historical_facts, 1):
                text_parts.append(f'{i}. "{fact}"')
        if self.all_nodes:
            text_parts.append(f"\n### [Entities involved]")
            for node in self.all_nodes:
                entity_type = next((l for l in node.labels if l not in ["Entity", "Node"]), "Entity")
                text_parts.append(f"- **{node.name}** ({entity_type})")
        return "\n".join(text_parts)


@dataclass
class AgentInterview:
    """Interview result for a single Agent"""
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
        text += f"_Bio: {self.agent_bio}_\n\n"
        text += f"**Q:** {self.question}\n\n"
        text += f"**A:** {self.response}\n"
        if self.key_quotes:
            text += "\n**Key quotes:**\n"
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
    """Interview result (Interview)"""
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
            "## In-Depth Interview Report",
            f"**Interview topic:** {self.interview_topic}",
            f"**Interviewees:** {self.interviewed_count} / {self.total_agents} simulated agents",
            "\n### Interviewee selection rationale",
            self.selection_reasoning or "(automatically selected)",
            "\n---",
            "\n### Interview transcript",
        ]
        if self.interviews:
            for i, interview in enumerate(self.interviews, 1):
                text_parts.append(f"\n#### Interview #{i}: {interview.agent_name}")
                text_parts.append(interview.to_text())
                text_parts.append("\n---")
        else:
            text_parts.append("(No interview records)\n\n---")
        text_parts.append("\n### Interview summary and key insights")
        text_parts.append(self.summary or "(No summary)")
        return "\n".join(text_parts)


# ────────────────────────────────────────────────────────────────
# Helper function: convert EntityEdge to internal data classes
# ────────────────────────────────────────────────────────────────

def _edge_to_edge_info(edge) -> EdgeInfo:
    """Convert a graphiti_core EntityEdge object to EdgeInfo"""
    uuid = str(getattr(edge, 'uuid', None) or "")
    name = str(getattr(edge, 'name', None) or "")
    fact = str(getattr(edge, 'fact', None) or "")
    source_node_name = str(getattr(edge, 'source_node_name', None) or "")
    target_node_name = str(getattr(edge, 'target_node_name', None) or "")

    # Graphiti's EntityEdge uses source_node_uuid / target_node_uuid;
    # may also be source_node_id / target_node_id — handle both forms
    source_node_uuid = str(
        getattr(edge, 'source_node_uuid', None)
        or getattr(edge, 'source_node_id', None)
        or source_node_name  # fallback: use name instead of uuid
        or ""
    )
    target_node_uuid = str(
        getattr(edge, 'target_node_uuid', None)
        or getattr(edge, 'target_node_id', None)
        or target_node_name
        or ""
    )

    # Temporal fields (natively supported by Graphiti EntityEdge)
    def _dt_str(val) -> Optional[str]:
        if val is None:
            return None
        if isinstance(val, datetime):
            return val.isoformat()
        return str(val)

    created_at = _dt_str(getattr(edge, 'created_at', None))
    valid_at = _dt_str(getattr(edge, 'valid_at', None))
    invalid_at = _dt_str(getattr(edge, 'invalid_at', None))
    # expired_at is Zep-specific; Graphiti does not have this field
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
    """Run an async coroutine from a synchronous context (compatible with Flask and other non-async frameworks)"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # An event loop is already running (e.g., inside Jupyter or an async framework)
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


# ────────────────────────────────────────────────────────────────
# Main service class
# ────────────────────────────────────────────────────────────────

class GraphitiToolsService:
    """
    Graphiti retrieval tools service (replaces the original ZepToolsService).

    All graph_id parameters are internally mapped to Graphiti's group_id,
    maintaining full compatibility with the original ZepToolsService interface.

    Core retrieval tools:
    1. insight_forge       - Deep insight retrieval
    2. panorama_search     - Breadth search
    3. quick_search        - Fast retrieval
    4. interview_agents    - In-depth interview (calls the real OASIS API)

    Basic tools:
    - search_graph         - Graph semantic search
    - get_all_nodes        - Get all nodes in the graph (approximate, inferred from edges)
    - get_all_edges        - Get all edges in the graph
    - get_entities_by_type - Get entities by type
    - get_entity_summary   - Get relationship summary for an entity
    - get_graph_statistics - Get graph statistics
    - get_simulation_context - Get simulation context
    """

    MAX_RETRIES = 3
    RETRY_DELAY = 2.0

    def __init__(self, llm_client: Optional[LLMClient] = None):
        # Graphiti is lazy-loaded via graphiti_client.get_graphiti(); no init needed here
        self._llm_client = llm_client
        logger.info("GraphitiToolsService initialized")

    @property
    def llm(self) -> LLMClient:
        """Lazily initialize the LLM client"""
        if self._llm_client is None:
            self._llm_client = LLMClient()
        return self._llm_client

    # ──────────────────────────────────────────────────────────────
    # Internal helper methods
    # ──────────────────────────────────────────────────────────────

    async def _search_async(
        self,
        group_id: str,
        query: str,
        num_results: int = 10,
    ) -> List:
        """Async call to graphiti.search(); returns a list of EntityEdge objects"""
        graphiti = await get_graphiti()
        try:
            results = await graphiti.search(
                query=query,
                group_ids=[group_id],
                num_results=num_results,
            )
            return results if results else []
        except Exception as e:
            logger.warning(f"Graphiti search failed (query={query[:40]}...): {e}")
            return []

    def _search(self, group_id: str, query: str, num_results: int = 10) -> List:
        """Synchronous wrapper for _search_async"""
        return _run_async(self._search_async(group_id, query, num_results))

    # ──────────────────────────────────────────────────────────────
    # Basic tools
    # ──────────────────────────────────────────────────────────────

    def search_graph(
        self,
        graph_id: str,
        query: str,
        limit: int = 10,
        scope: str = "edges",  # retained parameter; Graphiti searches edges by default
    ) -> SearchResult:
        """
        Graph semantic search.

        Graphiti's search() returns a list of EntityEdge objects, each containing a .fact field.
        The scope parameter is retained for compatibility; Graphiti does not differentiate
        between edges/nodes search.

        Args:
            graph_id: Graph ID (internally mapped to group_id)
            query: Search query
            limit: Number of results to return
            scope: Search scope (compatibility parameter; Graphiti searches edges only)

        Returns:
            SearchResult: Search results
        """
        group_id = graph_id
        logger.info(f"Graph search: group_id={group_id}, query={query[:50]}...")

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

            # Infer nodes from edges (Graphiti has no independent node API)
            for node_name in [edge_info.source_node_name, edge_info.target_node_name]:
                if node_name and node_name not in seen_nodes:
                    seen_nodes.add(node_name)
                    nodes.append({
                        "uuid": "",
                        "name": node_name,
                        "labels": [],
                        "summary": "",
                    })

        logger.info(f"Search complete: found {len(facts)} relevant facts")
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
        Get all edges in the graph.

        Note: Graphiti has no dedicated API for enumerating all edges.
        This method approximates it via search(query=""), returning all stored EntityEdges.
        If the num_results limit is too low, not all edges may be returned.

        Args:
            graph_id: Graph ID (internally mapped to group_id)
            include_temporal: Compatibility parameter; temporal info is always included

        Returns:
            List of edges
        """
        group_id = graph_id
        logger.info(f"Fetching all edges for graph {group_id} (via empty query approximation)...")

        raw_edges = self._search(group_id, query="", num_results=9999)

        result = []
        for edge in raw_edges:
            result.append(_edge_to_edge_info(edge))

        logger.info(f"Retrieved {len(result)} edges")
        return result

    def get_all_nodes(self, graph_id: str) -> List[NodeInfo]:
        """
        Get all nodes in the graph (inferred from edges).

        Graphiti does not provide a standalone node list API.
        This method fetches all edges and deduplicates source_node_name / target_node_name
        to build the node set.

        Args:
            graph_id: Graph ID (internally mapped to group_id)

        Returns:
            List of nodes (uuid is empty; name is used as identifier)
        """
        logger.info(f"Fetching all nodes for graph {graph_id} (inferred from edges)...")

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

        logger.info(f"Inferred {len(result)} nodes")
        return result

    def get_entities_by_type(
        self,
        graph_id: str,
        entity_type: str,
    ) -> List[NodeInfo]:
        """
        Get entities by type.

        Note: Graphiti does not carry entity type labels (labels) in EntityEdge.
        This method approximates the result by searching for the entity_type keyword
        and treating the resulting node names as entities of that type.

        Args:
            graph_id: Graph ID
            entity_type: Entity type keyword

        Returns:
            List of entities matching the type
        """
        logger.info(f"Fetching entities of type {entity_type} (via keyword search approximation)...")

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

        logger.info(f"Found {len(result)} entities related to {entity_type}")
        return result

    def get_entity_summary(
        self,
        graph_id: str,
        entity_name: str,
    ) -> Dict[str, Any]:
        """
        Get the relationship summary for a specified entity.

        Args:
            graph_id: Graph ID
            entity_name: Entity name

        Returns:
            Entity summary information
        """
        logger.info(f"Fetching relationship summary for entity {entity_name}...")

        search_result = self.search_graph(graph_id=graph_id, query=entity_name, limit=20)

        # Find edges directly related to the entity
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
        Get statistics for the graph.

        Note: Graphiti has no node type labels. entity_types and relation_types
        are counted from the name field of edges.

        Args:
            graph_id: Graph ID

        Returns:
            Statistics information
        """
        logger.info(f"Fetching statistics for graph {graph_id}...")

        all_edges = self.get_all_edges(graph_id)

        # Infer node set from edges
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
            # Graphiti has no type labels; entity_types is empty (field retained for compatibility)
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
        Get context information relevant to the simulation.

        Args:
            graph_id: Graph ID
            simulation_requirement: Description of the simulation requirement
            limit: Maximum number of items per category

        Returns:
            Simulation context information
        """
        logger.info(f"Fetching simulation context: {simulation_requirement[:50]}...")

        search_result = self.search_graph(
            graph_id=graph_id,
            query=simulation_requirement,
            limit=limit,
        )
        stats = self.get_graph_statistics(graph_id)
        all_nodes = self.get_all_nodes(graph_id)

        # Graphiti has no type labels; all nodes are treated as meaningful entities
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
    # Core retrieval tools
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
        [InsightForge - Deep Insight Retrieval]

        1. Use LLM to decompose the question into multiple sub-questions
        2. Run Graphiti semantic search for each sub-question
        3. Extract related entities from edges and summarize
        4. Build relationship chains
        5. Aggregate all results to produce deep insights

        Args:
            graph_id: Graph ID (corresponds to group_id)
            query: User question
            simulation_requirement: Description of simulation requirement
            report_context: Report context (optional)
            max_sub_queries: Maximum number of sub-questions

        Returns:
            InsightForgeResult: Deep insight retrieval result
        """
        logger.info(f"InsightForge deep insight retrieval: {query[:50]}...")

        result = InsightForgeResult(
            query=query,
            simulation_requirement=simulation_requirement,
            sub_queries=[],
        )

        # Step 1: Generate sub-questions using LLM
        sub_queries = self._generate_sub_queries(
            query=query,
            simulation_requirement=simulation_requirement,
            report_context=report_context,
            max_queries=max_sub_queries,
        )
        result.sub_queries = sub_queries
        logger.info(f"Generated {len(sub_queries)} sub-questions")

        # Step 2: Run semantic search for each sub-question
        all_facts: List[str] = []
        all_edge_dicts: List[Dict[str, Any]] = []
        seen_facts: set = set()
        # node name -> NodeInfo mapping (used for relationship chain construction)
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
                # Update node_map
                for name in [edge_info.source_node_name, edge_info.target_node_name]:
                    if name and name not in node_map:
                        node_map[name] = NodeInfo(uuid="", name=name, labels=[], summary="", attributes={})

        # Also search with the original question
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

        # Step 3: Aggregate entity insights
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

        # Step 4: Build relationship chains
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
            f"InsightForge complete: {result.total_facts} facts, "
            f"{result.total_entities} entities, {result.total_relationships} relationships"
        )
        return result

    def _generate_sub_queries(
        self,
        query: str,
        simulation_requirement: str,
        report_context: str = "",
        max_queries: int = 5,
    ) -> List[str]:
        """Use LLM to decompose a complex question into multiple independently searchable sub-questions"""
        system_prompt = (
            "You are a professional question analysis expert. Your task is to break down a complex question "
            "into multiple sub-questions that can each be independently observed in a simulated world.\n\n"
            "Requirements:\n"
            "1. Each sub-question should be specific enough to find relevant agent behaviors or events in the simulation\n"
            "2. Sub-questions should cover different dimensions of the original question (e.g., who, what, why, how, when, where)\n"
            "3. Sub-questions should be relevant to the simulation scenario\n"
            '4. Return JSON format: {"sub_queries": ["sub-question 1", "sub-question 2", ...]}'
        )
        user_prompt = (
            f"Simulation background:\n{simulation_requirement}\n\n"
            + (f"Report context: {report_context[:500]}\n\n" if report_context else "")
            + f"Please break down the following question into {max_queries} sub-questions:\n{query}\n\nReturn a JSON list of sub-questions."
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
            logger.warning(f"Failed to generate sub-questions: {str(e)}, using default sub-questions")
            return [
                query,
                f"Main participants in {query}",
                f"Causes and effects of {query}",
                f"How {query} developed",
            ][:max_queries]

    def panorama_search(
        self,
        graph_id: str,
        query: str,
        include_expired: bool = True,
        limit: int = 50,
    ) -> PanoramaResult:
        """
        [PanoramaSearch - Breadth Search]

        Gets a full panoramic view, including all related content:
        1. Fetch all related edges (via empty query + keyword query)
        2. Infer node set from edges
        3. Classify facts as currently active or historical based on valid_at / invalid_at

        Note: Unlike Zep, Graphiti does not track "expired" edges separately.
        This method treats edges where invalid_at is not None as "historical facts".
        The include_expired parameter is retained for compatibility.

        Args:
            graph_id: Graph ID
            query: Search query (used for relevance sorting)
            include_expired: Whether to include historical/invalid content (default True)
            limit: Maximum number of results to return

        Returns:
            PanoramaResult: Breadth search result
        """
        logger.info(f"PanoramaSearch breadth search: {query[:50]}...")

        result = PanoramaResult(query=query)

        # Two-round search: empty query (full scan) + keyword query (relevance)
        all_raw_edges = self._search(graph_id, query="", num_results=9999)
        if query.strip():
            keyword_raw = self._search(graph_id, query=query, num_results=limit)
            # Merge, deduplicating by uuid
            seen_uuids = set()
            merged = []
            for edge in keyword_raw + all_raw_edges:
                uid = str(getattr(edge, 'uuid', id(edge)))
                if uid not in seen_uuids:
                    seen_uuids.add(uid)
                    merged.append(edge)
            all_raw_edges = merged

        # Convert and build node_map
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

        # Classify facts
        active_facts: List[str] = []
        historical_facts: List[str] = []

        for edge in all_edges:
            if not edge.fact:
                continue
            # Graphiti: invalid_at not None means historical/invalid
            is_historical = edge.invalid_at is not None or edge.expired_at is not None
            if is_historical:
                valid_at = edge.valid_at or "unknown"
                invalid_at = edge.invalid_at or edge.expired_at or "unknown"
                historical_facts.append(f"[{valid_at} - {invalid_at}] {edge.fact}")
            else:
                active_facts.append(edge.fact)

        # Sort by relevance
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
            f"PanoramaSearch complete: {result.active_count} active, {result.historical_count} historical"
        )
        return result

    def quick_search(
        self,
        graph_id: str,
        query: str,
        limit: int = 10,
    ) -> SearchResult:
        """
        [QuickSearch - Simple Search]

        Fast, lightweight retrieval tool:
        Directly calls Graphiti semantic search and returns the most relevant results.

        Args:
            graph_id: Graph ID
            query: Search query
            limit: Number of results to return

        Returns:
            SearchResult: Search results
        """
        logger.info(f"QuickSearch simple search: {query[:50]}...")
        result = self.search_graph(graph_id=graph_id, query=query, limit=limit, scope="edges")
        logger.info(f"QuickSearch complete: {result.total_count} results")
        return result

    # ──────────────────────────────────────────────────────────────
    # Interview tool (same logic as ZepToolsService, only graph_id parameter name differs)
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
        [InterviewAgents - In-Depth Interview]

        Calls the real OASIS interview API to interview agents running in the simulation.
        Logic is identical to the original ZepToolsService.interview_agents; does not depend on Graphiti.

        Args:
            simulation_id: Simulation ID
            interview_requirement: Description of the interview requirement
            simulation_requirement: Simulation background context (optional)
            max_agents: Maximum number of agents to interview
            custom_questions: Custom interview questions (optional)

        Returns:
            InterviewResult: Interview result
        """
        from .simulation_runner import SimulationRunner

        logger.info(f"InterviewAgents in-depth interview (real API): {interview_requirement[:50]}...")

        result = InterviewResult(
            interview_topic=interview_requirement,
            interview_questions=custom_questions or [],
        )

        # Step 1: Load agent persona files
        profiles = self._load_agent_profiles(simulation_id)
        if not profiles:
            logger.warning(f"No persona files found for simulation {simulation_id}")
            result.summary = "No interviewable agent persona files found"
            return result

        result.total_agents = len(profiles)
        logger.info(f"Loaded {len(profiles)} agent personas")

        # Step 2: Use LLM to select agents for interview
        selected_agents, selected_indices, selection_reasoning = self._select_agents_for_interview(
            profiles=profiles,
            interview_requirement=interview_requirement,
            simulation_requirement=simulation_requirement,
            max_agents=max_agents,
        )
        result.selected_agents = selected_agents
        result.selection_reasoning = selection_reasoning
        logger.info(f"Selected {len(selected_agents)} agents for interview: {selected_indices}")

        # Step 3: Generate interview questions
        if not result.interview_questions:
            result.interview_questions = self._generate_interview_questions(
                interview_requirement=interview_requirement,
                simulation_requirement=simulation_requirement,
                selected_agents=selected_agents,
            )
            logger.info(f"Generated {len(result.interview_questions)} interview questions")

        combined_prompt = "\n".join(
            [f"{i+1}. {q}" for i, q in enumerate(result.interview_questions)]
        )

        INTERVIEW_PROMPT_PREFIX = (
            "You are being interviewed. Based on your persona, all past memories, and actions, "
            "please answer the following questions in plain text.\n"
            "Response requirements:\n"
            "1. Answer directly in natural language; do not call any tools\n"
            "2. Do not return JSON format or tool-call format\n"
            "3. Do not use Markdown headings (e.g., #, ##, ###)\n"
            "4. Answer each question by number, starting each answer with 'Question X:' (where X is the question number)\n"
            "5. Separate answers with a blank line\n"
            "6. Provide substantive content; answer each question with at least 2-3 sentences\n\n"
        )
        optimized_prompt = f"{INTERVIEW_PROMPT_PREFIX}{combined_prompt}"

        # Step 4: Call the real interview API
        try:
            interviews_request = []
            for agent_idx in selected_indices:
                interviews_request.append({
                    "agent_id": agent_idx,
                    "prompt": optimized_prompt,
                })

            logger.info(f"Calling batch interview API (dual platform): {len(interviews_request)} agents")

            api_result = SimulationRunner.interview_agents_batch(
                simulation_id=simulation_id,
                interviews=interviews_request,
                platform=None,
                timeout=180.0,
            )

            logger.info(
                f"Interview API response: {api_result.get('interviews_count', 0)} results, "
                f"success={api_result.get('success')}"
            )

            if not api_result.get("success", False):
                error_msg = api_result.get("error", "Unknown error")
                logger.warning(f"Interview API returned failure: {error_msg}")
                result.summary = f"Interview API call failed: {error_msg}. Please check the OASIS simulation environment status."
                return result

            # Step 5: Parse API response
            api_data = api_result.get("result", {})
            results_dict = api_data.get("results", {}) if isinstance(api_data, dict) else {}

            for i, agent_idx in enumerate(selected_indices):
                agent = selected_agents[i]
                agent_name = agent.get("realname", agent.get("username", f"Agent_{agent_idx}"))
                agent_role = agent.get("profession", "Unknown")
                agent_bio = agent.get("bio", "")

                twitter_result = results_dict.get(f"twitter_{agent_idx}", {})
                reddit_result = results_dict.get(f"reddit_{agent_idx}", {})

                twitter_response = self._clean_tool_call_response(
                    twitter_result.get("response", "")
                )
                reddit_response = self._clean_tool_call_response(
                    reddit_result.get("response", "")
                )

                twitter_text = twitter_response if twitter_response else "(No response from this platform)"
                reddit_text = reddit_response if reddit_response else "(No response from this platform)"
                response_text = f"[Twitter platform response]\n{twitter_text}\n\n[Reddit platform response]\n{reddit_text}"

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
            logger.warning(f"Interview API call failed (environment not running?): {e}")
            result.summary = f"Interview failed: {str(e)}. The simulation environment may have shut down; please ensure OASIS is running."
            return result
        except Exception as e:
            logger.error(f"Interview API call exception: {e}")
            import traceback
            logger.error(traceback.format_exc())
            result.summary = f"An error occurred during the interview: {str(e)}"
            return result

        # Step 6: Generate interview summary
        if result.interviews:
            result.summary = self._generate_interview_summary(
                interviews=result.interviews,
                interview_requirement=interview_requirement,
            )

        logger.info(f"InterviewAgents complete: interviewed {result.interviewed_count} agents (dual platform)")
        return result

    @staticmethod
    def _clean_tool_call_response(response: str) -> str:
        """Strip JSON tool-call wrappers from agent responses and extract the actual content"""
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
        """Load agent persona files for a simulation"""
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
                logger.info(f"Loaded {len(profiles)} personas from reddit_profiles.json")
                return profiles
            except Exception as e:
                logger.warning(f"Failed to read reddit_profiles.json: {e}")

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
                            "profession": "Unknown",
                        })
                logger.info(f"Loaded {len(profiles)} personas from twitter_profiles.csv")
                return profiles
            except Exception as e:
                logger.warning(f"Failed to read twitter_profiles.csv: {e}")

        return profiles

    def _select_agents_for_interview(
        self,
        profiles: List[Dict[str, Any]],
        interview_requirement: str,
        simulation_requirement: str,
        max_agents: int,
    ) -> tuple:
        """Use LLM to select agents for interview; returns (selected_agents, selected_indices, reasoning)"""
        agent_summaries = []
        for i, profile in enumerate(profiles):
            agent_summaries.append({
                "index": i,
                "name": profile.get("realname", profile.get("username", f"Agent_{i}")),
                "profession": profile.get("profession", "Unknown"),
                "bio": profile.get("bio", "")[:200],
                "interested_topics": profile.get("interested_topics", []),
            })

        system_prompt = (
            "You are a professional interview planning expert. Your task is to select the most suitable "
            "agents to interview from a list of simulated agents, based on the interview requirement.\n\n"
            "Selection criteria:\n"
            "1. The agent's identity/profession is relevant to the interview topic\n"
            "2. The agent likely holds a unique or valuable perspective\n"
            "3. Select a diverse range of viewpoints (e.g., supporters, opponents, neutrals, experts)\n"
            "4. Prioritize roles directly related to the event\n\n"
            "Return JSON format:\n"
            '{\n    "selected_indices": [list of selected agent indices],\n    "reasoning": "explanation of selection rationale"\n}'
        )
        user_prompt = (
            f"Interview requirement:\n{interview_requirement}\n\n"
            f"Simulation background:\n{simulation_requirement if simulation_requirement else 'Not provided'}\n\n"
            f"Available agents (total {len(agent_summaries)}):\n"
            + json.dumps(agent_summaries, ensure_ascii=False, indent=2)
            + f"\n\nPlease select up to {max_agents} agents best suited for interview and explain your selection rationale."
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
            reasoning = response.get("reasoning", "Automatically selected based on relevance")
            selected_agents = []
            valid_indices = []
            for idx in selected_indices:
                if 0 <= idx < len(profiles):
                    selected_agents.append(profiles[idx])
                    valid_indices.append(idx)
            return selected_agents, valid_indices, reasoning
        except Exception as e:
            logger.warning(f"LLM agent selection failed, using default selection: {e}")
            selected = profiles[:max_agents]
            indices = list(range(min(max_agents, len(profiles))))
            return selected, indices, "Using default selection strategy"

    def _generate_interview_questions(
        self,
        interview_requirement: str,
        simulation_requirement: str,
        selected_agents: List[Dict[str, Any]],
    ) -> List[str]:
        """Use LLM to generate interview questions"""
        agent_roles = [a.get("profession", "Unknown") for a in selected_agents]

        system_prompt = (
            "You are a professional journalist/interviewer. Generate 3-5 in-depth interview questions "
            "based on the interview requirement.\n\n"
            "Question requirements:\n"
            "1. Open-ended questions that encourage detailed answers\n"
            "2. Questions that may elicit different answers from different roles\n"
            "3. Cover multiple dimensions: facts, opinions, feelings, etc.\n"
            "4. Natural language, as if in a real interview\n"
            "5. Keep each question under 50 words, concise and clear\n"
            "6. Ask directly, without background explanations or prefixes\n\n"
            'Return in JSON format: {"questions": ["question 1", "question 2", ...]}'
        )
        user_prompt = (
            f"Interview requirement: {interview_requirement}\n\n"
            f"Simulation background: {simulation_requirement if simulation_requirement else 'Not provided'}\n\n"
            f"Interviewee roles: {', '.join(agent_roles)}\n\n"
            "Please generate 3-5 interview questions."
        )

        try:
            response = self.llm.chat_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.5,
            )
            return response.get("questions", [f"What is your view on {interview_requirement}?"])
        except Exception as e:
            logger.warning(f"Failed to generate interview questions: {e}")
            return [
                f"What is your perspective on {interview_requirement}?",
                "How has this affected you or the group you represent?",
                "How do you think this issue should be addressed or improved?",
            ]

    def _generate_interview_summary(
        self,
        interviews: List[AgentInterview],
        interview_requirement: str,
    ) -> str:
        """Generate an interview summary"""
        if not interviews:
            return "No interviews were completed"

        interview_texts = []
        for interview in interviews:
            interview_texts.append(
                f"[{interview.agent_name} ({interview.agent_role})]\n{interview.response[:500]}"
            )

        system_prompt = (
            "You are a professional news editor. Based on the responses from multiple interviewees, "
            "generate an interview summary.\n\n"
            "Summary requirements:\n"
            "1. Distill the main viewpoints from each party\n"
            "2. Identify points of consensus and disagreement\n"
            "3. Highlight valuable quotes\n"
            "4. Remain objective and neutral, without favoring any side\n"
            "5. Keep it within 1000 words\n\n"
            "Formatting constraints (must follow):\n"
            "- Use plain text paragraphs, separated by blank lines\n"
            "- Do not use Markdown headings (e.g., #, ##, ###)\n"
            "- Do not use dividers (e.g., ---, ***)\n"
            "- Use quotation marks when citing interviewees' words\n"
            "- You may use **bold** to highlight key terms, but no other Markdown syntax"
        )
        user_prompt = (
            f"Interview topic: {interview_requirement}\n\n"
            f"Interview content:\n{''.join(interview_texts)}\n\n"
            "Please generate the interview summary."
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
            logger.warning(f"Failed to generate interview summary: {e}")
            return f"Interviewed {len(interviews)} respondents, including: " + ", ".join(
                [i.agent_name for i in interviews]
            )
