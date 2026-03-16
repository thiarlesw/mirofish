"""
Graphiti CRUD API
Endpoints para operações diretas no grafo Graphiti (KuzuDB).
Usado pelo MCP graphiti-conector.
"""

import asyncio
import concurrent.futures
from datetime import datetime, timezone
from typing import Optional

from flask import Blueprint, jsonify, request

from ..services.graphiti_client import get_graphiti
from ..utils.logger import get_logger

graphiti_crud_bp = Blueprint("graphiti_crud", __name__)
logger = get_logger("mirofish.api.graphiti_crud")


def _run(coro):
    """Executa corrotina async em contexto Flask (sync)."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, coro).result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


# ─── Episódios ─────────────────────────────────────────────────────────────────

@graphiti_crud_bp.route("/episode", methods=["POST"])
def add_episode():
    """
    Adiciona um episódio ao grafo.
    Body: { group_id, content, name, source_description? }
    """
    data = request.get_json(force=True) or {}
    group_id = data.get("group_id")
    content = data.get("content")
    name = data.get("name")
    source_description = data.get("source_description", "Graphiti MCP input")

    if not group_id or not content or not name:
        return jsonify({"success": False, "error": "group_id, content e name são obrigatórios"}), 400

    try:
        from graphiti_core.nodes import EpisodeType

        async def _add():
            graphiti = await get_graphiti()
            await graphiti.add_episode(
                name=name,
                episode_body=content,
                source_description=source_description,
                source=EpisodeType.text,
                group_id=group_id,
                reference_time=datetime.now(tz=timezone.utc),
            )

        _run(_add())
        logger.info(f"Episódio adicionado: group_id={group_id}, name={name}")
        return jsonify({"success": True, "message": f"Episódio '{name}' adicionado ao grupo '{group_id}'"})
    except Exception as e:
        logger.error(f"Erro ao adicionar episódio: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@graphiti_crud_bp.route("/episode/<uuid>", methods=["DELETE"])
def delete_episode(uuid: str):
    """Deleta um episódio pelo UUID."""
    try:
        async def _delete():
            graphiti = await get_graphiti()
            await graphiti.delete_episode(uuid)

        _run(_delete())
        return jsonify({"success": True, "message": f"Episódio {uuid} deletado"})
    except Exception as e:
        logger.error(f"Erro ao deletar episódio {uuid}: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# ─── Busca ─────────────────────────────────────────────────────────────────────

@graphiti_crud_bp.route("/search", methods=["GET", "POST"])
def search():
    """
    Busca semântica no grafo.
    GET: ?q=query&group_ids=id1,id2&num_results=10
    POST: { query, group_ids?, num_results? }
    """
    if request.method == "POST":
        data = request.get_json(force=True) or {}
        query = data.get("query") or data.get("q", "")
        group_ids_raw = data.get("group_ids")
        num_results = int(data.get("num_results", 10))
    else:
        query = request.args.get("q", "")
        raw = request.args.get("group_ids", "")
        group_ids_raw = [g.strip() for g in raw.split(",") if g.strip()] if raw else None
        num_results = int(request.args.get("num_results", 10))

    if not query:
        return jsonify({"success": False, "error": "query é obrigatória"}), 400

    if isinstance(group_ids_raw, str):
        group_ids_raw = [g.strip() for g in group_ids_raw.split(",") if g.strip()]

    try:
        async def _search():
            graphiti = await get_graphiti()
            return await graphiti.search(
                query=query,
                group_ids=group_ids_raw or None,
                num_results=num_results,
            )

        results = _run(_search())

        items = []
        for r in results:
            node = getattr(r, "node", None)
            edge = getattr(r, "edge", None)
            fact = getattr(r, "fact", None)
            if node:
                items.append({
                    "type": "node",
                    "uuid": str(getattr(node, "uuid", "")),
                    "name": getattr(node, "name", ""),
                    "summary": getattr(node, "summary", ""),
                    "group_id": getattr(node, "group_id", ""),
                    "labels": list(getattr(node, "labels", [])),
                })
            elif edge:
                items.append({
                    "type": "edge",
                    "uuid": str(getattr(edge, "uuid", "")),
                    "name": getattr(edge, "name", ""),
                    "fact": getattr(edge, "fact", ""),
                    "group_id": getattr(edge, "group_id", ""),
                })
            elif fact:
                items.append({"type": "fact", "content": str(fact)})
            else:
                items.append({"type": "unknown", "raw": str(r)})

        return jsonify({"success": True, "results": items, "total": len(items)})
    except Exception as e:
        logger.error(f"Erro na busca: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# ─── Nós ───────────────────────────────────────────────────────────────────────

@graphiti_crud_bp.route("/nodes/<group_id>", methods=["GET"])
def list_nodes(group_id: str):
    """Lista todos os nós de um group_id."""
    limit = int(request.args.get("limit", 200))

    try:
        async def _nodes():
            graphiti = await get_graphiti()
            import kuzu
            conn = kuzu.Connection(graphiti.driver.db)
            result = conn.execute(
                f"MATCH (n:Entity) WHERE n.group_id = $g "
                f"RETURN n.uuid AS uuid, n.name AS name, n.summary AS summary, "
                f"labels(n) AS labels LIMIT {limit}",
                parameters={"g": group_id},
            )
            rows = []
            while result.has_next():
                row = result.get_next()
                rows.append({
                    "uuid": str(row[0]),
                    "name": row[1],
                    "summary": row[2],
                    "labels": row[3],
                })
            conn.close()
            return rows

        nodes = _run(_nodes())
        return jsonify({"success": True, "nodes": nodes, "total": len(nodes), "group_id": group_id})
    except Exception as e:
        logger.error(f"Erro ao listar nós ({group_id}): {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@graphiti_crud_bp.route("/node/<uuid>", methods=["DELETE"])
def delete_node(uuid: str):
    """Deleta um nó pelo UUID."""
    try:
        async def _delete():
            graphiti = await get_graphiti()
            import kuzu
            conn = kuzu.Connection(graphiti.driver.db)
            conn.execute("MATCH (n:Entity {uuid: $u}) DELETE n", parameters={"u": uuid})
            conn.close()

        _run(_delete())
        return jsonify({"success": True, "message": f"Nó {uuid} deletado"})
    except Exception as e:
        logger.error(f"Erro ao deletar nó {uuid}: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@graphiti_crud_bp.route("/node/<uuid>", methods=["PATCH"])
def update_node(uuid: str):
    """
    Atualiza campos de um nó.
    Body: { name?, summary? }
    """
    data = request.get_json(force=True) or {}
    name = data.get("name")
    summary = data.get("summary")

    if not name and not summary:
        return jsonify({"success": False, "error": "Informe name ou summary para atualizar"}), 400

    try:
        async def _update():
            graphiti = await get_graphiti()
            import kuzu
            conn = kuzu.Connection(graphiti.driver.db)
            sets = []
            params = {"u": uuid}
            if name:
                sets.append("n.name = $name")
                params["name"] = name
            if summary:
                sets.append("n.summary = $summary")
                params["summary"] = summary
            conn.execute(
                f"MATCH (n:Entity {{uuid: $u}}) SET {', '.join(sets)}",
                parameters=params,
            )
            conn.close()

        _run(_update())
        return jsonify({"success": True, "message": f"Nó {uuid} atualizado"})
    except Exception as e:
        logger.error(f"Erro ao atualizar nó {uuid}: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# ─── Arestas ───────────────────────────────────────────────────────────────────

@graphiti_crud_bp.route("/edges/<group_id>", methods=["GET"])
def list_edges(group_id: str):
    """Lista todas as arestas de um group_id."""
    limit = int(request.args.get("limit", 200))

    try:
        async def _edges():
            graphiti = await get_graphiti()
            import kuzu
            conn = kuzu.Connection(graphiti.driver.db)
            result = conn.execute(
                f"MATCH (s:Entity)-[r:RelatesToNode_]->(t:Entity) "
                f"WHERE r.group_id = $g "
                f"RETURN r.uuid AS uuid, r.name AS name, r.fact AS fact, "
                f"s.uuid AS src, t.uuid AS tgt LIMIT {limit}",
                parameters={"g": group_id},
            )
            rows = []
            while result.has_next():
                row = result.get_next()
                rows.append({
                    "uuid": str(row[0]),
                    "name": row[1],
                    "fact": row[2],
                    "source_uuid": str(row[3]),
                    "target_uuid": str(row[4]),
                })
            conn.close()
            return rows

        edges = _run(_edges())
        return jsonify({"success": True, "edges": edges, "total": len(edges), "group_id": group_id})
    except Exception as e:
        logger.error(f"Erro ao listar arestas ({group_id}): {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# ─── Grupos ────────────────────────────────────────────────────────────────────

@graphiti_crud_bp.route("/groups", methods=["GET"])
def list_groups():
    """Lista todos os group_ids distintos no grafo."""
    try:
        async def _groups():
            graphiti = await get_graphiti()
            import kuzu
            conn = kuzu.Connection(graphiti.driver.db)
            result = conn.execute(
                "MATCH (n:Entity) RETURN DISTINCT n.group_id AS group_id ORDER BY group_id"
            )
            groups = []
            while result.has_next():
                row = result.get_next()
                if row[0]:
                    groups.append(row[0])
            conn.close()
            return groups

        groups = _run(_groups())
        return jsonify({"success": True, "groups": groups, "total": len(groups)})
    except Exception as e:
        logger.error(f"Erro ao listar grupos: {e}")
        return jsonify({"success": False, "error": str(e)}), 500
