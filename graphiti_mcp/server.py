"""
Graphiti Connector MCP
MCP server com acesso total ao grafo Graphiti (leitura + escrita).
Usa auth JWT OAuth 2.1 (auth.rebote.cc).
"""

import base64
import hashlib
import hmac
import json
import os
import time
from typing import Optional

import httpx
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

# ─── Auth JWT ─────────────────────────────────────────────────────────────────

ISSUER = "https://auth.rebote.cc"
JWT_SECRET = os.environ.get("JWT_SECRET", "")
BACKEND_URL = os.environ.get("MIROFISH_BACKEND_URL", "http://localhost:5001")


def _b64decode_padding(s: str) -> bytes:
    s += "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s)


def verify_jwt(token: str) -> bool:
    if not JWT_SECRET or not token:
        return False
    try:
        header_b64, payload_b64, sig_b64 = token.split(".")
        expected_sig = (
            base64.urlsafe_b64encode(
                hmac.new(
                    JWT_SECRET.encode(),
                    f"{header_b64}.{payload_b64}".encode(),
                    hashlib.sha256,
                ).digest()
            )
            .rstrip(b"=")
            .decode()
        )
        if not hmac.compare_digest(sig_b64, expected_sig):
            return False
        payload = json.loads(_b64decode_padding(payload_b64))
        if payload.get("exp", 0) < time.time():
            return False
        if payload.get("iss") != ISSUER:
            return False
        return True
    except Exception:
        return False


class JWTAuthMiddleware(BaseHTTPMiddleware):
    BYPASS_PATHS = {
        "/mcp",
        "/.well-known/oauth-protected-resource",
        "/health",
    }

    async def dispatch(self, request, call_next):
        if request.url.path in self.BYPASS_PATHS:
            return await call_next(request)

        # OAuth 2.1 discovery / resource metadata
        if request.url.path.startswith("/.well-known"):
            return await call_next(request)

        auth_header = request.headers.get("Authorization", "")
        token = auth_header.removeprefix("Bearer ").strip()
        if not verify_jwt(token):
            return JSONResponse(
                status_code=401,
                content={"error": "Unauthorized"},
                headers={"WWW-Authenticate": f'Bearer realm="{ISSUER}"'},
            )
        return await call_next(request)


# ─── FastMCP ──────────────────────────────────────────────────────────────────

mcp = FastMCP(
    "Graphiti Connector",
    instructions=(
        "MCP com acesso direto ao grafo Graphiti da série de livros. "
        "Use graphiti_add_episode para adicionar conteúdo, graphiti_search para buscar, "
        "graphiti_list_nodes/edges para explorar o grafo, e graphiti_delete_* para remover."
    ),
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=False,
    ),
)
mcp.settings.host = "0.0.0.0"
mcp.settings.port = int(os.environ.get("PORT", "8766"))


def _api(method: str, path: str, **kwargs) -> dict:
    """Helper HTTP para chamar o backend MiroFish."""
    url = f"{BACKEND_URL}/api/graphiti{path}"
    with httpx.Client(timeout=60.0) as client:
        resp = getattr(client, method)(url, **kwargs)
        resp.raise_for_status()
        return resp.json()


# ─── Tools ────────────────────────────────────────────────────────────────────

@mcp.tool()
async def graphiti_add_episode(
    group_id: str,
    name: str,
    content: str,
    source_description: str = "Graphiti MCP input",
) -> dict:
    """
    Adiciona um episódio (chunk de texto) ao grafo Graphiti.
    O grafo extrai automaticamente entidades e relações do conteúdo.

    - group_id: identificador do grupo/projeto (ex: 'falta', 'sede', 'proj_abc123')
    - name: nome descritivo do episódio (ex: 'Cap 27 - Resolução')
    - content: texto a ser processado e indexado
    - source_description: origem do conteúdo (opcional)
    """
    return _api("post", "/episode", json={
        "group_id": group_id,
        "name": name,
        "content": content,
        "source_description": source_description,
    })


@mcp.tool()
async def graphiti_search(
    query: str,
    group_ids: Optional[list[str]] = None,
    num_results: int = 10,
) -> dict:
    """
    Busca semântica no grafo Graphiti.
    Retorna nós e arestas relevantes para a query.

    - query: texto de busca (ex: 'Enzo descobre sobre o passado de Lucas')
    - group_ids: filtrar por grupos específicos (None = todos)
    - num_results: número de resultados (default 10)
    """
    return _api("post", "/search", json={
        "query": query,
        "group_ids": group_ids,
        "num_results": num_results,
    })


@mcp.tool()
async def graphiti_list_groups() -> dict:
    """
    Lista todos os group_ids existentes no grafo.
    Útil para descobrir quais projetos/livros estão indexados.
    """
    return _api("get", "/groups")


@mcp.tool()
async def graphiti_list_nodes(
    group_id: str,
    limit: int = 100,
) -> dict:
    """
    Lista todos os nós (entidades) de um group_id.
    Cada nó é uma entidade extraída pelo Graphiti (personagem, lugar, evento, etc).

    - group_id: grupo a listar
    - limit: máximo de nós retornados (default 100)
    """
    return _api("get", f"/nodes/{group_id}", params={"limit": limit})


@mcp.tool()
async def graphiti_list_edges(
    group_id: str,
    limit: int = 100,
) -> dict:
    """
    Lista todas as arestas (relações) de um group_id.
    Cada aresta representa uma relação entre dois nós com um fato associado.

    - group_id: grupo a listar
    - limit: máximo de arestas retornadas (default 100)
    """
    return _api("get", f"/edges/{group_id}", params={"limit": limit})


@mcp.tool()
async def graphiti_update_node(
    uuid: str,
    name: Optional[str] = None,
    summary: Optional[str] = None,
) -> dict:
    """
    Atualiza o nome ou summary de um nó existente.
    Use para corrigir erros de extração ou enriquecer o grafo manualmente.

    - uuid: UUID do nó (obtido via graphiti_list_nodes)
    - name: novo nome (opcional)
    - summary: novo resumo/descrição (opcional)
    """
    body = {}
    if name:
        body["name"] = name
    if summary:
        body["summary"] = summary
    return _api("patch", f"/node/{uuid}", json=body)


@mcp.tool()
async def graphiti_delete_node(uuid: str) -> dict:
    """
    Deleta um nó do grafo pelo UUID.
    Use para remover entidades extraídas incorretamente.

    - uuid: UUID do nó (obtido via graphiti_list_nodes)
    """
    return _api("delete", f"/node/{uuid}")


@mcp.tool()
async def graphiti_delete_episode(uuid: str) -> dict:
    """
    Deleta um episódio do grafo pelo UUID.
    Remover o episódio não remove automaticamente os nós/arestas derivados.

    - uuid: UUID do episódio
    """
    return _api("delete", f"/episode/{uuid}")


@mcp.tool()
async def graphiti_list_episodes(
    group_id: str,
    limit: int = 100,
) -> dict:
    """
    Lista todos os episódios (chunks de texto) de um group_id.
    Cada episódio é um trecho que foi processado e originou nós/arestas.

    - group_id: grupo a listar (ex: 'falta', 'sede', 'mirofish_abc123')
    - limit: máximo de episódios (default 100)
    """
    return _api("get", f"/episodes/{group_id}", params={"limit": limit})


@mcp.tool()
async def graphiti_delete_group(group_id: str) -> dict:
    """
    Deleta TODOS os dados de um group_id — nós, arestas e episódios.
    Use para remover um livro/projeto inteiro do grafo.
    ATENÇÃO: irreversível.

    - group_id: grupo a deletar (ex: 'falta', 'sede')
    """
    return _api("delete", f"/group/{group_id}")


@mcp.tool()
async def graphiti_bulk_add_episodes(
    group_id: str,
    episodes: list[dict],
) -> dict:
    """
    Adiciona múltiplos episódios de uma vez — ideal para ingestão de livro completo.
    O Graphiti processa cada episódio e extrai entidades/relações automaticamente.

    - group_id: identificador do livro/projeto (ex: 'falta', 'sede', 'livro3')
    - episodes: lista de { name, content, source_description? }
      Exemplo:
        [
          { "name": "Cap 01 - Lucas", "content": "texto completo do capítulo..." },
          { "name": "Cap 02 - Enzo",  "content": "texto completo do capítulo..." }
        ]

    Retorna status de cada episódio individualmente.
    """
    return _api("post", "/episode/bulk", json={"group_id": group_id, "episodes": episodes})


# ─── Health & OAuth Discovery ─────────────────────────────────────────────────

@mcp.custom_route("/health", methods=["GET"])
async def health(request):
    from starlette.responses import JSONResponse
    try:
        result = _api("get", "/groups")
        return JSONResponse({"status": "ok", "groups": result.get("total", 0)})
    except Exception as e:
        return JSONResponse({"status": "degraded", "error": str(e)}, status_code=503)


@mcp.custom_route("/.well-known/oauth-protected-resource", methods=["GET"])
async def oauth_protected_resource(request):
    from starlette.responses import JSONResponse
    return JSONResponse({
        "resource": "https://mcptracker.rebote.cc",
        "authorization_servers": ["https://auth.rebote.cc"],
    })


@mcp.custom_route("/.well-known/oauth-authorization-server", methods=["GET"])
async def oauth_authorization_server(request):
    from starlette.responses import JSONResponse
    return JSONResponse({
        "issuer": "https://auth.rebote.cc",
        "authorization_endpoint": "https://auth.rebote.cc/authorize",
        "token_endpoint": "https://auth.rebote.cc/token",
        "registration_endpoint": "https://auth.rebote.cc/register",
        "response_types_supported": ["code"],
        "grant_types_supported": ["authorization_code"],
        "code_challenge_methods_supported": ["S256"],
    })


# ─── Entrypoint ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import asyncio

    try:
        mcp_app = getattr(mcp, "_mcp_server", None)
        mcp_app = getattr(mcp_app, "app", None) or getattr(mcp, "_app", None)
        if mcp_app:
            mcp_app.add_middleware(JWTAuthMiddleware)
    except Exception:
        pass

    asyncio.run(mcp.run_streamable_http_async())
