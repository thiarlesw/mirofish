"""
MiroFish MCP Server
Expõe as funcionalidades do MiroFish como tools MCP para Claude e outros clientes.
Com autenticação JWT OAuth 2.1.
"""

import base64
import hashlib
import hmac
import json
import os
import time
from typing import Optional

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
import httpx
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

# ─── Configuração de Autenticação JWT ─────────────────────────────────

ISSUER = "https://auth.rebote.cc"
JWT_SECRET = os.environ.get("JWT_SECRET", "")

MIROFISH_BASE_URL = os.environ.get("MIROFISH_BASE_URL", "http://localhost:5001")


def _b64decode_padding(s: str) -> bytes:
    """Decodifica base64url com padding automático."""
    s += "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s)


def verify_jwt(token: str) -> bool:
    """Valida JWT assinado com HMAC-SHA256 (compatível com Worker Cloudflare)."""
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


class MCPAuthMiddleware(BaseHTTPMiddleware):
    """Middleware para proteger rotas MCP com JWT."""

    async def dispatch(self, request, call_next):
        path = request.url.path
        # Rotas públicas (obrigatórias para OAuth discovery)
        if path.startswith("/.well-known"):
            return await call_next(request)
        # Validar Bearer token
        auth_header = request.headers.get("Authorization", "")
        token = (
            auth_header.removeprefix("Bearer ").strip()
            if auth_header.startswith("Bearer ")
            else auth_header.strip()
        )
        if not verify_jwt(token):
            return JSONResponse(
                status_code=401,
                content={"error": "unauthorized"},
                headers={
                    "WWW-Authenticate": (
                        'Bearer resource_metadata="https://auth.rebote.cc/.well-known/oauth-protected-resource"'
                    )
                },
            )
        return await call_next(request)


# ─── Fim da configuração JWT ──────────────────────────────────────────

mcp = FastMCP(
    "MiroFish",
    instructions=(
        "Motor de simulação de redes sociais com IA. "
        "Fluxo: create_project → build_knowledge_graph → create_simulation → "
        "prepare_simulation → start_simulation → get_simulation_status → "
        "generate_report → get_report. "
        "Use get_task_status para monitorar operações assíncronas."
    ),
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=False,
    ),
)


# ============== Projetos ==============


@mcp.tool()
async def list_projects(limit: int = 50) -> dict:
    """Lista todos os projetos de simulação MiroFish."""
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(
            f"{MIROFISH_BASE_URL}/api/graph/project/list", params={"limit": limit}
        )
        return r.json()


@mcp.tool()
async def get_project(project_id: str) -> dict:
    """Obtém detalhes de um projeto específico.

    Args:
        project_id: ID do projeto (ex: proj_xxxx)
    """
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(f"{MIROFISH_BASE_URL}/api/graph/project/{project_id}")
        return r.json()


@mcp.tool()
async def delete_project(project_id: str) -> dict:
    """Deleta um projeto e todos os seus dados.

    Args:
        project_id: ID do projeto a deletar
    """
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.delete(f"{MIROFISH_BASE_URL}/api/graph/project/{project_id}")
        return r.json()


@mcp.tool()
async def create_project(
    simulation_requirement: str,
    project_name: str = "Unnamed Project",
    content: str = "",
    filename: str = "document.txt",
    additional_context: str = "",
) -> dict:
    """Cria um novo projeto enviando um documento de texto e gerando a ontologia.

    O MiroFish requer ao menos um documento e uma descrição do requisito de
    simulação para criar o projeto e a ontologia automaticamente.

    Args:
        simulation_requirement: Descrição do requisito/objetivo da simulação (obrigatório)
        project_name: Nome do projeto
        content: Conteúdo textual do documento a ser analisado
        filename: Nome do arquivo (ex: relatorio.txt, artigo.pdf)
        additional_context: Contexto adicional opcional
    """
    async with httpx.AsyncClient(timeout=120) as client:
        files = {"files": (filename, content.encode("utf-8"), "text/plain")}
        data = {
            "simulation_requirement": simulation_requirement,
            "project_name": project_name,
        }
        if additional_context:
            data["additional_context"] = additional_context
        r = await client.post(
            f"{MIROFISH_BASE_URL}/api/graph/ontology/generate",
            files=files,
            data=data,
        )
        return r.json()


# ============== Grafo de Conhecimento ==============


@mcp.tool()
async def build_knowledge_graph(
    project_id: str,
    graph_name: str = "",
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    force: bool = False,
) -> dict:
    """Constrói o grafo de conhecimento a partir dos documentos do projeto.

    Operação assíncrona — retorna task_id para monitorar via get_task_status.

    Args:
        project_id: ID do projeto (deve ter ontologia gerada)
        graph_name: Nome opcional para o grafo
        chunk_size: Tamanho dos chunks de texto (padrão: 500)
        chunk_overlap: Sobreposição entre chunks (padrão: 50)
        force: Forçar reconstrução mesmo se já existir
    """
    async with httpx.AsyncClient(timeout=60) as client:
        payload: dict = {"project_id": project_id, "force": force}
        if graph_name:
            payload["graph_name"] = graph_name
        if chunk_size:
            payload["chunk_size"] = chunk_size
        if chunk_overlap:
            payload["chunk_overlap"] = chunk_overlap
        r = await client.post(f"{MIROFISH_BASE_URL}/api/graph/build", json=payload)
        return r.json()


@mcp.tool()
async def get_graph_data(graph_id: str) -> dict:
    """Obtém dados do grafo de conhecimento (nós, arestas, contagens).

    Args:
        graph_id: ID do grafo
    """
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(f"{MIROFISH_BASE_URL}/api/graph/data/{graph_id}")
        return r.json()


@mcp.tool()
async def get_task_status(task_id: str) -> dict:
    """Obtém o status de uma tarefa assíncrona (construção de grafo, preparação, etc.).

    Args:
        task_id: ID da tarefa retornado por build_knowledge_graph, prepare_simulation, etc.
    """
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(f"{MIROFISH_BASE_URL}/api/graph/task/{task_id}")
        return r.json()


@mcp.tool()
async def list_tasks() -> dict:
    """Lista todas as tarefas assíncronas e seus status."""
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(f"{MIROFISH_BASE_URL}/api/graph/tasks")
        return r.json()


@mcp.tool()
async def query_knowledge_graph(
    graph_id: str,
    query: str = "",
    num_results: int = 10,
) -> dict:
    """Busca entidades e fatos no grafo de conhecimento.

    Args:
        graph_id: ID do grafo
        query: Tipo de entidade a buscar (ex: 'Person', 'Organization') ou vazio para todas
        num_results: Número máximo de resultados (padrão: 10)
    """
    async with httpx.AsyncClient(timeout=60) as client:
        params: dict = {"enrich": "true"}
        if query:
            params["entity_types"] = query
        r = await client.get(
            f"{MIROFISH_BASE_URL}/api/simulation/entities/{graph_id}",
            params=params,
        )
        data = r.json()
        if data.get("success") and "data" in data and "entities" in data["data"]:
            entities = data["data"]["entities"]
            if len(entities) > num_results:
                data["data"]["entities"] = entities[:num_results]
                data["data"]["truncated"] = True
                data["data"]["total"] = len(entities)
        return data


# ============== Simulação ==============


@mcp.tool()
async def list_simulations(
    project_id: str = "",
    status: str = "",
    limit: int = 50,
) -> dict:
    """Lista todas as simulações existentes.

    Args:
        project_id: Filtrar por projeto (opcional)
        status: Filtrar por status — 'created', 'prepared', 'running', 'completed' (opcional)
        limit: Máximo de resultados (padrão: 50)
    """
    async with httpx.AsyncClient(timeout=30) as client:
        params: dict = {"limit": limit}
        if project_id:
            params["project_id"] = project_id
        if status:
            params["status"] = status
        r = await client.get(f"{MIROFISH_BASE_URL}/api/simulation/list", params=params)
        return r.json()


@mcp.tool()
async def get_simulation(simulation_id: str) -> dict:
    """Obtém estado completo de uma simulação.

    Args:
        simulation_id: ID da simulação
    """
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(f"{MIROFISH_BASE_URL}/api/simulation/{simulation_id}")
        return r.json()


@mcp.tool()
async def create_simulation(
    project_id: str,
    enable_twitter: bool = True,
    enable_reddit: bool = True,
    graph_id: str = "",
) -> dict:
    """Cria uma nova simulação de redes sociais para um projeto.

    Após criar, use prepare_simulation para gerar perfis e start_simulation para executar.

    Args:
        project_id: ID do projeto com grafo construído
        enable_twitter: Habilitar plataforma Twitter (padrão: True)
        enable_reddit: Habilitar plataforma Reddit (padrão: True)
        graph_id: ID do grafo (opcional — usa o do projeto se omitido)
    """
    async with httpx.AsyncClient(timeout=30) as client:
        payload: dict = {
            "project_id": project_id,
            "enable_twitter": enable_twitter,
            "enable_reddit": enable_reddit,
        }
        if graph_id:
            payload["graph_id"] = graph_id
        r = await client.post(
            f"{MIROFISH_BASE_URL}/api/simulation/create", json=payload
        )
        return r.json()


@mcp.tool()
async def prepare_simulation(
    simulation_id: str,
    use_llm_for_profiles: bool = True,
    parallel_profile_count: int = 5,
    force_regenerate: bool = False,
) -> dict:
    """Prepara o ambiente de simulação (gera perfis dos agentes via LLM).

    Operação assíncrona e demorada — retorna task_id para monitorar.

    Args:
        simulation_id: ID da simulação (obtido via create_simulation)
        use_llm_for_profiles: Usar LLM para gerar perfis (padrão: True)
        parallel_profile_count: Quantos perfis gerar em paralelo (padrão: 5)
        force_regenerate: Forçar regeneração mesmo se já preparado
    """
    async with httpx.AsyncClient(timeout=60) as client:
        payload: dict = {
            "simulation_id": simulation_id,
            "use_llm_for_profiles": use_llm_for_profiles,
            "parallel_profile_count": parallel_profile_count,
            "force_regenerate": force_regenerate,
        }
        r = await client.post(
            f"{MIROFISH_BASE_URL}/api/simulation/prepare", json=payload
        )
        return r.json()


@mcp.tool()
async def start_simulation(
    simulation_id: str,
    platform: str = "parallel",
    rounds: int = 0,
    enable_graph_memory_update: bool = False,
    force: bool = False,
) -> dict:
    """Inicia a execução de uma simulação de redes sociais.

    A simulação deve estar preparada (use prepare_simulation antes).

    Args:
        simulation_id: ID da simulação
        platform: 'twitter', 'reddit' ou 'parallel' (padrão)
        rounds: Número máximo de rodadas (0 = usa config do projeto)
        enable_graph_memory_update: Atualizar grafo de memória em tempo real
        force: Forçar reinício se já estiver rodando ou concluído
    """
    async with httpx.AsyncClient(timeout=60) as client:
        payload: dict = {
            "simulation_id": simulation_id,
            "platform": platform,
            "enable_graph_memory_update": enable_graph_memory_update,
            "force": force,
        }
        if rounds and rounds > 0:
            payload["max_rounds"] = rounds
        r = await client.post(f"{MIROFISH_BASE_URL}/api/simulation/start", json=payload)
        return r.json()


@mcp.tool()
async def stop_simulation(simulation_id: str) -> dict:
    """Para uma simulação em andamento.

    Args:
        simulation_id: ID da simulação
    """
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(
            f"{MIROFISH_BASE_URL}/api/simulation/stop",
            json={"simulation_id": simulation_id},
        )
        return r.json()


@mcp.tool()
async def get_simulation_status(simulation_id: str) -> dict:
    """Obtém o status atual de uma simulação (rodadas, progresso, ações).

    Args:
        simulation_id: ID da simulação
    """
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(
            f"{MIROFISH_BASE_URL}/api/simulation/{simulation_id}/run-status"
        )
        return r.json()


@mcp.tool()
async def get_simulation_posts(
    simulation_id: str,
    platform: str = "",
    limit: int = 50,
    agent_id: str = "",
) -> dict:
    """Obtém os posts gerados pelos agentes durante a simulação.

    Args:
        simulation_id: ID da simulação
        platform: Filtrar por plataforma — 'twitter' ou 'reddit' (opcional)
        limit: Máximo de posts retornados (padrão: 50)
        agent_id: Filtrar por agente específico (opcional)
    """
    async with httpx.AsyncClient(timeout=30) as client:
        params: dict = {"limit": limit}
        if platform:
            params["platform"] = platform
        if agent_id:
            params["agent_id"] = agent_id
        r = await client.get(
            f"{MIROFISH_BASE_URL}/api/simulation/{simulation_id}/posts",
            params=params,
        )
        return r.json()


@mcp.tool()
async def get_simulation_timeline(
    simulation_id: str,
    limit: int = 100,
) -> dict:
    """Obtém a timeline de eventos da simulação em ordem cronológica.

    Args:
        simulation_id: ID da simulação
        limit: Máximo de eventos (padrão: 100)
    """
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(
            f"{MIROFISH_BASE_URL}/api/simulation/{simulation_id}/timeline",
            params={"limit": limit},
        )
        return r.json()


@mcp.tool()
async def get_simulation_profiles(
    simulation_id: str,
    platform: str = "all",
) -> dict:
    """Obtém os perfis dos agentes gerados para a simulação.

    Args:
        simulation_id: ID da simulação
        platform: 'twitter', 'reddit' ou 'all' (padrão)
    """
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(
            f"{MIROFISH_BASE_URL}/api/simulation/{simulation_id}/profiles",
            params={"platform": platform},
        )
        return r.json()


@mcp.tool()
async def interview_agent(
    simulation_id: str,
    agent_id: str,
    message: str,
) -> dict:
    """Entrevista um agente específico da simulação — faz uma pergunta e recebe resposta no personagem.

    Args:
        simulation_id: ID da simulação
        agent_id: ID do agente a entrevistar
        message: Pergunta ou mensagem para o agente
    """
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            f"{MIROFISH_BASE_URL}/api/simulation/interview",
            json={
                "simulation_id": simulation_id,
                "agent_id": agent_id,
                "message": message,
            },
        )
        return r.json()


@mcp.tool()
async def interview_all_agents(simulation_id: str, message: str) -> dict:
    """Envia a mesma pergunta para TODOS os agentes da simulação simultaneamente.

    Args:
        simulation_id: ID da simulação
        message: Pergunta para todos os agentes
    """
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(
            f"{MIROFISH_BASE_URL}/api/simulation/interview/all",
            json={"simulation_id": simulation_id, "message": message},
        )
        return r.json()


# ============== Relatório ==============


@mcp.tool()
async def generate_report(
    simulation_id: str,
    force_regenerate: bool = False,
) -> dict:
    """Gera um relatório de análise para uma simulação concluída.

    Operação assíncrona — retorna task_id para monitorar via get_task_status.
    Se já existir relatório, retorna diretamente (a menos que force_regenerate=True).

    Args:
        simulation_id: ID da simulação concluída
        force_regenerate: Forçar regeneração do relatório
    """
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            f"{MIROFISH_BASE_URL}/api/report/generate",
            json={"simulation_id": simulation_id, "force_regenerate": force_regenerate},
        )
        return r.json()


@mcp.tool()
async def get_report(simulation_id: str) -> dict:
    """Obtém o conteúdo completo do relatório de uma simulação.

    Retorna o markdown_content com toda a análise gerada.

    Args:
        simulation_id: ID da simulação com relatório gerado
    """
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(
            f"{MIROFISH_BASE_URL}/api/report/by-simulation/{simulation_id}"
        )
        return r.json()


@mcp.tool()
async def get_report_section(report_id: str, section_index: int) -> dict:
    """Obtém uma seção específica do relatório.

    Args:
        report_id: ID do relatório
        section_index: Índice da seção (0, 1, 2...)
    """
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(
            f"{MIROFISH_BASE_URL}/api/report/{report_id}/section/{section_index}"
        )
        return r.json()


@mcp.tool()
async def chat_with_report(
    simulation_id: str,
    message: str,
) -> dict:
    """Conversa com o Report Agent sobre os resultados da simulação.

    O agente tem acesso ao grafo de conhecimento e pode responder perguntas
    analíticas sobre a simulação.

    Args:
        simulation_id: ID da simulação (relatório deve estar gerado)
        message: Pergunta ou pedido de análise
    """
    async with httpx.AsyncClient(timeout=90) as client:
        r = await client.post(
            f"{MIROFISH_BASE_URL}/api/report/chat",
            json={"simulation_id": simulation_id, "message": message},
        )
        return r.json()


# ─── OAuth 2.1 Discovery Endpoint ─────────────────────────────────────


@mcp.custom_route("/.well-known/oauth-protected-resource", methods=["GET"])
async def oauth_protected_resource(request):
    """Endpoint de discovery OAuth 2.1 (obrigatório pelo MCP spec)."""
    return JSONResponse(
        {
            "resource": "https://mirofish.rebote.cc",  # Ajustar quando expor publicamente
            "authorization_servers": ["https://auth.rebote.cc"],
        }
    )


if __name__ == "__main__":
    import asyncio

    # Adicionar middleware de autenticação JWT
    mcp_app = (
        mcp._mcp_server.app
        if hasattr(mcp, "_mcp_server")
        else mcp._app
        if hasattr(mcp, "_app")
        else None
    )
    if mcp_app:
        mcp_app.add_middleware(MCPAuthMiddleware)
        print("[INFO] Middleware de autenticação JWT ativado")
    else:
        print("[AVISO] Não foi possível adicionar middleware - app não encontrado")

    port = int(os.environ.get("PORT", "8765"))
    mcp.settings.host = "0.0.0.0"
    mcp.settings.port = port

    asyncio.run(mcp.run_streamable_http_async())
