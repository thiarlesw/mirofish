"""
MiroFish MCP Server
Expõe as funcionalidades do MiroFish como tools MCP para Claude e outros clientes.
"""
import os
from mcp.server.fastmcp import FastMCP
import httpx
from typing import Optional

MIROFISH_BASE_URL = os.environ.get("MIROFISH_BASE_URL", "http://localhost:5001")

mcp = FastMCP("MiroFish")


# ============== Projetos ==============

@mcp.tool()
async def list_projects() -> dict:
    """Lista todos os projetos de simulação MiroFish."""
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(f"{MIROFISH_BASE_URL}/api/graph/project/list")
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
        files = {
            "files": (filename, content.encode("utf-8"), "text/plain"),
        }
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


@mcp.tool()
async def upload_document(
    project_id: str,
    simulation_requirement: str,
    content: str,
    filename: str = "document.txt",
    additional_context: str = "",
) -> dict:
    """Faz upload de um documento para um novo projeto e gera a ontologia.

    Use esta tool para alimentar o MiroFish com textos (artigos, relatórios, etc.)
    que servirão de base para construção do grafo de conhecimento.

    Args:
        project_id: ID do projeto existente (não utilizado diretamente — a API
                    cria ou atualiza via /ontology/generate; passe o project_id
                    para referência semântica)
        simulation_requirement: Descrição do requisito de simulação
        content: Conteúdo textual do documento
        filename: Nome do arquivo
        additional_context: Contexto adicional opcional
    """
    # A API do MiroFish cria projeto + ontologia em uma só chamada.
    # Não há endpoint separado de upload sem criação de projeto.
    async with httpx.AsyncClient(timeout=120) as client:
        files = {
            "files": (filename, content.encode("utf-8"), "text/plain"),
        }
        data = {
            "simulation_requirement": simulation_requirement,
            "project_name": f"Upload para {project_id}",
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

    Esta é uma operação assíncrona. O retorno inclui um task_id para
    monitorar o progresso via get_task_status.

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

        r = await client.post(
            f"{MIROFISH_BASE_URL}/api/graph/build",
            json=payload,
        )
        return r.json()


@mcp.tool()
async def get_task_status(task_id: str) -> dict:
    """Obtém o status de uma tarefa assíncrona (construção de grafo, preparação, etc.).

    Args:
        task_id: ID da tarefa retornado por build_knowledge_graph ou prepare_simulation
    """
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(f"{MIROFISH_BASE_URL}/api/graph/task/{task_id}")
        return r.json()


@mcp.tool()
async def query_knowledge_graph(
    graph_id: str,
    query: str,
    num_results: int = 10,
) -> dict:
    """Busca entidades e fatos no grafo de conhecimento via filtro de tipo.

    Retorna as entidades do grafo filtradas, que podem ser usadas para
    análise ou para alimentar simulações.

    Args:
        graph_id: ID do grafo (obtido após build_knowledge_graph)
        query: Tipo de entidade a buscar (ex: 'Person', 'Organization') ou
               deixe vazio para retornar todas as entidades definidas
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

        # Limitar resultados se necessário
        if (
            data.get("success")
            and "data" in data
            and "entities" in data["data"]
        ):
            entities = data["data"]["entities"]
            if len(entities) > num_results:
                data["data"]["entities"] = entities[:num_results]
                data["data"]["truncated"] = True
                data["data"]["total"] = len(entities)

        return data


# ============== Simulação ==============

@mcp.tool()
async def create_simulation(
    project_id: str,
    enable_twitter: bool = True,
    enable_reddit: bool = True,
    graph_id: str = "",
) -> dict:
    """Cria uma nova simulação de redes sociais para um projeto.

    Após criar a simulação, use prepare_simulation para gerar os perfis
    dos agentes e depois start_simulation para iniciar a execução.

    Args:
        project_id: ID do projeto com grafo construído
        enable_twitter: Habilitar plataforma Twitter (padrão: True)
        enable_reddit: Habilitar plataforma Reddit (padrão: True)
        graph_id: ID do grafo (opcional — se não fornecido, usa o do projeto)
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
            f"{MIROFISH_BASE_URL}/api/simulation/create",
            json=payload,
        )
        return r.json()


@mcp.tool()
async def prepare_simulation(
    simulation_id: str,
    entity_types: Optional[list] = None,
    use_llm_for_profiles: bool = True,
    parallel_profile_count: int = 5,
    force_regenerate: bool = False,
) -> dict:
    """Prepara o ambiente de simulação (gera perfis dos agentes via LLM).

    Esta é uma operação assíncrona e demorada. Retorna um task_id para
    monitorar o progresso via get_task_status. Se já estiver preparado,
    retorna immediately sem reprocessar.

    Args:
        simulation_id: ID da simulação (obtido via create_simulation)
        entity_types: Lista de tipos de entidade a usar (ex: ['Person', 'Organization'])
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
        if entity_types:
            payload["entity_types"] = entity_types

        r = await client.post(
            f"{MIROFISH_BASE_URL}/api/simulation/prepare",
            json=payload,
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
    Monitore o progresso com get_simulation_status.

    Args:
        simulation_id: ID da simulação
        platform: Plataforma a simular — 'twitter', 'reddit' ou 'parallel' (padrão)
        rounds: Número máximo de rodadas (0 = sem limite, usa configuração do projeto)
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

        r = await client.post(
            f"{MIROFISH_BASE_URL}/api/simulation/start",
            json=payload,
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
async def list_simulations() -> dict:
    """Lista todas as simulações existentes."""
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(f"{MIROFISH_BASE_URL}/api/simulation/list")
        return r.json()


# ============== Relatório ==============

@mcp.tool()
async def generate_report(
    simulation_id: str,
    force_regenerate: bool = False,
) -> dict:
    """Gera um relatório de análise para uma simulação.

    Operação assíncrona — retorna task_id para monitorar via get_task_status.
    Se já existir um relatório concluído, retorna diretamente sem regenerar
    (a menos que force_regenerate=True).

    Args:
        simulation_id: ID da simulação concluída
        force_regenerate: Forçar regeneração do relatório
    """
    async with httpx.AsyncClient(timeout=60) as client:
        payload = {
            "simulation_id": simulation_id,
            "force_regenerate": force_regenerate,
        }
        r = await client.post(
            f"{MIROFISH_BASE_URL}/api/report/generate",
            json=payload,
        )
        return r.json()


if __name__ == "__main__":
    mcp.run(transport="sse", host="0.0.0.0", port=8765)
