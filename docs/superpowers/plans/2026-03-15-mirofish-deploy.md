# MiroFish Deploy — GitHub + OVH Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publicar o MiroFish no GitHub e fazer deploy no OVH com Graphiti + Gemini, acessível via mirofish.rebote.cc (UI) e mcp-miro.rebote.cc (MCP), protegidos por Cloudflare Access.

**Architecture:** GitHub repo público → clone no OVH → docker compose (backend Flask + frontend Vue + MCP FastMCP) → Cloudflare Tunnel ingress → Cloudflare Access por email.

**Tech Stack:** Python/Flask, Vue.js 3, Graphiti-core + Kuzu, Gemini API, FastMCP SSE, Docker Compose, Cloudflare Tunnel

**Servidor:** `ssh -i ~/.ssh/id_rsa ubuntu@15.204.207.79`
**Portas livres:** backend=5001, frontend=3010, mcp=8765

---

## Chunk 1: GitHub repo + push

### Task 1: Criar repo GitHub e fazer push

**Files:**
- Local: `C:\dev\mirofish\` (já commitado)

- [ ] **Step 1: Criar repo GitHub**

```bash
cd C:/dev/mirofish
gh repo create thiarlesw/mirofish \
  --public \
  --description "MiroFish — Multi-agent social simulation with Graphiti + Gemini (Zep-free)" \
  --source . \
  --remote origin \
  --push
```

Esperado: repo criado em `https://github.com/thiarlesw/mirofish` e push de `main` concluído.

- [ ] **Step 2: Verificar push**

```bash
gh repo view thiarlesw/mirofish --web
```

Esperado: abre no browser com todos os arquivos, incluindo `mcp_server/`, `backend/app/services/graphiti_*.py`.

---

## Chunk 2: Deploy OVH — clone + .env

### Task 2: Clonar repo e configurar .env no servidor

**Files no servidor:**
- Create: `/home/ubuntu/mirofish/` (clone)
- Create: `/home/ubuntu/mirofish/.env`

- [ ] **Step 1: Clonar no OVH**

```bash
ssh -i ~/.ssh/id_rsa ubuntu@15.204.207.79 \
  "git clone https://github.com/thiarlesw/mirofish.git /home/ubuntu/mirofish"
```

Esperado: `Cloning into '/home/ubuntu/mirofish'...` sem erros.

- [ ] **Step 2: Criar .env no servidor**

```bash
ssh -i ~/.ssh/id_rsa ubuntu@15.204.207.79 "cat > /home/ubuntu/mirofish/.env" << 'EOF'
# LLM para agentes de simulação (Gemini via OpenAI-compatible)
LLM_API_KEY=COLOCAR_GEMINI_API_KEY
LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
LLM_MODEL_NAME=gemini-2.0-flash

# Graphiti + Gemini
GEMINI_API_KEY=COLOCAR_GEMINI_API_KEY
GRAPHITI_LLM_MODEL=gemini-2.0-flash
GRAPHITI_DB_PATH=/app/data/graphiti_db

# Flask
SECRET_KEY=GERAR_UUID_AQUI
FLASK_DEBUG=false

# MCP
MIROFISH_BASE_URL=http://backend:5001

# Simulação
OASIS_DEFAULT_MAX_ROUNDS=10
REPORT_AGENT_MAX_TOOL_CALLS=5
REPORT_AGENT_MAX_REFLECTION_ROUNDS=2
REPORT_AGENT_TEMPERATURE=0.5
EOF
```

- [ ] **Step 3: Editar .env com chaves reais**

```bash
ssh -i ~/.ssh/id_rsa ubuntu@15.204.207.79 \
  "nano /home/ubuntu/mirofish/.env"
```

Substituir:
- `COLOCAR_GEMINI_API_KEY` → chave real do Google AI Studio
- `GERAR_UUID_AQUI` → `$(python3 -c 'import uuid; print(uuid.uuid4())')`

- [ ] **Step 4: Criar diretório de dados**

```bash
ssh -i ~/.ssh/id_rsa ubuntu@15.204.207.79 \
  "mkdir -p /home/ubuntu/mirofish/data /home/ubuntu/mirofish/uploads"
```

---

## Chunk 3: Verificar Dockerfiles do MiroFish

### Task 3: Confirmar Dockerfiles existentes

- [ ] **Step 1: Verificar Dockerfile do backend**

```bash
ssh -i ~/.ssh/id_rsa ubuntu@15.204.207.79 \
  "ls /home/ubuntu/mirofish/backend/Dockerfile /home/ubuntu/mirofish/frontend/Dockerfile 2>/dev/null || echo 'FALTANDO'"
```

Se faltar algum Dockerfile, o docker compose do repo original tem um monolítico — ajustar o docker-compose.yml para usar o Dockerfile raiz:

```bash
ssh -i ~/.ssh/id_rsa ubuntu@15.204.207.79 \
  "cat /home/ubuntu/mirofish/docker-compose.yml | head -30"
```

- [ ] **Step 2: Verificar se o Dockerfile raiz ainda existe**

```bash
ssh -i ~/.ssh/id_rsa ubuntu@15.204.207.79 \
  "cat /home/ubuntu/mirofish/Dockerfile | head -10"
```

Se o docker-compose.yml referenciar `/backend` ou `/frontend` separadamente mas não tiverem Dockerfiles próprios, criar symlinks ou copiar o Dockerfile raiz:

```bash
ssh -i ~/.ssh/id_rsa ubuntu@15.204.207.79 "
  cp /home/ubuntu/mirofish/Dockerfile /home/ubuntu/mirofish/backend/Dockerfile 2>/dev/null || true
"
```

---

## Chunk 4: Build e start dos containers

### Task 4: Docker compose build + up

- [ ] **Step 1: Build (pode demorar ~10min no primeiro run)**

```bash
ssh -i ~/.ssh/id_rsa ubuntu@15.204.207.79 "
  cd /home/ubuntu/mirofish
  sudo docker compose build --no-cache 2>&1 | tail -30
"
```

Esperado: `Successfully built ...` para cada serviço. Se falhar por dependência Python, verificar `backend/requirements.txt`.

- [ ] **Step 2: Subir containers**

```bash
ssh -i ~/.ssh/id_rsa ubuntu@15.204.207.79 "
  cd /home/ubuntu/mirofish
  sudo docker compose up -d
  sleep 5
  sudo docker ps | grep mirofish
"
```

Esperado:
```
mirofish-backend    Up    127.0.0.1:5001->5001/tcp
mirofish-frontend   Up    127.0.0.1:3010->80/tcp
mirofish-mcp        Up    127.0.0.1:8765->8765/tcp
```

- [ ] **Step 3: Verificar logs do backend**

```bash
ssh -i ~/.ssh/id_rsa ubuntu@15.204.207.79 "
  sudo docker logs mirofish-backend --tail 30
"
```

Esperado: `Running on http://0.0.0.0:5001` sem erros de import.

- [ ] **Step 4: Health check local**

```bash
ssh -i ~/.ssh/id_rsa ubuntu@15.204.207.79 "
  curl -s http://localhost:5001/api/health 2>/dev/null || \
  curl -s http://localhost:5001/ | head -5
"
```

---

## Chunk 5: Cloudflare Tunnel — ingress

### Task 5: Adicionar rotas no tunnel vps-nebulo

O tunnel `vps-nebulo` já existe e está ativo. Adicionar ingress para MiroFish via API da Cloudflare.

- [ ] **Step 1: Verificar tunnel ID atual**

```bash
CF_TOKEN="qHcoAc-oIp1EXtMFU7EUxN1J6YsM7RBAR8Pj4KYb"
CF_ACCOUNT="850ea93b2935bf5d1682f0dfa8735dd7"

curl -s "https://api.cloudflare.com/client/v4/accounts/$CF_ACCOUNT/cfd_tunnel?per_page=50" \
  -H "Authorization: Bearer $CF_TOKEN" | \
  python3 -c "
import sys,json
d=json.load(sys.stdin)
for t in d['result']:
    if 'nebulo' in t['name'].lower() or 'rebote' in t['name'].lower():
        print(t['name'], t['id'])
"
```

- [ ] **Step 2: Adicionar ingress mirofish.rebote.cc → localhost:3010**

Acessar Cloudflare Zero Trust → Networks → Tunnels → vps-nebulo → Configure → Public Hostname → Add:
- **Subdomain:** `mirofish`
- **Domain:** `rebote.cc`
- **Service:** `http://localhost:3010`
- **No TLS Verify:** ✓

Ou via arquivo de config do cloudflared no servidor:

```bash
ssh -i ~/.ssh/id_rsa ubuntu@15.204.207.79 "
  sudo cat /etc/cloudflared/*.yml 2>/dev/null || \
  sudo find /home -name '*.yml' | xargs grep -l 'ingress' 2>/dev/null | head -3
"
```

- [ ] **Step 3: Adicionar ingress mcp-miro.rebote.cc → localhost:8765**

Mesma operação para o MCP server:
- **Subdomain:** `mcp-miro`
- **Domain:** `rebote.cc`
- **Service:** `http://localhost:8765`

- [ ] **Step 4: Verificar DNS propagação**

```bash
curl -s -o /dev/null -w "%{http_code}" https://mirofish.rebote.cc
```

Esperado: `200` ou `302`.

---

## Chunk 6: Cloudflare Access

### Task 6: Proteger mirofish.rebote.cc e mcp-miro.rebote.cc

- [ ] **Step 1: Criar Access Application para mirofish.rebote.cc**

Cloudflare Zero Trust → Access → Applications → Add Application:
- Tipo: **Self-hosted**
- Nome: `MiroFish`
- Domain: `mirofish.rebote.cc`
- Session duration: 24h

- [ ] **Step 2: Criar policy**

- Policy name: `Owner only`
- Action: Allow
- Rule: **Emails** → seu email

- [ ] **Step 3: Criar Access Application para mcp-miro.rebote.cc**

Mesma configuração para `mcp-miro.rebote.cc`.

- [ ] **Step 4: Testar acesso**

Abrir `https://mirofish.rebote.cc` no browser — deve pedir autenticação Cloudflare Access.
Após login, deve exibir a UI do MiroFish.

---

## Chunk 7: MCP — conectar ao Claude

### Task 7: Configurar MCP no Claude Code local

- [ ] **Step 1: Adicionar MCP ao Claude Code**

```bash
claude mcp add mirofish \
  --transport sse \
  --url https://mcp-miro.rebote.cc/sse
```

Ou adicionar manualmente em `~/.claude/mcp_servers.json`:

```json
{
  "mirofish": {
    "transport": "sse",
    "url": "https://mcp-miro.rebote.cc/sse"
  }
}
```

- [ ] **Step 2: Verificar tools disponíveis**

```bash
claude mcp list
```

Esperado: `mirofish` listado com tools `list_projects`, `create_simulation`, etc.

- [ ] **Step 3: Teste end-to-end**

Numa sessão Claude Code: perguntar "liste os projetos do MiroFish" e verificar que o MCP responde corretamente.

---

## Resumo de portas e domínios

| Serviço | Porta local | Domínio |
|---------|------------|---------|
| Backend Flask | 127.0.0.1:5001 | (interno) |
| Frontend Vue | 127.0.0.1:3010 | mirofish.rebote.cc |
| MCP FastMCP | 127.0.0.1:8765 | mcp-miro.rebote.cc |

## Critérios de sucesso

- [ ] `https://mirofish.rebote.cc` abre a UI (com Cloudflare Access)
- [ ] `https://mcp-miro.rebote.cc/sse` responde com SSE headers
- [ ] Claude Code consegue chamar `list_projects` via MCP
- [ ] Criar um projeto de teste na UI e verificar que o Graphiti processa sem erros
