# Family Finance Assistant — dev commands
# Установить just: pacman -S just  (Manjaro) или https://github.com/casey/just
set dotenv-load := true

default:
    @just --list

# === Setup ===

install:
    uv sync

init-env:
    @test -f .env || (cp .env.example .env && echo "✅ .env создан, заполни TELEGRAM_BOT_TOKEN, TELEGRAM_ALLOWED_USER_IDS, OPENROUTER_API_KEY")

# === Infra ===

up:
    docker compose up -d postgres falkordb langfuse-postgres langfuse-clickhouse langfuse-redis langfuse-minio langfuse-worker langfuse-web
    @echo ""
    @echo "✅ Инфра поднята."
    @echo "   Postgres:     localhost:5432"
    @echo "   FalkorDB:     localhost:6380  (Graphiti episodic memory; Browser UI :3000)"
    @echo "   LangFuse UI:  http://localhost:3001  (admin@local.dev / admin12345)"
    @echo ""

down:
    docker compose --profile phase2 down

logs service="langfuse-web":
    docker compose logs -f {{service}}

ps:
    docker compose ps

nuke:
    docker compose --profile phase2 down -v

# === Dev ===

run:
    uv run python -m family_finance.bot

# Запустить MCP-сервер (read-only финанс-инструменты) поверх stdio — для Claude Desktop / внешних агентов
mcp:
    uv run python -m family_finance.mcp_server.server

# Сохранить актуальную схему графа LangGraph в docs/graph.mmd (mermaid, из кода; БД не нужна)
printgraph:
    @uv run python -c "from pathlib import Path; \
        from family_finance.agents.supervisor import build_supervisor_graph; \
        mermaid = build_supervisor_graph().get_graph().draw_mermaid(); \
        out = Path('docs/graph.mmd'); out.write_text(mermaid); \
        print(f'✅ Граф LangGraph сохранён: {out} ({len(mermaid)} символов)')"

shell:
    uv run ipython

# Проверить OpenRouter: валидность ключа + текущий баланс + наличие выбранных моделей
check-llm:
    @uv run python -c "import os, httpx; \
        key = os.environ['OPENROUTER_API_KEY']; \
        h = {'Authorization': f'Bearer {key}'}; \
        r = httpx.get('https://openrouter.ai/api/v1/auth/key', headers=h, timeout=10); \
        r.raise_for_status(); d = r.json()['data']; \
        print(f'✅ OpenRouter key ok'); \
        print(f'   label:        {d.get(\"label\", \"-\")}'); \
        print(f'   usage:        \${d.get(\"usage\", 0):.4f}'); \
        print(f'   limit:        {d.get(\"limit\") or \"unlimited\"}'); \
        print(f'   is_free_tier: {d.get(\"is_free_tier\", False)}'); \
        m = httpx.get('https://openrouter.ai/api/v1/models', timeout=10).json()['data']; \
        ids = {x['id'] for x in m}; \
        sup = os.environ.get('LLM_SUPERVISOR_MODEL', 'openai/gpt-5.4'); \
        wrk = os.environ.get('LLM_WORKER_MODEL', 'google/gemini-2.5-flash'); \
        print(f'   supervisor {sup}: {\"✅\" if sup in ids else \"❌ не найдена\"}'); \
        print(f'   worker     {wrk}: {\"✅\" if wrk in ids else \"❌ не найдена\"}')" \
        || echo "❌ Проверь OPENROUTER_API_KEY в .env"

# Проверить что LangFuse достижим
check-langfuse:
    @curl -sf http://localhost:3001/api/public/health > /dev/null && echo "✅ LangFuse healthy" || echo "❌ LangFuse не отвечает"

# Проверить что MCP-сервер стартует и отдаёт инструменты (БД не требуется)
check-mcp:
    @uv run python -c "import asyncio; from family_finance.infrastructure.mcp import get_finance_tools; \
        t = asyncio.run(get_finance_tools()); \
        print('✅ MCP server ok, tools:', ', '.join(sorted(t)))" \
        || echo "❌ MCP-сервер не стартует"

# Список топ-10 самых дешёвых моделей на OpenRouter
list-cheap-models:
    @uv run python -c "import httpx; \
        m = httpx.get('https://openrouter.ai/api/v1/models', timeout=10).json()['data']; \
        paid = [x for x in m if float(x['pricing']['prompt']) > 0]; \
        paid.sort(key=lambda x: float(x['pricing']['prompt'])); \
        print(f'{\"model\":<48} {\"in/1M\":>10} {\"out/1M\":>10}'); \
        [print(f'{x[\"id\"]:<48} \${float(x[\"pricing\"][\"prompt\"])*1e6:>9.2f} \${float(x[\"pricing\"][\"completion\"])*1e6:>9.2f}') for x in paid[:10]]"

# === Quality ===

lint:
    uv run ruff check src/ tests/
    uv run mypy src/

fmt:
    uv run ruff format src/ tests/
    uv run ruff check --fix src/ tests/

test:
    uv run pytest -m "unit" -v

eval:
    uv run pytest -m "eval" -v --no-cov

# Upload cases to LangFuse datasets + run scored experiments (needs `just up`)
eval-report *agents:
    uv run python -m tests.evals.experiment {{agents}}

test-all:
    uv run pytest -v

cov:
    uv run pytest --cov-report=html
    @echo "Открой htmlcov/index.html"

# === Smoke (Phase 0 finish line) ===

smoke: up
    @sleep 3
    @just check-langfuse
    @just check-llm
    @just check-mcp
    @echo ""
    @echo "Если всё ✅ — стартуй бота: just run"
    @echo "Потом напиши /start в Telegram и проверь LangFuse:"
    @echo "  http://localhost:3001/project/ff-project/traces"
