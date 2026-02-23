# Catalyst

**AI-Native API Framework — business logic defined by prompts, not code.**

Catalyst lets you build fully functional APIs by writing prompt files instead of code. Each endpoint's behaviour is described in a YAML file containing an LLM system prompt, and the framework handles everything else: routing, input parsing, LLM orchestration, tool calling with databases/services, and JSON response formatting.

**Zero hardcoded business logic.** Deploy a new API by dropping in a YAML file.

---

## How It Works

```
┌──────────────┐     ┌──────────────────┐     ┌──────────────┐
│  HTTP Client │────▶│  Catalyst Server │────▶│  LLM (GPT,   │
│  (JSON in)   │◀────│  (FastAPI)       │◀────│  Claude, etc) │
└──────────────┘     └────────┬─────────┘     └──────┬───────┘
                              │                       │
                     ┌────────▼─────────┐    ┌───────▼────────┐
                     │  Prompt Files    │    │  Connectors    │
                     │  (YAML — your   │    │  (DB, MCP,     │
                     │   business logic)│    │   HTTP, Redis) │
                     └─────────────────┘    └────────────────┘
```

1. **Request arrives** → FastAPI receives JSON input at a dynamically registered route
2. **Prompt is loaded** → The endpoint's YAML prompt file defines the system behaviour
3. **LLM processes** → Input + system prompt + available tools are sent to the LLM
4. **Tools are called** → If the LLM needs data, it calls connectors (DB queries, MCP tools, HTTP APIs)
5. **Response returned** → LLM output is parsed and returned as structured JSON

---

## Quick Start

### 1. Install

```bash
# Clone and install
git clone <repo-url> catalyst && cd catalyst
pip install -e ".[dev]"

# Or with specific connectors
pip install -e ".[postgres,mcp,redis]"
```

### 2. Set your API key

```bash
cp .env.example .env
# Edit .env and add your AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT
```

### 3. Create a prompt file

Create `prompts/hello.yaml`:

```yaml
path: /api/hello
method: POST
summary: "Greeting API"
system_prompt: |
  You are a friendly greeting API.
  Given a person's name and preferred language, generate a warm,
  culturally appropriate greeting.
  Return:
    - "greeting": the greeting text
    - "language": the language used
schema:
  input_body:
    type: object
    properties:
      name:
        type: string
      language:
        type: string
        default: "English"
    required: ["name"]
```

### 4. Run

```bash
catalyst
# or
python -m catalyst.app
```

### 5. Call your API

```bash
curl -X POST http://localhost:8000/api/hello \
  -H "Content-Type: application/json" \
  -d '{"name": "Alice", "language": "Spanish"}'
```

Response:
```json
{
  "data": {
    "greeting": "¡Hola Alice! ¡Qué gusto saludarte! Espero que estés teniendo un día maravilloso.",
    "language": "Spanish"
  }
}
```

---

## Prompt File Reference

Each YAML file defines one or more API endpoints:

```yaml
# Single endpoint
path: /api/endpoint
method: POST          # GET | POST | PUT | PATCH | DELETE
summary: "Short description"
description: "Detailed description for API docs"
tags: ["category"]

# The brain — describes what this endpoint does
system_prompt: |
  You are a [description] API.
  [Detailed business logic instructions]
  [Input/output format specifications]

# Optional: Jinja2 template for the user message
# If omitted, raw JSON input is forwarded to the LLM
user_prompt_template: |
  Process this order for {{ body.customer_name }}:
  Items: {{ body.items | tojson }}

# LLM settings
model: "gpt-4o"       # Override default model
temperature: 0.0      # 0.0 = deterministic, 1.0 = creative
max_tokens: 4096
response_format: "json"  # "json" | "text"

# Schema hints (included in context for the LLM)
schema:
  input_body:          # JSON Schema for request body
    type: object
    properties: { ... }
  output_schema:       # JSON Schema for response
    type: object
    properties: { ... }
  input_params:        # Path/query parameter descriptions
    - name: id
      type: string
      required: true
      description: "Resource ID"

# External services this endpoint can access
connectors:
  - main_db
  - cache
  - weather_api

# Guards
auth_required: false
rate_limit: "100/minute"
```

Multiple endpoints in one file:
```yaml
endpoints:
  - path: /api/users
    method: GET
    system_prompt: "List users..."
  - path: /api/users
    method: POST
    system_prompt: "Create a user..."
```

---

## Connectors

Connectors let your AI endpoints interact with real data. Configure them in `catalyst.yaml`:

### SQL Databases (Postgres, MySQL, SQLite)

```yaml
connectors:
  - name: main_db
    type: postgres  # or mysql, sqlite
    connection_string: "postgresql://user:pass@localhost:5432/mydb"
```

The LLM gets tools like `connector_main_db_query(sql, values)` and `connector_main_db_execute(sql, values)` and decides what SQL to run based on the prompt.

### MongoDB

```yaml
connectors:
  - name: mongo
    type: mongodb
    connection_string: "mongodb://localhost:27017"
    options:
      database: "myapp"
```

Tools: `find`, `insert_one`, `update`, `delete`, `aggregate`.

### Redis

```yaml
connectors:
  - name: cache
    type: redis
    connection_string: "redis://localhost:6379"
```

Tools: `get`, `set`, `delete`, `keys`, `hgetall`, `hset`.

### HTTP (External APIs)

```yaml
connectors:
  - name: weather
    type: http
    connection_string: "https://api.weather.com"
    options:
      headers:
        Authorization: "Bearer ${WEATHER_API_KEY}"
      timeout: 10
```

Tools: `request(method, url, headers, params, body)`.

### Elasticsearch

```yaml
connectors:
  - name: search
    type: elasticsearch
    connection_string: "http://localhost:9200"
```

Tools: `search`, `index`, `get`, `delete`.

### MCP (Model Context Protocol)

```yaml
connectors:
  # stdio transport
  - name: filesystem
    type: mcp
    mcp_server_command: "npx"
    mcp_server_args: ["-y", "@modelcontextprotocol/server-filesystem", "/data"]

  # SSE transport
  - name: remote_tools
    type: mcp
    mcp_server_url: "http://localhost:3001/sse"
```

MCP tools are **auto-discovered** — the connector queries the MCP server for available tools and exposes them to the LLM automatically.

---

## LLM Providers

Catalyst uses [LiteLLM](https://github.com/BerriAI/litellm) under the hood, supporting 100+ LLM providers:

```yaml
# Azure OpenAI (default)
llm:
  provider: azure_openai
  model: azure/gpt-4o
  api_key: "${AZURE_OPENAI_API_KEY}"
  api_base: "${AZURE_OPENAI_ENDPOINT}"    # e.g. https://your-resource.openai.azure.com
  api_version: "2024-12-01-preview"
  # azure_deployment: "my-gpt4o-deployment"  # if deployment name differs from model

# OpenAI (direct)
llm:
  provider: openai
  model: gpt-4o
  api_key: "${OPENAI_API_KEY}"

# Anthropic
llm:
  provider: anthropic
  model: claude-sonnet-4-20250514
  api_key: "${ANTHROPIC_API_KEY}"

# Ollama (local)
llm:
  provider: ollama
  model: ollama/llama3
  api_base: "http://localhost:11434"

# Any LiteLLM-supported model
llm:
  provider: litellm
  model: "bedrock/anthropic.claude-3-sonnet"
```

---

## Built-in Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Health check — shows connector status |
| `GET /meta/endpoints` | Lists all registered AI endpoints |
| `GET /docs` | Interactive Swagger UI (auto-generated) |
| `GET /redoc` | ReDoc API documentation |

---

## Examples

The `prompts/examples/` directory contains ready-to-use prompt files:

| File | Endpoints | Description |
|------|-----------|-------------|
| `calculator.yaml` | `/api/calculate`, `/api/convert-units` | Math operations, unit conversions |
| `text_processing.yaml` | `/api/text/summarize`, `/api/text/sentiment`, `/api/text/extract`, `/api/text/translate` | NLP tasks |
| `data_tools.yaml` | `/api/validate/email`, `/api/transform/json`, `/api/generate/mock-data` | Data validation & generation |
| `ecommerce_with_db.yaml` | `/api/products`, `/api/orders/analyze` | Full CRUD + analytics with SQL DB |

---

## Project Structure

```
catalyst/
├── catalyst.yaml              # App configuration
├── pyproject.toml              # Python package definition
├── prompts/                    # ← YOUR BUSINESS LOGIC LIVES HERE
│   ├── hello.yaml              #    Each file = one or more endpoints
│   └── examples/               #    Example prompt files
│       ├── calculator.yaml
│       ├── text_processing.yaml
│       ├── data_tools.yaml
│       └── ecommerce_with_db.yaml
├── src/
│   ├── __init__.py
│   ├── app.py                  # FastAPI application factory + CLI
│   ├── models.py               # Pydantic data models
│   ├── prompt_loader.py        # YAML prompt file parser
│   ├── engine.py               # LLM orchestration + tool loop
│   ├── connectors.py           # DB, MCP, HTTP, Redis connectors
│   └── router.py               # Dynamic route registration
└── tests/
    ├── test_models.py
    └── test_prompt_loader.py
```

---

## Philosophy

> **The prompt IS the program.**

Traditional APIs encode business logic in code — you write handlers, validators, queries, and transformations. Catalyst replaces all of that with natural language prompts:

| Traditional API | Catalyst |
|----------------|----------|
| Write route handler code | Write a YAML prompt file |
| Hand-code SQL queries | LLM generates SQL from your description |
| Write validation logic | Describe validation rules in the prompt |
| Write data transformations | Describe the transformation in English |
| Write business rules | Describe the rules in the prompt |
| Deploy new code for changes | Edit a YAML file and restart |

This means **anyone who can describe what an API should do can build one** — no programming required for the business logic layer.

---

## License

MIT
