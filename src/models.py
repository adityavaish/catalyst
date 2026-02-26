"""
Core data models for Catalyst.

Defines the schema for prompt files, API requests/responses,
connector configurations, and runtime context.
"""

from __future__ import annotations

import enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class HttpMethod(str, enum.Enum):
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    DELETE = "DELETE"


class ConnectorType(str, enum.Enum):
    """Supported external connector types."""
    MCP = "mcp"
    POSTGRES = "postgres"
    MYSQL = "mysql"
    SQLITE = "sqlite"
    MONGODB = "mongodb"
    REDIS = "redis"
    HTTP = "http"
    ELASTICSEARCH = "elasticsearch"
    MATH = "math"


class LLMProvider(str, enum.Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE_OPENAI = "azure_openai"
    OLLAMA = "ollama"
    LITELLM = "litellm"


# ---------------------------------------------------------------------------
# Connector config — defined in connectors.yaml or per-prompt
# ---------------------------------------------------------------------------

class ConnectorConfig(BaseModel):
    """Configuration for a single external connector."""
    name: str = Field(..., description="Unique name to reference this connector")
    type: ConnectorType
    connection_string: str | None = Field(None, description="DB connection string or URL")
    options: dict[str, Any] = Field(default_factory=dict, description="Extra driver-specific options")

    # MCP-specific
    mcp_server_command: str | None = Field(None, description="Command to start MCP server (stdio)")
    mcp_server_args: list[str] = Field(default_factory=list)
    mcp_server_url: str | None = Field(None, description="URL for SSE/streamable MCP server")
    mcp_server_env: dict[str, str] = Field(default_factory=dict, description="Environment vars for MCP server")


# ---------------------------------------------------------------------------
# Prompt endpoint definition — loaded from YAML files
# ---------------------------------------------------------------------------

class ParamSpec(BaseModel):
    """Describes a single parameter (path, query, header, body field)."""
    name: str
    type: str = "string"
    required: bool = True
    description: str = ""
    default: Any = None
    enum: list[str] | None = None


# ---------------------------------------------------------------------------
# Pre-LLM guards — validators & preconditions (YAML-driven, generic)
# ---------------------------------------------------------------------------

class InputValidator(BaseModel):
    """A pure-input validation rule evaluated before the LLM is called.

    The ``check`` field is a safe expression evaluated against the request
    context (``body``, ``params``, ``path``).  If it evaluates to falsy the
    request is rejected immediately with ``status`` and ``error``.

    Example YAML::

        validators:
          - check: "body.amount > 0"
            error: "Amount must be greater than 0"
            status: 400
    """
    check: str = Field(..., description="Safe expression to evaluate (e.g. 'body.amount > 0')")
    error: str = Field(..., description="Human-readable error message returned on failure")
    status: int = Field(400, description="HTTP status code returned on failure")


class PreconditionCheck(BaseModel):
    """A single assertion evaluated against SQL query result rows."""
    check: str = Field(..., description="Safe expression over 'rows' + request context")
    error: str = Field(..., description="Human-readable error message")
    status: int = Field(400, description="HTTP status code on failure")


class Precondition(BaseModel):
    """A SQL query + assertions run before the LLM.

    The query uses named bind parameters (``:_p0``, ``:_p1``, …) whose
    values are resolved from ``params`` via dotted context paths.

    Example YAML::

        preconditions:
          - query: "SELECT id, status FROM accounts WHERE id = :_p0"
            connector: main_db
            params: { _p0: "path.id" }
            checks:
              - check: "len(rows) > 0"
                error: "Account not found"
                status: 404
    """
    query: str = Field(..., description="SQL SELECT with named bind parameters")
    connector: str = Field(..., description="Connector name to execute the query against")
    params: dict[str, str] = Field(
        default_factory=dict,
        description="Map of bind-param name → dotted context path (e.g. _p0: 'path.id')",
    )
    checks: list[PreconditionCheck] = Field(
        default_factory=list,
        description="Assertions evaluated against the query result rows",
    )


class EndpointSchema(BaseModel):
    """Input/output schema hints included in the prompt context."""
    input_params: list[ParamSpec] = Field(default_factory=list)
    input_body: dict[str, Any] | None = Field(None, description="JSON Schema for request body")
    output_schema: dict[str, Any] | None = Field(None, description="JSON Schema for response")


class PromptEndpoint(BaseModel):
    """
    Fully describes a single AI-driven API endpoint.
    Loaded from a YAML prompt file.
    """
    # Routing
    path: str = Field(..., description="URL path, e.g. /api/orders/{id}")
    method: HttpMethod = HttpMethod.POST
    summary: str = ""
    description: str = ""
    tags: list[str] = Field(default_factory=list)

    # LLM behaviour
    system_prompt: str = Field(..., description="System prompt defining the business logic")
    user_prompt_template: str | None = Field(
        None,
        description="Optional Jinja2 template for user message. "
                    "If omitted, the raw JSON input is forwarded.",
    )
    model: str | None = Field(None, description="Override default model for this endpoint")
    temperature: float = 0.0
    max_tokens: int = 4096
    response_format: str = "json"  # "json" | "text"

    # Schema
    schema_: EndpointSchema = Field(default_factory=EndpointSchema, alias="schema")

    # Connectors this endpoint may use
    connectors: list[str] = Field(
        default_factory=list,
        description="Names of connectors this endpoint needs access to",
    )

    # Pre-LLM guards (generic, YAML-driven)
    validators: list[InputValidator] = Field(
        default_factory=list,
        description="Pure input validation rules — checked before LLM, no DB access",
    )
    preconditions: list[Precondition] = Field(
        default_factory=list,
        description="SQL-backed assertions — checked before LLM, require DB",
    )

    # Performance
    cache_ttl: float = 0  # seconds, 0 = no caching
    plan_cache_enabled: bool = True  # set False to skip plan caching for this endpoint
    streaming: bool = False  # enable SSE streaming

    # Middleware / guards
    auth_required: bool = False
    rate_limit: str | None = None  # e.g. "10/minute"

    model_config = {"populate_by_name": True}


# ---------------------------------------------------------------------------
# Runtime request / response wrappers
# ---------------------------------------------------------------------------

class CatalystRequest(BaseModel):
    """Internal representation of an incoming request."""
    path: str
    method: HttpMethod
    path_params: dict[str, str] = Field(default_factory=dict)
    query_params: dict[str, str] = Field(default_factory=dict)
    headers: dict[str, str] = Field(default_factory=dict)
    body: Any = None


class CatalystResponse(BaseModel):
    """Standard wrapper returned by the AI engine."""
    status_code: int = 200
    data: Any = None
    error: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# App-level configuration
# ---------------------------------------------------------------------------

class LLMConfig(BaseModel):
    provider: LLMProvider = LLMProvider.AZURE_OPENAI
    model: str = "azure/gpt-4o-mini"
    api_key: str | None = None
    api_base: str | None = Field(None, description="Azure OpenAI endpoint URL, e.g. https://your-resource.openai.azure.com")
    api_version: str = "2024-12-01-preview"
    azure_deployment: str | None = Field(None, description="Azure OpenAI deployment name (if different from model)")
    temperature: float = 0.0
    max_tokens: int = 4096
    extra: dict[str, Any] = Field(default_factory=dict)


class CacheConfig(BaseModel):
    """Response caching settings."""
    enabled: bool = True
    max_size: int = 1000
    default_ttl: float = 300  # seconds


class PlanConfig(BaseModel):
    """Execution plan caching settings."""
    enabled: bool = True
    plan_ttl: float = 3600       # seconds before a plan expires (0 = infinite)
    max_plans: int = 500         # max cached plans in memory
    max_errors: int = 3          # consecutive errors before plan is invalidated
    background_refresh: bool = True  # proactively regenerate plans before expiry


class PerformanceConfig(BaseModel):
    """Performance tuning settings."""
    parallel_tool_calls: bool = True
    circuit_breaker_enabled: bool = True
    circuit_breaker_threshold: int = 5
    circuit_breaker_recovery: float = 30.0
    llm_timeout: float = 60.0  # per-call timeout in seconds
    execution_plans: PlanConfig = Field(default_factory=PlanConfig)


class AppConfig(BaseModel):
    """Top-level application configuration."""
    app_name: str = "Catalyst"
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    prompts_dir: str = "prompts"
    connectors: list[ConnectorConfig] = Field(default_factory=list)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    log_level: str = "INFO"
