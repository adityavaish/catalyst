"""
Connectors — generic adapters for databases, MCP servers, and HTTP services.

Each connector exposes a simple ``execute(action, **kwargs)`` interface so the
LLM engine can invoke them uniformly via tool-calling.
"""

from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from typing import Any

from src.models import ConnectorConfig, ConnectorType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class BaseConnector(ABC):
    """All connectors share this interface."""

    def __init__(self, config: ConnectorConfig):
        self.config = config
        self.name = config.name
        self._connected = False

    async def connect(self):
        """Open connection / initialise client."""
        self._connected = True

    async def disconnect(self):
        """Tear down connection."""
        self._connected = False

    @abstractmethod
    async def execute(self, action: str, **kwargs: Any) -> Any:
        """Run an action and return the result."""
        ...

    def tool_definitions(self) -> list[dict[str, Any]]:
        """
        Return OpenAI-compatible tool/function definitions that describe
        what this connector can do.  The LLM uses these to decide when
        to call the connector.
        """
        return []


# ---------------------------------------------------------------------------
# SQL connector (Postgres / MySQL / SQLite)
# ---------------------------------------------------------------------------

class SQLConnector(BaseConnector):
    """
    Async SQL connector using ``databases`` + ``sqlalchemy`` core.
    Supports Postgres, MySQL, SQLite.
    """

    def __init__(self, config: ConnectorConfig):
        super().__init__(config)
        self._db: Any = None

    async def connect(self):
        try:
            import databases  # type: ignore
        except ImportError:
            raise ImportError(
                "Install 'databases[asyncpg]' (or aiosqlite/aiomysql) "
                "to use SQL connectors."
            )
        self._db = databases.Database(self.config.connection_string)
        await self._db.connect()
        await super().connect()
        logger.info("SQL connector '%s' connected.", self.name)

    async def disconnect(self):
        if self._db:
            await self._db.disconnect()
        await super().disconnect()

    async def execute(self, action: str, **kwargs: Any) -> Any:
        """
        Actions:
          - query: run a SELECT and return rows as dicts
          - execute: run INSERT/UPDATE/DELETE and return affected rows
        """
        sql = kwargs.get("sql", "")
        values = kwargs.get("values", {})

        if action == "query":
            rows = await self._db.fetch_all(query=sql, values=values)
            return [dict(r._mapping) for r in rows]
        elif action == "execute":
            result = await self._db.execute(query=sql, values=values)
            return {"last_id": result}
        else:
            raise ValueError(f"Unknown SQL action: {action}")

    def tool_definitions(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": f"connector_{self.name}_query",
                    "description": f"Run a read-only SQL SELECT query against the '{self.name}' database and return rows.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "sql": {"type": "string", "description": "The SQL SELECT statement to execute."},
                            "values": {"type": "object", "description": "Named bind parameters.", "default": {}},
                        },
                        "required": ["sql"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": f"connector_{self.name}_execute",
                    "description": f"Run a write SQL statement (INSERT/UPDATE/DELETE) against the '{self.name}' database.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "sql": {"type": "string", "description": "The SQL statement to execute."},
                            "values": {"type": "object", "description": "Named bind parameters.", "default": {}},
                        },
                        "required": ["sql"],
                    },
                },
            },
        ]


# ---------------------------------------------------------------------------
# MongoDB connector
# ---------------------------------------------------------------------------

class MongoDBConnector(BaseConnector):
    def __init__(self, config: ConnectorConfig):
        super().__init__(config)
        self._client: Any = None

    async def connect(self):
        try:
            from motor.motor_asyncio import AsyncIOMotorClient  # type: ignore
        except ImportError:
            raise ImportError("Install 'motor' to use the MongoDB connector.")
        self._client = AsyncIOMotorClient(self.config.connection_string)
        await super().connect()
        logger.info("MongoDB connector '%s' connected.", self.name)

    async def disconnect(self):
        if self._client:
            self._client.close()
        await super().disconnect()

    async def execute(self, action: str, **kwargs: Any) -> Any:
        db_name = kwargs.get("database", self.config.options.get("database", "default"))
        collection = kwargs["collection"]
        db = self._client[db_name]
        col = db[collection]

        if action == "find":
            cursor = col.find(kwargs.get("filter", {}), kwargs.get("projection"))
            docs = await cursor.to_list(length=kwargs.get("limit", 100))
            for d in docs:
                d["_id"] = str(d["_id"])
            return docs
        elif action == "insert_one":
            result = await col.insert_one(kwargs["document"])
            return {"inserted_id": str(result.inserted_id)}
        elif action == "insert_many":
            result = await col.insert_many(kwargs["documents"])
            return {"inserted_ids": [str(i) for i in result.inserted_ids]}
        elif action == "update":
            result = await col.update_many(kwargs["filter"], kwargs["update"])
            return {"matched": result.matched_count, "modified": result.modified_count}
        elif action == "delete":
            result = await col.delete_many(kwargs["filter"])
            return {"deleted": result.deleted_count}
        elif action == "aggregate":
            cursor = col.aggregate(kwargs["pipeline"])
            return await cursor.to_list(length=kwargs.get("limit", 100))
        else:
            raise ValueError(f"Unknown MongoDB action: {action}")

    def tool_definitions(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": f"connector_{self.name}_find",
                    "description": f"Find documents in a MongoDB collection in '{self.name}'.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "collection": {"type": "string"},
                            "filter": {"type": "object", "default": {}},
                            "projection": {"type": "object"},
                            "limit": {"type": "integer", "default": 100},
                        },
                        "required": ["collection"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": f"connector_{self.name}_insert_one",
                    "description": f"Insert a single document into a MongoDB collection in '{self.name}'.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "collection": {"type": "string"},
                            "document": {"type": "object"},
                        },
                        "required": ["collection", "document"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": f"connector_{self.name}_update",
                    "description": f"Update documents in a MongoDB collection in '{self.name}'.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "collection": {"type": "string"},
                            "filter": {"type": "object"},
                            "update": {"type": "object"},
                        },
                        "required": ["collection", "filter", "update"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": f"connector_{self.name}_delete",
                    "description": f"Delete documents from a MongoDB collection in '{self.name}'.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "collection": {"type": "string"},
                            "filter": {"type": "object"},
                        },
                        "required": ["collection", "filter"],
                    },
                },
            },
        ]


# ---------------------------------------------------------------------------
# Redis connector
# ---------------------------------------------------------------------------

class RedisConnector(BaseConnector):
    def __init__(self, config: ConnectorConfig):
        super().__init__(config)
        self._redis: Any = None

    async def connect(self):
        try:
            import redis.asyncio as aioredis  # type: ignore
        except ImportError:
            raise ImportError("Install 'redis' to use the Redis connector.")
        self._redis = aioredis.from_url(
            self.config.connection_string or "redis://localhost:6379",
            decode_responses=True,
        )
        await super().connect()
        logger.info("Redis connector '%s' connected.", self.name)

    async def disconnect(self):
        if self._redis:
            await self._redis.close()
        await super().disconnect()

    async def execute(self, action: str, **kwargs: Any) -> Any:
        if action == "get":
            return await self._redis.get(kwargs["key"])
        elif action == "set":
            return await self._redis.set(kwargs["key"], kwargs["value"], ex=kwargs.get("ttl"))
        elif action == "delete":
            return await self._redis.delete(kwargs["key"])
        elif action == "keys":
            return await self._redis.keys(kwargs.get("pattern", "*"))
        elif action == "hgetall":
            return await self._redis.hgetall(kwargs["key"])
        elif action == "hset":
            return await self._redis.hset(kwargs["key"], mapping=kwargs["mapping"])
        else:
            raise ValueError(f"Unknown Redis action: {action}")

    def tool_definitions(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": f"connector_{self.name}_get",
                    "description": f"Get a value by key from Redis '{self.name}'.",
                    "parameters": {
                        "type": "object",
                        "properties": {"key": {"type": "string"}},
                        "required": ["key"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": f"connector_{self.name}_set",
                    "description": f"Set a key-value pair in Redis '{self.name}'.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "key": {"type": "string"},
                            "value": {"type": "string"},
                            "ttl": {"type": "integer", "description": "TTL in seconds"},
                        },
                        "required": ["key", "value"],
                    },
                },
            },
        ]


# ---------------------------------------------------------------------------
# HTTP connector — call external REST APIs
# ---------------------------------------------------------------------------

class HTTPConnector(BaseConnector):
    def __init__(self, config: ConnectorConfig):
        super().__init__(config)
        self._session: Any = None

    async def connect(self):
        try:
            import httpx  # type: ignore
        except ImportError:
            raise ImportError("Install 'httpx' to use the HTTP connector.")
        import httpx
        self._session = httpx.AsyncClient(
            base_url=self.config.connection_string or "",
            timeout=float(self.config.options.get("timeout", 30)),
            headers=self.config.options.get("headers", {}),
        )
        await super().connect()
        logger.info("HTTP connector '%s' connected.", self.name)

    async def disconnect(self):
        if self._session:
            await self._session.aclose()
        await super().disconnect()

    async def execute(self, action: str, **kwargs: Any) -> Any:
        """action = HTTP method (get, post, put, patch, delete)."""
        method = action.upper()
        url = kwargs.get("url", "/")
        headers = kwargs.get("headers", {})
        params = kwargs.get("params", {})
        body = kwargs.get("body")

        resp = await self._session.request(
            method, url, headers=headers, params=params, json=body,
        )
        try:
            return {"status": resp.status_code, "body": resp.json()}
        except Exception:
            return {"status": resp.status_code, "body": resp.text}

    def tool_definitions(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": f"connector_{self.name}_request",
                    "description": f"Make an HTTP request via the '{self.name}' connector.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "method": {"type": "string", "enum": ["GET", "POST", "PUT", "PATCH", "DELETE"]},
                            "url": {"type": "string", "description": "Relative or absolute URL"},
                            "headers": {"type": "object", "default": {}},
                            "params": {"type": "object", "default": {}},
                            "body": {"type": "object"},
                        },
                        "required": ["method", "url"],
                    },
                },
            },
        ]


# ---------------------------------------------------------------------------
# Elasticsearch connector
# ---------------------------------------------------------------------------

class ElasticsearchConnector(BaseConnector):
    def __init__(self, config: ConnectorConfig):
        super().__init__(config)
        self._client: Any = None

    async def connect(self):
        try:
            from elasticsearch import AsyncElasticsearch  # type: ignore
        except ImportError:
            raise ImportError("Install 'elasticsearch[async]' to use the Elasticsearch connector.")
        self._client = AsyncElasticsearch(
            self.config.connection_string or "http://localhost:9200",
            **self.config.options,
        )
        await super().connect()
        logger.info("Elasticsearch connector '%s' connected.", self.name)

    async def disconnect(self):
        if self._client:
            await self._client.close()
        await super().disconnect()

    async def execute(self, action: str, **kwargs: Any) -> Any:
        if action == "search":
            result = await self._client.search(index=kwargs["index"], body=kwargs.get("query", {}))
            return result.body
        elif action == "index":
            result = await self._client.index(
                index=kwargs["index"],
                document=kwargs["document"],
                id=kwargs.get("id"),
            )
            return result.body
        elif action == "get":
            result = await self._client.get(index=kwargs["index"], id=kwargs["id"])
            return result.body
        elif action == "delete":
            result = await self._client.delete(index=kwargs["index"], id=kwargs["id"])
            return result.body
        else:
            raise ValueError(f"Unknown Elasticsearch action: {action}")

    def tool_definitions(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": f"connector_{self.name}_search",
                    "description": f"Search documents in Elasticsearch '{self.name}'.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "index": {"type": "string"},
                            "query": {"type": "object", "description": "ES query DSL"},
                        },
                        "required": ["index"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": f"connector_{self.name}_index",
                    "description": f"Index (upsert) a document in Elasticsearch '{self.name}'.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "index": {"type": "string"},
                            "document": {"type": "object"},
                            "id": {"type": "string"},
                        },
                        "required": ["index", "document"],
                    },
                },
            },
        ]


# ---------------------------------------------------------------------------
# MCP connector — Model Context Protocol
# ---------------------------------------------------------------------------

class MCPConnector(BaseConnector):
    """
    Connects to an MCP server (stdio or SSE) and exposes its tools
    as callable functions for the LLM.
    """

    def __init__(self, config: ConnectorConfig):
        super().__init__(config)
        self._session: Any = None
        self._client: Any = None
        self._tools: list[dict[str, Any]] = []
        self._exit_stack: Any = None

    async def connect(self):
        try:
            from mcp import ClientSession, StdioServerParameters  # type: ignore
            from mcp.client.stdio import stdio_client  # type: ignore
        except ImportError:
            raise ImportError("Install 'mcp' to use the MCP connector.")

        import contextlib

        self._exit_stack = contextlib.AsyncExitStack()

        if self.config.mcp_server_command:
            # stdio transport
            from mcp import StdioServerParameters
            from mcp.client.stdio import stdio_client

            server_params = StdioServerParameters(
                command=self.config.mcp_server_command,
                args=self.config.mcp_server_args,
                env=self.config.mcp_server_env or None,
            )
            read_stream, write_stream = await self._exit_stack.enter_async_context(
                stdio_client(server_params)
            )
        elif self.config.mcp_server_url:
            # SSE transport
            try:
                from mcp.client.sse import sse_client  # type: ignore
            except ImportError:
                raise ImportError("Install 'mcp[sse]' for SSE transport support.")
            read_stream, write_stream = await self._exit_stack.enter_async_context(
                sse_client(self.config.mcp_server_url)
            )
        else:
            raise ValueError(
                f"MCP connector '{self.name}' needs either "
                "mcp_server_command (stdio) or mcp_server_url (SSE)."
            )

        self._session = await self._exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        await self._session.initialize()

        # Discover available tools
        tools_result = await self._session.list_tools()
        self._tools = []
        for tool in tools_result.tools:
            self._tools.append({
                "type": "function",
                "function": {
                    "name": f"mcp_{self.name}_{tool.name}",
                    "description": tool.description or "",
                    "parameters": tool.inputSchema if hasattr(tool, 'inputSchema') else {"type": "object", "properties": {}},
                },
            })

        await super().connect()
        logger.info(
            "MCP connector '%s' connected with %d tools.",
            self.name,
            len(self._tools),
        )

    async def disconnect(self):
        if self._exit_stack:
            await self._exit_stack.aclose()
        await super().disconnect()

    async def execute(self, action: str, **kwargs: Any) -> Any:
        """Call an MCP tool by name."""
        # action is the tool name (without the mcp_{name}_ prefix)
        result = await self._session.call_tool(action, kwargs)
        # Extract text content from result
        if hasattr(result, 'content'):
            texts = []
            for block in result.content:
                if hasattr(block, 'text'):
                    texts.append(block.text)
            combined = "\n".join(texts)
            try:
                return json.loads(combined)
            except (json.JSONDecodeError, ValueError):
                return combined
        return str(result)

    def tool_definitions(self) -> list[dict[str, Any]]:
        return self._tools


# ---------------------------------------------------------------------------
# Math evaluator (safe Python math)
# ---------------------------------------------------------------------------

import ast
import math as _math
import operator


class MathConnector(BaseConnector):
    """
    Safe math expression evaluator using Python's AST.
    Allows the LLM to delegate arithmetic to deterministic code.
    """

    # Allowed operators
    _OPS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    # Allowed math functions
    _FUNCS: dict[str, Any] = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sqrt": _math.sqrt,
        "log": _math.log,
        "log10": _math.log10,
        "log2": _math.log2,
        "sin": _math.sin,
        "cos": _math.cos,
        "tan": _math.tan,
        "asin": _math.asin,
        "acos": _math.acos,
        "atan": _math.atan,
        "ceil": _math.ceil,
        "floor": _math.floor,
        "factorial": _math.factorial,
        "pi": _math.pi,
        "e": _math.e,
    }

    def _safe_eval(self, node: ast.AST) -> float | int:
        """Recursively evaluate an AST node safely."""
        if isinstance(node, ast.Expression):
            return self._safe_eval(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError(f"Unsupported constant: {node.value!r}")
        if isinstance(node, ast.BinOp):
            op = self._OPS.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
            return op(self._safe_eval(node.left), self._safe_eval(node.right))
        if isinstance(node, ast.UnaryOp):
            op = self._OPS.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsupported unary: {type(node.op).__name__}")
            return op(self._safe_eval(node.operand))
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in self._FUNCS:
                args = [self._safe_eval(a) for a in node.args]
                return self._FUNCS[node.func.id](*args)
            raise ValueError(f"Unsupported function call")
        if isinstance(node, ast.Name):
            if node.id in self._FUNCS and not callable(self._FUNCS[node.id]):
                return self._FUNCS[node.id]  # constants like pi, e
            raise ValueError(f"Unknown variable: {node.id}")
        raise ValueError(f"Unsupported expression: {ast.dump(node)}")

    async def execute(self, action: str, **kwargs: Any) -> Any:
        """
        Actions:
          - evaluate: safely evaluate a math expression string
        """
        if action == "evaluate":
            expression = kwargs.get("expression", "")
            try:
                tree = ast.parse(expression, mode="eval")
                result = self._safe_eval(tree)
                # Round to avoid floating point noise
                if isinstance(result, float) and result == int(result):
                    result = int(result)
                return {"result": result, "expression": expression}
            except Exception as e:
                return {"error": f"Cannot evaluate '{expression}': {e}"}
        return {"error": f"Unknown action: {action}"}

    def tool_definitions(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": f"connector_{self.name}_evaluate",
                    "description": (
                        "Evaluate a mathematical expression and return the exact numeric result. "
                        "Supports: +, -, *, /, //, **, %, and functions like sqrt(), log(), "
                        "sin(), cos(), tan(), abs(), round(), ceil(), floor(), factorial(). "
                        "Constants: pi, e. ALWAYS use this tool for any arithmetic computation."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {
                                "type": "string",
                                "description": (
                                    "Python math expression to evaluate, e.g. "
                                    "'20 + 8121 - 1824 / 12' or 'sqrt(144) + 3**2'"
                                ),
                            }
                        },
                        "required": ["expression"],
                    },
                },
            }
        ]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_CONNECTOR_MAP: dict[ConnectorType, type[BaseConnector]] = {
    ConnectorType.POSTGRES: SQLConnector,
    ConnectorType.MYSQL: SQLConnector,
    ConnectorType.SQLITE: SQLConnector,
    ConnectorType.MONGODB: MongoDBConnector,
    ConnectorType.REDIS: RedisConnector,
    ConnectorType.HTTP: HTTPConnector,
    ConnectorType.ELASTICSEARCH: ElasticsearchConnector,
    ConnectorType.MCP: MCPConnector,
    ConnectorType.MATH: MathConnector,
}


def create_connector(config: ConnectorConfig) -> BaseConnector:
    """Instantiate the right connector subclass from config."""
    cls = _CONNECTOR_MAP.get(config.type)
    if cls is None:
        raise ValueError(f"Unsupported connector type: {config.type}")
    return cls(config)


class ConnectorRegistry:
    """Holds all active connectors, keyed by name."""

    def __init__(self):
        self._connectors: dict[str, BaseConnector] = {}

    async def register(self, config: ConnectorConfig):
        conn = create_connector(config)
        await conn.connect()
        self._connectors[conn.name] = conn

    def get(self, name: str) -> BaseConnector | None:
        return self._connectors.get(name)

    def get_many(self, names: list[str]) -> list[BaseConnector]:
        return [c for n in names if (c := self._connectors.get(n)) is not None]

    async def disconnect_all(self):
        for conn in self._connectors.values():
            try:
                await conn.disconnect()
            except Exception:
                logger.exception("Error disconnecting %s", conn.name)
        self._connectors.clear()

    @property
    def all(self) -> list[BaseConnector]:
        return list(self._connectors.values())
