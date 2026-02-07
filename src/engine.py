"""
LLM Engine — the brain of Catalyst (optimized).

Takes a PromptEndpoint + CatalystRequest, builds the LLM messages
(including tool definitions from connectors), runs the completion loop
(handling tool calls), and returns a CatalystResponse.

Optimizations:
  - Response caching with configurable TTL (skip LLM for repeated inputs)
  - Parallel tool-call execution (when LLM emits multiple tool calls)
  - Pre-compiled system prompts (avoid rebuilding per-request)
  - Circuit breaker (fail fast when LLM is degraded)
  - SSE streaming support (deliver partial results to clients)
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Any, AsyncGenerator

from jinja2 import Template

from src.cache import ResponseCache
from src.circuit_breaker import CircuitBreaker, CircuitOpenError
from src.connectors import BaseConnector, ConnectorRegistry
from src.models import (
    CatalystRequest,
    CatalystResponse,
    LLMConfig,
    PerformanceConfig,
    PromptEndpoint,
)

logger = logging.getLogger(__name__)

# Maximum tool-calling round-trips to prevent infinite loops
MAX_TOOL_ROUNDS = 10


# ---------------------------------------------------------------------------
# Pre-compiled prompt cache  (avoids rebuilding system messages every request)
# ---------------------------------------------------------------------------

_compiled_prompts: dict[str, str] = {}


def precompile_system_prompt(endpoint: PromptEndpoint) -> str:
    """
    Build and cache the system message for an endpoint.
    Only rebuilt if the endpoint definition changes.
    """
    cache_key = f"{endpoint.path}:{endpoint.method.value}"
    if cache_key in _compiled_prompts:
        return _compiled_prompts[cache_key]

    prompt = SYSTEM_WRAPPER.format(
        system_prompt=endpoint.system_prompt,
        schema_hint=_build_schema_hint(endpoint),
    )
    _compiled_prompts[cache_key] = prompt
    return prompt


def clear_compiled_prompts():
    """Clear prompt cache (call on hot-reload)."""
    _compiled_prompts.clear()


# ---------------------------------------------------------------------------
# LLM client abstraction  (uses litellm for broadest provider support)
# ---------------------------------------------------------------------------

async def _call_llm(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    llm_config: LLMConfig,
    model_override: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    response_format: str = "json",
) -> dict[str, Any]:
    """
    Call the LLM via litellm (supports OpenAI, Anthropic, Azure, Ollama, etc.).
    Returns the raw response dict.
    """
    try:
        import litellm  # type: ignore
    except ImportError:
        raise ImportError("Install 'litellm' to use the LLM engine.")

    model = model_override or llm_config.model

    # Build kwargs
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"

    if response_format == "json":
        kwargs["response_format"] = {"type": "json_object"}

    if llm_config.api_key:
        kwargs["api_key"] = llm_config.api_key
    if llm_config.api_base:
        kwargs["api_base"] = llm_config.api_base

    # Merge any extra provider-specific kwargs
    kwargs.update(llm_config.extra)

    response = await litellm.acompletion(**kwargs)
    return response


async def _call_llm_streaming(
    messages: list[dict[str, Any]],
    llm_config: LLMConfig,
    model_override: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 4096,
) -> AsyncGenerator[str, None]:
    """Stream LLM response chunks for SSE delivery."""
    try:
        import litellm  # type: ignore
    except ImportError:
        raise ImportError("Install 'litellm' to use the LLM engine.")

    model = model_override or llm_config.model

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
    }

    if llm_config.api_key:
        kwargs["api_key"] = llm_config.api_key
    if llm_config.api_base:
        kwargs["api_base"] = llm_config.api_base
    kwargs.update(llm_config.extra)

    response = await litellm.acompletion(**kwargs)
    async for chunk in response:
        delta = chunk.choices[0].delta
        if hasattr(delta, "content") and delta.content:
            yield delta.content


# ---------------------------------------------------------------------------
# Tool-call dispatcher
# ---------------------------------------------------------------------------

def _parse_connector_tool_name(tool_name: str) -> tuple[str, str, str]:
    """
    Parse a tool call name into (prefix, connector_name, action).
    Handles patterns like:
      connector_mydb_query   -> ("connector", "mydb", "query")
      mcp_myserver_toolname  -> ("mcp", "myserver", "toolname")
    """
    if tool_name.startswith("connector_"):
        rest = tool_name[len("connector_"):]
    elif tool_name.startswith("mcp_"):
        rest = tool_name[len("mcp_"):]
    else:
        return ("", "", tool_name)

    # Split on first underscore after prefix to get connector name
    parts = rest.split("_", 1)
    if len(parts) == 2:
        prefix = "connector" if tool_name.startswith("connector_") else "mcp"
        return (prefix, parts[0], parts[1])
    return ("", "", tool_name)


async def _dispatch_tool_call(
    tool_name: str,
    arguments: dict[str, Any],
    connectors: list[BaseConnector],
) -> Any:
    """Route a tool call to the correct connector and action."""
    prefix, connector_name, action = _parse_connector_tool_name(tool_name)

    for conn in connectors:
        if conn.name == connector_name:
            logger.info("Dispatching tool %s → connector '%s' action '%s'", tool_name, connector_name, action)
            if prefix == "mcp":
                return await conn.execute(action, **arguments)
            elif prefix == "connector" and action == "request":
                method = arguments.pop("method", "GET")
                return await conn.execute(method, **arguments)
            else:
                return await conn.execute(action, **arguments)

    raise ValueError(f"No connector found for tool call: {tool_name}")


async def _dispatch_tool_calls_parallel(
    tool_calls: list[Any],
    connectors: list[BaseConnector],
) -> list[dict[str, Any]]:
    """
    Execute multiple tool calls concurrently using asyncio.gather.
    Returns tool-role messages in the same order as the input.
    """

    async def _run_one(tc) -> dict[str, Any]:
        fn_name = tc.function.name
        try:
            fn_args = json.loads(tc.function.arguments)
        except json.JSONDecodeError:
            fn_args = {}

        try:
            result = await _dispatch_tool_call(fn_name, fn_args, connectors)
        except Exception as e:
            logger.exception("Tool call failed: %s", fn_name)
            result = {"error": str(e)}

        return {
            "role": "tool",
            "tool_call_id": tc.id,
            "content": json.dumps(result, default=str),
        }

    results = await asyncio.gather(*[_run_one(tc) for tc in tool_calls])
    return list(results)


async def _dispatch_tool_calls_serial(
    tool_calls: list[Any],
    connectors: list[BaseConnector],
) -> list[dict[str, Any]]:
    """Execute tool calls one at a time (fallback if parallel is disabled)."""
    results = []
    for tc in tool_calls:
        fn_name = tc.function.name
        try:
            fn_args = json.loads(tc.function.arguments)
        except json.JSONDecodeError:
            fn_args = {}

        logger.info("Tool call: %s(%s)", fn_name, fn_args)

        try:
            result = await _dispatch_tool_call(fn_name, fn_args, connectors)
        except Exception as e:
            logger.exception("Tool call failed: %s", fn_name)
            result = {"error": str(e)}

        results.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "content": json.dumps(result, default=str),
        })
    return results


# ---------------------------------------------------------------------------
# Message builder
# ---------------------------------------------------------------------------

def _build_user_message(endpoint: PromptEndpoint, request: CatalystRequest) -> str:
    """Build the user message from the request data."""
    request_data = {
        "path_params": request.path_params,
        "query_params": request.query_params,
        "body": request.body,
        "method": request.method.value,
        "path": request.path,
    }

    if endpoint.user_prompt_template:
        template = Template(endpoint.user_prompt_template)
        return template.render(**request_data)
    else:
        return json.dumps(request_data, indent=2, default=str)


SYSTEM_WRAPPER = """\
You are an AI-powered API endpoint. You must process the incoming request \
and produce a response according to the business logic described below.

--- BUSINESS LOGIC ---
{system_prompt}
--- END BUSINESS LOGIC ---

RULES:
1. You MUST respond with valid JSON only (no markdown, no explanation outside JSON).
2. Your JSON response MUST include a "status_code" field (integer HTTP status) \
and a "data" field containing the response payload.
3. If the request is invalid or an error occurs, set an appropriate status_code \
(4xx/5xx) and include an "error" field with a message.
4. If you need data from external services or databases, use the provided tool \
functions. Do NOT fabricate data — if no tool is available, say so in the error.
5. Follow the business logic precisely. Do not add, remove, or change behaviour \
beyond what the logic specifies.

{schema_hint}
"""


def _build_schema_hint(endpoint: PromptEndpoint) -> str:
    """Build optional schema hint text."""
    parts = []
    if endpoint.schema_.input_body:
        parts.append(f"Expected request body schema:\n{json.dumps(endpoint.schema_.input_body, indent=2)}")
    if endpoint.schema_.output_schema:
        parts.append(f"Expected response data schema:\n{json.dumps(endpoint.schema_.output_schema, indent=2)}")
    if endpoint.schema_.input_params:
        param_descs = []
        for p in endpoint.schema_.input_params:
            param_descs.append(f"  - {p.name} ({p.type}): {p.description}")
        parts.append("Parameters:\n" + "\n".join(param_descs))
    return "\n\n".join(parts) if parts else ""


def _parse_llm_response(raw_content: str, response_format: str) -> CatalystResponse:
    """Parse the LLM's final text into a CatalystResponse."""
    try:
        parsed = json.loads(raw_content)
    except json.JSONDecodeError:
        json_match = re.search(r'\{.*\}', raw_content, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
            except json.JSONDecodeError:
                return CatalystResponse(
                    status_code=200 if response_format == "text" else 500,
                    data=raw_content if response_format == "text" else None,
                    error=None if response_format == "text" else f"Invalid JSON from LLM: {raw_content[:200]}",
                )
        else:
            return CatalystResponse(
                status_code=200 if response_format == "text" else 500,
                data=raw_content if response_format == "text" else None,
                error=None if response_format == "text" else f"No JSON in LLM response: {raw_content[:200]}",
            )

    return CatalystResponse(
        status_code=parsed.get("status_code", 200),
        data=parsed.get("data", parsed),
        error=parsed.get("error"),
        meta=parsed.get("meta", {}),
    )


# ---------------------------------------------------------------------------
# Main engine entry point
# ---------------------------------------------------------------------------

async def process_request(
    endpoint: PromptEndpoint,
    request: CatalystRequest,
    llm_config: LLMConfig,
    connector_registry: ConnectorRegistry,
    cache: ResponseCache | None = None,
    circuit_breaker: CircuitBreaker | None = None,
    perf_config: PerformanceConfig | None = None,
) -> CatalystResponse:
    """
    Process an incoming API request through the LLM.

    Optimized pipeline:
      1. Check cache → return immediately if hit
      2. Build messages (using pre-compiled system prompt)
      3. Gather tool definitions from connectors
      4. Run LLM through circuit breaker
      5. Execute tool calls in parallel
      6. Parse result, store in cache, return
    """
    start_time = time.monotonic()
    parallel_tools = perf_config.parallel_tool_calls if perf_config else True

    # ── 1. Cache check ──────────────────────────────────────────────────
    cache_key: str | None = None
    if cache and endpoint.cache_ttl > 0:
        input_data = {
            "path_params": request.path_params,
            "query_params": request.query_params,
            "body": request.body,
        }
        cache_key = ResponseCache.make_key(endpoint.path, request.method.value, input_data)
        cached = await cache.get(cache_key)
        if cached is not None:
            elapsed = (time.monotonic() - start_time) * 1000
            logger.info("Cache HIT for %s %s (%.1fms)", request.method.value, request.path, elapsed)
            cached.meta["cached"] = True
            cached.meta["latency_ms"] = round(elapsed, 1)
            return cached

    # ── 2. Build messages (pre-compiled system prompt) ──────────────────
    system_msg = precompile_system_prompt(endpoint)
    user_msg = _build_user_message(endpoint, request)

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    # ── 3. Gather tools ─────────────────────────────────────────────────
    active_connectors = connector_registry.get_many(endpoint.connectors)
    tools: list[dict[str, Any]] = []
    for conn in active_connectors:
        tools.extend(conn.tool_definitions())

    # ── 4. Completion loop with circuit breaker ─────────────────────────
    message = None
    round_num = 0
    for round_num in range(MAX_TOOL_ROUNDS):
        # Wrap LLM call with circuit breaker
        async def _do_llm_call():
            return await _call_llm(
                messages=messages,
                tools=tools if tools else None,
                llm_config=llm_config,
                model_override=endpoint.model,
                temperature=endpoint.temperature,
                max_tokens=endpoint.max_tokens,
                response_format=endpoint.response_format,
            )

        try:
            if circuit_breaker:
                response = await circuit_breaker.call(_do_llm_call)
            else:
                response = await _do_llm_call()
        except CircuitOpenError as e:
            return CatalystResponse(
                status_code=503,
                error=f"Service temporarily unavailable: {e}",
                meta={"circuit_breaker": "open"},
            )
        except asyncio.TimeoutError:
            return CatalystResponse(
                status_code=504,
                error="LLM request timed out",
            )
        except Exception as e:
            logger.exception("LLM call failed on round %d", round_num)
            return CatalystResponse(
                status_code=502,
                error=f"LLM call failed: {e}",
            )

        choice = response.choices[0]
        message = choice.message

        # If no tool calls, we have the final answer
        if not getattr(message, "tool_calls", None):
            break

        # ── 5. Execute tool calls (parallel or serial) ──────────────────
        messages.append(message.model_dump())

        if parallel_tools and len(message.tool_calls) > 1:
            logger.info(
                "Executing %d tool calls in parallel [round %d]",
                len(message.tool_calls),
                round_num,
            )
            tool_results = await _dispatch_tool_calls_parallel(
                message.tool_calls, active_connectors,
            )
        else:
            tool_results = await _dispatch_tool_calls_serial(
                message.tool_calls, active_connectors,
            )

        messages.extend(tool_results)
    else:
        return CatalystResponse(
            status_code=500,
            error="Max tool-calling rounds exceeded.",
        )

    # ── 6. Parse and cache ──────────────────────────────────────────────
    raw_content = message.content or "" if message else ""
    result = _parse_llm_response(raw_content, endpoint.response_format)

    elapsed = (time.monotonic() - start_time) * 1000
    result.meta["latency_ms"] = round(elapsed, 1)
    result.meta["cached"] = False

    # Store in cache if configured
    if cache and cache_key and endpoint.cache_ttl > 0 and result.status_code < 400:
        await cache.set(cache_key, result, ttl=endpoint.cache_ttl)
        logger.debug("Cached response for %s (ttl=%ds)", cache_key[:12], endpoint.cache_ttl)

    logger.info(
        "← %s %s → %d (%.0fms, %d tool rounds)",
        request.method.value,
        request.path,
        result.status_code,
        elapsed,
        round_num + 1,
    )

    return result


# ---------------------------------------------------------------------------
# Streaming entry point
# ---------------------------------------------------------------------------

async def process_request_streaming(
    endpoint: PromptEndpoint,
    request: CatalystRequest,
    llm_config: LLMConfig,
    connector_registry: ConnectorRegistry,
    cache: ResponseCache | None = None,
) -> AsyncGenerator[str, None]:
    """
    Stream the LLM response as Server-Sent Events (SSE).

    NOTE: Streaming bypasses tool-calling (returns direct LLM output).
    Best for endpoints that don't need connectors (pure prompt logic).
    """
    # Check cache first
    if cache and endpoint.cache_ttl > 0:
        input_data = {
            "path_params": request.path_params,
            "query_params": request.query_params,
            "body": request.body,
        }
        cache_key = ResponseCache.make_key(endpoint.path, request.method.value, input_data)
        cached = await cache.get(cache_key)
        if cached is not None:
            yield f"data: {json.dumps({'data': cached.data, 'cached': True})}\n\n"
            yield "data: [DONE]\n\n"
            return

    system_msg = precompile_system_prompt(endpoint)
    user_msg = _build_user_message(endpoint, request)

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    buffer = ""
    async for chunk in _call_llm_streaming(
        messages=messages,
        llm_config=llm_config,
        model_override=endpoint.model,
        temperature=endpoint.temperature,
        max_tokens=endpoint.max_tokens,
    ):
        buffer += chunk
        yield f"data: {json.dumps({'chunk': chunk})}\n\n"

    # Send the final assembled response
    yield f"data: {json.dumps({'complete': True, 'full_response': buffer})}\n\n"
    yield "data: [DONE]\n\n"
