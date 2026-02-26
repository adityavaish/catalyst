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
  - SQL transaction wrapping (atomic multi-statement operations)
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
from src.connectors import BaseConnector, ConnectorRegistry, SQLConnector
from src.execution_plan import ExecutionPlan, PlanCache, execute_plan, extract_plan_from_trace
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

    # Azure OpenAI specific parameters
    if llm_config.provider.value == "azure_openai":
        kwargs["api_version"] = llm_config.api_version
        if llm_config.azure_deployment:
            kwargs["model"] = f"azure/{llm_config.azure_deployment}"

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

    # Azure OpenAI specific parameters
    if llm_config.provider.value == "azure_openai":
        kwargs["api_version"] = llm_config.api_version
        if llm_config.azure_deployment:
            kwargs["model"] = f"azure/{llm_config.azure_deployment}"

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
    # Try to match against actual connector names (handles underscores in names)
    prefix = ""
    if tool_name.startswith("connector_"):
        prefix = "connector"
        rest = tool_name[len("connector_"):]
    elif tool_name.startswith("mcp_"):
        prefix = "mcp"
        rest = tool_name[len("mcp_"):]
    else:
        raise ValueError(f"No connector found for tool call: {tool_name}")

    # Match by checking if rest starts with a known connector name
    matched_conn = None
    action = ""
    for conn in connectors:
        candidate = conn.name + "_"
        if rest.startswith(candidate):
            matched_conn = conn
            action = rest[len(candidate):]
            break

    # Fallback to first-underscore split for simple names
    if matched_conn is None:
        parts = rest.split("_", 1)
        if len(parts) == 2:
            for conn in connectors:
                if conn.name == parts[0]:
                    matched_conn = conn
                    action = parts[1]
                    break

    if matched_conn is None:
        raise ValueError(f"No connector found for tool call: {tool_name}")

    logger.info("Dispatching tool %s → connector '%s' action '%s'", tool_name, matched_conn.name, action)
    if prefix == "mcp":
        return await matched_conn.execute(action, **arguments)
    elif prefix == "connector" and action == "request":
        method = arguments.pop("method", "GET")
        return await matched_conn.execute(method, **arguments)
    else:
        return await matched_conn.execute(action, **arguments)


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
    plan_cache: PlanCache | None = None,
) -> CatalystResponse:
    """
    Process an incoming API request through the LLM.

    Optimized pipeline:
      1. Check response cache → return immediately if hit
      2. Check execution plan cache → execute plan without LLM if hit
      3. Build messages (using pre-compiled system prompt)
      4. Gather tool definitions from connectors
      5. Run LLM through circuit breaker
      6. Execute tool calls in parallel
      7. Parse result, extract plan, store in caches, return
    """
    start_time = time.monotonic()
    parallel_tools = perf_config.parallel_tool_calls if perf_config else True
    plan_config = perf_config.execution_plans if perf_config else None

    # ── 1. Response cache check ─────────────────────────────────────────
    input_data = {
        "path_params": request.path_params,
        "query_params": request.query_params,
        "body": request.body,
    }
    cache_key: str | None = None
    if cache and endpoint.cache_ttl > 0:
        cache_key = ResponseCache.make_key(endpoint.path, request.method.value, input_data)
        cached = await cache.get(cache_key)
        if cached is not None:
            elapsed = (time.monotonic() - start_time) * 1000
            logger.info("Cache HIT for %s %s (%.1fms)", request.method.value, request.path, elapsed)
            cached.meta["cached"] = True
            cached.meta["latency_ms"] = round(elapsed, 1)
            return cached

    # ── 2. Execution plan check ─────────────────────────────────────────
    endpoint_key = f"{request.method.value}:{endpoint.path}"
    active_connectors = connector_registry.get_many(endpoint.connectors)

    if plan_cache and plan_config and plan_config.enabled and active_connectors and endpoint.plan_cache_enabled:
        plan = await plan_cache.get(endpoint_key, input_data=input_data)
        if plan is not None:
            try:
                logger.info("Plan HIT for %s — executing without LLM", endpoint_key)
                plan_response = await execute_plan(plan, input_data, active_connectors)
                await plan_cache.record_execution()

                result = CatalystResponse(
                    status_code=200,  # plans are only extracted from successful requests
                    data=plan_response.get("data", plan_response),
                    error=plan_response.get("error"),
                    meta=plan_response.get("meta", {}),
                )
                elapsed = (time.monotonic() - start_time) * 1000
                result.meta["latency_ms"] = round(elapsed, 1)
                result.meta["cached"] = False
                result.meta["plan_executed"] = True
                result.meta["plan_hit_count"] = plan.hit_count

                # Also store in response cache
                if cache and cache_key and endpoint.cache_ttl > 0 and result.status_code < 400:
                    await cache.set(cache_key, result, ttl=endpoint.cache_ttl)

                logger.info(
                    "← %s %s → %d (%.0fms, plan execution, no LLM)",
                    request.method.value, request.path, result.status_code, elapsed,
                )
                return result

            except Exception as e:
                logger.warning("Plan execution failed for %s: %s — falling back to LLM", endpoint_key, e)
                await plan_cache.record_error(endpoint_key, input_data=input_data)

    # ── 3. Build messages (pre-compiled system prompt) ──────────────────
    system_msg = precompile_system_prompt(endpoint)
    user_msg = _build_user_message(endpoint, request)

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    # ── 4. Gather tools ─────────────────────────────────────────────────
    tools: list[dict[str, Any]] = []
    for conn in active_connectors:
        tools.extend(conn.tool_definitions())

    # ── 5. Completion loop with circuit breaker ─────────────────────────
    # Track tool results for plan extraction
    all_tool_results_parsed: list[dict] = []
    message = None
    round_num = 0

    # Transaction wrapping: for write methods, wrap multi-statement DB
    # operations in a transaction so CHECK-constraint violations roll
    # back ALL mutations (no partial writes).
    is_write_method = request.method.value.upper() in ("POST", "PUT", "PATCH", "DELETE")
    sql_connectors = [c for c in active_connectors if isinstance(c, SQLConnector)]
    use_txn = is_write_method and len(sql_connectors) > 0
    txn_started = False
    constraint_error: str | None = None

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

        # ── 6. Execute tool calls (parallel or serial) ──────────────────
        messages.append(message.model_dump())

        # Start transaction before the first write tool call
        has_write = any(
            tc.function.name.endswith("_execute")
            for tc in message.tool_calls
        )
        if use_txn and has_write and not txn_started:
            for sc in sql_connectors:
                await sc.begin_transaction()
            txn_started = True

        if parallel_tools and len(message.tool_calls) > 1 and not txn_started:
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

        # Check for database constraint violations in tool results
        for tr in tool_results:
            content = tr.get("content", "")
            if "BANK_ERR:" in content or "CHECK constraint failed" in content:
                constraint_error = content
                break

        # If a constraint was violated, rollback and return error
        if constraint_error:
            if txn_started:
                for sc in sql_connectors:
                    try:
                        await sc.rollback_transaction()
                    except Exception:
                        pass  # transaction may already be dead after IntegrityError
                txn_started = False
            # Extract meaningful error message
            err_msg = constraint_error
            if "BANK_ERR:" in err_msg:
                try:
                    parsed_err = json.loads(err_msg)
                    err_msg = parsed_err.get("error", err_msg)
                except (json.JSONDecodeError, TypeError):
                    pass
                # Extract just the BANK_ERR message
                if "BANK_ERR:" in err_msg:
                    err_msg = err_msg.split("BANK_ERR:", 1)[1].strip().rstrip("'\"}")
            elif "CHECK constraint failed" in err_msg:
                if "balance" in err_msg.lower():
                    err_msg = "Insufficient funds — balance cannot go below zero"
                else:
                    err_msg = "Database constraint violation"

            elapsed = (time.monotonic() - start_time) * 1000
            return CatalystResponse(
                status_code=400,
                data={"error": err_msg, "status_code": 400},
                error=err_msg,
                meta={
                    "latency_ms": round(elapsed, 1),
                    "cached": False,
                    "plan_executed": False,
                    "constraint_guard": True,
                },
            )

        # Collect parsed tool results for plan extraction
        for tr in tool_results:
            try:
                all_tool_results_parsed.append(json.loads(tr["content"]))
            except (json.JSONDecodeError, KeyError):
                all_tool_results_parsed.append({})

        messages.extend(tool_results)
    else:
        # Max rounds exceeded — rollback any open transaction
        if txn_started:
            for sc in sql_connectors:
                try:
                    await sc.rollback_transaction()
                except Exception:
                    pass
        return CatalystResponse(
            status_code=500,
            error="Max tool-calling rounds exceeded.",
        )

    # ── 6b. Commit transaction if one was started ─────────────────────
    if txn_started:
        try:
            for sc in sql_connectors:
                await sc.commit_transaction()
        except Exception:
            logger.exception("Transaction commit failed — rolling back")
            for sc in sql_connectors:
                try:
                    await sc.rollback_transaction()
                except Exception:
                    pass
            return CatalystResponse(
                status_code=500,
                error="Database commit failed",
            )

    # ── 7. Parse, extract plan, cache  ──────────────────────────────────
    raw_content = message.content or "" if message else ""
    result = _parse_llm_response(raw_content, endpoint.response_format)

    elapsed = (time.monotonic() - start_time) * 1000
    result.meta["latency_ms"] = round(elapsed, 1)
    result.meta["cached"] = False
    result.meta["plan_executed"] = False

    # Store in response cache
    if cache and cache_key and endpoint.cache_ttl > 0 and result.status_code < 400:
        await cache.set(cache_key, result, ttl=endpoint.cache_ttl)
        logger.debug("Cached response for %s (ttl=%ds)", cache_key[:12], endpoint.cache_ttl)

    # Extract and cache execution plan (async, non-blocking)
    if (
        plan_cache
        and plan_config
        and plan_config.enabled
        and endpoint.plan_cache_enabled
        and active_connectors
        and all_tool_results_parsed
        and result.status_code < 400
    ):
        try:
            final_data = result.data if isinstance(result.data, dict) else {"data": result.data}
            # Only pass 'data' to the response template — status_code is a
            # framework field and must NOT be matched against tool-result
            # values (e.g. amount=200 ≠ HTTP 200).
            final_response = {
                "data": final_data,
            }
            plan = extract_plan_from_trace(
                endpoint_key=endpoint_key,
                input_data=input_data,
                messages=messages,
                tool_results_raw=all_tool_results_parsed,
                final_response=final_response,
                ttl=plan_config.plan_ttl,
                max_errors=plan_config.max_errors,
            )
            if plan:
                await plan_cache.put(plan)
                result.meta["plan_cached"] = True
        except Exception:
            logger.exception("Failed to extract execution plan for %s", endpoint_key)

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
