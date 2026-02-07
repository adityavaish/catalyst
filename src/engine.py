"""
LLM Engine — the brain of Catalyst.

Takes a PromptEndpoint + CatalystRequest, builds the LLM messages
(including tool definitions from connectors), runs the completion loop
(handling tool calls), and returns a CatalystResponse.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from jinja2 import Template

from src.connectors import BaseConnector, ConnectorRegistry
from src.models import (
    CatalystRequest,
    CatalystResponse,
    LLMConfig,
    PromptEndpoint,
)

logger = logging.getLogger(__name__)

# Maximum tool-calling round-trips to prevent infinite loops
MAX_TOOL_ROUNDS = 10


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
                # For MCP, the action is the original tool name
                return await conn.execute(action, **arguments)
            elif prefix == "connector" and action == "request":
                # HTTP connector: action = the HTTP method
                method = arguments.pop("method", "GET")
                return await conn.execute(method, **arguments)
            else:
                return await conn.execute(action, **arguments)

    raise ValueError(f"No connector found for tool call: {tool_name}")


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


# ---------------------------------------------------------------------------
# Main engine entry point
# ---------------------------------------------------------------------------

async def process_request(
    endpoint: PromptEndpoint,
    request: CatalystRequest,
    llm_config: LLMConfig,
    connector_registry: ConnectorRegistry,
) -> CatalystResponse:
    """
    Process an incoming API request through the LLM.

    1. Build system + user messages
    2. Gather tool definitions from connectors
    3. Run the LLM completion loop (with tool calling)
    4. Parse and return the result
    """
    # 1. Assemble connectors for this endpoint
    active_connectors = connector_registry.get_many(endpoint.connectors)

    # 2. Collect tool definitions
    tools: list[dict[str, Any]] = []
    for conn in active_connectors:
        tools.extend(conn.tool_definitions())

    # 3. Build messages
    system_msg = SYSTEM_WRAPPER.format(
        system_prompt=endpoint.system_prompt,
        schema_hint=_build_schema_hint(endpoint),
    )
    user_msg = _build_user_message(endpoint, request)

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    # 4. Completion loop with tool calling
    for round_num in range(MAX_TOOL_ROUNDS):
        try:
            response = await _call_llm(
                messages=messages,
                tools=tools if tools else None,
                llm_config=llm_config,
                model_override=endpoint.model,
                temperature=endpoint.temperature,
                max_tokens=endpoint.max_tokens,
                response_format=endpoint.response_format,
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

        # Process tool calls
        messages.append(message.model_dump())

        for tool_call in message.tool_calls:
            fn_name = tool_call.function.name
            try:
                fn_args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                fn_args = {}

            logger.info("Tool call [round %d]: %s(%s)", round_num, fn_name, fn_args)

            try:
                result = await _dispatch_tool_call(fn_name, fn_args, active_connectors)
            except Exception as e:
                logger.exception("Tool call failed: %s", fn_name)
                result = {"error": str(e)}

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result, default=str),
            })
    else:
        return CatalystResponse(
            status_code=500,
            error="Max tool-calling rounds exceeded.",
        )

    # 5. Parse the final response
    raw_content = message.content or ""

    try:
        parsed = json.loads(raw_content)
    except json.JSONDecodeError:
        # Try to extract JSON from the response
        json_match = re.search(r'\{.*\}', raw_content, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
            except json.JSONDecodeError:
                return CatalystResponse(
                    status_code=200 if endpoint.response_format == "text" else 500,
                    data=raw_content if endpoint.response_format == "text" else None,
                    error=None if endpoint.response_format == "text" else f"Invalid JSON from LLM: {raw_content[:200]}",
                )
        else:
            return CatalystResponse(
                status_code=200 if endpoint.response_format == "text" else 500,
                data=raw_content if endpoint.response_format == "text" else None,
                error=None if endpoint.response_format == "text" else f"No JSON in LLM response: {raw_content[:200]}",
            )

    return CatalystResponse(
        status_code=parsed.get("status_code", 200),
        data=parsed.get("data", parsed),
        error=parsed.get("error"),
        meta=parsed.get("meta", {}),
    )
