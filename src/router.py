"""
Dynamic Router — registers FastAPI routes from PromptEndpoint definitions.

No business logic lives here.  Each route is a thin shim that:
  1. Captures the incoming request (path params, query, body, headers)
  2. Wraps it into a CatalystRequest
  3. Passes it to the LLM engine
  4. Returns the CatalystResponse as JSON
"""

from __future__ import annotations

import logging
import re
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from starlette.responses import StreamingResponse

from src.cache import ResponseCache
from src.circuit_breaker import CircuitBreaker
from src.connectors import ConnectorRegistry
from src.engine import process_request, process_request_streaming
from src.execution_plan import PlanCache
from src.models import (
    CatalystRequest,
    HttpMethod,
    LLMConfig,
    PerformanceConfig,
    PromptEndpoint,
)

logger = logging.getLogger(__name__)


def _fastapi_method(method: HttpMethod) -> str:
    return method.value.lower()


# ---------------------------------------------------------------------------
# OpenAPI schema builder — makes Swagger show input fields
# ---------------------------------------------------------------------------

_TYPE_MAP = {
    "string": "string",
    "integer": "integer",
    "number": "number",
    "boolean": "boolean",
    "array": "array",
    "object": "object",
}


def _build_openapi_extra(endpoint: PromptEndpoint) -> dict[str, Any]:
    """
    Build an ``openapi_extra`` dict from the endpoint's YAML schema so that
    Swagger UI displays proper request body fields and parameters.
    """
    extra: dict[str, Any] = {}

    # ── Path parameters (extracted from URL pattern like /api/products/{id}) ──
    path_param_names = re.findall(r"\{(\w+)\}", endpoint.path)

    parameters: list[dict[str, Any]] = []

    for name in path_param_names:
        # Check if there's a matching ParamSpec for richer metadata
        param_spec = next(
            (p for p in endpoint.schema_.input_params if p.name == name), None
        )
        parameters.append({
            "name": name,
            "in": "path",
            "required": True,
            "schema": {
                "type": _TYPE_MAP.get(
                    param_spec.type if param_spec else "string", "string"
                ),
            },
            "description": param_spec.description if param_spec else "",
        })

    # ── Query parameters (from input_params that aren't path params) ──────────
    for p in endpoint.schema_.input_params:
        if p.name in path_param_names:
            continue  # already handled above
        param_schema: dict[str, Any] = {
            "type": _TYPE_MAP.get(p.type, "string"),
        }
        if p.default is not None:
            param_schema["default"] = p.default
        if p.enum:
            param_schema["enum"] = p.enum

        parameters.append({
            "name": p.name,
            "in": "query",
            "required": p.required,
            "schema": param_schema,
            "description": p.description,
        })

    if parameters:
        extra["parameters"] = parameters

    # ── Request body (from input_body, for POST/PUT/PATCH) ────────────────────
    if endpoint.schema_.input_body and endpoint.method in (
        HttpMethod.POST, HttpMethod.PUT, HttpMethod.PATCH,
    ):
        extra["requestBody"] = {
            "required": True,
            "content": {
                "application/json": {
                    "schema": endpoint.schema_.input_body,
                }
            },
        }

    return extra


def create_router(
    endpoints: list[PromptEndpoint],
    llm_config: LLMConfig,
    connector_registry: ConnectorRegistry,
    cache: ResponseCache | None = None,
    circuit_breaker: CircuitBreaker | None = None,
    perf_config: PerformanceConfig | None = None,
    plan_cache: PlanCache | None = None,
) -> APIRouter:
    """
    Build a FastAPI ``APIRouter`` with one route per ``PromptEndpoint``.
    """
    router = APIRouter()

    for ep in endpoints:
        _register_endpoint(
            router, ep, llm_config, connector_registry,
            cache, circuit_breaker, perf_config, plan_cache,
        )

    return router


def _register_endpoint(
    router: APIRouter,
    endpoint: PromptEndpoint,
    llm_config: LLMConfig,
    connector_registry: ConnectorRegistry,
    cache: ResponseCache | None = None,
    circuit_breaker: CircuitBreaker | None = None,
    perf_config: PerformanceConfig | None = None,
    plan_cache: PlanCache | None = None,
):
    """Register a single PromptEndpoint as a FastAPI route."""
    method = _fastapi_method(endpoint.method)
    path = endpoint.path

    # Build OpenAPI extra from YAML schema so Swagger shows parameters
    openapi_extra = _build_openapi_extra(endpoint)

    # We need to capture `endpoint` in a closure properly
    ep = endpoint

    async def handler(request: Request):
        # Gather all input
        path_params = dict(request.path_params)
        query_params = dict(request.query_params)
        headers = dict(request.headers)

        body: Any = None
        if request.method in ("POST", "PUT", "PATCH"):
            try:
                body = await request.json()
            except Exception:
                body_bytes = await request.body()
                body = body_bytes.decode("utf-8", errors="replace") if body_bytes else None

        cat_request = CatalystRequest(
            path=str(request.url.path),
            method=HttpMethod(request.method),
            path_params=path_params,
            query_params=query_params,
            headers=headers,
            body=body,
        )

        logger.info(
            "→ %s %s | body_size=%s",
            request.method,
            request.url.path,
            len(str(body)) if body else 0,
        )

        # Streaming path — return SSE stream
        if ep.streaming:
            return StreamingResponse(
                process_request_streaming(
                    endpoint=ep,
                    request=cat_request,
                    llm_config=llm_config,
                    connector_registry=connector_registry,
                    cache=cache,
                ),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        # Standard path — full response
        result = await process_request(
            endpoint=ep,
            request=cat_request,
            llm_config=llm_config,
            connector_registry=connector_registry,
            cache=cache,
            circuit_breaker=circuit_breaker,
            perf_config=perf_config,
            plan_cache=plan_cache,
        )

        response_body: dict[str, Any] = {"data": result.data}
        if result.error:
            response_body["error"] = result.error
        if result.meta:
            response_body["meta"] = result.meta

        return JSONResponse(
            status_code=result.status_code,
            content=response_body,
        )

    # Dynamically set the handler's name for OpenAPI docs
    handler.__name__ = f"handle_{method}_{path.replace('/', '_').strip('_')}"
    handler.__doc__ = ep.description or ep.summary or f"AI-powered endpoint: {method.upper()} {path}"

    # Register with the correct HTTP method
    route_decorator = getattr(router, method)
    route_kwargs: dict[str, Any] = {
        "summary": ep.summary or f"{method.upper()} {path}",
        "description": ep.description or ep.system_prompt[:200],
        "tags": ep.tags or ["catalyst"],
        "name": handler.__name__,
    }
    if openapi_extra:
        route_kwargs["openapi_extra"] = openapi_extra

    route_decorator(path, **route_kwargs)(handler)

    logger.info("Registered route: %s %s", method.upper(), path)
