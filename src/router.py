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
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from starlette.responses import StreamingResponse

from src.cache import ResponseCache
from src.circuit_breaker import CircuitBreaker
from src.connectors import ConnectorRegistry
from src.engine import process_request, process_request_streaming
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


def create_router(
    endpoints: list[PromptEndpoint],
    llm_config: LLMConfig,
    connector_registry: ConnectorRegistry,
    cache: ResponseCache | None = None,
    circuit_breaker: CircuitBreaker | None = None,
    perf_config: PerformanceConfig | None = None,
) -> APIRouter:
    """
    Build a FastAPI ``APIRouter`` with one route per ``PromptEndpoint``.
    """
    router = APIRouter()

    for ep in endpoints:
        _register_endpoint(
            router, ep, llm_config, connector_registry,
            cache, circuit_breaker, perf_config,
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
):
    """Register a single PromptEndpoint as a FastAPI route."""
    method = _fastapi_method(endpoint.method)
    path = endpoint.path

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
    route_decorator(
        path,
        summary=ep.summary or f"{method.upper()} {path}",
        description=ep.description or ep.system_prompt[:200],
        tags=ep.tags or ["catalyst"],
        name=handler.__name__,
    )(handler)

    logger.info("Registered route: %s %s", method.upper(), path)
