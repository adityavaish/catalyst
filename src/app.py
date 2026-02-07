"""
Catalyst — Main application entry point.

Boots the FastAPI server, loads config, initialises connectors,
reads prompt files, and registers dynamic routes.
"""

from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.cache import ResponseCache
from src.circuit_breaker import CircuitBreaker
from src.connectors import ConnectorRegistry
from src.models import AppConfig, ConnectorConfig
from src.prompt_loader import load_all_prompts
from src.router import create_router

logger = logging.getLogger("catalyst")


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _resolve_env(value: Any) -> Any:
    """Recursively resolve ${ENV_VAR} placeholders in config values."""
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        env_key = value[2:-1]
        return os.environ.get(env_key, value)
    if isinstance(value, dict):
        return {k: _resolve_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env(v) for v in value]
    return value


def load_config(config_path: str | Path | None = None) -> AppConfig:
    """
    Load app config from a YAML file.  Falls back to env vars and defaults.
    """
    path = Path(config_path) if config_path else Path("catalyst.yaml")

    if path.exists():
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}
        raw = _resolve_env(raw)
        return AppConfig.model_validate(raw)
    else:
        logger.warning("No config file found at %s — using defaults + env vars.", path)
        return AppConfig(
            llm=_resolve_env({
                "provider": os.getenv("CATALYST_LLM_PROVIDER", "openai"),
                "model": os.getenv("CATALYST_LLM_MODEL", "gpt-4o"),
                "api_key": os.getenv("OPENAI_API_KEY", ""),
            }),
            prompts_dir=os.getenv("CATALYST_PROMPTS_DIR", "prompts"),
        )


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

connector_registry = ConnectorRegistry()
response_cache: ResponseCache | None = None
circuit_breaker: CircuitBreaker | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    global response_cache, circuit_breaker
    config: AppConfig = app.state.config

    # Initialize response cache
    if config.cache.enabled:
        response_cache = ResponseCache(
            max_size=config.cache.max_size,
            default_ttl=config.cache.default_ttl,
        )
        logger.info(
            "Response cache enabled (max_size=%d, default_ttl=%ds)",
            config.cache.max_size,
            config.cache.default_ttl,
        )

    # Initialize circuit breaker
    if config.performance.circuit_breaker_enabled:
        circuit_breaker = CircuitBreaker(
            failure_threshold=config.performance.circuit_breaker_threshold,
            recovery_timeout=config.performance.circuit_breaker_recovery,
            timeout=config.performance.llm_timeout,
        )
        logger.info(
            "Circuit breaker enabled (threshold=%d, recovery=%ds, timeout=%ds)",
            config.performance.circuit_breaker_threshold,
            config.performance.circuit_breaker_recovery,
            config.performance.llm_timeout,
        )

    # Connect all configured connectors
    for cc in config.connectors:
        try:
            await connector_registry.register(cc)
            logger.info("Connector '%s' (%s) ready.", cc.name, cc.type.value)
        except Exception:
            logger.exception("Failed to connect '%s'", cc.name)

    # Load prompt files and register routes
    prompts_dir = Path(config.prompts_dir)
    endpoints = load_all_prompts(prompts_dir)

    if not endpoints:
        logger.warning(
            "No endpoints found in '%s'. "
            "Create YAML prompt files to define your API.",
            prompts_dir,
        )

    api_router = create_router(
        endpoints,
        config.llm,
        connector_registry,
        cache=response_cache,
        circuit_breaker=circuit_breaker,
        perf_config=config.performance,
    )
    app.include_router(api_router)

    logger.info(
        "Catalyst is live — %d endpoint(s) on http://%s:%s",
        len(endpoints),
        config.host,
        config.port,
    )

    yield

    # Shutdown
    await connector_registry.disconnect_all()
    if response_cache:
        await response_cache.clear()
    logger.info("Catalyst shut down.")


def create_app(config_path: str | Path | None = None) -> FastAPI:
    """Build and return the FastAPI application."""
    config = load_config(config_path)

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, config.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    app = FastAPI(
        title=config.app_name,
        description=(
            "AI-Native API — business logic defined by prompts, not code. "
            "Every endpoint is powered by an LLM with optional connectors "
            "to databases, MCP servers, and external services."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    app.state.config = config

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Health check (the only hardcoded endpoint!)
    @app.get("/health", tags=["system"])
    async def health():
        health_data: dict[str, Any] = {
            "status": "ok",
            "app": config.app_name,
            "connectors": [
                {"name": c.name, "connected": c._connected}
                for c in connector_registry.all
            ],
        }
        if response_cache:
            health_data["cache"] = response_cache.stats
        if circuit_breaker:
            health_data["circuit_breaker"] = circuit_breaker.stats
        return health_data

    # Meta endpoint — lists all registered AI endpoints
    @app.get("/meta/endpoints", tags=["system"])
    async def list_endpoints():
        routes = []
        for route in app.routes:
            if hasattr(route, "methods") and hasattr(route, "path"):
                if route.path not in ("/health", "/meta/endpoints", "/openapi.json", "/docs", "/redoc"):
                    routes.append({
                        "path": route.path,
                        "methods": list(route.methods) if route.methods else [],
                        "name": getattr(route, "name", ""),
                        "summary": getattr(route, "summary", ""),
                    })
        return {"endpoints": routes}

    return app


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    """Run the server from the command line."""
    import uvicorn

    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    config = load_config(config_path)

    uvicorn.run(
        "src.app:create_app",
        host=config.host,
        port=config.port,
        reload=config.debug,
        factory=True,
        log_level=config.log_level.lower(),
    )


if __name__ == "__main__":
    main()
