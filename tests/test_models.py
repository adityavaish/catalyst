"""
Tests for data models.
"""

import pytest
from pydantic import ValidationError

from src.models import (
    AppConfig,
    CatalystRequest,
    CatalystResponse,
    ConnectorConfig,
    ConnectorType,
    HttpMethod,
    LLMConfig,
    PromptEndpoint,
)


def test_prompt_endpoint_minimal():
    ep = PromptEndpoint(path="/test", system_prompt="Do something")
    assert ep.method == HttpMethod.POST
    assert ep.temperature == 0.0


def test_prompt_endpoint_full():
    ep = PromptEndpoint(
        path="/api/items/{id}",
        method="GET",
        summary="Get item",
        system_prompt="Return item details",
        tags=["items"],
        connectors=["db"],
        temperature=0.5,
        max_tokens=2048,
    )
    assert ep.method == HttpMethod.GET
    assert ep.connectors == ["db"]


def test_connector_config_sql():
    cc = ConnectorConfig(
        name="mydb",
        type=ConnectorType.POSTGRES,
        connection_string="postgresql://localhost/test",
    )
    assert cc.type == ConnectorType.POSTGRES


def test_connector_config_mcp():
    cc = ConnectorConfig(
        name="tools",
        type=ConnectorType.MCP,
        mcp_server_command="npx",
        mcp_server_args=["-y", "@modelcontextprotocol/server-filesystem"],
    )
    assert cc.type == ConnectorType.MCP
    assert cc.mcp_server_command == "npx"


def test_catalyst_request():
    req = CatalystRequest(
        path="/api/test",
        method=HttpMethod.POST,
        body={"key": "value"},
        query_params={"page": "1"},
    )
    assert req.body == {"key": "value"}


def test_catalyst_response():
    resp = CatalystResponse(status_code=201, data={"id": 1})
    assert resp.status_code == 201
    assert resp.error is None


def test_app_config_defaults():
    config = AppConfig()
    assert config.port == 8000
    assert config.llm.model == "gpt-4o"
