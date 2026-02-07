"""
Tests for the prompt loader.
"""

from pathlib import Path
from textwrap import dedent

import pytest
import yaml

from src.models import HttpMethod, PromptEndpoint
from src.prompt_loader import load_prompt_file


@pytest.fixture
def tmp_prompt(tmp_path: Path):
    """Helper to write a YAML prompt file and return its path."""

    def _write(content: str, name: str = "test.yaml") -> Path:
        p = tmp_path / name
        p.write_text(dedent(content))
        return p

    return _write


def test_single_endpoint(tmp_prompt):
    path = tmp_prompt("""\
        path: /api/hello
        method: GET
        summary: Say hello
        system_prompt: "You are a greeting API. Return a friendly greeting."
    """)

    endpoints = load_prompt_file(path)
    assert len(endpoints) == 1
    ep = endpoints[0]
    assert ep.path == "/api/hello"
    assert ep.method == HttpMethod.GET
    assert "greeting" in ep.system_prompt.lower()


def test_multiple_endpoints(tmp_prompt):
    path = tmp_prompt("""\
        endpoints:
          - path: /api/a
            method: POST
            system_prompt: "Endpoint A"
          - path: /api/b
            method: GET
            system_prompt: "Endpoint B"
    """)

    endpoints = load_prompt_file(path)
    assert len(endpoints) == 2
    assert endpoints[0].path == "/api/a"
    assert endpoints[1].path == "/api/b"


def test_default_values(tmp_prompt):
    path = tmp_prompt("""\
        path: /api/test
        system_prompt: "Test endpoint"
    """)

    endpoints = load_prompt_file(path)
    ep = endpoints[0]
    assert ep.method == HttpMethod.POST  # default
    assert ep.temperature == 0.0
    assert ep.max_tokens == 4096
    assert ep.connectors == []
    assert ep.auth_required is False


def test_with_connectors(tmp_prompt):
    path = tmp_prompt("""\
        path: /api/data
        method: GET
        system_prompt: "Query data"
        connectors:
          - main_db
          - cache
    """)

    endpoints = load_prompt_file(path)
    assert endpoints[0].connectors == ["main_db", "cache"]


def test_with_schema(tmp_prompt):
    path = tmp_prompt("""\
        path: /api/validate
        method: POST
        system_prompt: "Validate input"
        schema:
          input_body:
            type: object
            properties:
              email:
                type: string
            required: ["email"]
          output_schema:
            type: object
            properties:
              valid:
                type: boolean
    """)

    endpoints = load_prompt_file(path)
    ep = endpoints[0]
    assert ep.schema_.input_body is not None
    assert "email" in ep.schema_.input_body["properties"]
    assert ep.schema_.output_schema is not None
