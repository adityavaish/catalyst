"""
Prompt Loader — reads YAML prompt files and returns PromptEndpoint models.

Prompt files define the API's behaviour.  Each YAML file can contain one or
more endpoint definitions under a top-level ``endpoints`` key.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from src.models import PromptEndpoint

logger = logging.getLogger(__name__)


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a single YAML file and return its dict."""
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_prompt_file(path: Path) -> list[PromptEndpoint]:
    """
    Parse a single prompt YAML file and return a list of ``PromptEndpoint``.

    The file may be:
      - A single endpoint dict (with a ``path`` key at the root).
      - A mapping with an ``endpoints`` list containing multiple defs.
    """
    raw = _load_yaml(path)

    if "endpoints" in raw:
        items: list[dict[str, Any]] = raw["endpoints"]
    else:
        items = [raw]

    endpoints: list[PromptEndpoint] = []
    for item in items:
        try:
            ep = PromptEndpoint.model_validate(item)
            endpoints.append(ep)
            logger.info("Loaded endpoint %s %s from %s", ep.method.value, ep.path, path.name)
        except Exception:
            logger.exception("Failed to parse endpoint from %s: %s", path.name, item.get("path", "?"))
    return endpoints


def load_all_prompts(directory: str | Path) -> list[PromptEndpoint]:
    """
    Recursively scan *directory* for ``.yaml`` / ``.yml`` files and return
    all parsed ``PromptEndpoint`` objects.
    """
    root = Path(directory)
    if not root.is_dir():
        logger.warning("Prompts directory does not exist: %s", root)
        return []

    endpoints: list[PromptEndpoint] = []
    for path in sorted(root.rglob("*.y*ml")):
        if path.suffix in (".yaml", ".yml"):
            endpoints.extend(load_prompt_file(path))

    logger.info("Total endpoints loaded: %d", len(endpoints))
    return endpoints


def watch_prompts(directory: str | Path, callback):
    """
    Placeholder for hot-reload support.

    In production you'd use ``watchfiles`` or similar to detect changes
    and call ``callback(new_endpoints)`` so routes can be refreshed
    without restarting the server.
    """
    # TODO: implement with watchfiles
    pass
