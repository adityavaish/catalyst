"""
Execution Plan Cache — skips LLM for endpoints with deterministic tool-call patterns.

When an endpoint is first called, the LLM generates the response as usual.
Afterwards, the engine analyses the conversation trace to extract a reusable
"execution plan" — a recipe that maps inputs → tool calls → response.

On subsequent requests to the same endpoint, the plan is executed directly
without calling the LLM, providing near-instant responses.

Plans auto-refresh via:
  - TTL-based expiry
  - Error-count threshold (too many failures → regenerate)
  - Background refresh (proactive re-generation before expiry)
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Plan models
# ---------------------------------------------------------------------------

class ValueSource(str, Enum):
    INPUT = "input"              # Direct from request input (body, path_params, query_params)
    TOOL_RESULT = "tool_result"  # From a previous tool-call result
    LITERAL = "literal"          # Static / hardcoded value
    TEMPLATE = "template"        # String template with embedded input values


class ArgumentMapping(BaseModel):
    """Maps a single tool argument to its data source."""
    arg_name: str
    source: ValueSource = ValueSource.INPUT
    source_path: str = ""           # dot-notation, e.g. "body.expression"
    literal_value: Any = None       # only used when source == LITERAL
    template_string: str = ""       # only when source == TEMPLATE, e.g. "({%body.value%} * 9/5) + 32"
    template_inputs: dict[str, str] = Field(
        default_factory=dict,
        description="placeholder → input path, e.g. {'{%body.value%}': 'body.value'}",
    )


class PlanStep(BaseModel):
    """One tool invocation inside a plan."""
    tool_name: str
    arguments: list[ArgumentMapping]


class ExecutionPlan(BaseModel):
    """A reusable, LLM-free recipe for processing requests."""
    endpoint_key: str                          # base key, e.g. "POST:/api/calculate"
    variant_key: str = ""                      # with discriminators, e.g. "POST:/api/convert-units|body.from_unit=celsius|body.to_unit=fahrenheit"
    discriminator_fields: list[str] = Field(default_factory=list)  # input paths that vary plan shape
    is_static: bool = False                    # True when extracted from an empty-input request
    steps: list[PlanStep] = Field(default_factory=list)
    response_template: Any = Field(default_factory=dict)
    created_at: float = Field(default_factory=time.time)
    ttl: float = 3600                          # seconds (0 = infinite)
    hit_count: int = 0
    error_count: int = 0
    max_errors: int = 3

    @property
    def is_expired(self) -> bool:
        if self.ttl <= 0:
            return False
        return (time.time() - self.created_at) > self.ttl

    @property
    def needs_refresh(self) -> bool:
        return self.error_count >= self.max_errors or self.is_expired

    @property
    def age_seconds(self) -> float:
        return time.time() - self.created_at


# ---------------------------------------------------------------------------
# Plan Cache
# ---------------------------------------------------------------------------

class PlanCache:
    """
    In-memory LRU cache for execution plans.

    Plans are stored by *variant key* which may include discriminator values.
    A discriminator index maps base endpoint keys to their discriminator fields,
    so the cache can build the correct variant key on lookup.
    """

    def __init__(self, max_plans: int = 500):
        self._store: dict[str, ExecutionPlan] = {}
        self._discriminator_index: dict[str, list[str]] = {}  # endpoint_key → disc fields
        self._max_plans = max_plans
        self._lock = asyncio.Lock()
        self._stats = {
            "hits": 0, "misses": 0, "executions": 0,
            "errors": 0, "refreshes": 0, "variants": 0,
        }

    def _variant_key(self, endpoint_key: str, input_data: dict | None) -> str:
        """Build the variant key using known discriminator fields."""
        disc_fields = self._discriminator_index.get(endpoint_key)
        if not disc_fields or not input_data:
            return endpoint_key
        return build_variant_key(endpoint_key, input_data, disc_fields)

    async def get(self, endpoint_key: str, input_data: dict | None = None) -> ExecutionPlan | None:
        async with self._lock:
            vk = self._variant_key(endpoint_key, input_data)
            plan = self._store.get(vk)
            if plan is None:
                self._stats["misses"] += 1
                return None
            if plan.needs_refresh:
                del self._store[vk]
                self._stats["misses"] += 1
                self._stats["refreshes"] += 1
                logger.info("Plan expired/errored for %s — will regenerate", vk)
                return None
            # Static plans (extracted from empty-input requests) must NOT match
            # requests that carry query/body parameters — those need their own
            # LLM call to produce a filter-aware plan.
            if plan.is_static and input_data:
                leaves = _collect_leaf_values(input_data)
                if leaves:
                    self._stats["misses"] += 1
                    return None
            # Coverage check: reject plan if request has input parameters the
            # plan doesn't know about (not in its argument mappings, templates,
            # or discriminator fields).
            if input_data and not plan.is_static:
                if not self._plan_covers_input(plan, input_data):
                    self._stats["misses"] += 1
                    logger.debug(
                        "Plan %s rejected: request has uncovered input params", vk,
                    )
                    return None
            plan.hit_count += 1
            self._stats["hits"] += 1
            return plan

    @staticmethod
    def _plan_covers_input(plan: ExecutionPlan, input_data: dict) -> bool:
        """Return True if the plan accounts for every leaf in *input_data*."""
        request_leaves = set(_collect_leaf_values(input_data).keys())
        if not request_leaves:
            return True
        # Gather all input paths the plan references
        known_paths: set[str] = set(plan.discriminator_fields or [])
        for step in plan.steps:
            for arg in step.arguments:
                if arg.source == ValueSource.INPUT and arg.source_path:
                    known_paths.add(arg.source_path)
                if arg.source == ValueSource.TEMPLATE and arg.template_inputs:
                    known_paths.update(arg.template_inputs.values())
        return request_leaves <= known_paths

    async def put(self, plan: ExecutionPlan) -> None:
        async with self._lock:
            store_key = plan.variant_key or plan.endpoint_key
            # Evict oldest if at capacity
            if len(self._store) >= self._max_plans and store_key not in self._store:
                oldest_key = next(iter(self._store))
                del self._store[oldest_key]
            self._store[store_key] = plan
            # Index discriminators so future lookups can build variant keys
            if plan.discriminator_fields:
                self._discriminator_index[plan.endpoint_key] = plan.discriminator_fields
                self._stats["variants"] += 1
            logger.info(
                "Cached execution plan for %s (%d steps, ttl=%ds, discriminators=%s)",
                store_key, len(plan.steps), plan.ttl,
                plan.discriminator_fields or "none",
            )

    async def record_error(self, endpoint_key: str, input_data: dict | None = None) -> None:
        async with self._lock:
            vk = self._variant_key(endpoint_key, input_data)
            plan = self._store.get(vk)
            if plan:
                plan.error_count += 1
                self._stats["errors"] += 1
                logger.warning(
                    "Plan error for %s (count=%d/%d)",
                    vk, plan.error_count, plan.max_errors,
                )

    async def record_execution(self) -> None:
        async with self._lock:
            self._stats["executions"] += 1

    async def invalidate(self, endpoint_key: str) -> None:
        async with self._lock:
            self._store.pop(endpoint_key, None)

    async def clear(self) -> None:
        async with self._lock:
            self._store.clear()
            self._discriminator_index.clear()

    @property
    def stats(self) -> dict[str, int]:
        return self._stats.copy()


# ---------------------------------------------------------------------------
# Helpers — value lookup
# ---------------------------------------------------------------------------

def _resolve_path(data: Any, path: str) -> Any:
    """Resolve a dot-notation path (e.g. 'body.expression') in nested data."""
    if not path:
        return data
    parts = path.split(".")
    current = data
    for part in parts:
        if isinstance(current, dict):
            current = current.get(part)
        elif isinstance(current, (list, tuple)):
            try:
                current = current[int(part)]
            except (ValueError, IndexError):
                return None
        else:
            return None
    return current


def _values_match(a: Any, b: Any) -> bool:
    """Fuzzy comparison that handles str/num coercion."""
    if a is None or b is None:
        return False
    if isinstance(a, str) and isinstance(b, str):
        return a.strip() == b.strip()
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) < 1e-10
    return str(a).strip() == str(b).strip()


def _find_all_in_dict(value: Any, obj: Any, prefix: str = "") -> list[str]:
    """Recursively collect *all* dot-paths where *value* appears in a nested structure."""
    results: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            path = f"{prefix}.{k}" if prefix else k
            results.extend(_find_all_in_dict(value, v, path))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            path = f"{prefix}.{i}"
            results.extend(_find_all_in_dict(value, v, path))
    else:
        if _values_match(obj, value):
            results.append(prefix)
    return results


def _find_value_in_dict(value: Any, obj: Any, prefix: str = "") -> str | None:
    """Recursively search for *value* in a nested dict/list. Returns dot-path."""
    paths = _find_all_in_dict(value, obj, prefix)
    return paths[0] if paths else None


def _find_in_input(value: Any, input_data: dict, hint_key: str | None = None) -> str | None:
    """Search input_data (body, path_params, query_params) for a value.

    When *hint_key* is provided, prefer a path that ends with the same key.
    """
    if hint_key:
        # Try to find a path ending with the hint key first
        first_match: str | None = None
        for leaf_path, leaf_val in _collect_leaf_values(input_data).items():
            if _values_match(leaf_val, value):
                if leaf_path.endswith(f".{hint_key}") or leaf_path == hint_key:
                    return leaf_path
                if first_match is None:
                    first_match = leaf_path
        return first_match
    return _find_value_in_dict(value, input_data)


def _find_in_tool_results(value: Any, tool_results: list[dict], hint_key: str | None = None) -> str | None:
    """Search tool results for a value. Returns 'step_idx.field_path'.

    When *hint_key* is provided (the response dict key being mapped), prefer a
    tool-result path that ends with the same key name.  This prevents ambiguous
    mappings when multiple fields share the same value (e.g. id=1, in_stock=1).
    """
    all_paths: list[str] = []
    for idx, result in enumerate(tool_results):
        all_paths.extend(_find_all_in_dict(value, result, str(idx)))

    if not all_paths:
        return None
    if hint_key:
        for p in all_paths:
            if p.endswith(f".{hint_key}"):
                return p
    return all_paths[0]


# ---------------------------------------------------------------------------
# Template detection — finds input values *embedded* within strings
# ---------------------------------------------------------------------------

def _collect_leaf_values(data: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten a nested dict/list into {dot_path: leaf_value} pairs."""
    leaves: dict[str, Any] = {}
    if isinstance(data, dict):
        for k, v in data.items():
            path = f"{prefix}.{k}" if prefix else k
            leaves.update(_collect_leaf_values(v, path))
    elif isinstance(data, (list, tuple)):
        for i, v in enumerate(data):
            path = f"{prefix}.{i}"
            leaves.update(_collect_leaf_values(v, path))
    else:
        if data is not None:
            leaves[prefix] = data
    return leaves


def _try_build_template(
    arg_value: str,
    input_leaves: dict[str, Any],
) -> tuple[str, dict[str, str]] | None:
    """
    Check whether any input leaf values appear *embedded* inside ``arg_value``.

    Returns ``(template_string, {placeholder: input_path})`` or ``None``.

    Example::

        arg_value    = "(100 * 9/5) + 32"
        input_leaves = {"body.value": 100, "body.from_unit": "celsius"}
        →  ("({%body.value%} * 9/5) + 32", {"{%body.value%}": "body.value"})
    """
    template = str(arg_value)
    embedded: dict[str, str] = {}

    # Build candidates: (input_path, string_repr)
    # Sort longest first to avoid partial replacements (e.g. "100" before "10")
    candidates: list[tuple[str, str]] = []
    for path, value in input_leaves.items():
        str_val = str(value)
        if len(str_val) < 1:
            continue
        if str_val in template:
            candidates.append((path, str_val))

    candidates.sort(key=lambda x: len(x[1]), reverse=True)

    for path, str_val in candidates:
        # Word-boundary-aware matching to avoid partial matches
        # e.g. don't match "3" inside "32" or "10" inside "100"
        escaped = re.escape(str_val)
        pattern = r'(?<![a-zA-Z0-9_.])' + escaped + r'(?![a-zA-Z0-9_.])'
        if re.search(pattern, template):
            placeholder = "{%" + path + "%}"
            # Only replace the FIRST occurrence — subsequent matches are likely
            # constants that coincide with the input value (e.g. "5280 / 5280"
            # where the first 5280 is the input and the second is a factor).
            template = re.sub(pattern, placeholder, template, count=1)
            embedded[placeholder] = path

    if embedded:
        return template, embedded
    return None


def build_variant_key(
    endpoint_key: str,
    input_data: dict,
    discriminator_fields: list[str],
) -> str:
    """Build a variant key by appending discriminator field values."""
    parts = [endpoint_key]
    for field in sorted(discriminator_fields):
        val = _resolve_path(input_data, field)
        parts.append(f"{field}={val}")
    return "|".join(parts)


# ---------------------------------------------------------------------------
# Plan extraction from LLM conversation trace
# ---------------------------------------------------------------------------

def extract_plan_from_trace(
    endpoint_key: str,
    input_data: dict,
    messages: list[dict],
    tool_results_raw: list[dict],
    final_response: dict,
    ttl: float = 3600,
    max_errors: int = 3,
) -> ExecutionPlan | None:
    """
    Analyse an LLM conversation trace and extract a reusable execution plan.

    The extractor performs three kinds of argument mapping:

    1. **Direct input** — tool arg value matches an input field exactly.
    2. **Template** — tool arg *contains* an input value embedded in a larger
       string (e.g. ``"(100 * 9/5) + 32"`` with ``body.value = 100``).
    3. **Literal** — cannot be traced to any input → stored as a constant.

    Input fields that are neither directly mapped nor embedded in templates
    become **discriminators** — they vary the plan structure, so each
    combination of discriminator values maps to a separate cached plan variant.

    Returns None if the trace isn't suitable for plan caching.
    """
    input_leaves = _collect_leaf_values(input_data)
    used_input_paths: set[str] = set()

    # ── Collect tool-call steps from the assistant messages ──────────────
    steps: list[PlanStep] = []

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") != "assistant":
            continue
        tool_calls = msg.get("tool_calls") or []
        for tc in tool_calls:
            fn = tc if isinstance(tc, dict) else tc
            fn_info = fn.get("function", fn)
            fn_name = fn_info.get("name", "")
            try:
                fn_args = json.loads(fn_info.get("arguments", "{}"))
            except (json.JSONDecodeError, TypeError):
                fn_args = {}

            arg_mappings: list[ArgumentMapping] = []
            for arg_name, arg_value in fn_args.items():

                # 1. Direct input match
                input_path = _find_in_input(arg_value, input_data)
                if input_path is not None:
                    used_input_paths.add(input_path)
                    arg_mappings.append(ArgumentMapping(
                        arg_name=arg_name,
                        source=ValueSource.INPUT,
                        source_path=input_path,
                    ))
                    continue

                # 2. Template match — input values embedded in the string
                if isinstance(arg_value, str):
                    tmpl_result = _try_build_template(arg_value, input_leaves)
                    if tmpl_result is not None:
                        tmpl_str, tmpl_inputs = tmpl_result
                        for path in tmpl_inputs.values():
                            used_input_paths.add(path)
                        arg_mappings.append(ArgumentMapping(
                            arg_name=arg_name,
                            source=ValueSource.TEMPLATE,
                            template_string=tmpl_str,
                            template_inputs=tmpl_inputs,
                        ))
                        continue

                # 3. Literal fallback
                arg_mappings.append(ArgumentMapping(
                    arg_name=arg_name,
                    source=ValueSource.LITERAL,
                    literal_value=arg_value,
                ))

            steps.append(PlanStep(tool_name=fn_name, arguments=arg_mappings))

    if not steps:
        logger.debug("No tool calls in trace for %s — skipping plan extraction", endpoint_key)
        return None

    # Check whether the plan has any dynamic (input/template) argument mappings.
    has_dynamic_mapping = any(
        any(a.source in (ValueSource.INPUT, ValueSource.TEMPLATE) for a in step.arguments)
        for step in steps
    )
    # Allow all-literal "static" plans when the request had NO input at all.
    # These replay the exact same tool calls every time — valid because the
    # same empty input always produces the same result.
    has_input = bool(input_leaves)
    if not has_dynamic_mapping and has_input:
        logger.debug("No dynamic arguments for %s — skipping plan", endpoint_key)
        return None

    # ── Identify discriminators ─────────────────────────────────────────
    # Input fields NOT used as direct or template args may affect plan
    # structure (e.g. from_unit/to_unit determine the formula).
    all_input_paths = set(input_leaves.keys())
    discriminator_fields = sorted(all_input_paths - used_input_paths)

    variant_key = build_variant_key(endpoint_key, input_data, discriminator_fields)

    # ── Build response template ─────────────────────────────────────────
    response_template = _build_response_template(final_response, input_data, tool_results_raw)

    plan = ExecutionPlan(
        endpoint_key=endpoint_key,
        variant_key=variant_key,
        discriminator_fields=discriminator_fields,
        is_static=not has_input and not has_dynamic_mapping,
        steps=steps,
        response_template=response_template,
        ttl=ttl,
        max_errors=max_errors,
    )
    logger.info(
        "Extracted execution plan for %s: %d steps, %d arg-mappings, discriminators=%s",
        variant_key, len(steps),
        sum(len(s.arguments) for s in steps),
        discriminator_fields or "none",
    )
    return plan


def _try_detect_dynamic_list(processed_items: list) -> dict | None:
    """Detect when list items all map to consecutive tool_result row indices.

    This is a fallback detector that works on already-processed items.  The
    primary detector ``_try_detect_dynamic_list_raw`` works on raw values
    before processing and handles the common case where duplicate field
    values across rows cause all items to map to the same row index.

    Returns a compact ``tool_result_list`` template or ``None``.
    """
    if len(processed_items) < 2:
        return None
    for item in processed_items:
        if not isinstance(item, dict) or "__source__" in item:
            return None

    first = processed_items[0]
    if not first:
        return None
    step_idx: int | None = None
    first_row: int | None = None
    row_template: dict[str, str] = {}

    for key, ref in first.items():
        if not isinstance(ref, dict) or ref.get("__source__") != "tool_result":
            return None
        path: str = ref.get("__path__", "")
        parts = path.split(".", 2)
        if len(parts) < 3:
            return None
        try:
            s, r = int(parts[0]), int(parts[1])
        except (ValueError, TypeError):
            return None
        field = parts[2]
        if step_idx is None:
            step_idx, first_row = s, r
        elif s != step_idx or r != first_row:
            return None
        row_template[key] = field

    if step_idx is None or first_row is None:
        return None

    second = processed_items[1]
    expected_row = first_row + 1
    for key, ref in second.items():
        if not isinstance(ref, dict) or ref.get("__source__") != "tool_result":
            return None
        path = ref.get("__path__", "")
        parts = path.split(".", 2)
        if len(parts) < 3:
            return None
        try:
            s, r = int(parts[0]), int(parts[1])
        except (ValueError, TypeError):
            return None
        if s != step_idx or r != expected_row:
            return None

    logger.debug(
        "Detected dynamic list pattern (processed): step=%d, %d fields, %d items",
        step_idx, len(row_template), len(processed_items),
    )
    return {
        "__source__": "tool_result_list",
        "__step__": step_idx,
        "__row_template__": row_template,
    }


def _try_detect_dynamic_list_raw(
    raw_items: list,
    tool_results: list,
) -> dict | None:
    """Detect list-of-tool-result-rows directly from raw response values.

    When multiple rows share the same value for a field (e.g. all accounts
    have ``status='active'``), the per-item ``_find_in_tool_results`` maps
    every item to row 0's path.  This function sidesteps that problem by
    matching raw response items against raw tool-result rows using value
    equality, preferring exact key matches.

    Returns a ``tool_result_list`` template or ``None``.
    """
    if len(raw_items) < 2:
        return None
    if not all(isinstance(item, dict) for item in raw_items):
        return None

    for step_idx, step_result in enumerate(tool_results):
        if not isinstance(step_result, list) or len(step_result) < 2:
            continue
        if not isinstance(step_result[0], dict):
            continue

        # Build field_map from first response item → first tool-result row
        row = step_result[0]
        field_map: dict[str, str] = {}
        matched = True
        for resp_key, resp_value in raw_items[0].items():
            # Prefer exact key match, then any value match
            if resp_key in row and _values_match(resp_value, row[resp_key]):
                field_map[resp_key] = resp_key
            else:
                found = False
                for row_key, row_value in row.items():
                    if _values_match(resp_value, row_value):
                        field_map[resp_key] = row_key
                        found = True
                        break
                if not found:
                    matched = False
                    break

        if not matched or not field_map:
            continue

        # Verify second item matches second tool-result row
        if len(step_result) < 2:
            continue
        second_row = step_result[1]
        if not isinstance(second_row, dict):
            continue
        second_ok = True
        for resp_key, resp_value in raw_items[1].items():
            row_key = field_map.get(resp_key)
            if row_key is None or not _values_match(resp_value, second_row.get(row_key)):
                second_ok = False
                break

        if not second_ok:
            continue

        logger.debug(
            "Detected dynamic list pattern (raw): step=%d, %d fields, %d items",
            step_idx, len(field_map), len(raw_items),
        )
        return {
            "__source__": "tool_result_list",
            "__step__": step_idx,
            "__row_template__": field_map,
        }

    return None


def _build_response_template(
    response: Any,
    input_data: dict,
    tool_results: list[dict],
) -> Any:
    """
    Walk the final response and replace each leaf value with a source reference.

    Source references are dicts with ``__source__`` key:
      - ``{"__source__": "tool_result", "__path__": "0.result"}``
      - ``{"__source__": "input",       "__path__": "body.expression"}``
      - ``{"__source__": "template",    "__template__": "...", "__inputs__": {...}}``
      - ``{"__source__": "literal",     "__value__": 200}``
      - ``{"__source__": "omit"}``  (all-literal lists → LLM-generated content)
    """
    # Build combined leaf lookup for template detection in response values
    combined_leaves: dict[str, Any] = {}
    for path, val in _collect_leaf_values(input_data).items():
        combined_leaves[f"input.{path}"] = val
    for idx, tr in enumerate(tool_results):
        for path, val in _collect_leaf_values(tr).items():
            combined_leaves[f"tool_result.{idx}.{path}"] = val

    def _process(obj: Any, is_list_item: bool = False, hint_key: str | None = None) -> Any:
        if isinstance(obj, dict):
            return {k: _process(v, hint_key=k) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            # Primary detector: check RAW response values against raw
            # tool-result rows.  This correctly handles duplicate values
            # across rows (e.g. all rows have status='active') that would
            # confuse the per-item _find_in_tool_results approach.
            dyn_raw = _try_detect_dynamic_list_raw(list(obj), tool_results)
            if dyn_raw is not None:
                return dyn_raw

            processed = [_process(v, is_list_item=True) for v in obj]
            # If ALL items in the list are unmappable literals, the list was
            # LLM-generated content (e.g. "steps").  Mark it so the renderer
            # can omit it rather than return stale data.
            if processed and all(
                isinstance(p, dict)
                and p.get("__source__") in ("literal", "omit")
                for p in processed
            ):
                return {"__source__": "omit"}
            # Fallback detector: works on already-processed items when all
            # fields have unique values per row (no duplicates across rows).
            dyn = _try_detect_dynamic_list(processed)
            if dyn is not None:
                return dyn
            return processed

        # Leaf — try exact matches first (tool_result, then input)
        # Pass hint_key to prefer paths ending with the same key name,
        # avoiding ambiguous mappings (e.g. id=1 vs in_stock=1).
        tp = _find_in_tool_results(obj, tool_results, hint_key=hint_key)
        if tp is not None:
            return {"__source__": "tool_result", "__path__": tp}

        ip = _find_in_input(obj, input_data, hint_key=hint_key)
        if ip is not None:
            return {"__source__": "input", "__path__": ip}

        # Try template match — value may embed multiple source values
        str_obj = str(obj) if not isinstance(obj, str) else obj
        if len(str_obj) > 1:
            tmpl_result = _try_build_template(str_obj, combined_leaves)
            if tmpl_result is not None:
                tmpl_str, tmpl_inputs = tmpl_result
                return {
                    "__source__": "template",
                    "__template__": tmpl_str,
                    "__inputs__": tmpl_inputs,
                }

        return {"__source__": "literal", "__value__": obj}

    return _process(response)


# ---------------------------------------------------------------------------
# Plan execution (LLM-free)
# ---------------------------------------------------------------------------

async def execute_plan(
    plan: ExecutionPlan,
    input_data: dict,
    connectors: list,                          # list[BaseConnector]
) -> dict:
    """
    Execute a cached plan without calling the LLM.

    1. For each step, resolve arguments from input and call the tool.
    2. Collect tool results.
    3. Render the response template with input + tool results.

    Raises on failure so the caller can fall back to LLM.
    """
    # Import here to avoid circular dependency
    from src.engine import _dispatch_tool_call

    tool_results: list[dict] = []

    for step_idx, step in enumerate(plan.steps):
        # ── Resolve arguments ───────────────────────────────────────────
        resolved_args: dict[str, Any] = {}
        for mapping in step.arguments:
            if mapping.source == ValueSource.INPUT:
                value = _resolve_path(input_data, mapping.source_path)
                if value is None:
                    raise ValueError(
                        f"Plan step {step_idx}: input field '{mapping.source_path}' "
                        f"not found for arg '{mapping.arg_name}'"
                    )
                resolved_args[mapping.arg_name] = value

            elif mapping.source == ValueSource.TEMPLATE:
                rendered = mapping.template_string
                for placeholder, path in mapping.template_inputs.items():
                    value = _resolve_path(input_data, path)
                    if value is None:
                        raise ValueError(
                            f"Plan step {step_idx}: template input '{path}' "
                            f"not found for arg '{mapping.arg_name}'"
                        )
                    rendered = rendered.replace(placeholder, str(value))
                resolved_args[mapping.arg_name] = rendered

            elif mapping.source == ValueSource.TOOL_RESULT:
                parts = mapping.source_path.split(".", 1)
                prev_idx = int(parts[0])
                field_path = parts[1] if len(parts) > 1 else ""
                if prev_idx >= len(tool_results):
                    raise ValueError(
                        f"Plan step {step_idx}: references tool result {prev_idx} "
                        f"but only {len(tool_results)} results available"
                    )
                prev = tool_results[prev_idx]
                value = _resolve_path(prev, field_path) if field_path else prev
                resolved_args[mapping.arg_name] = value

            elif mapping.source == ValueSource.LITERAL:
                resolved_args[mapping.arg_name] = mapping.literal_value

        # ── Execute tool ────────────────────────────────────────────────
        try:
            result = await _dispatch_tool_call(step.tool_name, resolved_args, connectors)
        except Exception as e:
            raise RuntimeError(
                f"Plan step {step_idx} ({step.tool_name}) failed: {e}"
            ) from e

        if isinstance(result, str):
            try:
                result = json.loads(result)
            except (json.JSONDecodeError, TypeError):
                result = {"raw": result}

        # Keep the result structure as-is (including lists) so it matches
        # the paths built during plan extraction from tool_results_raw.
        # Wrapping lists in {"raw": ...} would break template paths like
        # "0.0.field" that expect list[index].field traversal.
        if isinstance(result, (dict, list, tuple)):
            tool_results.append(result)
        else:
            tool_results.append({"raw": result})

    # ── Render response template ────────────────────────────────────────
    response = _render_template(plan.response_template, input_data, tool_results)
    return response


def _render_template(
    template: Any,
    input_data: dict,
    tool_results: list[dict],
) -> Any:
    """Recursively resolve source references in a response template."""
    if isinstance(template, dict):
        if "__source__" in template:
            src = template["__source__"]
            if src == "input":
                return _resolve_path(input_data, template["__path__"])
            elif src == "tool_result":
                path = template["__path__"]
                parts = path.split(".", 1)
                step_idx = int(parts[0])
                field_path = parts[1] if len(parts) > 1 else ""
                result = tool_results[step_idx] if step_idx < len(tool_results) else {}
                return _resolve_path(result, field_path) if field_path else result
            elif src == "template":
                rendered = template["__template__"]
                for placeholder, path in template.get("__inputs__", {}).items():
                    # Determine source from path prefix
                    if path.startswith("input."):
                        val = _resolve_path(input_data, path[len("input."):])
                    elif path.startswith("tool_result."):
                        rest = path[len("tool_result."):]
                        parts = rest.split(".", 1)
                        idx = int(parts[0])
                        field = parts[1] if len(parts) > 1 else ""
                        val = _resolve_path(
                            tool_results[idx] if idx < len(tool_results) else {},
                            field,
                        )
                    else:
                        val = _resolve_path(input_data, path)
                    rendered = rendered.replace(placeholder, str(val) if val is not None else "")
                # Try to parse as number if result looks numeric
                try:
                    if "." in rendered:
                        return float(rendered)
                    return int(rendered)
                except (ValueError, TypeError):
                    return rendered
            elif src == "literal":
                return template["__value__"]
            elif src == "omit":
                return None  # Signal to parent to skip this key
            elif src == "tool_result_list":
                # Dynamic list: iterate over actual tool-result rows using
                # a single row template, so the number of rendered items
                # matches the real query result (not the cache-time count).
                step = template["__step__"]
                row_tmpl = template["__row_template__"]
                raw = tool_results[step] if step < len(tool_results) else []
                if not isinstance(raw, list):
                    raw = [raw]
                rendered_rows = []
                for row in raw:
                    rendered_item = {}
                    for key, field_path in row_tmpl.items():
                        val = _resolve_path(row, field_path) if field_path else row
                        if val is not None:
                            rendered_item[key] = val
                    if rendered_item:
                        rendered_rows.append(rendered_item)
                return rendered_rows
            return None
        # Regular dict — recurse, but drop keys whose value resolved to _OMIT
        rendered = {}
        for k, v in template.items():
            val = _render_template(v, input_data, tool_results)
            if val is not None:
                rendered[k] = val
        return rendered

    if isinstance(template, (list, tuple)):
        return [_render_template(v, input_data, tool_results) for v in template]

    return template
