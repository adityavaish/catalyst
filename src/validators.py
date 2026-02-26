"""
Validators & Preconditions — generic pre-LLM request guards.

Evaluated BEFORE the LLM is invoked, these allow endpoint authors
to declare business rules directly in YAML that short-circuit invalid
requests without burning an LLM call.

Two guard types:
  1. **Validators** — pure input checks on path/query/body values.
     No database access required.
  2. **Preconditions** — parameterised SQL queries whose results are
     checked against assertions.  Useful for state-dependent rules
     like "account must be active" or "balance >= amount".

Both are fully generic: they operate on the standard request context
(body, params, path) and can be attached to *any* PromptEndpoint
via YAML.
"""

from __future__ import annotations

import ast
import logging
import re
from typing import Any

from src.connectors import BaseConnector, ConnectorRegistry, SQLConnector
from src.models import CatalystRequest, CatalystResponse, PromptEndpoint

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Safe expression evaluator (AST-based, no exec/eval)
# ---------------------------------------------------------------------------

# Only these built-in functions are allowed in check expressions.
_SAFE_BUILTINS = {
    "len": len,
    "abs": abs,
    "min": min,
    "max": max,
    "int": int,
    "float": float,
    "str": str,
    "bool": bool,
    "round": round,
}


class _SafeEvaluator:
    """
    Evaluate a small subset of Python expressions safely.

    Supported constructs:
      - Constants (int, float, str, bool, None)
      - Names bound in context dict
      - Attribute access  (``body.amount``)
      - Subscript access  (``rows[0]['status']``)
      - Comparisons       (``==``, ``!=``, ``<``, ``<=``, ``>``, ``>=``,
                           ``in``, ``not in``, ``is``, ``is not``)
      - Boolean operators (``and``, ``or``, ``not``)
      - Arithmetic        (``+``, ``-``, ``*``, ``/``, ``//``, ``%``)
      - Unary minus       (``-x``)
      - Function calls    (whitelist: len, abs, min, max, int, float,
                           str, bool, round)
      - Ternary           (``x if cond else y``)
    """

    def __init__(self, context: dict[str, Any]):
        self._ctx = context

    def evaluate(self, expr: str) -> Any:
        tree = ast.parse(expr, mode="eval")
        return self._eval(tree.body)

    # ── recursive evaluator ─────────────────────────────────────────────
    def _eval(self, node: ast.AST) -> Any:
        if isinstance(node, ast.Constant):
            return node.value

        if isinstance(node, ast.Name):
            if node.id == "None":
                return None
            if node.id == "True":
                return True
            if node.id == "False":
                return False
            if node.id in self._ctx:
                return self._ctx[node.id]
            raise NameError(f"Unknown variable: {node.id}")

        if isinstance(node, ast.Attribute):
            obj = self._eval(node.value)
            if isinstance(obj, dict):
                return obj.get(node.attr)
            return getattr(obj, node.attr, None)

        if isinstance(node, ast.Subscript):
            obj = self._eval(node.value)
            key = self._eval(node.slice)
            return obj[key]

        if isinstance(node, ast.Compare):
            left = self._eval(node.left)
            for op, comp in zip(node.ops, node.comparators):
                right = self._eval(comp)
                if not self._cmp(op, left, right):
                    return False
                left = right
            return True

        if isinstance(node, ast.BoolOp):
            if isinstance(node.op, ast.And):
                return all(self._eval(v) for v in node.values)
            if isinstance(node.op, ast.Or):
                return any(self._eval(v) for v in node.values)

        if isinstance(node, ast.UnaryOp):
            operand = self._eval(node.operand)
            if isinstance(node.op, ast.Not):
                return not operand
            if isinstance(node.op, ast.USub):
                return -operand
            if isinstance(node.op, ast.UAdd):
                return +operand

        if isinstance(node, ast.BinOp):
            left = self._eval(node.left)
            right = self._eval(node.right)
            ops = {
                ast.Add: lambda a, b: a + b,
                ast.Sub: lambda a, b: a - b,
                ast.Mult: lambda a, b: a * b,
                ast.Div: lambda a, b: a / b,
                ast.FloorDiv: lambda a, b: a // b,
                ast.Mod: lambda a, b: a % b,
            }
            op_fn = ops.get(type(node.op))
            if op_fn:
                return op_fn(left, right)
            raise ValueError(f"Unsupported binary operator: {type(node.op).__name__}")

        if isinstance(node, ast.Call):
            fname = self._func_name(node.func)
            if fname not in _SAFE_BUILTINS:
                raise ValueError(f"Function not allowed: {fname}")
            args = [self._eval(a) for a in node.args]
            return _SAFE_BUILTINS[fname](*args)

        if isinstance(node, ast.IfExp):
            return self._eval(node.body) if self._eval(node.test) else self._eval(node.orelse)

        raise ValueError(f"Unsupported expression node: {type(node).__name__}")

    # ── comparison operators ────────────────────────────────────────────
    @staticmethod
    def _cmp(op: ast.cmpop, left: Any, right: Any) -> bool:
        dispatch = {
            ast.Eq: lambda: left == right,
            ast.NotEq: lambda: left != right,
            ast.Lt: lambda: left < right,
            ast.LtE: lambda: left <= right,
            ast.Gt: lambda: left > right,
            ast.GtE: lambda: left >= right,
            ast.In: lambda: left in right,
            ast.NotIn: lambda: left not in right,
            ast.Is: lambda: left is right,
            ast.IsNot: lambda: left is not right,
        }
        fn = dispatch.get(type(op))
        if fn is None:
            raise ValueError(f"Unsupported comparison: {type(op).__name__}")
        return fn()

    @staticmethod
    def _func_name(node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        raise ValueError(f"Only simple function names are allowed, got: {type(node).__name__}")


def safe_eval(expr: str, context: dict[str, Any]) -> Any:
    """Public entry point: evaluate *expr* against *context* safely."""
    return _SafeEvaluator(context).evaluate(expr)


# ---------------------------------------------------------------------------
# Request context builder
# ---------------------------------------------------------------------------

def _build_context(request: CatalystRequest) -> dict[str, Any]:
    """
    Build the evaluation context from a request.

    Available in expressions:
      - ``body.<field>``   — request body fields
      - ``params.<field>`` — query parameters
      - ``path.<field>``   — path parameters
    """
    return {
        "body": request.body if isinstance(request.body, dict) else {},
        "params": dict(request.query_params),
        "path": dict(request.path_params),
    }


# ---------------------------------------------------------------------------
# Resolve a dotted path like "body.amount" from the context
# ---------------------------------------------------------------------------

def _resolve_path(dotted: str, context: dict[str, Any]) -> Any:
    """Resolve ``'body.amount'`` → context['body']['amount']."""
    parts = dotted.split(".")
    obj: Any = context
    for part in parts:
        if isinstance(obj, dict):
            obj = obj.get(part)
        else:
            obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


# ---------------------------------------------------------------------------
# 1. Validators — pure input checks (no DB)
# ---------------------------------------------------------------------------

async def evaluate_validators(
    endpoint: PromptEndpoint,
    request: CatalystRequest,
) -> CatalystResponse | None:
    """
    Run all input validators declared on the endpoint.

    Returns a ``CatalystResponse`` with the first failing check's error
    and status, or ``None`` if all checks pass.
    """
    if not endpoint.validators:
        return None

    ctx = _build_context(request)

    for v in endpoint.validators:
        try:
            result = safe_eval(v.check, ctx)
        except Exception as exc:
            # Expression error → treat as validation failure
            logger.debug("Validator expression error (%s): %s", v.check, exc)
            result = False

        if not result:
            logger.info(
                "Validator FAILED for %s %s: %s → %d %s",
                request.method.value, request.path, v.check, v.status, v.error,
            )
            return CatalystResponse(
                status_code=v.status,
                data=None,
                error=v.error,
                meta={"validator_rejected": True, "check": v.check},
            )

    return None


# ---------------------------------------------------------------------------
# 2. Preconditions — SQL-backed assertions
# ---------------------------------------------------------------------------

def _build_parameterised_query(
    query_template: str,
    param_map: dict[str, str],
    context: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """
    Build a parameterised SQL query from a template + param map.

    ``param_map`` maps SQL bind-parameter names to dotted context paths:
        { "_p0": "path.id", "_p1": "body.amount" }

    Returns ``(sql, values)`` ready for the ``databases`` library.
    """
    values: dict[str, Any] = {}
    for param_name, ctx_path in param_map.items():
        values[param_name] = _resolve_path(ctx_path, context)
    return query_template, values


async def evaluate_preconditions(
    endpoint: PromptEndpoint,
    request: CatalystRequest,
    connector_registry: ConnectorRegistry,
) -> CatalystResponse | None:
    """
    Run all SQL precondition queries declared on the endpoint.

    For each precondition:
      1. Resolve the connector
      2. Substitute parameters and execute the query
      3. Evaluate each check expression against ``rows`` + request context

    Returns a ``CatalystResponse`` on the first failing check,
    or ``None`` if everything passes.
    """
    if not endpoint.preconditions:
        return None

    ctx = _build_context(request)

    for pre in endpoint.preconditions:
        # Resolve connector
        conn = connector_registry.get(pre.connector)
        if conn is None:
            logger.error("Precondition connector '%s' not found", pre.connector)
            continue
        if not isinstance(conn, SQLConnector):
            logger.warning("Precondition connector '%s' is not a SQL connector — skipping", pre.connector)
            continue

        # Build parameterised query
        sql, values = _build_parameterised_query(pre.query, pre.params, ctx)

        # Execute query
        try:
            rows = await conn.execute("query", sql=sql, values=values)
        except Exception as exc:
            logger.error("Precondition query failed (%s): %s", sql, exc)
            # Query error → return a generic 500 or skip
            return CatalystResponse(
                status_code=500,
                data=None,
                error=f"Precondition query failed: {exc}",
                meta={"precondition_rejected": True},
            )

        # Evaluate checks with rows in context
        check_ctx = {**ctx, "rows": rows}

        for chk in pre.checks:
            try:
                result = safe_eval(chk.check, check_ctx)
            except Exception as exc:
                logger.debug("Precondition check error (%s): %s", chk.check, exc)
                result = False

            if not result:
                logger.info(
                    "Precondition FAILED for %s %s: %s → %d %s",
                    request.method.value, request.path, chk.check, chk.status, chk.error,
                )
                return CatalystResponse(
                    status_code=chk.status,
                    data=None,
                    error=chk.error,
                    meta={"precondition_rejected": True, "check": chk.check},
                )

    return None
