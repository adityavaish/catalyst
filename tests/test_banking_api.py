"""
Comprehensive end-to-end test suite for the Banking Transaction System API.

Tests 6 endpoints across 7 sections (45 tests):
  1. GET  /api/accounts             — list / filter / sort / paginate
  2. GET  /api/accounts/{id}        — lookup by ID
  3. POST /api/accounts/{id}/deposit — deposit money
  4. POST /api/transfers             — fund transfers (the core banking op)
  5. GET  /api/transactions          — transaction history
  6. POST /api/analytics/summary     — AI financial analytics
  7. Cross-cutting: balance integrity checks after mutating operations

Seed data (from scripts/seed_banking_db.py):
  8 accounts (6 active, 1 frozen, 1 closed), 15 transactions

Usage:
    # server must be running at http://127.0.0.1:8000
    python tests/test_banking_api.py
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

BASE_URL = "http://127.0.0.1:8000"
TIMEOUT = 60  # seconds per request (LLM can be slow on first call)
DB_PATH = None  # set in main()  — path to catalyst.db for integrity checks

# ── Known seed data constants ───────────────────────────────────────────
SEED_ACCOUNT_COUNT = 8
SEED_ACTIVE_ACCOUNTS = 6
SEED_CHECKING = 3          # ids 1, 3, 6
SEED_SAVINGS = 3           # ids 2, 5, 8
SEED_BUSINESS = 2          # ids 4, 7
SEED_FROZEN_ID = 6         # Frank Wilson, $150.25
SEED_CLOSED_ID = 8         # Heidi Clark, $0.00
SEED_TXN_COUNT = 15

# Initial balances (from seed)
SEED_BALANCES = {
    1: 5240.50,   # Alice Johnson  - checking - active
    2: 12800.00,  # Bob Smith      - savings  - active
    3: 890.75,    # Charlie Brown  - checking - active
    4: 45000.00,  # Diana Prince   - business - active
    5: 3200.00,   # Eve Martinez   - savings  - active
    6: 150.25,    # Frank Wilson   - checking - frozen
    7: 28750.00,  # Grace Lee      - business - active
    8: 0.00,      # Heidi Clark    - savings  - closed
}


# ── Test infrastructure ─────────────────────────────────────────────────

@dataclass
class TestResult:
    name: str
    passed: bool
    latency_ms: float = 0.0
    plan_hit: bool = False
    cached: bool = False
    detail: str = ""


class TestSuite:
    def __init__(self):
        self.results: list[TestResult] = []
        # Track balance mutations for integrity checks
        self._balance_deltas: dict[int, float] = {}
        self._created_txn_ids: list[int] = []

    def record(self, r: TestResult):
        self.results.append(r)
        source = "CACHE" if r.cached else ("PLAN" if r.plan_hit else "LLM")
        status = "PASS" if r.passed else "FAIL"
        print(f"  [{status}] {r.name:<62s} {r.latency_ms:>8.1f}ms  [{source}]  ({r.detail})")

    def track_balance_change(self, account_id: int, delta: float):
        self._balance_deltas[account_id] = self._balance_deltas.get(account_id, 0.0) + delta

    def expected_balance(self, account_id: int) -> float:
        return SEED_BALANCES[account_id] + self._balance_deltas.get(account_id, 0.0)

    def summary(self) -> bool:
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed = total - passed
        lats = [r.latency_ms for r in self.results if r.latency_ms > 0]
        llm = sum(1 for r in self.results if not r.plan_hit and not r.cached)
        plan = sum(1 for r in self.results if r.plan_hit)
        cache = sum(1 for r in self.results if r.cached)

        print(f"\n{'=' * 80}")
        print(f"  RESULTS: {passed}/{total} passed, {failed} failed")
        print(f"  SOURCE:  {llm} LLM | {plan} PLAN | {cache} CACHE")
        if lats:
            print(f"  LATENCY: min={min(lats):.1f}ms  avg={sum(lats)/len(lats):.1f}ms  max={max(lats):.1f}ms")
        print(f"{'=' * 80}")

        if failed:
            print(f"\n  FAILURES:")
            for r in self.results:
                if not r.passed:
                    print(f"    - {r.name}: {r.detail}")
        print()
        return failed == 0


def _meta(resp: dict) -> dict:
    return resp.get("meta", {}) or {}


def _get(path: str, params: dict | None = None) -> tuple[dict, float]:
    t0 = time.perf_counter()
    try:
        r = httpx.get(f"{BASE_URL}{path}", params=params, timeout=TIMEOUT)
        lat = (time.perf_counter() - t0) * 1000
        try:
            return r.json(), lat
        except Exception:
            return {"error": r.text, "status_code": r.status_code}, lat
    except Exception as e:
        return {"error": str(e)}, (time.perf_counter() - t0) * 1000


def _post(path: str, body: dict) -> tuple[dict, float]:
    t0 = time.perf_counter()
    try:
        r = httpx.post(f"{BASE_URL}{path}", json=body, timeout=TIMEOUT)
        lat = (time.perf_counter() - t0) * 1000
        try:
            return r.json(), lat
        except Exception:
            return {"error": r.text, "status_code": r.status_code}, lat
    except Exception as e:
        return {"error": str(e)}, (time.perf_counter() - t0) * 1000


# ══════════════════════════════════════════════════════════════════════════
# 1. GET /api/accounts — List / Filter / Sort
# ══════════════════════════════════════════════════════════════════════════

def test_list_all_accounts(suite: TestSuite):
    """List all accounts — should return 8."""
    resp, lat = _get("/api/accounts")
    meta = _meta(resp)
    data = resp.get("data", {}) or {}
    accounts = data.get("accounts", []) or []

    ok = len(accounts) == SEED_ACCOUNT_COUNT
    suite.record(TestResult(
        name="GET /api/accounts — list all",
        passed=ok,
        latency_ms=meta.get("latency_ms", lat),
        plan_hit=meta.get("plan_executed", False),
        cached=meta.get("cached", False),
        detail=f"{len(accounts)} accounts (expected {SEED_ACCOUNT_COUNT})",
    ))


def test_filter_checking(suite: TestSuite):
    """Filter by account_type=checking."""
    resp, lat = _get("/api/accounts", {"account_type": "checking"})
    meta = _meta(resp)
    data = resp.get("data", {}) or {}
    accounts = data.get("accounts", []) or []

    all_checking = all(a.get("account_type") == "checking" for a in accounts)
    ok = len(accounts) == SEED_CHECKING and all_checking
    suite.record(TestResult(
        name="GET /api/accounts?account_type=checking",
        passed=ok,
        latency_ms=meta.get("latency_ms", lat),
        plan_hit=meta.get("plan_executed", False),
        cached=meta.get("cached", False),
        detail=f"{len(accounts)} checking accounts",
    ))


def test_filter_savings(suite: TestSuite):
    """Filter by account_type=savings."""
    resp, lat = _get("/api/accounts", {"account_type": "savings"})
    meta = _meta(resp)
    data = resp.get("data", {}) or {}
    accounts = data.get("accounts", []) or []

    all_savings = all(a.get("account_type") == "savings" for a in accounts)
    ok = len(accounts) == SEED_SAVINGS and all_savings
    suite.record(TestResult(
        name="GET /api/accounts?account_type=savings",
        passed=ok,
        latency_ms=meta.get("latency_ms", lat),
        plan_hit=meta.get("plan_executed", False),
        cached=meta.get("cached", False),
        detail=f"{len(accounts)} savings accounts",
    ))


def test_filter_active(suite: TestSuite):
    """Filter by status=active — should exclude frozen and closed."""
    resp, lat = _get("/api/accounts", {"status": "active"})
    meta = _meta(resp)
    data = resp.get("data", {}) or {}
    accounts = data.get("accounts", []) or []

    all_active = all(a.get("status") == "active" for a in accounts)
    ok = len(accounts) == SEED_ACTIVE_ACCOUNTS and all_active
    suite.record(TestResult(
        name="GET /api/accounts?status=active",
        passed=ok,
        latency_ms=meta.get("latency_ms", lat),
        plan_hit=meta.get("plan_executed", False),
        cached=meta.get("cached", False),
        detail=f"{len(accounts)} active accounts",
    ))


def test_filter_frozen(suite: TestSuite):
    """Filter by status=frozen — should return only Frank Wilson."""
    resp, lat = _get("/api/accounts", {"status": "frozen"})
    meta = _meta(resp)
    data = resp.get("data", {}) or {}
    accounts = data.get("accounts", []) or []

    ok = len(accounts) == 1 and accounts[0].get("id") == SEED_FROZEN_ID
    suite.record(TestResult(
        name="GET /api/accounts?status=frozen",
        passed=ok,
        latency_ms=meta.get("latency_ms", lat),
        plan_hit=meta.get("plan_executed", False),
        cached=meta.get("cached", False),
        detail=f"{len(accounts)} frozen accounts, id={accounts[0].get('id') if accounts else '?'}",
    ))


def test_filter_min_balance(suite: TestSuite):
    """Filter by min_balance=10000 — Bob, Diana, Grace."""
    resp, lat = _get("/api/accounts", {"min_balance": 10000})
    meta = _meta(resp)
    data = resp.get("data", {}) or {}
    accounts = data.get("accounts", []) or []

    all_above = all(a.get("balance", 0) >= 10000 for a in accounts)
    ok = len(accounts) == 3 and all_above
    suite.record(TestResult(
        name="GET /api/accounts?min_balance=10000",
        passed=ok,
        latency_ms=meta.get("latency_ms", lat),
        plan_hit=meta.get("plan_executed", False),
        cached=meta.get("cached", False),
        detail=f"{len(accounts)} accounts >= $10k",
    ))


def test_filter_max_balance(suite: TestSuite):
    """Filter by max_balance=1000 — Charlie ($890.75), Frank ($150.25), Heidi ($0)."""
    resp, lat = _get("/api/accounts", {"max_balance": 1000})
    meta = _meta(resp)
    data = resp.get("data", {}) or {}
    accounts = data.get("accounts", []) or []

    all_below = all(a.get("balance", 99999) <= 1000 for a in accounts)
    ok = len(accounts) == 3 and all_below
    suite.record(TestResult(
        name="GET /api/accounts?max_balance=1000",
        passed=ok,
        latency_ms=meta.get("latency_ms", lat),
        plan_hit=meta.get("plan_executed", False),
        cached=meta.get("cached", False),
        detail=f"{len(accounts)} accounts <= $1k",
    ))


def test_sort_balance_desc(suite: TestSuite):
    """Sort by balance descending."""
    resp, lat = _get("/api/accounts", {"sort": "balance_desc"})
    meta = _meta(resp)
    data = resp.get("data", {}) or {}
    accounts = data.get("accounts", []) or []

    balances = [a.get("balance", 0) for a in accounts]
    sorted_ok = all(balances[i] >= balances[i + 1] for i in range(len(balances) - 1))
    ok = len(accounts) == SEED_ACCOUNT_COUNT and sorted_ok
    suite.record(TestResult(
        name="GET /api/accounts?sort=balance_desc",
        passed=ok,
        latency_ms=meta.get("latency_ms", lat),
        plan_hit=meta.get("plan_executed", False),
        cached=meta.get("cached", False),
        detail=f"{'sorted' if sorted_ok else 'NOT SORTED'}, top=${balances[0] if balances else '?'}",
    ))


def test_sort_name(suite: TestSuite):
    """Sort by holder name alphabetically."""
    resp, lat = _get("/api/accounts", {"sort": "name"})
    meta = _meta(resp)
    data = resp.get("data", {}) or {}
    accounts = data.get("accounts", []) or []

    names = [a.get("holder_name", "") for a in accounts]
    sorted_ok = all(names[i].lower() <= names[i + 1].lower() for i in range(len(names) - 1))
    ok = len(accounts) >= SEED_ACCOUNT_COUNT and sorted_ok
    suite.record(TestResult(
        name="GET /api/accounts?sort=name",
        passed=ok,
        latency_ms=meta.get("latency_ms", lat),
        plan_hit=meta.get("plan_executed", False),
        cached=meta.get("cached", False),
        detail=f"{'sorted' if sorted_ok else 'NOT SORTED'}, first='{names[0][:20] if names else '?'}'",
    ))


def test_pagination(suite: TestSuite):
    """Pagination: limit=3."""
    resp, lat = _get("/api/accounts", {"limit": 3})
    meta = _meta(resp)
    data = resp.get("data", {}) or {}
    accounts = data.get("accounts", []) or []

    ok = len(accounts) == 3
    suite.record(TestResult(
        name="GET /api/accounts?limit=3",
        passed=ok,
        latency_ms=meta.get("latency_ms", lat),
        plan_hit=meta.get("plan_executed", False),
        cached=meta.get("cached", False),
        detail=f"{len(accounts)} returned",
    ))


# ══════════════════════════════════════════════════════════════════════════
# 2. GET /api/accounts/{id} — Lookup by ID
# ══════════════════════════════════════════════════════════════════════════

def test_get_account_alice(suite: TestSuite):
    """Get Alice Johnson (id=1)."""
    resp, lat = _get("/api/accounts/1")
    meta = _meta(resp)
    data = resp.get("data", {}) or {}

    ok = (
        data.get("id") == 1
        and "Alice" in data.get("holder_name", "")
        and data.get("account_type") == "checking"
        and data.get("status") == "active"
    )
    suite.record(TestResult(
        name="GET /api/accounts/1 — Alice Johnson",
        passed=ok,
        latency_ms=meta.get("latency_ms", lat),
        plan_hit=meta.get("plan_executed", False),
        cached=meta.get("cached", False),
        detail=f"name='{data.get('holder_name', '?')[:25]}', balance=${data.get('balance', '?')}",
    ))


def test_get_account_diana(suite: TestSuite):
    """Get Diana Prince (id=4, business, highest balance)."""
    resp, lat = _get("/api/accounts/4")
    meta = _meta(resp)
    data = resp.get("data", {}) or {}

    ok = (
        data.get("id") == 4
        and data.get("account_type") == "business"
        and data.get("balance") == 45000.00
    )
    suite.record(TestResult(
        name="GET /api/accounts/4 — Diana Prince (business)",
        passed=ok,
        latency_ms=meta.get("latency_ms", lat),
        plan_hit=meta.get("plan_executed", False),
        cached=meta.get("cached", False),
        detail=f"balance=${data.get('balance', '?')}, type={data.get('account_type', '?')}",
    ))


def test_get_frozen_account(suite: TestSuite):
    """Get frozen account (id=6, Frank Wilson)."""
    resp, lat = _get(f"/api/accounts/{SEED_FROZEN_ID}")
    meta = _meta(resp)
    data = resp.get("data", {}) or {}

    ok = data.get("id") == SEED_FROZEN_ID and data.get("status") == "frozen"
    suite.record(TestResult(
        name=f"GET /api/accounts/{SEED_FROZEN_ID} — frozen account",
        passed=ok,
        latency_ms=meta.get("latency_ms", lat),
        plan_hit=meta.get("plan_executed", False),
        cached=meta.get("cached", False),
        detail=f"status={data.get('status', '?')}",
    ))


def test_get_account_not_found(suite: TestSuite):
    """Get non-existent account (id=9999)."""
    resp, lat = _get("/api/accounts/9999")
    meta = _meta(resp)

    data = resp.get("data", {}) or {}
    is_empty = data in ({}, None)
    not_found = "not found" in str(data).lower() or "not found" in str(resp.get("error", "")).lower()
    status = resp.get("status_code", data.get("status_code", 200))
    ok = is_empty or not_found or status == 404
    suite.record(TestResult(
        name="GET /api/accounts/9999 — not found",
        passed=ok,
        latency_ms=meta.get("latency_ms", lat),
        plan_hit=meta.get("plan_executed", False),
        cached=meta.get("cached", False),
        detail=f"empty={is_empty}, not_found={not_found}, status={status}",
    ))


# ══════════════════════════════════════════════════════════════════════════
# 3. POST /api/accounts/{id}/deposit — Deposits
# ══════════════════════════════════════════════════════════════════════════

def test_deposit_valid(suite: TestSuite):
    """Deposit $500 into Alice's account (id=1)."""
    resp, lat = _post("/api/accounts/1/deposit", {"amount": 500.00, "description": "Test deposit"})
    meta = _meta(resp)
    data = resp.get("data", {}) or {}

    # Check transaction was created — LLM may return various keys
    has_ref = bool(
        data.get("reference_number")
        or data.get("transaction", {}).get("reference_number")
        or data.get("transaction_id")
        or data.get("id")
    )
    # Check balance updated — try all known response shapes
    new_balance = (
        data.get("new_balance")
        or data.get("balance")
        or data.get("account", {}).get("balance")
        or data.get("account", {}).get("new_balance")
        or data.get("updated_balance")
    )

    expected = SEED_BALANCES[1] + 500.00
    balance_ok = new_balance is not None and abs(float(new_balance) - expected) < 0.01
    ok = has_ref and balance_ok

    # Track balance mutation if deposit likely succeeded (any positive signal)
    probably_succeeded = has_ref or (new_balance is not None and float(new_balance) > SEED_BALANCES[1])
    if probably_succeeded:
        suite.track_balance_change(1, 500.00)
    suite.record(TestResult(
        name="POST /api/accounts/1/deposit — $500",
        passed=ok,
        latency_ms=meta.get("latency_ms", lat),
        plan_hit=meta.get("plan_executed", False),
        cached=meta.get("cached", False),
        detail=f"balance=${new_balance}, ref={has_ref}",
    ))


def test_deposit_large(suite: TestSuite):
    """Deposit $10,000 into Eve's savings (id=5)."""
    resp, lat = _post("/api/accounts/5/deposit", {"amount": 10000.00, "description": "Large deposit"})
    meta = _meta(resp)
    data = resp.get("data", {}) or {}

    new_balance = (
        data.get("new_balance")
        or data.get("balance")
        or data.get("account", {}).get("balance")
        or data.get("account", {}).get("new_balance")
        or data.get("updated_balance")
    )
    expected = SEED_BALANCES[5] + 10000.00
    balance_ok = new_balance is not None and abs(float(new_balance) - expected) < 0.01
    has_ref = bool(
        data.get("reference_number")
        or data.get("transaction", {}).get("reference_number")
        or data.get("transaction_id")
        or data.get("id")
    )
    ok = has_ref and balance_ok

    # Track balance mutation if deposit likely succeeded
    probably_succeeded = has_ref or (new_balance is not None and float(new_balance) > SEED_BALANCES[5])
    if probably_succeeded:
        suite.track_balance_change(5, 10000.00)
    suite.record(TestResult(
        name="POST /api/accounts/5/deposit — $10,000",
        passed=ok,
        latency_ms=meta.get("latency_ms", lat),
        plan_hit=meta.get("plan_executed", False),
        cached=meta.get("cached", False),
        detail=f"balance=${new_balance}, expected=${expected}",
    ))


def test_deposit_frozen_account(suite: TestSuite):
    """Deposit into frozen account (id=6) — should fail."""
    resp, lat = _post(f"/api/accounts/{SEED_FROZEN_ID}/deposit", {"amount": 100.00})
    meta = _meta(resp)

    data = resp.get("data", {}) or {}
    status = data.get("status_code", resp.get("status_code", 200))
    has_error = (
        bool(resp.get("error"))
        or bool(data.get("error"))
        or status in (400, 403, 422)
        or "frozen" in str(data).lower()
        or "frozen" in str(resp).lower()
    )
    ok = has_error
    suite.record(TestResult(
        name=f"POST /api/accounts/{SEED_FROZEN_ID}/deposit — frozen → error",
        passed=ok,
        latency_ms=meta.get("latency_ms", lat),
        plan_hit=meta.get("plan_executed", False),
        cached=meta.get("cached", False),
        detail=f"error={has_error}, status={status}",
    ))


def test_deposit_negative_amount(suite: TestSuite):
    """Deposit negative amount — should fail."""
    resp, lat = _post("/api/accounts/1/deposit", {"amount": -100.00})
    meta = _meta(resp)

    data = resp.get("data", {}) or {}
    status = data.get("status_code", resp.get("status_code", 200))
    has_error = (
        bool(resp.get("error"))
        or bool(data.get("error"))
        or status in (400, 422)
        or "negative" in str(data).lower()
        or "must be" in str(data).lower()
        or "greater than" in str(data).lower()
        or "invalid" in str(data).lower()
    )
    ok = has_error
    suite.record(TestResult(
        name="POST /api/accounts/1/deposit — negative amount → error",
        passed=ok,
        latency_ms=meta.get("latency_ms", lat),
        plan_hit=meta.get("plan_executed", False),
        cached=meta.get("cached", False),
        detail=f"error={has_error}, status={status}",
    ))


def test_deposit_zero_amount(suite: TestSuite):
    """Deposit $0 — should fail."""
    resp, lat = _post("/api/accounts/1/deposit", {"amount": 0})
    meta = _meta(resp)

    data = resp.get("data", {}) or {}
    status = data.get("status_code", resp.get("status_code", 200))
    has_error = (
        bool(resp.get("error"))
        or bool(data.get("error"))
        or status in (400, 422)
        or "zero" in str(data).lower()
        or "must be" in str(data).lower()
        or "greater than" in str(data).lower()
    )
    ok = has_error
    suite.record(TestResult(
        name="POST /api/accounts/1/deposit — $0 → error",
        passed=ok,
        latency_ms=meta.get("latency_ms", lat),
        plan_hit=meta.get("plan_executed", False),
        cached=meta.get("cached", False),
        detail=f"error={has_error}, status={status}",
    ))


# ══════════════════════════════════════════════════════════════════════════
# 4. POST /api/transfers — Fund Transfers (the core test!)
# ══════════════════════════════════════════════════════════════════════════

def test_transfer_valid(suite: TestSuite):
    """Transfer $200 from Alice (1) to Charlie (3)."""
    resp, lat = _post("/api/transfers", {
        "from_account_id": 1,
        "to_account_id": 3,
        "amount": 200.00,
        "description": "Test transfer Alice→Charlie",
    })
    meta = _meta(resp)
    data = resp.get("data", {}) or {}
    txn = data.get("transaction", {}) or {}

    # Check a transaction record came back
    has_amount = (
        data.get("amount") == 200.00
        or txn.get("amount") == 200.00
    )
    has_ref = bool(
        data.get("reference_number")
        or txn.get("reference_number")
        or data.get("transaction_id")
        or txn.get("id")
        or data.get("id")
    )
    status_val = data.get("status") or txn.get("status")
    transfer_ok = status_val in ("completed", "success", None)

    ok = has_amount and has_ref
    # Track mutation if transfer likely succeeded
    probably_succeeded = has_ref or has_amount or (status_val == "completed")
    if probably_succeeded and "error" not in str(data).lower()[:200]:
        suite.track_balance_change(1, -200.00)
        suite.track_balance_change(3, +200.00)
    suite.record(TestResult(
        name="POST /api/transfers — $200 Alice→Charlie",
        passed=ok,
        latency_ms=meta.get("latency_ms", lat),
        plan_hit=meta.get("plan_executed", False),
        cached=meta.get("cached", False),
        detail=f"amount={txn.get('amount', data.get('amount'))}, ref={has_ref}, status={status_val}",
    ))


def test_transfer_large(suite: TestSuite):
    """Transfer $5,000 from Diana (4) to Bob (2)."""
    resp, lat = _post("/api/transfers", {
        "from_account_id": 4,
        "to_account_id": 2,
        "amount": 5000.00,
        "description": "Large business transfer",
    })
    meta = _meta(resp)
    data = resp.get("data", {}) or {}
    txn = data.get("transaction", {}) or {}

    has_amount = (
        data.get("amount") == 5000.00
        or txn.get("amount") == 5000.00
    )
    has_ref = bool(
        data.get("reference_number")
        or txn.get("reference_number")
        or data.get("transaction_id")
        or txn.get("id")
        or data.get("id")
    )
    ok = has_amount and has_ref

    # Track mutation if transfer likely succeeded
    probably_succeeded = has_ref or has_amount
    if probably_succeeded and "error" not in str(data).lower()[:200]:
        suite.track_balance_change(4, -5000.00)
        suite.track_balance_change(2, +5000.00)
    suite.record(TestResult(
        name="POST /api/transfers — $5,000 Diana→Bob",
        passed=ok,
        latency_ms=meta.get("latency_ms", lat),
        plan_hit=meta.get("plan_executed", False),
        cached=meta.get("cached", False),
        detail=f"amount={txn.get('amount', data.get('amount'))}, ref={has_ref}",
    ))


def test_transfer_insufficient_funds(suite: TestSuite):
    """Transfer more than Charlie's balance — should fail."""
    charlie_expected = suite.expected_balance(3)
    over_amount = charlie_expected + 5000  # definitely more than Charlie has
    resp, lat = _post("/api/transfers", {
        "from_account_id": 3,
        "to_account_id": 1,
        "amount": over_amount,
    })
    meta = _meta(resp)
    data = resp.get("data", {}) or {}

    status = data.get("status_code", resp.get("status_code", 200))
    resp_str = str(data).lower() + str(resp.get("error", "")).lower()
    has_error = (
        bool(resp.get("error"))
        or bool(data.get("error"))
        or status in (400, 422)
        or "insufficient" in resp_str
        or "enough" in resp_str
        or "exceed" in resp_str
        or "failed" in resp_str
        or "cannot" in resp_str
        or "not enough" in resp_str
    )
    # If the LLM didn't reject and actually performed the transfer, track it
    if not has_error:
        txn = data.get("transaction", {}) or {}
        if txn.get("status") == "completed" or data.get("status") == "completed":
            suite.track_balance_change(3, -over_amount)
            suite.track_balance_change(1, +over_amount)
    ok = has_error
    suite.record(TestResult(
        name="POST /api/transfers — insufficient funds → error",
        passed=ok,
        latency_ms=meta.get("latency_ms", lat),
        plan_hit=meta.get("plan_executed", False),
        cached=meta.get("cached", False),
        detail=f"attempted=${over_amount:.2f}, error={has_error}",
    ))


def test_transfer_to_self(suite: TestSuite):
    """Transfer to same account — should fail."""
    resp, lat = _post("/api/transfers", {
        "from_account_id": 1,
        "to_account_id": 1,
        "amount": 100.00,
    })
    meta = _meta(resp)
    data = resp.get("data", {}) or {}

    status = data.get("status_code", resp.get("status_code", 200))
    has_error = (
        bool(resp.get("error"))
        or bool(data.get("error"))
        or status in (400, 422)
        or "same" in str(data).lower()
        or "different" in str(data).lower()
        or "cannot" in str(data).lower()
    )
    ok = has_error
    suite.record(TestResult(
        name="POST /api/transfers — same account → error",
        passed=ok,
        latency_ms=meta.get("latency_ms", lat),
        plan_hit=meta.get("plan_executed", False),
        cached=meta.get("cached", False),
        detail=f"error={has_error}",
    ))


def test_transfer_from_frozen(suite: TestSuite):
    """Transfer FROM frozen account — should fail."""
    resp, lat = _post("/api/transfers", {
        "from_account_id": SEED_FROZEN_ID,
        "to_account_id": 1,
        "amount": 50.00,
    })
    meta = _meta(resp)
    data = resp.get("data", {}) or {}

    status = data.get("status_code", resp.get("status_code", 200))
    has_error = (
        bool(resp.get("error"))
        or bool(data.get("error"))
        or status in (400, 403, 422)
        or "frozen" in str(data).lower()
        or "active" in str(data).lower()
        or "not active" in str(data).lower()
    )
    ok = has_error
    suite.record(TestResult(
        name="POST /api/transfers — from frozen account → error",
        passed=ok,
        latency_ms=meta.get("latency_ms", lat),
        plan_hit=meta.get("plan_executed", False),
        cached=meta.get("cached", False),
        detail=f"error={has_error}",
    ))


def test_transfer_to_closed(suite: TestSuite):
    """Transfer TO closed account — should fail."""
    resp, lat = _post("/api/transfers", {
        "from_account_id": 1,
        "to_account_id": SEED_CLOSED_ID,
        "amount": 50.00,
    })
    meta = _meta(resp)
    data = resp.get("data", {}) or {}

    status = data.get("status_code", resp.get("status_code", 200))
    has_error = (
        bool(resp.get("error"))
        or bool(data.get("error"))
        or status in (400, 403, 422)
        or "closed" in str(data).lower()
        or "active" in str(data).lower()
        or "not active" in str(data).lower()
    )
    ok = has_error
    suite.record(TestResult(
        name="POST /api/transfers — to closed account → error",
        passed=ok,
        latency_ms=meta.get("latency_ms", lat),
        plan_hit=meta.get("plan_executed", False),
        cached=meta.get("cached", False),
        detail=f"error={has_error}",
    ))


def test_transfer_negative_amount(suite: TestSuite):
    """Transfer negative amount — should fail."""
    resp, lat = _post("/api/transfers", {
        "from_account_id": 1,
        "to_account_id": 2,
        "amount": -100.00,
    })
    meta = _meta(resp)
    data = resp.get("data", {}) or {}

    status = data.get("status_code", resp.get("status_code", 200))
    has_error = (
        bool(resp.get("error"))
        or bool(data.get("error"))
        or status in (400, 422)
    )
    ok = has_error
    suite.record(TestResult(
        name="POST /api/transfers — negative amount → error",
        passed=ok,
        latency_ms=meta.get("latency_ms", lat),
        plan_hit=meta.get("plan_executed", False),
        cached=meta.get("cached", False),
        detail=f"error={has_error}",
    ))


def test_transfer_nonexistent_account(suite: TestSuite):
    """Transfer from non-existent account — should fail."""
    resp, lat = _post("/api/transfers", {
        "from_account_id": 9999,
        "to_account_id": 1,
        "amount": 100.00,
    })
    meta = _meta(resp)
    data = resp.get("data", {}) or {}

    status = data.get("status_code", resp.get("status_code", 200))
    has_error = (
        bool(resp.get("error"))
        or bool(data.get("error"))
        or status in (400, 404, 422)
        or "not found" in str(data).lower()
        or "does not exist" in str(data).lower()
        or "exist" in str(data).lower()
    )
    ok = has_error
    suite.record(TestResult(
        name="POST /api/transfers — non-existent account → error",
        passed=ok,
        latency_ms=meta.get("latency_ms", lat),
        plan_hit=meta.get("plan_executed", False),
        cached=meta.get("cached", False),
        detail=f"error={has_error}",
    ))


# ══════════════════════════════════════════════════════════════════════════
# 5. GET /api/transactions — Transaction History
# ══════════════════════════════════════════════════════════════════════════

def test_list_all_transactions(suite: TestSuite):
    """List all transactions — at least seed count + our new ones."""
    resp, lat = _get("/api/transactions")
    meta = _meta(resp)
    data = resp.get("data", {}) or {}
    txns = data.get("transactions", []) or []

    # We added at least 2 deposits + 2 transfers = 4 new txns
    ok = len(txns) >= SEED_TXN_COUNT
    suite.record(TestResult(
        name="GET /api/transactions — list all",
        passed=ok,
        latency_ms=meta.get("latency_ms", lat),
        plan_hit=meta.get("plan_executed", False),
        cached=meta.get("cached", False),
        detail=f"{len(txns)} transactions (seed={SEED_TXN_COUNT})",
    ))


def test_filter_transfers(suite: TestSuite):
    """Filter by type=transfer."""
    resp, lat = _get("/api/transactions", {"type": "transfer"})
    meta = _meta(resp)
    data = resp.get("data", {}) or {}
    txns = data.get("transactions", []) or []

    all_transfers = all(t.get("type") == "transfer" for t in txns)
    ok = len(txns) >= 10 and all_transfers  # seed has 10 transfers + our 2
    suite.record(TestResult(
        name="GET /api/transactions?type=transfer",
        passed=ok,
        latency_ms=meta.get("latency_ms", lat),
        plan_hit=meta.get("plan_executed", False),
        cached=meta.get("cached", False),
        detail=f"{len(txns)} transfers, all_correct={all_transfers}",
    ))


def test_filter_deposits(suite: TestSuite):
    """Filter by type=deposit."""
    resp, lat = _get("/api/transactions", {"type": "deposit"})
    meta = _meta(resp)
    data = resp.get("data", {}) or {}
    txns = data.get("transactions", []) or []

    all_deposits = all(t.get("type") == "deposit" for t in txns)
    ok = len(txns) >= 4 and all_deposits  # seed has 4 deposits + our 2
    suite.record(TestResult(
        name="GET /api/transactions?type=deposit",
        passed=ok,
        latency_ms=meta.get("latency_ms", lat),
        plan_hit=meta.get("plan_executed", False),
        cached=meta.get("cached", False),
        detail=f"{len(txns)} deposits",
    ))


def test_filter_by_account(suite: TestSuite):
    """Filter transactions for Alice (account_id=1) — she's involved in many."""
    resp, lat = _get("/api/transactions", {"account_id": 1})
    meta = _meta(resp)
    data = resp.get("data", {}) or {}
    txns = data.get("transactions", []) or []

    # Alice is party to seed TXN-0001,0003,0004,0007,0011,0015 + our transfers/deposits
    all_involve_alice = all(
        t.get("from_account_id") == 1 or t.get("to_account_id") == 1
        for t in txns
    )
    ok = len(txns) >= 6 and all_involve_alice
    suite.record(TestResult(
        name="GET /api/transactions?account_id=1 — Alice's history",
        passed=ok,
        latency_ms=meta.get("latency_ms", lat),
        plan_hit=meta.get("plan_executed", False),
        cached=meta.get("cached", False),
        detail=f"{len(txns)} txns involving Alice, all_correct={all_involve_alice}",
    ))


def test_filter_completed(suite: TestSuite):
    """Filter by status=completed."""
    resp, lat = _get("/api/transactions", {"status": "completed"})
    meta = _meta(resp)
    data = resp.get("data", {}) or {}
    txns = data.get("transactions", []) or []

    all_completed = all(t.get("status") == "completed" for t in txns)
    ok = len(txns) >= 10 and all_completed  # seed has 11 completed + our new ones
    suite.record(TestResult(
        name="GET /api/transactions?status=completed",
        passed=ok,
        latency_ms=meta.get("latency_ms", lat),
        plan_hit=meta.get("plan_executed", False),
        cached=meta.get("cached", False),
        detail=f"{len(txns)} completed, all_correct={all_completed}",
    ))


def test_filter_large_transactions(suite: TestSuite):
    """Filter by min_amount=1000 — large transactions."""
    resp, lat = _get("/api/transactions", {"min_amount": 1000})
    meta = _meta(resp)
    data = resp.get("data", {}) or {}
    txns = data.get("transactions", []) or []

    all_above = all(t.get("amount", 0) >= 1000 for t in txns)
    # Seed: TXN-0003($1500), TXN-0005($1000), TXN-0009($5000), TXN-0010($2500), TXN-0012($3000) + our $5000 + $10000
    ok = len(txns) >= 5 and all_above
    suite.record(TestResult(
        name="GET /api/transactions?min_amount=1000",
        passed=ok,
        latency_ms=meta.get("latency_ms", lat),
        plan_hit=meta.get("plan_executed", False),
        cached=meta.get("cached", False),
        detail=f"{len(txns)} transactions >= $1000",
    ))


# ══════════════════════════════════════════════════════════════════════════
# 6. POST /api/analytics/summary — Financial Analytics
# ══════════════════════════════════════════════════════════════════════════

def test_analytics_total_balance(suite: TestSuite):
    """Analytics: total balance across all accounts."""
    resp, lat = _post("/api/analytics/summary", {
        "question": "What is the total balance across all accounts?"
    })
    meta = _meta(resp)
    data = resp.get("data", {}) or {}

    raw_data = data.get("raw_data", []) or []
    has_analysis = len(str(data.get("analysis", ""))) > 5
    ok = len(raw_data) >= 1 or has_analysis
    suite.record(TestResult(
        name="POST /api/analytics/summary — total balance",
        passed=ok,
        latency_ms=meta.get("latency_ms", lat),
        plan_hit=meta.get("plan_executed", False),
        cached=meta.get("cached", False),
        detail=f"raw={raw_data[:1]}, analysis={has_analysis}",
    ))


def test_analytics_avg_by_type(suite: TestSuite):
    """Analytics: average balance by account type."""
    resp, lat = _post("/api/analytics/summary", {
        "question": "What is the average balance by account type (checking, savings, business)?"
    })
    meta = _meta(resp)
    data = resp.get("data", {}) or {}

    raw_data = data.get("raw_data", []) or []
    has_analysis = len(str(data.get("analysis", ""))) > 5
    ok = len(raw_data) >= 2 or has_analysis
    suite.record(TestResult(
        name="POST /api/analytics/summary — avg balance by type",
        passed=ok,
        latency_ms=meta.get("latency_ms", lat),
        plan_hit=meta.get("plan_executed", False),
        cached=meta.get("cached", False),
        detail=f"{len(raw_data)} types, analysis={has_analysis}",
    ))


def test_analytics_txn_volume(suite: TestSuite):
    """Analytics: transaction volume by type."""
    resp, lat = _post("/api/analytics/summary", {
        "question": "How many transactions are there for each type (transfer, deposit, withdrawal)?"
    })
    meta = _meta(resp)
    data = resp.get("data", {}) or {}

    raw_data = data.get("raw_data", []) or []
    has_analysis = len(str(data.get("analysis", ""))) > 5
    ok = len(raw_data) >= 2 or has_analysis
    suite.record(TestResult(
        name="POST /api/analytics/summary — txn volume by type",
        passed=ok,
        latency_ms=meta.get("latency_ms", lat),
        plan_hit=meta.get("plan_executed", False),
        cached=meta.get("cached", False),
        detail=f"{len(raw_data)} types, analysis={has_analysis}",
    ))


def test_analytics_largest_transfers(suite: TestSuite):
    """Analytics: top 3 largest transfers."""
    resp, lat = _post("/api/analytics/summary", {
        "question": "What are the top 3 largest transfers by amount?"
    })
    meta = _meta(resp)
    data = resp.get("data", {}) or {}

    raw_data = data.get("raw_data", []) or []
    has_analysis = len(str(data.get("analysis", ""))) > 5
    ok = len(raw_data) >= 1 or has_analysis
    suite.record(TestResult(
        name="POST /api/analytics/summary — largest transfers",
        passed=ok,
        latency_ms=meta.get("latency_ms", lat),
        plan_hit=meta.get("plan_executed", False),
        cached=meta.get("cached", False),
        detail=f"{len(raw_data)} rows, analysis={has_analysis}",
    ))


# ══════════════════════════════════════════════════════════════════════════
# 7. Cross-cutting: Balance Integrity Checks (direct DB verification)
# ══════════════════════════════════════════════════════════════════════════

def _db_balance(account_id: int) -> float | None:
    """Read current balance directly from SQLite (bypasses API/plan cache)."""
    import sqlite3
    if DB_PATH is None:
        return None
    try:
        conn = sqlite3.connect(str(DB_PATH))
        c = conn.cursor()
        c.execute("SELECT balance FROM accounts WHERE id = ?", (account_id,))
        row = c.fetchone()
        conn.close()
        return row[0] if row else None
    except Exception:
        return None


def test_balance_integrity_alice(suite: TestSuite):
    """Verify Alice's balance reflects deposit + transfer out."""
    actual = _db_balance(1)
    expected = suite.expected_balance(1)
    ok = actual is not None and abs(float(actual) - expected) < 0.01
    suite.record(TestResult(
        name="INTEGRITY: Alice's balance after operations",
        passed=ok,
        latency_ms=0,
        detail=f"actual=${actual}, expected=${expected:.2f}",
    ))


def test_balance_integrity_charlie(suite: TestSuite):
    """Verify Charlie's balance reflects transfer in."""
    actual = _db_balance(3)
    expected = suite.expected_balance(3)
    ok = actual is not None and abs(float(actual) - expected) < 0.01
    suite.record(TestResult(
        name="INTEGRITY: Charlie's balance after transfer in",
        passed=ok,
        latency_ms=0,
        detail=f"actual=${actual}, expected=${expected:.2f}",
    ))


def test_balance_integrity_eve(suite: TestSuite):
    """Verify Eve's balance reflects large deposit."""
    actual = _db_balance(5)
    expected = suite.expected_balance(5)
    ok = actual is not None and abs(float(actual) - expected) < 0.01
    suite.record(TestResult(
        name="INTEGRITY: Eve's balance after $10k deposit",
        passed=ok,
        latency_ms=0,
        detail=f"actual=${actual}, expected=${expected:.2f}",
    ))


def test_balance_integrity_diana(suite: TestSuite):
    """Verify Diana's balance reflects large transfer out."""
    actual = _db_balance(4)
    expected = suite.expected_balance(4)
    ok = actual is not None and abs(float(actual) - expected) < 0.01
    suite.record(TestResult(
        name="INTEGRITY: Diana's balance after $5k transfer out",
        passed=ok,
        latency_ms=0,
        detail=f"actual=${actual}, expected=${expected:.2f}",
    ))


def test_balance_integrity_bob(suite: TestSuite):
    """Verify Bob's balance reflects transfer in."""
    actual = _db_balance(2)
    expected = suite.expected_balance(2)
    ok = actual is not None and abs(float(actual) - expected) < 0.01
    suite.record(TestResult(
        name="INTEGRITY: Bob's balance after $5k transfer in",
        passed=ok,
        latency_ms=0,
        detail=f"actual=${actual}, expected=${expected:.2f}",
    ))


def test_frozen_balance_unchanged(suite: TestSuite):
    """Verify frozen account balance hasn't changed."""
    actual = _db_balance(SEED_FROZEN_ID)
    expected = SEED_BALANCES[SEED_FROZEN_ID]  # no deltas expected
    ok = actual is not None and abs(float(actual) - expected) < 0.01
    suite.record(TestResult(
        name="INTEGRITY: Frozen account balance unchanged",
        passed=ok,
        latency_ms=0,
        detail=f"actual=${actual}, expected=${expected:.2f}",
    ))


# ──────────────────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────────────────

def main():
    global DB_PATH
    suite = TestSuite()

    # Reseed the banking database before running tests to ensure known state
    import subprocess, pathlib
    seed_script = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "seed_banking_db.py"
    DB_PATH = pathlib.Path(__file__).resolve().parent.parent / "data" / "catalyst.db"
    if seed_script.exists():
        print(f"  Reseeding database from {seed_script.name} ...")
        subprocess.run([sys.executable, str(seed_script)], check=True,
                       capture_output=True, text=True)
        print("  Database reseeded.\n")

    # Verify server
    try:
        httpx.get(f"{BASE_URL}/health", timeout=5)
    except Exception as e:
        print(f"ERROR: Cannot reach server at {BASE_URL}: {e}")
        sys.exit(1)

    # ── Section 1: List / Filter / Sort ─────────────────────────────────
    print(f"\n{'─' * 80}")
    print("  1. GET /api/accounts — List / Filter / Sort / Paginate")
    print(f"{'─' * 80}")
    for t in [
        test_list_all_accounts,
        test_filter_checking,
        test_filter_savings,
        test_filter_active,
        test_filter_frozen,
        test_filter_min_balance,
        test_filter_max_balance,
        test_sort_balance_desc,
        test_sort_name,
        test_pagination,
    ]:
        try:
            t(suite)
        except Exception as e:
            suite.record(TestResult(name=t.__name__, passed=False, detail=f"EXCEPTION: {e}"))

    # ── Section 2: Get by ID ────────────────────────────────────────────
    print(f"\n{'─' * 80}")
    print("  2. GET /api/accounts/{{id}} — Lookup by ID")
    print(f"{'─' * 80}")
    for t in [
        test_get_account_alice,
        test_get_account_diana,
        test_get_frozen_account,
        test_get_account_not_found,
    ]:
        try:
            t(suite)
        except Exception as e:
            suite.record(TestResult(name=t.__name__, passed=False, detail=f"EXCEPTION: {e}"))

    # ── Section 3: Deposits ─────────────────────────────────────────────
    print(f"\n{'─' * 80}")
    print("  3. POST /api/accounts/{{id}}/deposit — Deposits")
    print(f"{'─' * 80}")
    for t in [
        test_deposit_valid,
        test_deposit_large,
        test_deposit_frozen_account,
        test_deposit_negative_amount,
        test_deposit_zero_amount,
    ]:
        try:
            t(suite)
        except Exception as e:
            suite.record(TestResult(name=t.__name__, passed=False, detail=f"EXCEPTION: {e}"))

    # ── Section 4: Transfers ────────────────────────────────────────────
    print(f"\n{'─' * 80}")
    print("  4. POST /api/transfers — Fund Transfers")
    print(f"{'─' * 80}")
    for t in [
        test_transfer_valid,
        test_transfer_large,
        test_transfer_insufficient_funds,
        test_transfer_to_self,
        test_transfer_from_frozen,
        test_transfer_to_closed,
        test_transfer_negative_amount,
        test_transfer_nonexistent_account,
    ]:
        try:
            t(suite)
        except Exception as e:
            suite.record(TestResult(name=t.__name__, passed=False, detail=f"EXCEPTION: {e}"))

    # ── Section 5: Transaction History ──────────────────────────────────
    print(f"\n{'─' * 80}")
    print("  5. GET /api/transactions — Transaction History")
    print(f"{'─' * 80}")
    for t in [
        test_list_all_transactions,
        test_filter_transfers,
        test_filter_deposits,
        test_filter_by_account,
        test_filter_completed,
        test_filter_large_transactions,
    ]:
        try:
            t(suite)
        except Exception as e:
            suite.record(TestResult(name=t.__name__, passed=False, detail=f"EXCEPTION: {e}"))

    # ── Section 6: Analytics ────────────────────────────────────────────
    print(f"\n{'─' * 80}")
    print("  6. POST /api/analytics/summary — Financial Analytics")
    print(f"{'─' * 80}")
    for t in [
        test_analytics_total_balance,
        test_analytics_avg_by_type,
        test_analytics_txn_volume,
        test_analytics_largest_transfers,
    ]:
        try:
            t(suite)
        except Exception as e:
            suite.record(TestResult(name=t.__name__, passed=False, detail=f"EXCEPTION: {e}"))

    # ── Section 7: Balance Integrity ────────────────────────────────────
    print(f"\n{'─' * 80}")
    print("  7. Balance Integrity Checks")
    print(f"{'─' * 80}")
    for t in [
        test_balance_integrity_alice,
        test_balance_integrity_charlie,
        test_balance_integrity_eve,
        test_balance_integrity_diana,
        test_balance_integrity_bob,
        test_frozen_balance_unchanged,
    ]:
        try:
            t(suite)
        except Exception as e:
            suite.record(TestResult(name=t.__name__, passed=False, detail=f"EXCEPTION: {e}"))

    # ── Final ───────────────────────────────────────────────────────────
    all_passed = suite.summary()
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
