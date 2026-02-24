"""
Extensive E-Commerce API Test Suite

Tests all 4 ecommerce endpoints against a live Catalyst server:
  1. GET  /api/products       — list / search / filter / sort / paginate
  2. POST /api/products       — create product (valid + invalid)
  3. GET  /api/products/{id}  — get by ID (found + not found)
  4. POST /api/orders/analyze — natural language analytics

Requires:
  - Server running at BASE_URL
  - Database seeded via scripts/seed_db.py (12 products, 15 orders)

Usage:
    python tests/test_ecommerce_api.py
"""

import json
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

BASE_URL = "http://127.0.0.1:8000"

# Known seed data for validation
SEED_PRODUCT_COUNT = 12
SEED_CATEGORIES = {"electronics", "food", "books", "furniture", "fitness", "appliances", "outdoors"}
SEED_ELECTRONICS_IN_STOCK = 4   # IDs: 1, 5, 8, 12 (10 is out of stock)
SEED_ELECTRONICS_TOTAL = 5      # IDs: 1, 5, 8, 10, 12
SEED_ORDER_COUNT = 15

# ──────────────────────────────────────────────────────────────────────────
# Test infrastructure
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class TestResult:
    name: str
    passed: bool
    latency_ms: float = 0.0
    plan_hit: bool = False
    cached: bool = False
    detail: str = ""


@dataclass
class TestSuite:
    results: list[TestResult] = field(default_factory=list)
    _created_product_ids: list[int] = field(default_factory=list)

    def record(self, r: TestResult):
        self.results.append(r)
        status = "PASS" if r.passed else "FAIL"
        source = "CACHE" if r.cached else ("PLAN" if r.plan_hit else "LLM")
        detail_str = f"  ({r.detail})" if r.detail else ""
        print(f"  [{status}] {r.name:<55} {r.latency_ms:>9.1f}ms  [{source}]{detail_str}")

    def summary(self):
        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed)
        total = len(self.results)
        plan_hits = sum(1 for r in self.results if r.plan_hit)
        cache_hits = sum(1 for r in self.results if r.cached)
        llm_calls = total - plan_hits - cache_hits

        print(f"\n{'=' * 80}")
        print(f"  RESULTS: {passed}/{total} passed, {failed} failed")
        print(f"  SOURCE:  {llm_calls} LLM | {plan_hits} PLAN | {cache_hits} CACHE")
        if self.results:
            lats = [r.latency_ms for r in self.results]
            print(f"  LATENCY: min={min(lats):.1f}ms  avg={sum(lats)/len(lats):.1f}ms  max={max(lats):.1f}ms")
        print(f"{'=' * 80}")

        if failed:
            print("\n  FAILURES:")
            for r in self.results:
                if not r.passed:
                    print(f"    - {r.name}: {r.detail}")
        return failed == 0


def _get(path: str, params: dict | None = None, timeout: float = 60) -> tuple[dict, float]:
    """GET request, returns (response_json, latency_ms)."""
    t0 = time.perf_counter()
    resp = httpx.get(f"{BASE_URL}{path}", params=params, timeout=timeout)
    lat = (time.perf_counter() - t0) * 1000
    try:
        return resp.json(), lat
    except Exception:
        return {"error": resp.text, "status_code": resp.status_code}, lat


def _post(path: str, body: dict, timeout: float = 60) -> tuple[dict, float]:
    """POST request, returns (response_json, latency_ms)."""
    t0 = time.perf_counter()
    resp = httpx.post(f"{BASE_URL}{path}", json=body, timeout=timeout)
    lat = (time.perf_counter() - t0) * 1000
    try:
        return resp.json(), lat
    except Exception:
        return {"error": resp.text, "status_code": resp.status_code}, lat


def _meta(resp: dict) -> dict:
    return resp.get("meta", {})


# ──────────────────────────────────────────────────────────────────────────
# 1. GET /api/products — list / search / filter / sort / paginate
# ──────────────────────────────────────────────────────────────────────────

def test_list_all(suite: TestSuite):
    """List all products with no filters."""
    resp, lat = _get("/api/products")
    meta = _meta(resp)
    data = resp.get("data", {}) or {}
    products = data.get("products", []) or []

    ok = len(products) >= SEED_PRODUCT_COUNT
    suite.record(TestResult(
        name="GET /api/products — list all",
        passed=ok,
        latency_ms=meta.get("latency_ms", lat),
        plan_hit=meta.get("plan_executed", False),
        cached=meta.get("cached", False),
        detail=f"{len(products)} products",
    ))


def test_filter_category_electronics(suite: TestSuite):
    """Filter by category=electronics."""
    resp, lat = _get("/api/products", {"category": "electronics"})
    meta = _meta(resp)
    data = resp.get("data", {}) or {}
    products = data.get("products", []) or []

    all_electronics = all(p.get("category") == "electronics" for p in products)
    ok = len(products) >= SEED_ELECTRONICS_TOTAL and all_electronics
    suite.record(TestResult(
        name="GET /api/products?category=electronics",
        passed=ok,
        latency_ms=meta.get("latency_ms", lat),
        plan_hit=meta.get("plan_executed", False),
        cached=meta.get("cached", False),
        detail=f"{len(products)} electronics" + ("" if all_electronics else " WRONG CATEGORY"),
    ))


def test_filter_category_food(suite: TestSuite):
    """Filter by category=food."""
    resp, lat = _get("/api/products", {"category": "food"})
    meta = _meta(resp)
    data = resp.get("data", {}) or {}
    products = data.get("products", []) or []

    all_food = all(p.get("category") == "food" for p in products)
    ok = len(products) >= 1 and all_food
    suite.record(TestResult(
        name="GET /api/products?category=food",
        passed=ok,
        latency_ms=meta.get("latency_ms", lat),
        plan_hit=meta.get("plan_executed", False),
        cached=meta.get("cached", False),
        detail=f"{len(products)} food products",
    ))


def test_filter_category_fitness(suite: TestSuite):
    """Filter by category=fitness."""
    resp, lat = _get("/api/products", {"category": "fitness"})
    meta = _meta(resp)
    data = resp.get("data", {}) or {}
    products = data.get("products", []) or []

    all_fitness = all(p.get("category") == "fitness" for p in products)
    ok = len(products) >= 2 and all_fitness  # Yoga Mat + Water Bottle
    suite.record(TestResult(
        name="GET /api/products?category=fitness",
        passed=ok,
        latency_ms=meta.get("latency_ms", lat),
        plan_hit=meta.get("plan_executed", False),
        cached=meta.get("cached", False),
        detail=f"{len(products)} fitness products",
    ))


def test_filter_in_stock(suite: TestSuite):
    """Filter by in_stock=true — should exclude Vintage Vinyl (id=10)."""
    resp, lat = _get("/api/products", {"in_stock": "true"})
    meta = _meta(resp)
    data = resp.get("data", {}) or {}
    products = data.get("products", []) or []

    all_in_stock = all(p.get("in_stock") in (True, 1) for p in products)
    ok = len(products) >= 11 and all_in_stock
    suite.record(TestResult(
        name="GET /api/products?in_stock=true",
        passed=ok,
        latency_ms=meta.get("latency_ms", lat),
        plan_hit=meta.get("plan_executed", False),
        cached=meta.get("cached", False),
        detail=f"{len(products)} in-stock" + ("" if all_in_stock else " HAS OUT-OF-STOCK"),
    ))


def test_filter_electronics_in_stock(suite: TestSuite):
    """Combined: category=electronics + in_stock=true."""
    resp, lat = _get("/api/products", {"category": "electronics", "in_stock": "true"})
    meta = _meta(resp)
    data = resp.get("data", {}) or {}
    products = data.get("products", []) or []

    correct = all(
        p.get("category") == "electronics" and p.get("in_stock") in (True, 1)
        for p in products
    )
    ok = len(products) == SEED_ELECTRONICS_IN_STOCK and correct
    suite.record(TestResult(
        name="GET /api/products?category=electronics&in_stock=true",
        passed=ok,
        latency_ms=meta.get("latency_ms", lat),
        plan_hit=meta.get("plan_executed", False),
        cached=meta.get("cached", False),
        detail=f"{len(products)} electronics in-stock (expected {SEED_ELECTRONICS_IN_STOCK})",
    ))


def test_filter_min_price(suite: TestSuite):
    """Price range: min_price=100."""
    resp, lat = _get("/api/products", {"min_price": 100})
    meta = _meta(resp)
    data = resp.get("data", {}) or {}
    products = data.get("products", []) or []

    all_above = all(p.get("price", 0) >= 100 for p in products)
    ok = len(products) >= 4 and all_above
    suite.record(TestResult(
        name="GET /api/products?min_price=100",
        passed=ok,
        latency_ms=meta.get("latency_ms", lat),
        plan_hit=meta.get("plan_executed", False),
        cached=meta.get("cached", False),
        detail=f"{len(products)} products >= $100" + ("" if all_above else " HAS CHEAPER"),
    ))


def test_filter_max_price(suite: TestSuite):
    """Price range: max_price=30."""
    resp, lat = _get("/api/products", {"max_price": 30})
    meta = _meta(resp)
    data = resp.get("data", {}) or {}
    products = data.get("products", []) or []

    all_below = all(p.get("price", 999) <= 30 for p in products)
    ok = len(products) >= 3 and all_below
    suite.record(TestResult(
        name="GET /api/products?max_price=30",
        passed=ok,
        latency_ms=meta.get("latency_ms", lat),
        plan_hit=meta.get("plan_executed", False),
        cached=meta.get("cached", False),
        detail=f"{len(products)} products <= $30" + ("" if all_below else " HAS PRICIER"),
    ))


def test_filter_price_range(suite: TestSuite):
    """Price range: min_price=50, max_price=150."""
    resp, lat = _get("/api/products", {"min_price": 50, "max_price": 150})
    meta = _meta(resp)
    data = resp.get("data", {}) or {}
    products = data.get("products", []) or []

    in_range = all(50 <= p.get("price", 0) <= 150 for p in products)
    ok = len(products) >= 3 and in_range
    suite.record(TestResult(
        name="GET /api/products?min_price=50&max_price=150",
        passed=ok,
        latency_ms=meta.get("latency_ms", lat),
        plan_hit=meta.get("plan_executed", False),
        cached=meta.get("cached", False),
        detail=f"{len(products)} products in $50-$150" + ("" if in_range else " OUT OF RANGE"),
    ))


def test_sort_price_asc(suite: TestSuite):
    """Sort by price ascending."""
    resp, lat = _get("/api/products", {"sort": "price_asc"})
    meta = _meta(resp)
    data = resp.get("data", {}) or {}
    products = data.get("products", []) or []

    prices = [p.get("price", 0) for p in products]
    sorted_correctly = all(prices[i] <= prices[i + 1] for i in range(len(prices) - 1))
    ok = len(products) >= SEED_PRODUCT_COUNT and sorted_correctly
    suite.record(TestResult(
        name="GET /api/products?sort=price_asc",
        passed=ok,
        latency_ms=meta.get("latency_ms", lat),
        plan_hit=meta.get("plan_executed", False),
        cached=meta.get("cached", False),
        detail=f"{'sorted' if sorted_correctly else 'NOT sorted'}, first=${prices[0] if prices else '?'}",
    ))


def test_sort_price_desc(suite: TestSuite):
    """Sort by price descending."""
    resp, lat = _get("/api/products", {"sort": "price_desc"})
    meta = _meta(resp)
    data = resp.get("data", {}) or {}
    products = data.get("products", []) or []

    prices = [p.get("price", 0) for p in products]
    sorted_correctly = all(prices[i] >= prices[i + 1] for i in range(len(prices) - 1))
    ok = len(products) >= SEED_PRODUCT_COUNT and sorted_correctly
    suite.record(TestResult(
        name="GET /api/products?sort=price_desc",
        passed=ok,
        latency_ms=meta.get("latency_ms", lat),
        plan_hit=meta.get("plan_executed", False),
        cached=meta.get("cached", False),
        detail=f"{'sorted' if sorted_correctly else 'NOT sorted'}, first=${prices[0] if prices else '?'}",
    ))


def test_sort_name(suite: TestSuite):
    """Sort by name alphabetically."""
    resp, lat = _get("/api/products", {"sort": "name"})
    meta = _meta(resp)
    data = resp.get("data", {}) or {}
    products = data.get("products", []) or []

    names = [p.get("name", "") for p in products]
    sorted_correctly = all(names[i].lower() <= names[i + 1].lower() for i in range(len(names) - 1))
    ok = len(products) >= SEED_PRODUCT_COUNT and sorted_correctly
    suite.record(TestResult(
        name="GET /api/products?sort=name",
        passed=ok,
        latency_ms=meta.get("latency_ms", lat),
        plan_hit=meta.get("plan_executed", False),
        cached=meta.get("cached", False),
        detail=f"{'sorted' if sorted_correctly else 'NOT sorted'}, first='{names[0][:20] if names else '?'}'",
    ))


def test_pagination_limit(suite: TestSuite):
    """Pagination: limit=3."""
    resp, lat = _get("/api/products", {"limit": 3})
    meta = _meta(resp)
    data = resp.get("data", {}) or {}
    products = data.get("products", []) or []

    ok = len(products) == 3
    suite.record(TestResult(
        name="GET /api/products?limit=3",
        passed=ok,
        latency_ms=meta.get("latency_ms", lat),
        plan_hit=meta.get("plan_executed", False),
        cached=meta.get("cached", False),
        detail=f"{len(products)} returned",
    ))


def test_pagination_offset(suite: TestSuite):
    """Pagination: limit=5, offset=5."""
    resp, lat = _get("/api/products", {"limit": 5, "offset": 5})
    meta = _meta(resp)
    data = resp.get("data", {}) or {}
    products = data.get("products", []) or []

    ok = len(products) == 5
    suite.record(TestResult(
        name="GET /api/products?limit=5&offset=5",
        passed=ok,
        latency_ms=meta.get("latency_ms", lat),
        plan_hit=meta.get("plan_executed", False),
        cached=meta.get("cached", False),
        detail=f"{len(products)} returned (page 2)",
    ))


def test_combined_filter_sort_page(suite: TestSuite):
    """Combined: category + sort + pagination."""
    resp, lat = _get("/api/products", {
        "category": "electronics",
        "sort": "price_desc",
        "limit": 2,
    })
    meta = _meta(resp)
    data = resp.get("data", {}) or {}
    products = data.get("products", []) or []

    all_electronics = all(p.get("category") == "electronics" for p in products)
    prices = [p.get("price", 0) for p in products]
    sorted_desc = len(prices) < 2 or prices[0] >= prices[1]
    ok = len(products) == 2 and all_electronics and sorted_desc
    suite.record(TestResult(
        name="GET /api/products?category=electronics&sort=price_desc&limit=2",
        passed=ok,
        latency_ms=meta.get("latency_ms", lat),
        plan_hit=meta.get("plan_executed", False),
        cached=meta.get("cached", False),
        detail=f"{len(products)} products, prices={prices}",
    ))


def test_filter_nonexistent_category(suite: TestSuite):
    """Filter by non-existent category — should return empty."""
    resp, lat = _get("/api/products", {"category": "unicorns"})
    meta = _meta(resp)
    data = resp.get("data", {}) or {}
    products = data.get("products", []) or []

    ok = len(products) == 0
    suite.record(TestResult(
        name="GET /api/products?category=unicorns (empty result)",
        passed=ok,
        latency_ms=meta.get("latency_ms", lat),
        plan_hit=meta.get("plan_executed", False),
        cached=meta.get("cached", False),
        detail=f"{len(products)} products returned",
    ))


# ──────────────────────────────────────────────────────────────────────────
# 2. POST /api/products — create product
# ──────────────────────────────────────────────────────────────────────────

def test_create_product_valid(suite: TestSuite):
    """Create a valid product."""
    body = {
        "name": "Test Widget Alpha",
        "description": "A test product for automated testing",
        "price": 19.99,
        "category": "testing",
        "in_stock": True,
    }
    resp, lat = _post("/api/products", body)
    meta = _meta(resp)
    data = resp.get("data", {}) or {}

    has_id = "id" in data
    name_ok = "Test Widget" in str(data.get("name", "")) or "Test Widget" in str(data.get("message", ""))
    ok = has_id and data.get("price") in (19.99, "19.99")
    if has_id:
        suite._created_product_ids.append(data["id"])
    suite.record(TestResult(
        name="POST /api/products — valid product",
        passed=ok,
        latency_ms=meta.get("latency_ms", lat),
        plan_hit=meta.get("plan_executed", False),
        cached=meta.get("cached", False),
        detail=f"id={data.get('id')}, name='{data.get('name')}'",
    ))


def test_create_product_minimal(suite: TestSuite):
    """Create product with only required fields."""
    body = {"name": "Minimal Item", "price": 5.00, "category": "testing"}
    resp, lat = _post("/api/products", body)
    meta = _meta(resp)
    data = resp.get("data", {}) or {}

    has_id = "id" in data
    ok = has_id and data.get("name") == "Minimal Item"
    if has_id:
        suite._created_product_ids.append(data["id"])
    suite.record(TestResult(
        name="POST /api/products — minimal (required fields only)",
        passed=ok,
        latency_ms=meta.get("latency_ms", lat),
        plan_hit=meta.get("plan_executed", False),
        cached=meta.get("cached", False),
        detail=f"id={data.get('id')}",
    ))


def test_create_product_expensive(suite: TestSuite):
    """Create a high-price product."""
    body = {
        "name": "Diamond Encrusted Keyboard",
        "description": "Luxury mechanical keyboard with diamond keycaps",
        "price": 9999.99,
        "category": "luxury",
    }
    resp, lat = _post("/api/products", body)
    meta = _meta(resp)
    data = resp.get("data", {}) or {}

    ok = "id" in data and data.get("price") == 9999.99
    if "id" in data:
        suite._created_product_ids.append(data["id"])
    suite.record(TestResult(
        name="POST /api/products — high price ($9999.99)",
        passed=ok,
        latency_ms=meta.get("latency_ms", lat),
        plan_hit=meta.get("plan_executed", False),
        cached=meta.get("cached", False),
        detail=f"price={data.get('price')}",
    ))


def test_create_product_missing_name(suite: TestSuite):
    """Validation: missing name → 400."""
    body = {"price": 10.00, "category": "test"}
    resp, lat = _post("/api/products", body)
    meta = _meta(resp)

    data = resp.get("data", {}) or {}
    status = data.get("status_code", resp.get("status_code", 200))
    has_error = bool(resp.get("error")) or bool(data.get("error")) or status in (400, 422)
    ok = has_error
    suite.record(TestResult(
        name="POST /api/products — missing name → error",
        passed=ok,
        latency_ms=meta.get("latency_ms", lat),
        plan_hit=meta.get("plan_executed", False),
        cached=meta.get("cached", False),
        detail=f"status={status}, error={has_error}",
    ))


def test_create_product_negative_price(suite: TestSuite):
    """Validation: negative price → error."""
    body = {"name": "Bad Product", "price": -5.00, "category": "test"}
    resp, lat = _post("/api/products", body)
    meta = _meta(resp)

    data = resp.get("data", {}) or {}
    status = data.get("status_code", resp.get("status_code", 200))
    has_error = bool(resp.get("error")) or bool(data.get("error")) or status in (400, 422)
    ok = has_error
    suite.record(TestResult(
        name="POST /api/products — negative price → error",
        passed=ok,
        latency_ms=meta.get("latency_ms", lat),
        plan_hit=meta.get("plan_executed", False),
        cached=meta.get("cached", False),
        detail=f"status={status}, error={has_error}",
    ))


def test_create_product_missing_category(suite: TestSuite):
    """Validation: missing category → error."""
    body = {"name": "No Category", "price": 10.00}
    resp, lat = _post("/api/products", body)
    meta = _meta(resp)

    data = resp.get("data", {}) or {}
    status = data.get("status_code", resp.get("status_code", 200))
    has_error = bool(resp.get("error")) or bool(data.get("error")) or status in (400, 422)
    ok = has_error
    suite.record(TestResult(
        name="POST /api/products — missing category → error",
        passed=ok,
        latency_ms=meta.get("latency_ms", lat),
        plan_hit=meta.get("plan_executed", False),
        cached=meta.get("cached", False),
        detail=f"status={status}, error={has_error}",
    ))


def test_create_product_zero_price(suite: TestSuite):
    """Validation: zero price → error (price must be > 0)."""
    body = {"name": "Free Thing", "price": 0, "category": "test"}
    resp, lat = _post("/api/products", body)
    meta = _meta(resp)

    data = resp.get("data", {}) or {}
    status = data.get("status_code", resp.get("status_code", 200))
    has_error = bool(resp.get("error")) or bool(data.get("error")) or status in (400, 422)
    ok = has_error
    suite.record(TestResult(
        name="POST /api/products — zero price → error",
        passed=ok,
        latency_ms=meta.get("latency_ms", lat),
        plan_hit=meta.get("plan_executed", False),
        cached=meta.get("cached", False),
        detail=f"status={status}, error={has_error}",
    ))


# ──────────────────────────────────────────────────────────────────────────
# 3. GET /api/products/{id} — get by ID
# ──────────────────────────────────────────────────────────────────────────

def test_get_product_by_id_1(suite: TestSuite):
    """Get product ID 1 (Wireless Bluetooth Headphones)."""
    resp, lat = _get("/api/products/1")
    meta = _meta(resp)
    data = resp.get("data", {}) or {}

    ok = data.get("id") == 1 and "Headphones" in data.get("name", "")
    suite.record(TestResult(
        name="GET /api/products/1 — Bluetooth Headphones",
        passed=ok,
        latency_ms=meta.get("latency_ms", lat),
        plan_hit=meta.get("plan_executed", False),
        cached=meta.get("cached", False),
        detail=f"name='{data.get('name', '?')[:30]}'",
    ))


def test_get_product_by_id_7(suite: TestSuite):
    """Get product ID 7 (Espresso Machine)."""
    resp, lat = _get("/api/products/7")
    meta = _meta(resp)
    data = resp.get("data", {}) or {}

    ok = data.get("id") == 7 and data.get("price") == 249.99
    suite.record(TestResult(
        name="GET /api/products/7 — Espresso Machine",
        passed=ok,
        latency_ms=meta.get("latency_ms", lat),
        plan_hit=meta.get("plan_executed", False),
        cached=meta.get("cached", False),
        detail=f"price={data.get('price')}",
    ))


def test_get_product_by_id_12(suite: TestSuite):
    """Get product ID 12 (Noise-Cancelling Earbuds)."""
    resp, lat = _get("/api/products/12")
    meta = _meta(resp)
    data = resp.get("data", {}) or {}

    ok = data.get("id") == 12 and "Earbuds" in data.get("name", "")
    suite.record(TestResult(
        name="GET /api/products/12 — Earbuds",
        passed=ok,
        latency_ms=meta.get("latency_ms", lat),
        plan_hit=meta.get("plan_executed", False),
        cached=meta.get("cached", False),
        detail=f"name='{data.get('name', '?')[:30]}'",
    ))


def test_get_product_by_id_10_out_of_stock(suite: TestSuite):
    """Get product ID 10 (out of stock Vintage Vinyl)."""
    resp, lat = _get("/api/products/10")
    meta = _meta(resp)
    data = resp.get("data", {}) or {}

    ok = data.get("id") == 10 and data.get("in_stock") in (False, 0, "0", "false")
    suite.record(TestResult(
        name="GET /api/products/10 — out-of-stock Vinyl Player",
        passed=ok,
        latency_ms=meta.get("latency_ms", lat),
        plan_hit=meta.get("plan_executed", False),
        cached=meta.get("cached", False),
        detail=f"in_stock={data.get('in_stock')}",
    ))


def test_get_product_not_found(suite: TestSuite):
    """Get non-existent product ID 9999."""
    resp, lat = _get("/api/products/9999")
    meta = _meta(resp)

    data = resp.get("data", {}) or {}
    is_empty = data in ({}, None, [], [{}])
    status = resp.get("status_code", 200)
    has_error_msg = bool(resp.get("error"))
    # Accept: (1) 404 with error, (2) empty data, (3) data containing "not found" message
    not_found_text = "not found" in str(data).lower() or "not found" in str(resp.get("error", "")).lower()
    ok = is_empty or (status == 404) or not_found_text
    suite.record(TestResult(
        name="GET /api/products/9999 — not found",
        passed=ok,
        latency_ms=meta.get("latency_ms", lat),
        plan_hit=meta.get("plan_executed", False),
        cached=meta.get("cached", False),
        detail=f"status={status}, data_empty={is_empty}, not_found={not_found_text}",
    ))


def test_get_created_product(suite: TestSuite):
    """Get a product we just created via POST."""
    if not suite._created_product_ids:
        suite.record(TestResult(
            name="GET /api/products/{created_id} — read-after-write",
            passed=False, detail="no created products to verify",
        ))
        return

    pid = suite._created_product_ids[0]
    resp, lat = _get(f"/api/products/{pid}")
    meta = _meta(resp)
    data = resp.get("data", {}) or {}

    ok = data.get("id") == pid and data.get("name") == "Test Widget Alpha"
    suite.record(TestResult(
        name=f"GET /api/products/{pid} — read-after-write",
        passed=ok,
        latency_ms=meta.get("latency_ms", lat),
        plan_hit=meta.get("plan_executed", False),
        cached=meta.get("cached", False),
        detail=f"name='{data.get('name', '?')}'",
    ))


# ──────────────────────────────────────────────────────────────────────────
# 4. POST /api/orders/analyze — AI analytics
# ──────────────────────────────────────────────────────────────────────────

def test_analytics_top_selling(suite: TestSuite):
    """Analytics: top selling products."""
    resp, lat = _post("/api/orders/analyze", {
        "question": "What are the top 3 best-selling products by total quantity?"
    })
    meta = _meta(resp)
    data = resp.get("data", {}) or {}

    has_query = "query_used" in data or "sql" in str(data).lower()
    raw_data = data.get("raw_data", []) or []
    has_analysis = len(str(data.get("analysis", data.get("summary", "")))) > 5
    ok = len(raw_data) > 0 or has_analysis
    suite.record(TestResult(
        name="POST /api/orders/analyze — top selling products",
        passed=ok,
        latency_ms=meta.get("latency_ms", lat),
        plan_hit=meta.get("plan_executed", False),
        cached=meta.get("cached", False),
        detail=f"rows={len(raw_data)}, has_analysis={has_analysis}",
    ))


def test_analytics_revenue_by_category(suite: TestSuite):
    """Analytics: revenue by category."""
    resp, lat = _post("/api/orders/analyze", {
        "question": "What is the total revenue by product category?"
    })
    meta = _meta(resp)
    data = resp.get("data", {}) or {}

    raw_data = data.get("raw_data", []) or []
    has_analysis = len(str(data.get("analysis", data.get("summary", "")))) > 5
    ok = len(raw_data) >= 3 or has_analysis
    suite.record(TestResult(
        name="POST /api/orders/analyze — revenue by category",
        passed=ok,
        latency_ms=meta.get("latency_ms", lat),
        plan_hit=meta.get("plan_executed", False),
        cached=meta.get("cached", False),
        detail=f"{len(raw_data)} categories, analysis={has_analysis}",
    ))


def test_analytics_order_status(suite: TestSuite):
    """Analytics: order count by status."""
    resp, lat = _post("/api/orders/analyze", {
        "question": "How many orders are there for each status (pending, shipped, completed)?"
    })
    meta = _meta(resp)
    data = resp.get("data", {}) or {}

    raw_data = data.get("raw_data", []) or []
    has_analysis = len(str(data.get("analysis", data.get("summary", "")))) > 5
    ok = len(raw_data) >= 2 or has_analysis
    suite.record(TestResult(
        name="POST /api/orders/analyze — orders by status",
        passed=ok,
        latency_ms=meta.get("latency_ms", lat),
        plan_hit=meta.get("plan_executed", False),
        cached=meta.get("cached", False),
        detail=f"{len(raw_data)} statuses, analysis={has_analysis}",
    ))


def test_analytics_avg_order_value(suite: TestSuite):
    """Analytics: average order value."""
    resp, lat = _post("/api/orders/analyze", {
        "question": "What is the average order value across all orders?"
    })
    meta = _meta(resp)
    data = resp.get("data", {}) or {}

    raw_data = data.get("raw_data", []) or []
    has_analysis = len(str(data.get("analysis", data.get("summary", "")))) > 5
    ok = len(raw_data) >= 1 or has_analysis
    suite.record(TestResult(
        name="POST /api/orders/analyze — average order value",
        passed=ok,
        latency_ms=meta.get("latency_ms", lat),
        plan_hit=meta.get("plan_executed", False),
        cached=meta.get("cached", False),
        detail=f"raw_data={raw_data[:1]}, analysis={has_analysis}",
    ))


def test_analytics_top_customers(suite: TestSuite):
    """Analytics: top customers by spending."""
    resp, lat = _post("/api/orders/analyze", {
        "question": "Who are the top 3 customers by total spending?"
    })
    meta = _meta(resp)
    data = resp.get("data", {}) or {}

    raw_data = data.get("raw_data", []) or []
    has_analysis = len(str(data.get("analysis", data.get("summary", "")))) > 5
    ok = len(raw_data) >= 1 or has_analysis
    suite.record(TestResult(
        name="POST /api/orders/analyze — top customers",
        passed=ok,
        latency_ms=meta.get("latency_ms", lat),
        plan_hit=meta.get("plan_executed", False),
        cached=meta.get("cached", False),
        detail=f"{len(raw_data)} customers, analysis={has_analysis}",
    ))


def test_analytics_pending_orders(suite: TestSuite):
    """Analytics: pending orders summary."""
    resp, lat = _post("/api/orders/analyze", {
        "question": "How many orders are pending and what is their total value?"
    })
    meta = _meta(resp)
    data = resp.get("data", {}) or {}

    raw_data = data.get("raw_data", []) or []
    has_analysis = len(str(data.get("analysis", data.get("summary", "")))) > 5
    ok = len(raw_data) >= 1 or has_analysis
    suite.record(TestResult(
        name="POST /api/orders/analyze — pending orders summary",
        passed=ok,
        latency_ms=meta.get("latency_ms", lat),
        plan_hit=meta.get("plan_executed", False),
        cached=meta.get("cached", False),
        detail=f"data={raw_data[:1]}, analysis={has_analysis}",
    ))


# ──────────────────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────────────────

def main():
    suite = TestSuite()

    # Verify server is reachable
    try:
        httpx.get(f"{BASE_URL}/health", timeout=5)
    except Exception as e:
        print(f"ERROR: Cannot reach server at {BASE_URL}: {e}")
        sys.exit(1)

    # ── Section 1: GET /api/products (list/filter/sort/paginate) ────────
    print(f"\n{'─' * 80}")
    print("  1. GET /api/products — List / Filter / Sort / Paginate")
    print(f"{'─' * 80}")

    tests_list = [
        test_list_all,
        test_filter_category_electronics,
        test_filter_category_food,
        test_filter_category_fitness,
        test_filter_in_stock,
        test_filter_electronics_in_stock,
        test_filter_min_price,
        test_filter_max_price,
        test_filter_price_range,
        test_sort_price_asc,
        test_sort_price_desc,
        test_sort_name,
        test_pagination_limit,
        test_pagination_offset,
        test_combined_filter_sort_page,
        test_filter_nonexistent_category,
    ]
    for t in tests_list:
        try:
            t(suite)
        except Exception as e:
            suite.record(TestResult(name=t.__name__, passed=False, detail=f"EXCEPTION: {e}"))

    # ── Section 2: POST /api/products (create / validate) ──────────────
    print(f"\n{'─' * 80}")
    print("  2. POST /api/products — Create / Validate")
    print(f"{'─' * 80}")

    tests_create = [
        test_create_product_valid,
        test_create_product_minimal,
        test_create_product_expensive,
        test_create_product_missing_name,
        test_create_product_negative_price,
        test_create_product_missing_category,
        test_create_product_zero_price,
    ]
    for t in tests_create:
        try:
            t(suite)
        except Exception as e:
            suite.record(TestResult(name=t.__name__, passed=False, detail=f"EXCEPTION: {e}"))

    # ── Section 3: GET /api/products/{id} (lookup) ─────────────────────
    print(f"\n{'─' * 80}")
    print("  3. GET /api/products/{id} — Lookup by ID")
    print(f"{'─' * 80}")

    tests_get_id = [
        test_get_product_by_id_1,
        test_get_product_by_id_7,
        test_get_product_by_id_12,
        test_get_product_by_id_10_out_of_stock,
        test_get_product_not_found,
        test_get_created_product,
    ]
    for t in tests_get_id:
        try:
            t(suite)
        except Exception as e:
            suite.record(TestResult(name=t.__name__, passed=False, detail=f"EXCEPTION: {e}"))

    # ── Section 4: POST /api/orders/analyze (analytics) ────────────────
    print(f"\n{'─' * 80}")
    print("  4. POST /api/orders/analyze — AI Analytics")
    print(f"{'─' * 80}")

    tests_analytics = [
        test_analytics_top_selling,
        test_analytics_revenue_by_category,
        test_analytics_order_status,
        test_analytics_avg_order_value,
        test_analytics_top_customers,
        test_analytics_pending_orders,
    ]
    for t in tests_analytics:
        try:
            t(suite)
        except Exception as e:
            suite.record(TestResult(name=t.__name__, passed=False, detail=f"EXCEPTION: {e}"))

    # ── Final summary ──────────────────────────────────────────────────
    all_passed = suite.summary()
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
