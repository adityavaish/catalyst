"""Debug script to test plan cache behavior with filter params."""
import requests
import json

BASE = "http://127.0.0.1:8000"

# First, get status=active to trigger plan creation
print("=== GET /api/accounts?status=active ===")
r = requests.get(f"{BASE}/api/accounts", params={"status": "active"}, timeout=60)
d = r.json()
meta = d.get("meta", {})
accounts = d.get("data", {}).get("accounts", [])
print(f"  count={len(accounts)}, plan_cached={meta.get('plan_cached')}, plan_exec={meta.get('plan_executed')}")

# Then status=frozen — should replay plan
print("\n=== GET /api/accounts?status=frozen ===")
r2 = requests.get(f"{BASE}/api/accounts", params={"status": "frozen"}, timeout=60)
d2 = r2.json()
meta2 = d2.get("meta", {})
accounts2 = d2.get("data", {}).get("accounts", [])
print(f"  count={len(accounts2)}, plan_exec={meta2.get('plan_executed')}")
for a in accounts2:
    print(f"    {json.dumps(a)}")

# Check: does the response have the right number of items?
print(f"\nExpected: 1 frozen account (Frank Wilson)")
print(f"Got: {len(accounts2)} accounts")
