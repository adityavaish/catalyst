"""Debug script: test plan cache behavior for transaction type filter."""
import requests
import json

BASE = "http://127.0.0.1:8000"

# First call: type=deposit → populate plan cache
r1 = requests.get(f"{BASE}/api/transactions", params={"type": "deposit"})
d1 = r1.json()
m1 = d1.get("meta", {})
t1 = d1.get("data", {}).get("transactions", [])
print(f"[1] type=deposit: count={len(t1)} plan={m1.get('plan_executed')} cached={m1.get('cached')}")

# Second call: type=transfer → should reuse plan
r2 = requests.get(f"{BASE}/api/transactions", params={"type": "transfer"})
d2 = r2.json()
m2 = d2.get("meta", {})
t2 = d2.get("data", {}).get("transactions", [])
print(f"[2] type=transfer: count={len(t2)} plan={m2.get('plan_executed')} cached={m2.get('cached')}")

if len(t2) < 10:
    print(f"  ISSUE: only {len(t2)} transfers (expected 13)")
    for t in t2:
        print(f"    id={t.get('id')} type={t.get('type')} amount={t.get('amount')}")
else:
    print(f"  OK: {len(t2)} transfers returned")

# Third call: type=transfer again → check consistency
r3 = requests.get(f"{BASE}/api/transactions", params={"type": "transfer"})
d3 = r3.json()
m3 = d3.get("meta", {})
t3 = d3.get("data", {}).get("transactions", [])
print(f"[3] type=transfer (retry): count={len(t3)} plan={m3.get('plan_executed')} cached={m3.get('cached')}")
