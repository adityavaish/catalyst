"""Quick test that all DB constraints and triggers work."""
import sqlite3

conn = sqlite3.connect("data/catalyst.db")
cur = conn.cursor()

print("=== Test 1: CHECK(balance >= 0) — insufficient funds ===")
try:
    cur.execute("UPDATE accounts SET balance = balance - 999999 WHERE id = 3")
    print("  FAIL: update went through")
except Exception as e:
    print(f"  PASS: {e}")
conn.rollback()

print("\n=== Test 2: Trigger on frozen account balance UPDATE ===")
try:
    cur.execute("UPDATE accounts SET balance = balance + 100 WHERE id = 6")
    print("  FAIL: update went through on frozen acct")
except Exception as e:
    print(f"  PASS: {e}")
conn.rollback()

print("\n=== Test 3: Trigger on INSERT txn to frozen account ===")
try:
    cur.execute(
        "INSERT INTO transactions (from_account_id, to_account_id, amount, currency, type, description, status, reference_number)"
        " VALUES (1, 6, 100, 'USD', 'transfer', 'test', 'completed', 'TEST-001')"
    )
    print("  FAIL: insert went through")
except Exception as e:
    print(f"  PASS: {e}")
conn.rollback()

print("\n=== Test 4: CHECK(amount > 0) — negative amount ===")
try:
    cur.execute(
        "INSERT INTO transactions (from_account_id, to_account_id, amount, currency, type, description, status, reference_number)"
        " VALUES (NULL, 1, -100, 'USD', 'deposit', 'test', 'completed', 'TEST-002')"
    )
    print("  FAIL: insert went through")
except Exception as e:
    print(f"  PASS: {e}")
conn.rollback()

print("\n=== Test 5: Valid deposit on active account (should succeed) ===")
try:
    cur.execute("UPDATE accounts SET balance = balance + 100 WHERE id = 1")
    print("  PASS: update succeeded")
    conn.rollback()
except Exception as e:
    print(f"  FAIL: {e}")

conn.close()
print("\nAll constraint tests complete.")
