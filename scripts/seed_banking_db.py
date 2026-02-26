"""
Seed the SQLite database with banking system data.

Preserves existing e-commerce tables and adds banking tables alongside them.

Usage:
    python scripts/seed_banking_db.py
"""
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "catalyst.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

conn = sqlite3.connect(str(DB_PATH))
cur = conn.cursor()

# ── Create banking tables (drop only banking-specific ones) ─────────────
cur.executescript("""
DROP TABLE IF EXISTS transactions;
DROP TABLE IF EXISTS accounts;

CREATE TABLE accounts (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    account_number  TEXT    NOT NULL UNIQUE,
    holder_name     TEXT    NOT NULL,
    email           TEXT    NOT NULL,
    account_type    TEXT    NOT NULL CHECK(account_type IN ('checking', 'savings', 'business')),
    balance         REAL    NOT NULL DEFAULT 0.00 CHECK(balance >= 0),
    currency        TEXT    NOT NULL DEFAULT 'USD',
    status          TEXT    NOT NULL DEFAULT 'active' CHECK(status IN ('active', 'frozen', 'closed')),
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE transactions (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    from_account_id   INTEGER,
    to_account_id     INTEGER,
    amount            REAL    NOT NULL CHECK(amount > 0),
    currency          TEXT    NOT NULL DEFAULT 'USD',
    type              TEXT    NOT NULL CHECK(type IN ('transfer', 'deposit', 'withdrawal')),
    description       TEXT,
    status            TEXT    NOT NULL DEFAULT 'completed' CHECK(status IN ('completed', 'pending', 'failed')),
    reference_number  TEXT    NOT NULL UNIQUE,
    created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (from_account_id) REFERENCES accounts(id),
    FOREIGN KEY (to_account_id)   REFERENCES accounts(id)
);

-- Prevent any UPDATE/INSERT on frozen or closed accounts.
-- This is the HARD guardrail: even if the LLM tries to modify a
-- frozen/closed account, the database itself will reject it.
CREATE TRIGGER prevent_frozen_account_update
BEFORE UPDATE OF balance ON accounts
WHEN NEW.balance != OLD.balance AND OLD.status != 'active'
BEGIN
    SELECT RAISE(ABORT, 'BANK_ERR: Cannot modify balance of a non-active account (frozen or closed)');
END;

-- Prevent deposits to non-active accounts via transactions table
CREATE TRIGGER prevent_transaction_to_frozen
BEFORE INSERT ON transactions
WHEN NEW.to_account_id IS NOT NULL
     AND (SELECT status FROM accounts WHERE id = NEW.to_account_id) != 'active'
BEGIN
    SELECT RAISE(ABORT, 'BANK_ERR: Cannot create transaction targeting a non-active account');
END;

-- Prevent transfers from non-active accounts via transactions table
CREATE TRIGGER prevent_transaction_from_frozen
BEFORE INSERT ON transactions
WHEN NEW.from_account_id IS NOT NULL
     AND (SELECT status FROM accounts WHERE id = NEW.from_account_id) != 'active'
BEGIN
    SELECT RAISE(ABORT, 'BANK_ERR: Cannot create transaction from a non-active account');
END;
""")

# Enable foreign key enforcement
cur.execute("PRAGMA foreign_keys = ON;")

# ── Seed accounts ───────────────────────────────────────────────────────
# 8 accounts across 3 types, various balances, 1 frozen, 1 closed
accounts = [
    # (account_number, holder_name, email, account_type, balance, currency, status)
    ("ACC-1001", "Alice Johnson",   "alice@example.com",   "checking",  5240.50,  "USD", "active"),
    ("ACC-1002", "Bob Smith",       "bob@example.com",     "savings",   12800.00, "USD", "active"),
    ("ACC-1003", "Charlie Brown",   "charlie@example.com", "checking",  890.75,   "USD", "active"),
    ("ACC-1004", "Diana Prince",    "diana@example.com",   "business",  45000.00, "USD", "active"),
    ("ACC-1005", "Eve Martinez",    "eve@example.com",     "savings",   3200.00,  "USD", "active"),
    ("ACC-1006", "Frank Wilson",    "frank@example.com",   "checking",  150.25,   "USD", "frozen"),
    ("ACC-1007", "Grace Lee",       "grace@example.com",   "business",  28750.00, "USD", "active"),
    ("ACC-1008", "Heidi Clark",     "heidi@example.com",   "savings",   0.00,     "USD", "closed"),
]

cur.executemany(
    "INSERT INTO accounts (account_number, holder_name, email, account_type, balance, currency, status) VALUES (?, ?, ?, ?, ?, ?, ?)",
    accounts,
)

# ── Seed transactions ──────────────────────────────────────────────────
# Mix of transfers, deposits, and withdrawals with known reference numbers
transactions = [
    # (from_account_id, to_account_id, amount, currency, type, description, status, reference_number)
    (1, 2,    500.00, "USD", "transfer",   "Rent payment",          "completed", "TXN-0001"),
    (2, 3,    200.00, "USD", "transfer",   "Loan repayment",        "completed", "TXN-0002"),
    (4, 1,   1500.00, "USD", "transfer",   "Salary payment",        "completed", "TXN-0003"),
    (None, 1, 2000.00, "USD", "deposit",   "Cash deposit",          "completed", "TXN-0004"),
    (None, 5, 1000.00, "USD", "deposit",   "Birthday gift",         "completed", "TXN-0005"),
    (3, None,  50.00,  "USD", "withdrawal", "ATM withdrawal",       "completed", "TXN-0006"),
    (1, 4,    750.00, "USD", "transfer",   "Invoice #1234 payment", "completed", "TXN-0007"),
    (5, 3,    300.00, "USD", "transfer",   "Shared dinner",         "completed", "TXN-0008"),
    (None, 7, 5000.00, "USD", "deposit",   "Client payment",        "completed", "TXN-0009"),
    (7, 4,   2500.00, "USD", "transfer",   "Partnership draw",      "completed", "TXN-0010"),
    (1, 5,    100.00, "USD", "transfer",   "Gift",                  "pending",   "TXN-0011"),
    (4, 7,   3000.00, "USD", "transfer",   "Investment capital",    "pending",   "TXN-0012"),
    (3, 1,     75.00, "USD", "transfer",   "Coffee reimbursement",  "failed",    "TXN-0013"),
    (None, 2, 500.00, "USD", "deposit",    "Interest payment",      "completed", "TXN-0014"),
    (2, 1,    250.00, "USD", "transfer",   "Birthday present",      "completed", "TXN-0015"),
]

cur.executemany(
    "INSERT INTO transactions (from_account_id, to_account_id, amount, currency, type, description, status, reference_number) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
    transactions,
)

conn.commit()

# ── Verify ──────────────────────────────────────────────────────────────
account_count = cur.execute("SELECT COUNT(*) FROM accounts").fetchone()[0]
txn_count     = cur.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
total_balance = cur.execute("SELECT SUM(balance) FROM accounts").fetchone()[0]

print(f"Banking database seeded at {DB_PATH}")
print(f"  accounts:      {account_count} rows")
print(f"  transactions:  {txn_count} rows")
print(f"  total balance: ${total_balance:,.2f}")

# Quick verification of account balances
print("\n  Account balances:")
for row in cur.execute("SELECT id, account_number, holder_name, balance, status FROM accounts ORDER BY id"):
    print(f"    {row[0]:2d}. {row[1]} — {row[2]:<18s} ${row[3]:>10,.2f}  [{row[4]}]")

conn.close()
