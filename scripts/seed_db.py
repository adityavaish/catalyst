"""
Seed the SQLite database with sample e-commerce data.

Usage:
    python scripts/seed_db.py
"""
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "catalyst.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

conn = sqlite3.connect(str(DB_PATH))
cur = conn.cursor()

# ── Create tables ───────────────────────────────────────────────────────
cur.executescript("""
DROP TABLE IF EXISTS orders;
DROP TABLE IF EXISTS products;

CREATE TABLE products (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL,
    description TEXT,
    price       REAL    NOT NULL,
    category    TEXT    NOT NULL,
    in_stock    BOOLEAN NOT NULL DEFAULT 1,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE orders (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    product_id      INTEGER NOT NULL,
    quantity        INTEGER NOT NULL,
    total_price     REAL    NOT NULL,
    customer_email  TEXT    NOT NULL,
    status          TEXT    NOT NULL DEFAULT 'pending',
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (product_id) REFERENCES products(id)
);
""")

# ── Seed products ───────────────────────────────────────────────────────
products = [
    ("Wireless Bluetooth Headphones",  "Premium noise-cancelling headphones with 30hr battery",   79.99,  "electronics", True),
    ("Organic Green Tea (100 bags)",   "Japanese sencha green tea, USDA organic certified",       14.99,  "food",        True),
    ("Python Programming Book",        "Comprehensive guide to Python 3.12 — 800 pages",         49.99,  "books",       True),
    ("Standing Desk Converter",        "Adjustable sit-stand desk riser, fits 32\" monitors",    189.99,  "furniture",   True),
    ("Mechanical Keyboard",            "Cherry MX Brown switches, RGB backlit, USB-C",            129.99, "electronics", True),
    ("Yoga Mat (6mm)",                 "Non-slip TPE yoga mat, eco-friendly, carry strap",        29.99,  "fitness",     True),
    ("Espresso Machine",               "15-bar pump, milk frother, stainless steel",             249.99,  "appliances",  True),
    ("LED Desk Lamp",                  "Touch control, 5 brightness levels, USB charging port",   34.99,  "electronics", True),
    ("Hiking Backpack (40L)",          "Waterproof, ergonomic frame, rain cover included",        89.99,  "outdoors",    True),
    ("Vintage Vinyl Record Player",    "Belt-drive turntable, built-in speakers, Bluetooth",     149.99,  "electronics", False),
    ("Stainless Steel Water Bottle",   "Vacuum insulated, keeps drinks cold 24hr/hot 12hr",      24.99,  "fitness",     True),
    ("Noise-Cancelling Earbuds",       "True wireless, ANC, 8hr battery, IPX4 waterproof",       59.99,  "electronics", True),
]

cur.executemany(
    "INSERT INTO products (name, description, price, category, in_stock) VALUES (?, ?, ?, ?, ?)",
    products,
)

# ── Seed orders ─────────────────────────────────────────────────────────
orders = [
    (1, 2, 159.98, "alice@example.com",   "completed"),
    (3, 1,  49.99, "bob@example.com",     "completed"),
    (5, 1, 129.99, "alice@example.com",   "completed"),
    (7, 1, 249.99, "charlie@example.com", "completed"),
    (2, 3,  44.97, "diana@example.com",   "shipped"),
    (4, 1, 189.99, "bob@example.com",     "shipped"),
    (1, 1,  79.99, "eve@example.com",     "pending"),
    (6, 2,  59.98, "frank@example.com",   "pending"),
    (8, 1,  34.99, "alice@example.com",   "completed"),
    (9, 1,  89.99, "charlie@example.com", "shipped"),
    (12, 2, 119.98, "diana@example.com",  "completed"),
    (5, 1, 129.99, "george@example.com",  "completed"),
    (1, 3, 239.97, "heidi@example.com",   "pending"),
    (11, 4,  99.96, "ivan@example.com",   "completed"),
    (3, 2,  99.98, "charlie@example.com", "shipped"),
]

cur.executemany(
    "INSERT INTO orders (product_id, quantity, total_price, customer_email, status) VALUES (?, ?, ?, ?, ?)",
    orders,
)

conn.commit()

# ── Verify ──────────────────────────────────────────────────────────────
product_count = cur.execute("SELECT COUNT(*) FROM products").fetchone()[0]
order_count   = cur.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
print(f"Database seeded at {DB_PATH}")
print(f"  products: {product_count} rows")
print(f"  orders:   {order_count} rows")

conn.close()
