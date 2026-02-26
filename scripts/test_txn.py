"""Test transaction approaches with the databases library."""
import asyncio


async def test_raw_sql_txn():
    """Test using raw BEGIN/COMMIT via databases library."""
    import databases
    db = databases.Database("sqlite:///data/catalyst.db")
    await db.connect()

    print("=== Test 1: Raw SQL BEGIN/COMMIT ===")
    try:
        await db.execute(query="BEGIN IMMEDIATE")
        print("  BEGIN IMMEDIATE: OK")
        rows = await db.fetch_all(query="SELECT id, balance FROM accounts WHERE id = 2")
        print(f"  SELECT: balance = {rows[0]._mapping['balance']}")
        await db.execute(query="UPDATE accounts SET balance = balance + 0.01 WHERE id = 2")
        print("  UPDATE: OK")
        await db.execute(query="COMMIT")
        print("  COMMIT: OK")
        # Undo
        await db.execute(query="UPDATE accounts SET balance = balance - 0.01 WHERE id = 2")
        print("  Undo: OK")
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")
        try:
            await db.execute(query="ROLLBACK")
        except:
            pass

    print("\n=== Test 2: databases Transaction API ===")
    try:
        txn = db.transaction()
        await txn.start()
        print("  transaction.start(): OK")
        rows = await db.fetch_all(query="SELECT id, balance FROM accounts WHERE id = 2")
        print(f"  SELECT: balance = {rows[0]._mapping['balance']}")
        await db.execute(query="UPDATE accounts SET balance = balance + 0.01 WHERE id = 2")
        print("  UPDATE: OK")
        await txn.commit()
        print("  transaction.commit(): OK")
        # Undo
        await db.execute(query="UPDATE accounts SET balance = balance - 0.01 WHERE id = 2")
        print("  Undo: OK")
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")
        try:
            await txn.rollback()
        except:
            pass

    print("\n=== Test 3: constraint failure + rollback ===")
    try:
        txn = db.transaction()
        await txn.start()
        print("  transaction.start(): OK")
        await db.execute(query="UPDATE accounts SET balance = balance + 100 WHERE id = 1")
        print("  UPDATE (Alice +100): OK")
        try:
            await db.execute(query="UPDATE accounts SET balance = balance + 100 WHERE id = 6")
            print("  UPDATE (Frank +100): OK (SHOULD HAVE FAILED!)")
        except Exception as inner:
            print(f"  UPDATE (Frank): raised {type(inner).__name__}: {inner}")
            print("  Rolling back...")
            await txn.rollback()
            print("  Rollback: OK")
            # Check Alice's balance wasn't changed
            rows = await db.fetch_all(query="SELECT balance FROM accounts WHERE id = 1")
            print(f"  Alice balance after rollback: {rows[0]._mapping['balance']} (should be 5240.5)")
            await db.disconnect()
            return
        await txn.commit()
        print("  COMMIT: OK (shouldn't get here)")
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")

    await db.disconnect()

asyncio.run(test_raw_sql_txn())
