"""
Stage 2: SQLite Storage + Analytics Layer
Loads OCR-extracted records into SQLite and runs analytical SQL queries.
"""

import sqlite3
import os
import json
import pandas as pd

DB_PATH = "/content/drive/MyDrive/AI_OCR_dataset/receipts.db"


def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return sqlite3.connect(DB_PATH)


def create_tables(conn):
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS receipts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_name TEXT NOT NULL,
            store_name TEXT,
            store_confidence REAL,
            receipt_date TEXT,
            date_confidence REAL,
            total_amount REAL,
            total_confidence REAL,
            has_low_confidence_flags INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS line_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            receipt_id INTEGER,
            item_name TEXT,
            price REAL,
            item_confidence REAL,
            FOREIGN KEY (receipt_id) REFERENCES receipts(id)
        )
    """)
    conn.commit()
    print("✅ Tables created")


def safe_float(value):
    try:
        return float(str(value).replace(',', '').replace('RM', '').strip())
    except:
        return None


def insert_records(conn, all_records):
    cursor = conn.cursor()
    inserted, skipped = 0, 0
    for record in all_records:
        try:
            store_name = record.get('store_name', {}).get('value')
            store_conf = record.get('store_name', {}).get('confidence', 0.0)
            date_val   = record.get('date', {}).get('value')
            date_conf  = record.get('date', {}).get('confidence', 0.0)
            total_val  = safe_float(record.get('total_amount', {}).get('value'))
            total_conf = record.get('total_amount', {}).get('confidence', 0.0)
            has_flags  = 1 if record.get('low_confidence_flags') else 0

            cursor.execute("""
                INSERT INTO receipts
                (file_name, store_name, store_confidence, receipt_date,
                 date_confidence, total_amount, total_confidence, has_low_confidence_flags)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (record['file'], store_name, store_conf, date_val,
                  date_conf, total_val, total_conf, has_flags))

            receipt_id = cursor.lastrowid
            for item in record.get('items', []):
                cursor.execute("""
                    INSERT INTO line_items (receipt_id, item_name, price, item_confidence)
                    VALUES (?, ?, ?, ?)
                """, (receipt_id, item.get('name'),
                      safe_float(item.get('price')), item.get('confidence', 0.0)))
            inserted += 1
        except Exception as e:
            skipped += 1
            print(f"Skipped {record.get('file', 'unknown')}: {e}")

    conn.commit()
    print(f"✅ Inserted: {inserted} | ⚠️ Skipped: {skipped}")


def run_analytics(conn):
    queries = {
        "01_overall_summary": """
            SELECT COUNT(*) as total_receipts,
                   ROUND(SUM(total_amount), 2) as total_spend,
                   ROUND(AVG(total_amount), 2) as avg_transaction
            FROM receipts WHERE total_amount IS NOT NULL
        """,
        "02_top_stores_by_spend": """
            SELECT store_name, COUNT(*) as visit_count,
                   ROUND(SUM(total_amount), 2) as total_spend,
                   ROUND(AVG(total_amount), 2) as avg_spend
            FROM receipts
            WHERE store_name IS NOT NULL AND total_amount IS NOT NULL
            GROUP BY store_name ORDER BY total_spend DESC LIMIT 10
        """,
        "03_store_visit_frequency": """
            SELECT store_name, COUNT(*) as visits,
                   RANK() OVER (ORDER BY COUNT(*) DESC) as rank
            FROM receipts WHERE store_name IS NOT NULL
            GROUP BY store_name ORDER BY visits DESC LIMIT 10
        """,
        "04_missing_totals": """
            SELECT COUNT(*) as missing_totals
            FROM receipts WHERE total_amount IS NULL
        """,
        "05_top_transactions": """
            SELECT file_name, store_name, total_amount, receipt_date
            FROM receipts WHERE total_amount IS NOT NULL
            ORDER BY total_amount DESC LIMIT 5
        """,
        "06_spend_buckets": """
            SELECT store_name,
                   CASE
                       WHEN total_amount < 20  THEN 'Under 20'
                       WHEN total_amount <= 50 THEN '20-50'
                       WHEN total_amount <= 100 THEN '50-100'
                       ELSE 'Over 100'
                   END as spend_bucket,
                   COUNT(*) as count
            FROM receipts
            WHERE total_amount IS NOT NULL AND store_name IS NOT NULL
            GROUP BY store_name, spend_bucket
            ORDER BY store_name
        """,
        "07_top_line_items": """
            SELECT item_name, COUNT(*) as frequency,
                   ROUND(AVG(price), 2) as avg_price
            FROM line_items
            WHERE item_name IS NOT NULL AND price IS NOT NULL
            GROUP BY item_name ORDER BY frequency DESC LIMIT 10
        """,
        "08_store_join_summary": """
            SELECT r.store_name, COUNT(DISTINCT r.id) as receipt_count,
                   COUNT(l.id) as total_items,
                   ROUND(AVG(l.price), 2) as avg_item_price
            FROM receipts r JOIN line_items l ON r.id = l.receipt_id
            WHERE r.store_name IS NOT NULL
            GROUP BY r.store_name ORDER BY total_items DESC LIMIT 10
        """,
        "09_confidence_breakdown": """
            SELECT has_low_confidence_flags,
                   COUNT(*) as count,
                   ROUND(AVG(total_amount), 2) as avg_amount
            FROM receipts GROUP BY has_low_confidence_flags
        """,
        "10_running_total_per_store": """
            SELECT store_name, total_amount,
                   ROUND(SUM(total_amount) OVER (
                       PARTITION BY store_name ORDER BY id
                   ), 2) as running_total
            FROM receipts
            WHERE store_name IS NOT NULL AND total_amount IS NOT NULL
            ORDER BY store_name, id LIMIT 20
        """
    }

    export_dir = "/content/drive/MyDrive/AI_OCR_dataset/analytics_exports"
    os.makedirs(export_dir, exist_ok=True)

    for name, query in queries.items():
        df = pd.read_sql_query(query, conn)
        print(f"\n📊 {name}\n{df.to_string(index=False)}")
        df.to_csv(os.path.join(export_dir, f"{name}.csv"), index=False)

    print(f"\n✅ All analytics saved to {export_dir}")


if __name__ == "__main__":
    OUTPUT_PATH = "/content/drive/MyDrive/AI_OCR_dataset/outputs"
    with open(os.path.join(OUTPUT_PATH, "all_receipts.json")) as f:
        all_records = json.load(f)

    conn = get_connection()
    create_tables(conn)
    insert_records(conn, all_records)
    run_analytics(conn)
    conn.close()
