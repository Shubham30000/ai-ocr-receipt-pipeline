"""
Stage 3: Automated Pipeline Watcher
Polls a folder for new receipt images and processes them incrementally.
Implements the same pattern as an Airflow DAG with idempotency checks.

To use real Airflow locally:
    docker run -p 8080:8080 apache/airflow standalone
    Then convert watch_folder() into a DAG with a FileSensor trigger.
"""

import os
import time
import sqlite3
import json
from pathlib import Path

# Import your existing pipeline
from pipeline import process_receipt

DB_PATH = "/content/drive/MyDrive/AI_OCR_dataset/receipts.db"


def get_processed_files(conn):
    """Idempotency check — fetch files already in DB"""
    cursor = conn.cursor()
    cursor.execute("SELECT file_name FROM receipts")
    return set(row[0] for row in cursor.fetchall())


def insert_single_record(conn, record):
    """Insert one processed receipt into DB"""
    cursor = conn.cursor()

    def safe_float(v):
        try:
            return float(str(v).replace(',', '').replace('RM', '').strip())
        except:
            return None

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

    conn.commit()


def watch_folder(folder_path, check_interval=10, max_checks=3):
    """
    Core watcher loop — equivalent to an Airflow DAG with:
      - FileSensor: detects new images in folder
      - PythonOperator: runs OCR + extraction
      - PythonOperator: inserts result into DB
      - Idempotency: skips already-processed files
    """
    print(f"👁️  Watching: {folder_path}")
    print(f"⏱️  Interval: {check_interval}s | Max checks: {max_checks}\n")

    conn = sqlite3.connect(DB_PATH)
    already_processed = get_processed_files(conn)
    print(f"📦 Already in DB: {len(already_processed)} files\n")

    new_count = 0
    for check in range(1, max_checks + 1):
        all_images = set(
            f for f in os.listdir(folder_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        )
        new_files = all_images - already_processed

        if new_files:
            print(f"[Check {check}] 🆕 {len(new_files)} new file(s) found")
            for fname in new_files:
                img_path = os.path.join(folder_path, fname)
                print(f"  🔄 Processing: {fname}")
                record = process_receipt(img_path)
                insert_single_record(conn, record)
                already_processed.add(fname)
                new_count += 1
                print(f"  ✅ Done: {fname}")
        else:
            print(f"[Check {check}/{max_checks}] No new files.")

        if check < max_checks:
            time.sleep(check_interval)

    conn.close()
    print(f"\n✅ Watcher done. Processed {new_count} new receipts.")
    print("ℹ️  In production: remove max_checks to run indefinitely,")
    print("   or deploy as an Airflow DAG using FileSensor + PythonOperator.")


if __name__ == "__main__":
    WATCH_PATH = "/content/drive/MyDrive/AI_OCR_dataset"
    watch_folder(WATCH_PATH, check_interval=5, max_checks=3)
