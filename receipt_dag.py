from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.filesystem import FileSensor
from datetime import datetime, timedelta
import os
import sqlite3
import json

DB_PATH = "/opt/airflow/receipts.db"
WATCH_FOLDER = "/opt/airflow/receipts_inbox"
OUTPUT_FOLDER = "/opt/airflow/receipts_processed"

default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

def check_new_images(**context):
    """Scan inbox for unprocessed images"""
    os.makedirs(WATCH_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Get already processed files
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS receipts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_name TEXT NOT NULL,
            store_name TEXT,
            total_amount REAL,
            processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    cursor.execute("SELECT file_name FROM receipts")
    done = set(r[0] for r in cursor.fetchall())
    conn.close()

    new_files = [
        f for f in os.listdir(WATCH_FOLDER)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        and f not in done
    ]
    print(f"Found {len(new_files)} new files: {new_files}")
    context['ti'].xcom_push(key='new_files', value=new_files)

def process_images(**context):
    """Run OCR + insert into DB for each new file"""
    new_files = context['ti'].xcom_pull(key='new_files', task_ids='check_new_images')

    if not new_files:
        print("No new files to process.")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    for fname in new_files:
        img_path = os.path.join(WATCH_FOLDER, fname)
        print(f"Processing: {fname}")

        # Simulate extraction (replace with your real pipeline.process_receipt call)
        # In real use: from pipeline import process_receipt; record = process_receipt(img_path)
        record = {
            "file": fname,
            "store_name": {"value": "Test Store", "confidence": 0.9},
            "total_amount": {"value": "25.50", "confidence": 0.85}
        }

        cursor.execute("""
            INSERT INTO receipts (file_name, store_name, total_amount)
            VALUES (?, ?, ?)
        """, (
            record['file'],
            record['store_name']['value'],
            float(record['total_amount']['value'])
        ))

        # Move to processed folder
        os.rename(img_path, os.path.join(OUTPUT_FOLDER, fname))
        print(f"✅ Done: {fname}")

    conn.commit()
    conn.close()

def generate_report(**context):
    """Print summary of what was processed"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*), ROUND(SUM(total_amount),2) FROM receipts")
    count, total = cursor.fetchone()
    conn.close()
    print(f"📊 Total receipts in DB: {count} | Total spend: {total}")

with DAG(
    dag_id="receipt_ocr_pipeline",
    default_args=default_args,
    description="Auto-process receipt images from inbox folder",
    schedule_interval=timedelta(minutes=5),
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["ocr", "receipts", "pipeline"],
) as dag:

    t1 = PythonOperator(
        task_id="check_new_images",
        python_callable=check_new_images,
    )

    t2 = PythonOperator(
        task_id="process_images",
        python_callable=process_images,
    )

    t3 = PythonOperator(
        task_id="generate_report",
        python_callable=generate_report,
    )

    t1 >> t2 >> t3  # DAG dependency chain
