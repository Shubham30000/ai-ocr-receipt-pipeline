"""
Financial Summary Generator
Aggregates extracted receipt data into spending reports.
"""

import json
import os
import pandas as pd


def generate_summary(records):
    rows = []
    for r in records:
        if "error" in r:
            continue
        store  = r['store_name']['value'] or "Unknown"
        total  = r['total_amount']['value']
        date   = r['date']['value']
        t_conf = r['total_amount']['confidence']
        try:
            amount = float(str(total).replace(',', '')) if total else None
        except:
            amount = None
        if amount and amount > 10000:   # sanity cap
            amount = None
        rows.append({
            "store":      store,
            "total":      amount,
            "date":       date,
            "file":       r['file'],
            "total_conf": t_conf,
            "flagged":    len(r.get('low_confidence_flags', [])) > 0
        })

    df    = pd.DataFrame(rows)
    valid = df.dropna(subset=['total'])
    high  = valid[valid['total_conf'] >= 0.5]

    summary = {
        "total_receipts_processed":      len(records),
        "receipts_with_total_extracted": len(valid),
        "high_confidence_extractions":   len(high),
        "total_spend_all":               round(valid['total'].sum(), 2),
        "total_spend_high_confidence":   round(high['total'].sum(), 2),
        "avg_transaction":               round(valid['total'].mean(), 2),
        "flagged_receipts":              int(df['flagged'].sum()),
        "spend_per_store": (
            valid.groupby('store')['total']
                 .agg(['sum', 'count'])
                 .rename(columns={'sum': 'total_spend', 'count': 'transactions'})
                 .round(2)
                 .to_dict()
        )
    }
    return summary, df


if __name__ == "__main__":
    OUTPUT_PATH = "/content/drive/MyDrive/AI-OCR dataset/outputs"
    with open(os.path.join(OUTPUT_PATH, "all_receipts.json")) as f:
        all_records = json.load(f)

    summary, df = generate_summary(all_records)
    print(json.dumps({k: v for k, v in summary.items() if k != 'spend_per_store'}, indent=2))

    out_path = os.path.join(OUTPUT_PATH, "expense_summary.json")
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {out_path}")
