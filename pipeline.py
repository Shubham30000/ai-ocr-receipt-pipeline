"""
AI-OCR Receipt Extraction Pipeline
Carbon Crunch Shortlisting Assignment
"""

import os
import re
import json
import cv2
import numpy as np
import easyocr
from pathlib import Path

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
DATASET_PATH = "/content/drive/MyDrive/AI-OCR dataset"
OUTPUT_PATH  = "/content/drive/MyDrive/AI-OCR dataset/outputs"
os.makedirs(OUTPUT_PATH, exist_ok=True)

reader = easyocr.Reader(['en'], gpu=True)

# ─────────────────────────────────────────
# PATTERNS & KEYWORDS
# ─────────────────────────────────────────
DATE_PATTERNS = [
    r'\b(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})\b',
    r'\b(\d{4}[\/\-\.]\d{1,2}[\/\-\.]\d{1,2})\b',
    r'\b([A-Za-z]{3,9}\.?\s+\d{1,2},?\s+\d{4})\b',
    r'\b(\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4})\b',
]
PRICE_PATTERN  = r'\$?\s*(\d{1,4}[,\d]*\.\d{2})\b'
TOTAL_PRIORITY = [
    'grand total', 'total amount', 'jumlah besar', 'amaun',
    'total', 'amount due', 'balance due', 'jumlah',
    'subtotal', 'sub total'
]
SKIP_KEYWORDS = [
    'tax', 'gst', 'sst', 'cash', 'change', 'tender',
    'thank', 'welcome', 'please', 'survey', 'phone',
    'tel', 'fax', 'address', 'manager', 'open', 'hours',
    'store', 'stw', 'tc#', 'trx', 'network', 'terminal',
    'receipt', 'invoice', 'reg', 'cashier'
]
STORE_KEYWORDS = [
    'walmart', 'wal-mart', 'wal mart', 'target', 'costco',
    'kroger', "trader joe", 'whole foods', 'walgreens', 'cvs',
    'aldi', 'safeway', 'publix', "sam's club", 'sams club',
    'dollar', '7-eleven', 'starbucks', 'mcdonald',
    'pasaraya', 'borong', 'unihakka', 'mydin', 'giant',
    'tesco', 'aeon', 'lotus', 'econsave', 'sdn bhd', 'berhad',
    '99 speed', 'popular', 'kaison', 'bens independent',
]
STORE_CANONICAL = {
    "99 speed mart":       "99 SPEED MART",
    "aeon":                "AEON",
    "mydin":               "MYDIN",
    "pasaraya borong":     "PASARAYA BORONG SUPER SEVEN",
    "unihakka":            "UNIHAKKA INTERNATIONAL SDN BHD",
    "walmart":             "Walmart",
    "wal mart":            "Walmart",
    "wal-mart":            "Walmart",
    "whole foods":         "Whole Foods",
    "trader joe":          "Trader Joe's",
    "popular":             "POPULAR",
    "ikano handel":        "IKANO HANDEL SDN BHD",
    "kaison":              "KAISON FURNISHING SDN BHD",
    "bens independent":    "BENS INDEPENDENT GROCER",
    "lightroom gallery":   "LIGHTROOM GALLERY SDN",
    "syarikat perniagaan": "SYARIKAT PERNIAGAAN GIN KEE",
    "kedai papan yew":     "KEDAI PAPAN YEW CHUAN",
    "beyond brothers":     "BEYOND BROTHERS HARDWARE",
    "gerbang alaf":        "Gerbang Alaf Restaurants Sdn Bhd",
    "golden arches":       "Golden Arches Restaurants Sdn Bhd",
    "old town":            "Old Town Kopitiam",
    "subway":              "Subway",
    "mcdonald":            "McDonald's",
}
ADDRESS_PATTERNS = [
    r'\b(jalan|jln|lorong|lrg|taman|bandar|no\.?|lot)\b',
    r'\b(selangor|kuala lumpur|johor|penang|perak|sabah|sarawak)\b',
    r'\b\d{5}\b',
    r'\b(sdn bhd|s/b|plt)\s*$',
]


# ─────────────────────────────────────────
# STEP 1: IMAGE PREPROCESSING
# ─────────────────────────────────────────
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Cannot read: {img_path}")
    gray     = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=15)
    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Deskew
    coords = np.column_stack(np.where(binary > 0))
    if len(coords) > 0:
        angle = cv2.minAreaRect(coords)[-1]
        angle = -(90 + angle) if angle < -45 else -angle
        if abs(angle) < 30:
            h, w = binary.shape
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            binary = cv2.warpAffine(binary, M, (w, h),
                                    flags=cv2.INTER_CUBIC,
                                    borderMode=cv2.BORDER_REPLICATE)
    return binary


# ─────────────────────────────────────────
# STEP 2: OCR
# ─────────────────────────────────────────
def run_ocr(img_path):
    processed = preprocess_image(img_path)
    results   = reader.readtext(processed, detail=1)
    return [(res[1].strip(), float(res[2])) for res in results if res[1].strip()]


# ─────────────────────────────────────────
# STEP 3: EXTRACTION HELPERS
# ─────────────────────────────────────────
def is_address(text):
    lower = text.lower()
    return any(re.search(p, lower) for p in ADDRESS_PATTERNS)

def normalize_store(raw):
    if not raw:
        return "Unknown"
    lower = raw.lower()
    for key, canonical in STORE_CANONICAL.items():
        if key in lower:
            return canonical
    if is_address(raw):
        return "Unknown"
    return raw.strip()

def extract_store(lines):
    for text, conf in lines[:10]:
        if is_address(text):
            continue
        lower = text.lower()
        for kw in STORE_KEYWORDS:
            if kw in lower:
                return text.strip(), round(min(conf + 0.15, 1.0), 3)
    for text, conf in lines[:8]:
        if is_address(text):
            continue
        if len(text.strip()) < 4:
            continue
        if re.fullmatch(r'[\d\s\W]+', text.strip()):
            continue
        return text.strip(), round(conf * 0.6, 3)
    return None, 0.0

def extract_date(lines):
    for text, conf in lines:
        for pat in DATE_PATTERNS:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                return m.group(1), round(0.5 * conf + 0.5, 3)
    return None, 0.0

def extract_total(lines):
    found = []
    for i, (text, conf) in enumerate(lines):
        lower = text.lower()
        for rank, kw in enumerate(TOTAL_PRIORITY):
            if kw in lower:
                m = re.search(PRICE_PATTERN, text)
                if m:
                    val = float(m.group(1).replace(',', ''))
                    if val > 0:
                        found.append((rank, val, conf, 'same_line'))
                elif i + 1 < len(lines):
                    m2 = re.search(PRICE_PATTERN, lines[i+1][0])
                    if m2:
                        val = float(m2.group(1).replace(',', ''))
                        if val > 0:
                            found.append((rank, val, conf * 0.9, 'next_line'))
                break
    if found:
        found.sort(key=lambda x: (x[0], -x[2]))
        best  = found[0]
        score = round(0.4 * best[2] + 0.35 * (1 - best[0]/len(TOTAL_PRIORITY)) + 0.25, 3)
        return str(round(best[1], 2)), min(score, 1.0)
    # Fallback: largest plausible price
    all_prices = []
    for text, conf in lines:
        for m in re.finditer(PRICE_PATTERN, text):
            try:
                val = float(m.group(1).replace(',', ''))
                if 0.5 < val < 9999:
                    all_prices.append((val, conf))
            except:
                pass
    if all_prices:
        best_val, best_conf = max(all_prices, key=lambda x: x[0])
        return str(round(best_val, 2)), round(best_conf * 0.35, 3)
    return None, 0.0

def extract_items(lines):
    items = []
    HARD_SKIP = SKIP_KEYWORDS + [kw for kw in TOTAL_PRIORITY]
    i = 0
    while i < len(lines):
        text, conf = lines[i]
        lower = text.lower()
        if any(kw in lower for kw in HARD_SKIP):
            i += 1; continue
        if len(text.strip()) < 3:
            i += 1; continue
        if re.fullmatch(r'[\d\s\*\#\-\.]+', text.strip()):
            i += 1; continue
        m = re.search(PRICE_PATTERN, text)
        if m:
            price = m.group(1).replace(',', '')
            name  = re.sub(PRICE_PATTERN, '', text).strip(' .,-\t*#@')
            name  = re.sub(r'\s{2,}', ' ', name)
            try:
                if 0.01 <= float(price) <= 999 and len(name) > 2:
                    if not re.fullmatch(r'[\d\s]+', name):
                        items.append({"name": name, "price": price, "confidence": round(conf, 3)})
            except:
                pass
        elif i + 1 < len(lines):
            next_text, next_conf = lines[i + 1]
            m2 = re.search(r'^\s*\$?\s*(\d{1,4}\.\d{2})\s*$', next_text.strip())
            if m2:
                name = text.strip(' .,-\t*#@')
                price = m2.group(1)
                try:
                    if 0.01 <= float(price) <= 999 and len(name) > 2:
                        items.append({"name": name, "price": price,
                                      "confidence": round((conf + next_conf) / 2, 3)})
                except:
                    pass
                i += 2; continue
        i += 1
    return items


# ─────────────────────────────────────────
# STEP 4: CONFIDENCE & FLAGGING
# ─────────────────────────────────────────
def flag_low_confidence(record, threshold=0.7):
    flags = []
    for field in ['store_name', 'date', 'total_amount']:
        val  = record.get(field, {}).get('value')
        conf = record.get(field, {}).get('confidence', 0)
        if val is None:
            flags.append(f"{field} (NOT EXTRACTED)")
        elif conf < threshold:
            flags.append(f"{field} (conf={conf})")
    return flags


# ─────────────────────────────────────────
# FULL PIPELINE
# ─────────────────────────────────────────
def process_receipt(img_path):
    img_name = Path(img_path).name
    try:
        lines = run_ocr(img_path)
    except Exception as e:
        return {"file": img_name, "error": str(e)}

    store_raw, store_conf = extract_store(lines)
    store_clean           = normalize_store(store_raw)
    date,  date_conf      = extract_date(lines)
    total, total_conf     = extract_total(lines)
    items                 = extract_items(lines)

    record = {
        "file":         img_name,
        "store_name":   {"value": store_clean, "confidence": store_conf},
        "date":         {"value": date,        "confidence": date_conf},
        "items":        items,
        "total_amount": {"value": total,       "confidence": total_conf},
    }
    record["low_confidence_flags"] = flag_low_confidence(record)
    return record


# ─────────────────────────────────────────
# RUN ON ALL IMAGES
# ─────────────────────────────────────────
if __name__ == "__main__":
    import pandas as pd

    image_files = [
        f for f in os.listdir(DATASET_PATH)
        if f.lower().endswith(('.jpg', '.png', '.jpeg'))
    ]
    all_records = []

    for i, fname in enumerate(image_files):
        img_path = os.path.join(DATASET_PATH, fname)
        print(f"[{i+1}/{len(image_files)}] {fname}", end='\r')
        record = process_receipt(img_path)
        all_records.append(record)
        json_name = Path(fname).stem + ".json"
        with open(os.path.join(OUTPUT_PATH, json_name), 'w') as f:
            json.dump(record, f, indent=2)

    with open(os.path.join(OUTPUT_PATH, "all_receipts.json"), 'w') as f:
        json.dump(all_records, f, indent=2)

    print(f"\nDone. {len(all_records)} receipts processed.")
