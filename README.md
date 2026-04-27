# AI-OCR Receipt Extraction Pipeline

---

## Overview

An end-to-end pipeline that extracts structured information from receipt images using classical image preprocessing, deep-learning OCR, and rule-based NLP — with per-field confidence scoring throughout.

**Dataset:** 371 real-world receipt images (Walmart, Trader Joe's, Whole Foods, Malaysian stores — Pasaraya, AEON, 99 Speed Mart, Unihakka, and more)

---

## Results

| Metric | Value |
|---|---|
| Receipts processed | 371 |
| Totals extracted | 323 / 371 (87%) |
| High-confidence extractions (≥ 0.5) | 166 |
| Flagged low-confidence fields | 361 receipts had ≥ 1 flag |
| Avg transaction value | ~RM 93.81 |

---

## Approach

### 1. Image Preprocessing (`pipeline.py → preprocess_image`)
Raw receipt photos have noise, uneven lighting, and skew. Each image goes through:
- **Grayscale conversion** — removes colour noise
- **Denoising** — `cv2.fastNlMeansDenoising` smooths sensor noise
- **CLAHE** — adaptive histogram equalisation fixes uneven lighting
- **Otsu binarisation** — converts to clean black/white for OCR
- **Deskew** — detects rotation via `minAreaRect` and corrects angles up to ±30°

### 2. OCR (`pipeline.py → run_ocr`)
**EasyOCR** (CRAFT text detector + CRNN recogniser) was chosen over Tesseract for its superior accuracy on real-world, noisy receipts and its native confidence scores per detected region.

Output per image: a list of `(text, confidence)` tuples.

### 3. Key Information Extraction (`pipeline.py → extract_*`)
Three-pass extraction for each field:

**Store Name**
- Pass 1: keyword match against known store list in the first 10 lines
- Pass 2: first non-address, non-numeric line as fallback
- Address filter: regex patterns detect street addresses (Jalan, postcode, state names) and discard them
- Canonical normaliser maps OCR variants (`"wal mart"`, `"walmart:"`, `"Walmart-"`) to one clean name

**Date**
- Four regex patterns covering `DD/MM/YYYY`, `YYYY-MM-DD`, `Month DD YYYY`, and `DD Month YYYY`

**Total Amount**
- Pass 1: line with a total keyword (ranked: `grand total` > `total` > `subtotal`) that also contains a price on the same line
- Pass 2: price on the line immediately following a total keyword
- Pass 3 (fallback): largest plausible price in the document (capped at 9,999 to exclude barcodes)

**Items**
- Lines containing a price pattern that are not header/footer/summary lines
- Two-line matching: item name on one line, price on the next

### 4. Data Structuring
Each receipt produces a JSON with confidence scores per field:
```json
{
  "file": "0.jpg",
  "store_name":   { "value": "Walmart",    "confidence": 0.701 },
  "date":         { "value": "05/25/10",   "confidence": 0.875 },
  "items":        [{ "name": "BANANAS", "price": "0.49", "confidence": 0.91 }],
  "total_amount": { "value": "5.11",       "confidence": 0.82  },
  "low_confidence_flags": []
}
```

### 5. Confidence Scoring
Field confidence is a weighted combination of:
- **OCR confidence** (0–1) from EasyOCR's detector
- **Pattern match score** — did a regex match? (binary 0 or 1)
- **Keyword score** — was a strong keyword like "TOTAL" present?

Formula used for total:
```
confidence = 0.4 × ocr_conf + 0.35 × keyword_rank_score + 0.25
```

Fields below 0.7 are flagged in `low_confidence_flags`.

### 6. Financial Summary (`summary.py`)
Aggregated across all receipts using Pandas:
- Total spend (all extractions + high-confidence only)
- Transaction count
- Per-store spend breakdown

---

## Tools Used

| Tool | Purpose |
|---|---|
| Python 3.10 | Core language |
| EasyOCR 1.7 | OCR engine (CRAFT + CRNN) |
| OpenCV 4.8 | Image preprocessing |
| Pillow | Image I/O |
| NumPy | Array operations |
| Pandas | Summary aggregation |
| Google Colab | Runtime environment (free GPU) |
| Google Drive | Dataset storage |

---

## Project Structure

```
ai-ocr-pipeline/
├── pipeline.py          # Full preprocessing → OCR → extraction → JSON
├── summary.py           # Financial summary generator
├── requirements.txt     # Python dependencies
├── README.md
└── outputs/             # Generated at runtime (not committed)
    ├── 0.json
    ├── 18.json
    ├── ...
    ├── all_receipts.json
    └── expense_summary.json
```

---

## How to Run

### Option A — Google Colab (recommended)
1. Open `AI_OCR_Pipeline.ipynb` in Colab
2. Mount your Google Drive
3. Set `DATASET_PATH` to your Drive folder
4. Run all cells top to bottom

### Option B — Local
```bash
pip install -r requirements.txt
python pipeline.py
python summary.py
```

> Note: EasyOCR downloads ~1 GB of model weights on first run. GPU is strongly recommended.

---

## Challenges Faced

**1. OCR noise on store names**
The biggest challenge. Many receipts are photographed at angles, with motion blur or low resolution. EasyOCR would read `"99 SPEED HART S/8 (519537-X )"` and `"99 SPEEd Hart SV8 (519537-X)"` as separate stores. Solved with a canonical normaliser that maps known OCR variants to a clean name.

**2. Multi-language receipts**
The dataset contains both English and Malay receipts. Keywords like "JUMLAH" (total) and "AMAUN" (amount) were added to the extraction rules, and address filters were extended with Malaysian geographic terms.

**3. Total vs. subtotal ambiguity**
Receipts often show SUBTOTAL, TAX, and TOTAL on adjacent lines. A priority ranking system ensures GRAND TOTAL > TOTAL > SUBTOTAL so the correct value is always preferred.

**4. Address lines picked as store names**
EasyOCR confidently reads address text, which appeared before the actual store name in many receipts. An address-detection regex filter (checking for `Jalan`, postcodes, state names) was built to skip these lines.

**5. Items = 0 on many receipts**
Early versions filtered too aggressively. The fix was a two-line lookahead: if a line has no price but the next line is a standalone price, they are paired as an item.

---

## Potential Improvements

- **Fuzzy store name deduplication** using `rapidfuzz` or `difflib` to merge remaining OCR variants (`"AEON C0_"` → `"AEON"`) at the summary stage
- **Fine-tuning EasyOCR** on receipt-specific data (e.g. SROIE dataset) to improve character-level accuracy on numbers and currency
- **LLM post-processing layer** using a small LLM (e.g. via LangChain + GPT-3.5 or Claude Haiku) to clean and validate extracted fields
- **Layout-aware parsing** using a document layout model (e.g. LayoutLM) to understand column structure rather than relying purely on text patterns
- **Better deskew** using Hough line transform for more robust angle detection on heavily rotated photos

---

## Evaluation Mapping

| Criterion | Implementation |
|---|---|
| Extraction Accuracy (30%) | EasyOCR + 3-pass extraction + keyword priority |
| Robustness to Noise (15%) | CLAHE + denoising + deskew preprocessing |
| Data Structuring (10%) | JSON with confidence fields per receipt |
| Financial Summary (10%) | `summary.py` — total, per-store, avg |
| Confidence Scoring (20%) | Weighted formula, flagging < 0.7 |
| Code Quality (10%) | Modular functions, clear separation of concerns |
| Edge Case Handling (5%) | Missing receipts, address filtering, price sanity caps |
