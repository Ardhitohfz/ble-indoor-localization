# ML Pipeline (Preprocessing → Tuning → Evaluasi)

Pipeline ini memproses dataset RSSI, melatih model LightGBM, lalu menghasilkan evaluasi komprehensif.

## Setup

```bash
cd ml_pipeline
pip install -r requirements.txt
```

## Jalankan Pipeline Lengkap

```bash
python run_pipeline.py
```

## Jalankan Per Stage

```bash
python training/prepare_dataset.py
python tuning/tune_lgbm.py
python scripts/evaluation/comprehensive_testing.py
```

## Struktur Data

- **Input canonical:** `data/raw/`
- **Generated:** `data/processed/` (gitignored)
- Detail struktur data ada di `data/README.md`.

## Artefak Model dan Report

- `models/tuned/` menyimpan artefak model terlatih.
- `reports/` berisi output evaluasi yang dapat di-regenerate.
- Kebijakan tracking report dijelaskan di `reports/README.md`.
