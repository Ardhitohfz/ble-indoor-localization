# Reports Directory Policy

Folder ini berisi artefak evaluasi yang **generated** dari pipeline (gambar, CSV metrik, ringkasan teks).

## Kebijakan Repository

- Konten report dianggap output run, bukan source code.
- File report tidak dijadikan source of truth.
- Untuk membagikan hasil, gunakan release artifacts atau unggah hasil run yang relevan.

## Regenerasi

Jalankan:

```bash
cd ml_pipeline
python scripts/evaluation/comprehensive_testing.py
```

Semua output akan dibuat ulang di `reports/comprehensive_testing/`.
