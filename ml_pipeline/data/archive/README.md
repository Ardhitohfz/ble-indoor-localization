# Archive Data

Folder ini menyimpan file dataset historis yang tidak dipakai sebagai input default pipeline.

- `raspi4_sink_legacy/`: file CSV lama dari `raspi4_sink/data/dataset/` yang kontennya berbeda dari canonical `data/raw/`.
- Pipeline tetap menggunakan `data/raw/` kecuali `--data-dir` diubah manual.
