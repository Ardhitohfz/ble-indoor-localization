# Raspberry Pi 4 Sink (Data Acquisition)

Modul ini melakukan akuisisi RSSI paralel dari 4 anchor BLE dan menulis hasil ke CSV.

## Setup

```bash
cd raspi4_sink
pip install -r requirements.txt
```

## Jalankan

```bash
python rpi_sink_parallel.py --samples 100 --interactive
```

Atau non-interactive:

```bash
python rpi_sink_parallel.py --samples 100 --interval 1.0 --cell A1
```

## Kebijakan Data

- **Single source of truth dataset:** `../ml_pipeline/data/raw/`
- Default output collector sekarang diarahkan ke folder tersebut (file `dataset_YYYYMMDD_HHMMSS.csv`).
- Jika folder `ml_pipeline/` tidak tersedia (mis. deploy minimal), collector fallback ke `raspi4_sink/data/dataset/`.

## Opsi Penting

- `--output`: path file output CSV (absolute/relative)
- `--samples`: jumlah sampel
- `--interval`: jeda antar sampel
- `--cell`, `--ground-truth-x`, `--ground-truth-y`: metadata label untuk supervised learning
