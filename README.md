# Sistem Lokalisasi Indoor Berbasis BLE Pada Area Parkir Menggunakan Machine Learning Fingerprinting

> Source Code skripsi — Program Studi Teknik Komputer, Universitas Brawijaya

![License](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/python-3.11-blue?logo=python&logoColor=white)
![Platform](https://img.shields.io/badge/platform-ESP32%20%7C%20Raspberry%20Pi%204-orange?logo=espressif&logoColor=white)
![Framework](https://img.shields.io/badge/ML-LightGBM%20%2B%20Optuna-blue?logo=python&logoColor=white)

## Gambaran Umum

Sistem ini mengimplementasikan lokalisasi indoor menggunakan teknik **RSSI fingerprinting** berbasis Bluetooth Low Energy (BLE). Beacon bergerak memancarkan sinyal BLE; empat anchor ESP32 di sudut area membaca RSSI secara paralel dan mengirimkan data ke Raspberry Pi 4 (sink). Sink mengekstrak 36 fitur statistik dari RSSI dan mengklasifikasikan posisi ke salah satu dari 25 sel grid menggunakan model LightGBM yang dioptimasi dengan Optuna.

### Arsitektur Sistem

```text
[Beacon ESP32-S3]
        │ BLE Advertisement (100 ms interval, 0 dBm)
        ▼
┌───────────────────────────────┐
│  Anchor ESP32 × 4             │  A:(2,2)  B:(8,2)
│  (antena eksternal 5 dBi)     │  C:(2,8)  D:(8,8)
└──────────────┬────────────────┘
               │ Serial / WiFi
               ▼
    [Sink: Raspberry Pi 4 Model B]
    ├─ Akuisisi RSSI (bleak, parallel)
    ├─ Ekstraksi 36 fitur statistik
    └─ Inferensi LightGBM → Grid (A1–E5)
```

Area uji: **10 m × 10 m** (basement parkir), grid **5×5 = 25 sel**, jarak antar sel 2 meter.

---

## Hasil Model

| Metrik | Nilai |
| --- | --- |
| Akurasi test set | **99,60%** (249/250 sampel) |
| 5-fold Stratified CV | **100,00%** (std dev 0,00%) |
| Generalization gap | 0,40% |
| Error spasial median | 0,00 m |

Model: LightGBM (multi-class, 25 kelas) — dioptimasi Optuna (100 trials, objective F1-score makro).

---

## Struktur Direktori

```text
thesis-repo/
├── ble-localizer-updated/      # Firmware ESP32 (PlatformIO)
│   ├── README.md
│   ├── platformio.ini
│   ├── firmware/
│   │   ├── anchor/             # Firmware anchor: baca RSSI beacon
│   │   │   ├── include/
│   │   │   └── src/
│   │   └── beacon/             # Firmware beacon: BLE advertiser
│   │       ├── include/
│   │       └── src/
│   └── shared/                 # Header dan utilitas bersama anchor-beacon
│       ├── include/
│       └── src/
├── raspi4_sink/                 # Python: akuisisi data BLE di Raspberry Pi 4
│   ├── README.md
│   ├── rpi_sink_parallel.py    # Entry point: baca RSSI dari 4 anchor paralel
│   ├── ble_utils.py
│   ├── config.py
│   ├── requirements.txt
├── ml_pipeline/                 # Pipeline ML: preprocessing → tuning → evaluasi
│   ├── README.md
│   ├── run_pipeline.py         # Entry point: jalankan semua stage
│   ├── config_ml.py            # Konfigurasi path dan parameter global
│   ├── requirements.txt
│   ├── data/
│   │   ├── README.md
│   │   ├── raw/                # Single source of truth dataset
│   │   └── archive/            # Dataset historis (bukan input default)
│   ├── models/
│   │   └── tuned/              # Model terlatih dan artefak evaluasi
│   ├── reports/                # Generated report artifacts (re-runnable)
│   ├── training/               # Modul training
│   ├── tuning/                 # Modul Optuna hyperparameter tuning
│   ├── inference/              # Modul inferensi/prediksi
│   ├── scripts/                # Script preprocessing dan evaluasi
│   └── core/                   # Utilitas inti pipeline
├── .github/workflows/ci.yml     # CI smoke check (syntax + entrypoint help)
├── CONTRIBUTING.md
└── .gitignore
```

---

## Hardware

| Komponen | Spesifikasi |
| --- | --- |
| **Beacon** | ESP32-S3 DevKitC-1 — BLE advertiser, non-connectable, interval 100 ms, daya 0 dBm, baterai Li-ion 18650 |
| **Anchor** | 4× ESP32 + antena eksternal 5 dBi — posisi tetap di sudut area |
| **Sink** | Raspberry Pi 4 Model B — Python 3.x, Bluetooth 5.0 built-in, inferensi LightGBM |

---

## Software dan Dependensi

### Firmware ESP32 (`ble-localizer-updated/`)

- Framework: Arduino (PlatformIO)
- Library: NimBLE 2.3.6
- Bahasa: C++

Build menggunakan PlatformIO:

```bash
cd ble-localizer-updated
pio run --environment beacon    # build firmware beacon
pio run --environment anchor_a  # build firmware anchor A
pio run --environment anchor_b  # build firmware anchor B
pio run --environment anchor_c  # build firmware anchor C
pio run --environment anchor_d  # build firmware anchor D
pio run --environment beacon --target upload
pio run --environment anchor_a --target upload
```

### Sink Raspberry Pi 4 (`raspi4_sink/`)

- Python 3.x
- Dependensi: lihat `raspi4_sink/requirements.txt` (`bleak`)

```bash
cd raspi4_sink
pip install -r requirements.txt
python rpi_sink_parallel.py
```

> Default output collector sekarang diarahkan ke `ml_pipeline/data/raw/` agar dataset punya satu sumber utama.

### Pipeline ML (`ml_pipeline/`)

- Python 3.11
- Dependensi: lihat `ml_pipeline/requirements.txt`

```bash
cd ml_pipeline
pip install -r requirements.txt
python run_pipeline.py
```

Pipeline berjalan dalam tiga stage berurutan:

```text
Stage 1 — prepare_dataset.py      → data/processed/
Stage 2 — tuning/tune_lgbm.py     → models/tuned/
Stage 3 — comprehensive_testing.py → reports/
```

## Quality Gates

- CI GitHub Actions: `.github/workflows/ci.yml`
- Smoke checks:
  - `python -m compileall ml_pipeline raspi4_sink`
  - `python ml_pipeline/run_pipeline.py --help`
  - `python ml_pipeline/training/prepare_dataset.py --help`
  - `python raspi4_sink/rpi_sink_parallel.py --help`
- Unit tests:
  - `python -m unittest discover -s tests -p "test_*.py" -v`
- Hygiene policy:
  - `python tools/check_repo_hygiene.py`
- Pre-commit gate:
  - `pre-commit run --all-files`

---

## Dataset

- **Single source of truth:** `ml_pipeline/data/raw/`
- Contoh dataset baseline: **25 file CSV** (`A1.csv` — `E5.csv`), satu file per sel grid
- Dataset historis lama disimpan terpisah di `ml_pipeline/data/archive/` (tidak dipakai default).
- **1.250 sampel** total — 50 sampel per sel × 25 sel
- **Split:** 80% training (1.000 sampel) / 20% test (250 sampel), stratified
- **46 kolom per file:** 4 ground truth + 4 RSSI raw + 32 fitur statistik + 6 metadata

### Fitur Input Model (36 fitur)

| Kelompok | Jumlah | Keterangan |
| --- | --- | --- |
| RSSI raw | 4 | RSSI langsung dari anchor A, B, C, D |
| Statistik per anchor | 32 | Mean, median, std, min, max, Q25, Q75, outlier\_IQR × 4 anchor |

---

## Model Terlatih

File model tersedia di `ml_pipeline/models/tuned/`:

| File | Keterangan |
| --- | --- |
| `lgbm_tuned.pkl` | Model LightGBM terlatih (pickle) |
| `best_params.json` | Hyperparameter terbaik hasil Optuna |
| `evaluation_results.json` | Metrik evaluasi lengkap |
| `feature_info.json` | Daftar fitur dan urutan kolom input |
| `label_encoder.pkl` | Label encoder untuk 25 kelas grid |
| `lgbm_tuned.txt` | Representasi teks model (human-readable) |

---

## Penulis

**Ardhito Hafiz Prathama**
Program Studi Teknik Komputer, Fakultas Ilmu Komputer
Universitas Brawijaya — NIM 215150300111005
