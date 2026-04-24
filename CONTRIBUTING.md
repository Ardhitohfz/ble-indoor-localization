# Contributing Guidelines

## Repository Conventions

1. Gunakan naming konsisten:
   - Folder: `snake_case` atau `kebab-case` sesuai folder existing (jangan campur dalam modul yang sama).
   - File Python: `snake_case.py`.
2. Dataset mentah canonical hanya di `ml_pipeline/data/raw/`.
3. Output generated (`ml_pipeline/data/processed/`, `ml_pipeline/reports/`, logs) tidak dijadikan sumber data utama.
4. Hindari commit file hasil eksperimen besar yang bisa di-regenerate.

## Workflow Ringkas

1. Install dependency per modul.
2. Install pre-commit hook:
   - `pip install pre-commit`
   - `pre-commit install`
3. Jalankan quality gates lokal:
   - `python tools/check_repo_hygiene.py`
   - `python -m unittest discover -s tests -p "test_*.py" -v`
   - `pre-commit run --all-files`
4. Pastikan perubahan dokumentasi ikut diupdate jika perilaku/path berubah.

## Pull Request

- Jelaskan perubahan fungsional dan dampaknya.
- Sertakan command run yang digunakan.
- Jangan campur refactor besar dengan perubahan fitur/bugfix kecil.
