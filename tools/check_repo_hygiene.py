from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ALLOWED_REPORT_TRACKED = {"ml_pipeline/reports/README.md"}
BLOCKED_PATTERNS = (
    "ml_pipeline/reports/**",
    "raspi4_sink/data/dataset/**",
)


def tracked_files(pattern: str) -> list[str]:
    output = subprocess.check_output(
        ["git", "ls-files", pattern],
        cwd=ROOT,
        text=True,
    )
    return [line.strip() for line in output.splitlines() if line.strip()]


def main() -> int:
    violations: list[str] = []

    for pattern in BLOCKED_PATTERNS:
        files = tracked_files(pattern)
        if pattern == "ml_pipeline/reports/**":
            files = [path for path in files if path not in ALLOWED_REPORT_TRACKED]
        if files:
            violations.extend(files)

    raw_csv_files = list((ROOT / "ml_pipeline" / "data" / "raw").glob("*.csv"))
    if not raw_csv_files:
        print("ERROR: ml_pipeline/data/raw has no CSV files (canonical dataset missing).")
        return 1

    if violations:
        print("ERROR: Blocked generated artifacts are tracked by git:")
        for path in violations:
            print(f" - {path}")
        return 1

    print("Repo hygiene check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
