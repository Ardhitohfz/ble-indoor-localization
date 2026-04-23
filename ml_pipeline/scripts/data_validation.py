import glob
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from config_ml import (
    DATA_DIR,
    REPORTS_DIR,
    ANCHOR_COLUMNS,
    TARGET_COLUMN,
    TIMESTAMP_COLUMN,
    REQUIRED_COLUMNS,
    CSV_COLUMNS,
    VALID_CELLS,
    SENTINEL_VALUES,
    MIN_VALID_RSSI,
    MAX_VALID_RSSI,
    MIN_SAMPLES_PER_CELL,
    MAX_MISSING_PER_ANCHOR,
    MAX_MISSING_OVERALL,
    MIN_ANCHOR_DETECTION_RATE,
    MAX_CELL_IMBALANCE,
    AREA_WIDTH,
    AREA_HEIGHT,
    VALIDATION_REPORT_FILENAME,
    VERBOSE,
)

@dataclass
class ValidationResult:
    name: str
    passed: bool
    message: str
    details: Optional[Dict] = None

@dataclass
class ValidationReport:
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    total_files: int = 0
    total_samples: int = 0
    results: List[ValidationResult] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def passed_count(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed_count(self) -> int:
        return sum(1 for r in self.results if not r.passed)

def log(message: str, level: str = "INFO") -> None:
    if not VERBOSE and level == "INFO":
        return

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    prefix = {
        "INFO": "[INFO]",
        "OK": "[OK]",
        "WARN": "[WARN]",
        "ERROR": "[ERROR]",
        "FAIL": "[FAIL]",
    }.get(level, "[INFO]")

    print(f"[{timestamp}] {prefix} {message}")

def load_all_csv_files(data_dir: Path = DATA_DIR) -> Tuple[Optional[pd.DataFrame], int]:
    log(f"Searching for CSV files in: {data_dir}")

    csv_pattern = str(data_dir / "*.csv")
    csv_files = glob.glob(csv_pattern)

    if not csv_files:
        log(f"No CSV files found in {data_dir}", "ERROR")
        return None, 0

    log(f"Found {len(csv_files)} CSV files")

    dfs = []
    for filepath in csv_files:
        try:
            df = pd.read_csv(filepath)
            dfs.append(df)
            log(f"  Loaded: {Path(filepath).name} ({len(df)} rows)")
        except Exception as e:
            log(f"  Failed to load {filepath}: {e}", "WARN")

    if not dfs:
        log("No valid CSV files could be loaded", "ERROR")
        return None, 0

    merged_df = pd.concat(dfs, ignore_index=True)

    if TIMESTAMP_COLUMN in merged_df.columns:
        try:
            merged_df = merged_df.sort_values(TIMESTAMP_COLUMN)
        except Exception:
            pass

    log(f"Merged dataset: {len(merged_df)} total samples", "OK")

    return merged_df, len(csv_files)

def validate_csv_schema(df: pd.DataFrame) -> ValidationResult:
    log("Checking CSV schema...")

    missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)

    if missing_cols:
        return ValidationResult(
            name="CSV Schema",
            passed=False,
            message=f"Missing required columns: {missing_cols}",
            details={"missing": list(missing_cols), "present": list(df.columns)},
        )

    log("All required columns present", "OK")
    return ValidationResult(
        name="CSV Schema",
        passed=True,
        message=f"All {len(REQUIRED_COLUMNS)} required columns present",
        details={"columns": list(df.columns)},
    )

def check_sample_count_per_cell(df: pd.DataFrame) -> ValidationResult:
    log("Checking sample count per cell...")

    if TARGET_COLUMN not in df.columns:
        return ValidationResult(
            name="Sample Count",
            passed=False,
            message=f"Column '{TARGET_COLUMN}' not found",
        )

    counts = df.groupby(TARGET_COLUMN).size()

    insufficient = counts[counts < MIN_SAMPLES_PER_CELL]

    details = {
        "counts": counts.to_dict(),
        "min_required": MIN_SAMPLES_PER_CELL,
        "min_actual": int(counts.min()) if len(counts) > 0 else 0,
        "max_actual": int(counts.max()) if len(counts) > 0 else 0,
        "total_cells": len(counts),
    }

    if not insufficient.empty:
        log(f"Cells with insufficient samples: {list(insufficient.index)}", "WARN")
        return ValidationResult(
            name="Sample Count",
            passed=False,
            message=f"{len(insufficient)} cells have < {MIN_SAMPLES_PER_CELL} samples",
            details=details,
        )

    log(f"All {len(counts)} cells have >= {MIN_SAMPLES_PER_CELL} samples", "OK")
    return ValidationResult(
        name="Sample Count",
        passed=True,
        message=f"All cells have >= {MIN_SAMPLES_PER_CELL} samples (min: {counts.min()}, max: {counts.max()})",
        details=details,
    )

def check_missing_data(df: pd.DataFrame) -> ValidationResult:
    log("Checking missing data percentage...")

    missing_stats = {}
    max_missing = 0.0

    for col in ANCHOR_COLUMNS:
        if col not in df.columns:
            missing_stats[col] = 1.0
            continue

        missing_count = df[col].isin(SENTINEL_VALUES).sum()
        missing_pct = missing_count / len(df)
        missing_stats[col] = missing_pct
        max_missing = max(max_missing, missing_pct)

        status = "OK" if missing_pct < MAX_MISSING_PER_ANCHOR else "WARN"
        log(f"  {col}: {missing_pct*100:.1f}% missing", status)

    details = {
        "missing_percentage": {k: f"{v*100:.1f}%" for k, v in missing_stats.items()},
        "max_allowed": f"{MAX_MISSING_PER_ANCHOR*100:.0f}%",
        "max_actual": f"{max_missing*100:.1f}%",
    }

    if max_missing > MAX_MISSING_PER_ANCHOR:
        return ValidationResult(
            name="Missing Data",
            passed=False,
            message=f"Max missing {max_missing*100:.1f}% exceeds threshold {MAX_MISSING_PER_ANCHOR*100:.0f}%",
            details=details,
        )

    log(f"Missing data within threshold (<{MAX_MISSING_PER_ANCHOR*100:.0f}%)", "OK")
    return ValidationResult(
        name="Missing Data",
        passed=True,
        message=f"All anchors have < {MAX_MISSING_PER_ANCHOR*100:.0f}% missing data",
        details=details,
    )

def validate_ground_truth(df: pd.DataFrame) -> ValidationResult:
    log("Validating ground truth values...")

    issues = []

    if "ground_truth_x" in df.columns:
        x_valid = df["ground_truth_x"].dropna()
        if len(x_valid) > 0:
            x_min, x_max = x_valid.min(), x_valid.max()
            if x_min < 0 or x_max > AREA_WIDTH:
                issues.append(
                    f"X out of range: [{x_min:.1f}, {x_max:.1f}] (expected [0, {AREA_WIDTH}])"
                )

    if "ground_truth_y" in df.columns:
        y_valid = df["ground_truth_y"].dropna()
        if len(y_valid) > 0:
            y_min, y_max = y_valid.min(), y_valid.max()
            if y_min < 0 or y_max > AREA_HEIGHT:
                issues.append(
                    f"Y out of range: [{y_min:.1f}, {y_max:.1f}] (expected [0, {AREA_HEIGHT}])"
                )

    if TARGET_COLUMN in df.columns:
        unique_cells = set(df[TARGET_COLUMN].dropna().unique())
        invalid_cells = unique_cells - set(VALID_CELLS)
        if invalid_cells:
            issues.append(f"Invalid cell labels: {invalid_cells}")

    details = {
        "valid_cells": VALID_CELLS,
        "found_cells": list(unique_cells) if TARGET_COLUMN in df.columns else [],
        "issues": issues,
    }

    if issues:
        return ValidationResult(
            name="Ground Truth",
            passed=False,
            message=f"{len(issues)} ground truth issues found",
            details=details,
        )

    log("Ground truth values valid", "OK")
    return ValidationResult(
        name="Ground Truth",
        passed=True,
        message="All ground truth values within valid range",
        details=details,
    )

def check_rssi_range(df: pd.DataFrame) -> ValidationResult:
    log("Checking RSSI value ranges...")

    outlier_counts = {}
    total_outliers = 0
    total_values = 0

    for col in ANCHOR_COLUMNS:
        if col not in df.columns:
            continue

        valid_data = df[col][~df[col].isin(SENTINEL_VALUES)]
        total_values += len(valid_data)

        outliers = ((valid_data < MIN_VALID_RSSI) | (valid_data > MAX_VALID_RSSI)).sum()
        outlier_counts[col] = outliers
        total_outliers += outliers

        if outliers > 0:
            log(
                f"  {col}: {outliers} values outside [{MIN_VALID_RSSI}, {MAX_VALID_RSSI}] dBm",
                "WARN",
            )

    outlier_pct = total_outliers / total_values if total_values > 0 else 0

    details = {
        "outlier_counts": outlier_counts,
        "total_outliers": total_outliers,
        "outlier_percentage": f"{outlier_pct*100:.2f}%",
        "valid_range": f"[{MIN_VALID_RSSI}, {MAX_VALID_RSSI}] dBm",
    }

    if outlier_pct > 0.05:
        return ValidationResult(
            name="RSSI Range",
            passed=False,
            message=f"{outlier_pct*100:.1f}% outliers exceed 5% threshold",
            details=details,
        )

    log(f"RSSI values within valid range ({outlier_pct*100:.2f}% outliers)", "OK")
    return ValidationResult(
        name="RSSI Range",
        passed=True,
        message=f"RSSI values valid ({outlier_pct*100:.2f}% outliers)",
        details=details,
    )

def check_anchor_coverage(df: pd.DataFrame) -> ValidationResult:
    log("Checking anchor detection rates...")

    detection_rates = {}
    min_rate = 1.0

    for col in ANCHOR_COLUMNS:
        if col not in df.columns:
            detection_rates[col] = 0.0
            continue

        valid_count = (~df[col].isin(SENTINEL_VALUES)).sum()
        rate = valid_count / len(df)
        detection_rates[col] = rate
        min_rate = min(min_rate, rate)

        status = "OK" if rate >= MIN_ANCHOR_DETECTION_RATE else "WARN"
        log(f"  {col}: {rate*100:.1f}% detection rate", status)

    details = {
        "detection_rates": {k: f"{v*100:.1f}%" for k, v in detection_rates.items()},
        "min_required": f"{MIN_ANCHOR_DETECTION_RATE*100:.0f}%",
        "min_actual": f"{min_rate*100:.1f}%",
    }

    if min_rate < MIN_ANCHOR_DETECTION_RATE:
        return ValidationResult(
            name="Anchor Coverage",
            passed=False,
            message=f"Detection rate {min_rate*100:.1f}% below {MIN_ANCHOR_DETECTION_RATE*100:.0f}% threshold",
            details=details,
        )

    log(
        f"All anchors have >= {MIN_ANCHOR_DETECTION_RATE*100:.0f}% detection rate", "OK"
    )
    return ValidationResult(
        name="Anchor Coverage",
        passed=True,
        message=f"All anchors have >= {MIN_ANCHOR_DETECTION_RATE*100:.0f}% detection rate",
        details=details,
    )

def check_duplicates(df: pd.DataFrame) -> ValidationResult:
    log("Checking for duplicate samples...")

    key_cols = (
        [TIMESTAMP_COLUMN, TARGET_COLUMN]
        if TARGET_COLUMN in df.columns
        else [TIMESTAMP_COLUMN]
    )
    key_cols = [c for c in key_cols if c in df.columns]

    if not key_cols:
        return ValidationResult(
            name="Duplicates",
            passed=True,
            message="No key columns available for duplicate check",
        )

    dup_count = df.duplicated(subset=key_cols).sum()
    dup_pct = dup_count / len(df) if len(df) > 0 else 0

    details = {
        "duplicate_count": int(dup_count),
        "duplicate_percentage": f"{dup_pct*100:.2f}%",
        "key_columns": key_cols,
    }

    if dup_count > 0:
        log(f"Found {dup_count} duplicate samples ({dup_pct*100:.2f}%)", "WARN")
        return ValidationResult(
            name="Duplicates",
            passed=True,
            message=f"{dup_count} duplicates found (will be removed during preprocessing)",
            details=details,
        )

    log("No duplicate samples found", "OK")
    return ValidationResult(
        name="Duplicates",
        passed=True,
        message="No duplicate samples found",
        details=details,
    )

def check_cell_distribution(df: pd.DataFrame) -> ValidationResult:
    log("Checking cell distribution balance...")

    if TARGET_COLUMN not in df.columns:
        return ValidationResult(
            name="Cell Distribution",
            passed=False,
            message=f"Column '{TARGET_COLUMN}' not found",
        )

    counts = df.groupby(TARGET_COLUMN).size()
    mean_count = counts.mean()

    max_deviation = (
        (counts - mean_count).abs().max() / mean_count if mean_count > 0 else 0
    )

    details = {
        "mean_samples": f"{mean_count:.1f}",
        "min_samples": int(counts.min()),
        "max_samples": int(counts.max()),
        "imbalance": f"{max_deviation*100:.1f}%",
        "max_allowed": f"{MAX_CELL_IMBALANCE*100:.0f}%",
    }

    if max_deviation > MAX_CELL_IMBALANCE:
        return ValidationResult(
            name="Cell Distribution",
            passed=False,
            message=f"Imbalance {max_deviation*100:.1f}% exceeds {MAX_CELL_IMBALANCE*100:.0f}% threshold",
            details=details,
        )

    log(f"Cell distribution balanced (imbalance: {max_deviation*100:.1f}%)", "OK")
    return ValidationResult(
        name="Cell Distribution",
        passed=True,
        message=f"Distribution balanced within {MAX_CELL_IMBALANCE*100:.0f}% tolerance",
        details=details,
    )

def check_boundary_constraints(df: pd.DataFrame) -> ValidationResult:
    """Boundary constraints feature has been removed from the pipeline."""
    return ValidationResult(
        name="Boundary Constraints",
        passed=True,
        message="Boundary constraints feature removed (augmentation disabled)",
    )

def generate_validation_report(
    report: ValidationReport, output_dir: Path = REPORTS_DIR
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / VALIDATION_REPORT_FILENAME

    lines = [
        "=" * 70,
        "BLE FINGERPRINTING - DATA VALIDATION REPORT",
        "=" * 70,
        f"Generated: {report.timestamp}",
        f"Total Files: {report.total_files}",
        f"Total Samples: {report.total_samples}",
        "",
        "-" * 70,
        "VALIDATION RESULTS",
        "-" * 70,
        "",
    ]

    for result in report.results:
        status = "[PASS]" if result.passed else "[FAIL]"
        lines.append(f"{status} {result.name}")
        lines.append(f"       {result.message}")
        if result.details:
            for key, value in result.details.items():
                if isinstance(value, dict):
                    lines.append(f"       {key}:")
                    for k, v in value.items():
                        lines.append(f"         {k}: {v}")
                else:
                    lines.append(f"       {key}: {value}")
        lines.append("")

    lines.extend(
        [
            "-" * 70,
            "SUMMARY",
            "-" * 70,
            f"Passed: {report.passed_count}/{len(report.results)}",
            f"Failed: {report.failed_count}/{len(report.results)}",
            f"Status: {'READY FOR TRAINING' if report.all_passed else 'FIX ISSUES BEFORE TRAINING'}",
            "=" * 70,
        ]
    )

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    log(f"Report saved: {output_path}", "OK")
    return output_path

def run_all_validations(data_dir: Path = DATA_DIR) -> ValidationReport:
    print()
    log("=" * 60)
    log("BLE FINGERPRINTING - DATA VALIDATION")
    log("=" * 60)
    print()

    report = ValidationReport()

    df, file_count = load_all_csv_files(data_dir)
    report.total_files = file_count

    if df is None or len(df) == 0:
        log("No data to validate", "ERROR")
        report.results.append(
            ValidationResult(
                name="Data Loading",
                passed=False,
                message="No CSV files found or loaded",
            )
        )
        return report

    report.total_samples = len(df)
    print()

    validations = [
        validate_csv_schema,
        check_sample_count_per_cell,
        check_missing_data,
        validate_ground_truth,
        check_rssi_range,
        check_anchor_coverage,
        check_duplicates,
        check_cell_distribution,
        check_boundary_constraints,
    ]

    for validation_func in validations:
        result = validation_func(df)
        report.results.append(result)
        print()

    generate_validation_report(report)

    print()
    log("=" * 60)
    log("VALIDATION SUMMARY")
    log("=" * 60)
    log(f"Total files: {report.total_files}")
    log(f"Total samples: {report.total_samples}")
    log(f"Checks passed: {report.passed_count}/{len(report.results)}")

    if report.all_passed:
        log("STATUS: READY FOR TRAINING", "OK")
    else:
        log("STATUS: FIX ISSUES BEFORE TRAINING", "FAIL")
        log("Failed checks:")
        for r in report.results:
            if not r.passed:
                log(f"  - {r.name}: {r.message}", "ERROR")

    log("=" * 60)

    return report

def run_full_validation(
    df: pd.DataFrame, save_report: bool = False
) -> ValidationReport:
    report = ValidationReport()
    report.total_samples = len(df)
    report.total_files = 1

    validations = [
        validate_csv_schema,
        check_sample_count_per_cell,
        check_missing_data,
        validate_ground_truth,
        check_rssi_range,
        check_anchor_coverage,
        check_duplicates,
        check_cell_distribution,
        check_boundary_constraints,
    ]

    for validation_func in validations:
        result = validation_func(df)
        report.results.append(result)

    if save_report:
        generate_validation_report(report)

    return report

def main():
    try:
        report = run_all_validations()

        sys.exit(0 if report.all_passed else 1)

    except KeyboardInterrupt:
        print("\n\nValidation cancelled by user")
        sys.exit(130)
    except Exception as e:
        log(f"Validation failed with error: {e}", "ERROR")
        import traceback

        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()