import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from sklearn.model_selection import train_test_split

from config_ml import (
    DATA_DIR,
    PROCESSED_DIR,
    MODELS_DIR,
    TARGET_COLUMN,
    ANCHOR_COLUMNS,
    TRAIN_SIZE,
    TEST_SIZE,
    RANDOM_STATE,
    TRAIN_FILENAME,
    TEST_FILENAME,
    MERGED_FILENAME,
    CLEANED_FILENAME,
    MIN_SAMPLES_PER_CELL,
    VERBOSE,
)
from scripts.data_validation import (
    load_all_csv_files,
    run_full_validation,
    ValidationReport,
)
from core.preprocessing import BLEDataPreprocessor

def log(message: str, level: str = "INFO") -> None:
    if not VERBOSE and level == "INFO":
        return
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")

def load_and_merge_data(data_dir: Path) -> pd.DataFrame:
    log(f"Loading data from {data_dir}")

    df, _ = load_all_csv_files(data_dir)

    if df is None or df.empty:
        raise FileNotFoundError(f"No valid CSV files found in {data_dir}")

    log(f"  Loaded {len(df)} samples from CSV files")

    return df

def validate_data(df: pd.DataFrame, strict: bool = False) -> ValidationReport:
    log("Running validation checks...")

    report = run_full_validation(df, save_report=True)

    errors = [r.message for r in report.results if not r.passed]

    if errors:
        log(f"  VALIDATION FAILED: {len(errors)} errors", "ERROR")
        for error in errors:
            log(f"    - {error}", "ERROR")
        raise ValueError("Data validation failed. Fix errors before proceeding.")

    log("  Validation PASSED")
    return report

def preprocess_data(
    df: pd.DataFrame, preprocessor: Optional[BLEDataPreprocessor] = None
) -> Tuple[pd.DataFrame, Optional[BLEDataPreprocessor]]:
    fit_new = preprocessor is None

    if fit_new:
        log("Preprocessing data (FIT + TRANSFORM)...")
        preprocessor = BLEDataPreprocessor(min_samples_per_cell=0)
        preprocessor.fit(df)
        log(f"  Fitted preprocessor on {len(df)} samples")
    else:
        log("Preprocessing data (TRANSFORM only)...")

    processed_df = preprocessor.transform(df)

    columns_to_drop = []

    if "timestamp" in processed_df.columns:
        columns_to_drop.append("timestamp")

    metadata_cols = [
        "anchors_valid",
        "sample_number",
        "collection_time_ms",
        "beacon_moving",
        "environment",
        "notes",
    ]
    columns_to_drop.extend(
        [col for col in metadata_cols if col in processed_df.columns]
    )

    if columns_to_drop:
        processed_df = processed_df.drop(columns=columns_to_drop)
        if fit_new:
            log(
                f"  Dropped {len(columns_to_drop)} unused columns (memory optimization)"
            )

    if {"ground_truth_x", "ground_truth_y"}.issubset(processed_df.columns):
        before = len(processed_df)
        processed_df = processed_df.dropna(subset=["ground_truth_x", "ground_truth_y"])
        removed = before - len(processed_df)
        if removed > 0:
            log(f"  Removed {removed} samples without coordinates", "WARN")

    if fit_new:
        processed_df = preprocessor.filter_low_sample_cells(processed_df)

    log(f"  Preprocessed: {len(df)} -> {len(processed_df)} samples")
    if fit_new:
        log(f"  Features: {len(preprocessor.feature_names_)}")

    return processed_df, preprocessor if fit_new else None

def split_data(
    df: pd.DataFrame,
    train_size: float = TRAIN_SIZE,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
  
    log("Splitting data (80/20 strategy)...")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in DataFrame")

    total = train_size + test_size
    if abs(total - 1.0) > 0.01:
        raise ValueError(f"Split proportions must sum to 1.0, got {total}")

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df[TARGET_COLUMN],
        random_state=random_state,
    )

    log(f"  Train: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
    log(f"  Test:  {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")

    train_cells = set(train_df[TARGET_COLUMN].unique())
    test_cells = set(test_df[TARGET_COLUMN].unique())

    if train_cells != test_cells:
        log("  Warning: Not all cells represented in both splits", "WARN")

    # Log samples per cell for verification
    train_counts = train_df[TARGET_COLUMN].value_counts()
    test_counts = test_df[TARGET_COLUMN].value_counts()
    log(f"  Train samples per cell: min={train_counts.min()}, max={train_counts.max()}, mean={train_counts.mean():.1f}")
    log(f"  Test samples per cell:  min={test_counts.min()}, max={test_counts.max()}, mean={test_counts.mean():.1f}")

    return train_df, test_df

def save_datasets(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    preprocessor: BLEDataPreprocessor,
    output_dir: Path,
    merged_df: Optional[pd.DataFrame] = None,
) -> None:
    log(f"Saving datasets to {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(output_dir / TRAIN_FILENAME, index=False)
    test_df.to_csv(output_dir / TEST_FILENAME, index=False)

    log(f"  Saved {TRAIN_FILENAME}, {TEST_FILENAME}")

    if merged_df is not None:
        merged_df.to_csv(output_dir / MERGED_FILENAME, index=False)
        log(f"  Saved {MERGED_FILENAME}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    preprocessor.save()

    log(f"  Saved preprocessor to {MODELS_DIR}")

def generate_summary(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    preprocessor: BLEDataPreprocessor,
) -> str:
    total = len(train_df) + len(test_df)

    lines = [
        "=" * 60,
        "DATASET PREPARATION SUMMARY (80/20 Split)",
        "=" * 60,
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "SPLIT STRATEGY:",
        "  80/20 (Train/Test) - Best Practice for Small Datasets",
        "  Validation: 5-Fold StratifiedKFold CV during tuning",
        "",
        "SAMPLE COUNTS:",
        f"  Train: {len(train_df):,} ({len(train_df)/total*100:.1f}%)",
        f"  Test:  {len(test_df):,} ({len(test_df)/total*100:.1f}%)",
        f"  Total: {total:,}",
        "",
        "SAMPLES PER CELL:",
        f"  Train: {len(train_df)//25} samples/cell (avg)",
        f"  Test:  {len(test_df)//25} samples/cell (avg)",
        "",
        "FEATURES:",
        f"  Count: {len(preprocessor.feature_names_)}",
        f"  Names: {preprocessor.feature_names_}",
        "",
        "CLASSES:",
        f"  Count: {preprocessor.n_classes_}",
        f"  Labels: {list(preprocessor.encoder.classes_)}",
        "",
        "CELL DISTRIBUTION (Train):",
    ]

    cell_counts = train_df[TARGET_COLUMN].value_counts().sort_index()
    for cell, count in cell_counts.items():
        lines.append(f"  {cell}: {count}")

    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser(
        description="Prepare BLE fingerprinting dataset for training"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help=f"Directory containing CSV files (default: {DATA_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROCESSED_DIR,
        help=f"Output directory for processed data (default: {PROCESSED_DIR})",
    )
    parser.add_argument(
        "--skip-validation", action="store_true", help="Skip data validation step"
    )
    parser.add_argument(
        "--strict", action="store_true", help="Treat validation warnings as errors"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("BLE FINGERPRINTING - DATASET PREPARATION")
    print("=" * 60)

    try:
        merged_df = load_and_merge_data(args.data_dir)

        if not args.skip_validation:
            validate_data(merged_df, strict=args.strict)
        else:
            log("Skipping validation (--skip-validation)", "WARN")

        train_df_raw, test_df_raw = split_data(merged_df)

        train_df, preprocessor = preprocess_data(train_df_raw)

        assert (
            preprocessor is not None
        ), "Preprocessor should have been fitted on training data"

        test_df, _ = preprocess_data(test_df_raw, preprocessor)

        save_datasets(
            train_df,
            test_df,
            preprocessor,
            args.output_dir,
            merged_df=merged_df,
        )

        summary = generate_summary(train_df, test_df, preprocessor)
        print("\n" + summary)

        summary_path = args.output_dir / "preparation_summary.txt"
        summary_path.write_text(summary)
        log(f"Saved summary to {summary_path}")

        print("\nDataset preparation COMPLETE! (80/20 Split)")
        print(f"  Train: {args.output_dir / TRAIN_FILENAME}")
        print(f"  Test:  {args.output_dir / TEST_FILENAME}")

        return 0

    except FileNotFoundError as e:
        log(str(e), "ERROR")
        print("\nHint: Place CSV files in the data/ directory first.")
        return 1

    except ValueError as e:
        log(str(e), "ERROR")
        return 1

    except Exception as e:
        log(f"Unexpected error: {e}", "ERROR")
        raise

if __name__ == "__main__":
    sys.exit(main())