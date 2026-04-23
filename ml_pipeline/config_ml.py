import json
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).parent

# Data directories (following Cookiecutter Data Science standard)
DATA_DIR = BASE_DIR / "data" / "raw"  # Immutable raw data (25 CSV files)
PROCESSED_DIR = BASE_DIR / "data" / "processed"  # Preprocessed train/test splits
REPORTS_DIR = BASE_DIR / "reports"
MODELS_DIR = BASE_DIR / "models"

BASELINE_MODEL_DIR = MODELS_DIR / "baseline"
TUNED_MODEL_DIR = MODELS_DIR / "tuned"
LOGS_DIR = BASE_DIR / "logs"  # Execution logs and debug traces
LOGS_DIR = BASE_DIR / "logs"  # Execution logs and debug traces

def create_experiment_dir(base_dir: Path, experiment_name: str = "experiment") -> tuple[Path, dict]:
    """
    Create timestamped experiment directory for reproducible runs.
    
    Args:
        base_dir: Base directory for experiments (typically REPORTS_DIR)
        experiment_name: Human-readable experiment identifier
    
    Returns:
        tuple: (experiment_directory_path, metadata_dict)
        
    Example:
        >>> exp_dir, meta = create_experiment_dir(REPORTS_DIR, "comprehensive_testing")
        >>> # Creates: reports/experiments/20251224_153045_comprehensive_testing/
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = base_dir / "experiments" / f"{timestamp}_{experiment_name}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    metadata = {
        "timestamp": timestamp,
        "experiment_name": experiment_name,
        "created_at": datetime.now().isoformat(),
    }
    
    # Save metadata to JSON
    metadata_file = exp_dir / "metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    
    return exp_dir, metadata

BOUNDARY_CONSTRAINTS_FILE = BASE_DIR / "data" / "rssi_boundaries.json"
USE_BOUNDARY_CONSTRAINTS = False  # ❌ DISABLED: No augmentation for K-Fold CV (sufficient natural data)

SENTINEL_VALUES = [-999, -127]
MISSING_RSSI_VALUE = -110
MIN_VALID_RSSI = -120
MAX_VALID_RSSI = -10

ANCHOR_COLUMNS = ["rssi_A", "rssi_B", "rssi_C", "rssi_D"]

STATS_COLUMNS = [
    "rssiA_mean",
    "rssiA_median",
    "rssiA_std",
    "rssiA_min",
    "rssiA_max",
    "rssiA_q25",
    "rssiA_q75",
    "rssiA_outliers",
    "rssiB_mean",
    "rssiB_median",
    "rssiB_std",
    "rssiB_min",
    "rssiB_max",
    "rssiB_q25",
    "rssiB_q75",
    "rssiB_outliers",
    "rssiC_mean",
    "rssiC_median",
    "rssiC_std",
    "rssiC_min",
    "rssiC_max",
    "rssiC_q25",
    "rssiC_q75",
    "rssiC_outliers",
    "rssiD_mean",
    "rssiD_median",
    "rssiD_std",
    "rssiD_min",
    "rssiD_max",
    "rssiD_q25",
    "rssiD_q75",
    "rssiD_outliers",
]

# Conservative feature set: Only most valuable ESP32 statistics
# Removes redundant features (mean overlaps with raw RSSI, quartiles overlap with std)
CONSERVATIVE_STATS_COLUMNS = [
    "rssiA_median",
    "rssiA_std",
    "rssiA_outliers",
    "rssiB_median",
    "rssiB_std",
    "rssiB_outliers",
    "rssiC_median",
    "rssiC_std",
    "rssiC_outliers",
    "rssiD_median",
    "rssiD_std",
    "rssiD_outliers",
]

ALL_FEATURE_COLUMNS = ANCHOR_COLUMNS + STATS_COLUMNS
CONSERVATIVE_FEATURE_COLUMNS = ANCHOR_COLUMNS + CONSERVATIVE_STATS_COLUMNS

TARGET_COLUMN = "ground_truth_cell"
TIMESTAMP_COLUMN = "timestamp"

CSV_COLUMNS = [
    "timestamp",
    "ground_truth_x",
    "ground_truth_y",
    "ground_truth_cell",
    "rssi_A",
    "rssi_B",
    "rssi_C",
    "rssi_D",
    "anchors_valid",
    "sample_number",
    "collection_time_ms",
    "beacon_moving",
    "environment",
    "notes",
]
REQUIRED_COLUMNS = [
    "timestamp",
    "ground_truth_cell",
    "rssi_A",
    "rssi_B",
    "rssi_C",
    "rssi_D",
]

GRID_COLS = 5
GRID_ROWS = 5
CELL_WIDTH = 2.0
CELL_HEIGHT = 2.0
AREA_WIDTH = 10.0
AREA_HEIGHT = 10.0

VALID_CELLS = [f"{col}{row}" for col in "ABCDE" for row in "12345"]

MIN_SAMPLES_PER_CELL = 50
RECOMMENDED_SAMPLES_PER_CELL = 100
TOTAL_MINIMUM_SAMPLES = 1250

MAX_MISSING_PER_ANCHOR = 0.20
MAX_MISSING_PER_CELL = 0.10
MAX_MISSING_OVERALL = 0.15

MAX_OUTLIER_PERCENTAGE = 0.05

MAX_RSSI_STD_PER_CELL = 15.0

MIN_ANCHOR_DETECTION_RATE = 0.80
CRITICAL_ANCHOR_DETECTION_RATE = 0.50

MAX_CELL_IMBALANCE = 0.50

SMOOTHING_WINDOW_SIZE = 5

# 80/20 Split Strategy (Best Practice for Small Datasets < 2,000 samples)
# Validation provided by 5-Fold StratifiedKFold CV during hyperparameter tuning
TRAIN_SIZE = 0.80
TEST_SIZE = 0.20
RANDOM_STATE = 42

USE_ADAPTIVE_AUGMENTATION = False  # ❌ DISABLED: No augmentation for K-Fold CV

LGBM_PARAMS = {
    "objective": "multiclass",
    "metric": "multi_logloss",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,  # ✅ FIXED: 0.0005 was too low for 1,250 samples
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 20,
    "verbose": -1,
    "n_jobs": 2,
    "random_state": RANDOM_STATE,
}

NUM_BOOST_ROUND = 1000
LOG_EVALUATION_PERIOD = 10

ADAPTIVE_AUGMENTATION_ENABLED = False  # ❌ DISABLED: No augmentation for K-Fold CV

NOISE_HIGH_RSSI = 2.0
NOISE_MEDIUM_RSSI = 3.0
NOISE_LOW_RSSI = 4.0

RSSI_HIGH_THRESHOLD = -60
RSSI_MEDIUM_THRESHOLD = -75

MIN_SAMPLES_FOR_AUGMENTATION = 30
TARGET_SAMPLES_PER_CLASS = 35
MAX_SAMPLES_PER_CLASS = 40

LGBM_REG_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 63,
    "learning_rate": 0.0001,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 20,
    "verbose": -1,
    "n_jobs": 2,
    "random_state": RANDOM_STATE,
}

LGBM_REG_SEARCH_SPACE = {
    "num_leaves": [31, 63, 127],
    "learning_rate": [0.05, 0.07, 0.01],
    "max_depth": [-1, 10, 20],
    "n_estimators": [200, 400, 800],
}

EXTRA_TREES_PARAMS = {
    "n_estimators": 400,
    "max_depth": None,
    "max_features": 0.8,
    "min_samples_leaf": 1,
    "n_jobs": 2,
    "random_state": RANDOM_STATE,
}

EXTRA_TREES_SEARCH_SPACE = {
    "n_estimators": [200, 400, 600],
    "max_depth": [None, 20, 40],
    "max_features": [0.6, 0.8, 1.0],
}

ENSEMBLE_WEIGHT_GRID = [0.2, 0.4, 0.5, 0.6, 0.8]
NOISE_STD_FOR_ROBUSTNESS = 2.0

MODEL_FILENAME = "ble_localizer_lgbm.pkl"
ENCODER_FILENAME = "label_encoder.pkl"
SCALER_FILENAME = "feature_scaler.pkl"
ENSEMBLE_MODEL_FILENAME = "ble_localizer_ensemble.pkl"

MERGED_FILENAME = "dataset_merged.csv"
CLEANED_FILENAME = "dataset_cleaned.csv"
TRAIN_FILENAME = "train.csv"
TEST_FILENAME = "test.csv"

EDA_REPORT_FILENAME = "eda_report.html"
VALIDATION_REPORT_FILENAME = "validation_report.txt"

FIGURE_DPI = 150
FIGURE_SIZE_SMALL = (8, 6)
FIGURE_SIZE_MEDIUM = (10, 8)
FIGURE_SIZE_LARGE = (12, 10)

HEATMAP_CMAP = "YlGnBu"
RSSI_HEATMAP_CMAP = "RdYlGn_r"
CORRELATION_CMAP = "coolwarm"

PARALLEL_INFERENCE = True
PARALLEL_MAX_WORKERS = 4
PARALLEL_TIMEOUT_SEC = 1.0

import os

GPU_CONFIG = {
    "use_gpu": True,
    "gpu_device_id": 0,
    "gpu_platform_id": 0,
    "gpu_use_dp": False,
    "max_bin": 63,
    "num_threads": 1,
}

PARALLEL_TRAINING = True
TRAINING_N_JOBS = 2
TRAINING_VERBOSE = 10

OPTUNA_TRIAL_PRESETS = {"quick": 20, "normal": 50, "thorough": 100, "production": 200}

TUNE_MODE = os.getenv("TUNE_MODE", "normal")
DEFAULT_N_TRIALS = OPTUNA_TRIAL_PRESETS.get(TUNE_MODE, 50)

OPTUNA_EARLY_STOPPING_ROUNDS = 20

EARLY_STOPPING_CONFIGS = {
    "aggressive": 20,
    "balanced": 30,
    "conservative": 50,
    "patient": 100,
}

EARLY_STOP_MODE = os.getenv("EARLY_STOP_MODE", "balanced")
EARLY_STOPPING_ROUNDS = EARLY_STOPPING_CONFIGS.get(EARLY_STOP_MODE, 30)

LOG_FORMAT = "[%(asctime)s] %(levelname)s: %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
VERBOSE = True

def get_full_path(directory: Path, filename: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    return directory / filename

def get_model_path() -> Path:
    return get_full_path(MODELS_DIR, MODEL_FILENAME)

def get_encoder_path() -> Path:
    return get_full_path(MODELS_DIR, ENCODER_FILENAME)

def get_scaler_path() -> Path:
    return get_full_path(MODELS_DIR, SCALER_FILENAME)

AUGMENTATION_WEAKNESS_MULTIPLIER = 0.5
AUGMENTATION_BASE_STD_MIN = 0.5

AUGMENTATION_DRIFT_MIN = -1.0
AUGMENTATION_DRIFT_MAX = +1.0

AUGMENTATION_RSSI_MARGIN = 5
AUGMENTATION_GLOBAL_MIN_RSSI = -101
AUGMENTATION_GLOBAL_MAX_RSSI = -48

AUGMENTATION_MEAN_DIFF_THRESHOLD = 2.0
AUGMENTATION_STD_DIFF_THRESHOLD = 1.5
AUGMENTATION_BALANCE_STD_THRESHOLD = 5.0

AUGMENTATION_TARGET_SAMPLES = 100

EVAL_CELL_SIZE_METERS = 2.0

EVAL_ADJACENT_MANHATTAN_DISTANCE = 1

EVAL_SPATIAL_ERROR_TARGET = 2.0
EVAL_SPATIAL_ERROR_ACCEPTABLE = 3.0

EVAL_CONFUSION_MATRIX_DPI = 150
EVAL_HEATMAP_VMIN = 0.7
EVAL_HEATMAP_VMAX = 1.0

EVAL_TARGET_ACCURACY = 0.85
EVAL_MIN_ACCEPTABLE_ACCURACY = 0.82
EVAL_TARGET_F1_WEIGHTED = 0.83
EVAL_TARGET_ADJACENT_TOLERANCE = 0.93

if __name__ == "__main__":
    print("=" * 60)
    print("ML PIPELINE CONFIGURATION")
    print("=" * 60)
    print(f"\nDirectories:")
    print(f"  Base:      {BASE_DIR}")
    print(f"  Data:      {DATA_DIR}")
    print(f"  Processed: {PROCESSED_DIR}")
    print(f"  Reports:   {REPORTS_DIR}")
    print(f"  Models:    {MODELS_DIR}")
    print(f"\nQuality Criteria:")
    print(f"  Min samples/cell:  {MIN_SAMPLES_PER_CELL}")
    print(f"  Max missing/anchor: {MAX_MISSING_PER_ANCHOR*100:.0f}%")
    print(f"  Max outliers:       {MAX_OUTLIER_PERCENTAGE*100:.0f}%")
    print(f"\nSplit Ratios:")
    print(f"  Train: {TRAIN_SIZE*100:.0f}%")
    print(f"  Test:  {TEST_SIZE*100:.0f}%")
    print(f"\nValid Cells: {len(VALID_CELLS)} ({VALID_CELLS[0]} to {VALID_CELLS[-1]})")
    print("=" * 60)