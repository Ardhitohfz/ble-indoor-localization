import joblib
from pathlib import Path
from typing import List, Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder

from config_ml import (
    ANCHOR_COLUMNS,
    STATS_COLUMNS,
    CONSERVATIVE_STATS_COLUMNS,
    ALL_FEATURE_COLUMNS,
    TARGET_COLUMN,
    TIMESTAMP_COLUMN,
    SENTINEL_VALUES,
    MISSING_RSSI_VALUE,
    MIN_VALID_RSSI,
    MAX_VALID_RSSI,
    SMOOTHING_WINDOW_SIZE,
    MIN_SAMPLES_PER_CELL,
    MODELS_DIR,
    ENCODER_FILENAME,
)
from .logger import get_logger
from .errors import PreprocessingError

logger = get_logger(__name__)

FEATURE_MODES = {
    "basic": {
        "generate_diffs": False,
        "generate_ratios": False,
        "generate_stats": False,
        "use_conservative_stats": False,
        "description": "Basic: RSSI values only (4 features)",
    },
    "conservative": {
        "generate_diffs": True,
        "generate_ratios": False,
        "generate_stats": True,
        "use_conservative_stats": True,
        "description": "Conservative: RSSI + diffs + essential stats (28 features, optimal)",
    },
    "standard": {
        "generate_diffs": True,
        "generate_ratios": False,
        "generate_stats": True,
        "use_conservative_stats": False,
        "description": "Standard: RSSI + diffs + all statistics (48 features)",
    },
    "advanced": {
        "generate_diffs": True,
        "generate_ratios": True,
        "generate_stats": True,
        "use_conservative_stats": False,
        "description": "Advanced: All features including ratios (54 features)",
    },
}

VALID_SMOOTHING = ["none", "moving_avg", "ewm"]

class BLEDataPreprocessor(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        feature_mode: str = "standard",
        smoothing: str = "ewm",
        smoothing_window: int = SMOOTHING_WINDOW_SIZE,
        min_samples_per_cell: int = MIN_SAMPLES_PER_CELL,
    ) -> None:
        if feature_mode not in FEATURE_MODES:
            raise ValueError(
                f"Invalid feature_mode '{feature_mode}'. "
                f"Choose from: {list(FEATURE_MODES.keys())}"
            )

        if smoothing not in VALID_SMOOTHING:
            raise ValueError(
                f"Invalid smoothing '{smoothing}'. " f"Choose from: {VALID_SMOOTHING}"
            )

        self.feature_mode = feature_mode
        self.smoothing = smoothing
        self.smoothing_window = smoothing_window
        self.min_samples_per_cell = min_samples_per_cell

        mode_config = FEATURE_MODES[feature_mode]
        self.generate_diffs = mode_config["generate_diffs"]
        self.generate_ratios = mode_config["generate_ratios"]
        self.generate_stats = mode_config["generate_stats"]
        self.use_conservative_stats = mode_config.get("use_conservative_stats", False)

        self.encoder = LabelEncoder()
        self.is_fitted = False
        self.feature_names_: List[str] = []
        self.n_classes_: int = 0

        logger.debug(
            f"Initialized preprocessor: mode={feature_mode}, smoothing={smoothing}, "
            f"window={smoothing_window}"
        )

    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> "BLEDataPreprocessor":
        logger.info("Fitting preprocessor...")

        try:
            if TARGET_COLUMN in X.columns:
                targets = X[TARGET_COLUMN].dropna().unique()
                targets = sorted([t for t in targets if t is not None])

                if len(targets) > 0:
                    self.encoder.fit(targets)
                    self.n_classes_ = len(targets)
                    self.is_fitted = True
                    logger.info(f"Fitted encoder with {self.n_classes_} classes")
                else:
                    logger.warning("No valid target labels found")
            else:
                logger.warning(f"Target column '{TARGET_COLUMN}' not found in data")

            return self

        except Exception as e:
            logger.error(f"Fitting failed: {e}")
            raise PreprocessingError(f"Fitting failed: {e}") from e

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        logger.info("Transforming data...")

        try:
            df = X.copy()
            original_size = len(df)

            df = self._remove_duplicates(df)

            df = self._clean_data(df)

            df = self._smooth_data(df)

            df = self._engineer_features(df)

            if TARGET_COLUMN in df.columns and self.is_fitted:
                mask = df[TARGET_COLUMN].isin(self.encoder.classes_)
                df = df[mask].copy()
                df["target_encoded"] = self.encoder.transform(df[TARGET_COLUMN])

            self.feature_names_ = self.get_feature_names(df)

            logger.info(
                f"Transformed: {original_size} -> {len(df)} samples, "
                f"{len(self.feature_names_)} features"
            )

            return df

        except Exception as e:
            logger.error(f"Transformation failed: {e}")
            raise PreprocessingError(f"Transformation failed: {e}") from e

    def fit_transform(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        key_cols = []
        if TIMESTAMP_COLUMN in df.columns:
            key_cols.append(TIMESTAMP_COLUMN)
        if TARGET_COLUMN in df.columns:
            key_cols.append(TARGET_COLUMN)

        if not key_cols:
            return df

        before = len(df)
        df = df.drop_duplicates(subset=key_cols, keep="first")
        after = len(df)

        if before > after:
            logger.info(f"Removed {before - after} duplicates")

        return df

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in ANCHOR_COLUMNS:
            if col not in df.columns:
                continue

            df[col] = df[col].replace(SENTINEL_VALUES, MISSING_RSSI_VALUE)

            df.loc[df[col] < MIN_VALID_RSSI, col] = MIN_VALID_RSSI
            df.loc[df[col] > MAX_VALID_RSSI, col] = MISSING_RSSI_VALUE

        valid_cols = [c for c in ANCHOR_COLUMNS if c in df.columns]
        df[valid_cols] = df[valid_cols].fillna(MISSING_RSSI_VALUE)

        return df

    def _smooth_data(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.smoothing == "none":
            return df

        valid_cols = [c for c in ANCHOR_COLUMNS if c in df.columns]

        if self.smoothing == "moving_avg":
            if TARGET_COLUMN in df.columns:
                df[valid_cols] = df.groupby(TARGET_COLUMN)[valid_cols].transform(
                    lambda x: x.rolling(
                        window=self.smoothing_window, min_periods=1
                    ).mean()
                )
            else:
                df[valid_cols] = (
                    df[valid_cols]
                    .rolling(window=self.smoothing_window, min_periods=1)
                    .mean()
                )

        elif self.smoothing == "ewm":
            if TARGET_COLUMN in df.columns:
                df[valid_cols] = df.groupby(TARGET_COLUMN)[valid_cols].transform(
                    lambda x: x.ewm(span=self.smoothing_window, adjust=False).mean()
                )
            else:
                df[valid_cols] = (
                    df[valid_cols].ewm(span=self.smoothing_window, adjust=False).mean()
                )

        return df

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        valid_cols = [c for c in ANCHOR_COLUMNS if c in df.columns]

        new_features = {}

        if self.generate_diffs:
            for i in range(len(valid_cols)):
                for j in range(i + 1, len(valid_cols)):
                    col_a = valid_cols[i]
                    col_b = valid_cols[j]
                    name_a = col_a.split("_")[1]
                    name_b = col_b.split("_")[1]
                    new_features[f"diff_{name_a}_{name_b}"] = df[col_a] - df[col_b]

        if self.generate_ratios:
            for i in range(len(valid_cols)):
                for j in range(i + 1, len(valid_cols)):
                    col_a = valid_cols[i]
                    col_b = valid_cols[j]
                    name_a = col_a.split("_")[1]
                    name_b = col_b.split("_")[1]
                    new_features[f"ratio_{name_a}_{name_b}"] = df[col_a] / (
                        df[col_b] - 1e-6
                    )

        if self.generate_stats:
            rssi_values = df[valid_cols]
            new_features["rssi_mean"] = rssi_values.mean(axis=1)
            new_features["rssi_std"] = rssi_values.std(axis=1)
            new_features["rssi_min"] = rssi_values.min(axis=1)
            new_features["rssi_max"] = rssi_values.max(axis=1)
            new_features["rssi_range"] = rssi_values.max(axis=1) - rssi_values.min(
                axis=1
            )

        threshold = MISSING_RSSI_VALUE + 5
        new_features["visible_anchors"] = (df[valid_cols] > threshold).sum(axis=1)

        return df.assign(**new_features)

    def get_feature_names(self, df: Optional[pd.DataFrame] = None) -> List[str]:
        if df is None:
            return self.feature_names_

        features = []

        # 1. Raw RSSI values (4 features)
        features.extend([c for c in ANCHOR_COLUMNS if c in df.columns])

        # 2. Statistical features from ESP32 anchors
        if self.use_conservative_stats:
            # Conservative: Only median, std, outliers (12 features)
            # Removes redundant: mean (overlaps raw RSSI), min/max (noisy), quartiles (overlaps std)
            stats_to_use = CONSERVATIVE_STATS_COLUMNS
        else:
            # Standard/Advanced: All 32 statistical features
            stats_to_use = STATS_COLUMNS
        
        features.extend([c for c in stats_to_use if c in df.columns])

        # 3. Engineered difference features (6 features)
        features.extend([c for c in df.columns if c.startswith("diff_")])

        # 4. Engineered ratio features (6 features if enabled)
        features.extend([c for c in df.columns if c.startswith("ratio_")])

        # 5. Aggregate statistics across anchors (5 features)
        # Note: These are computed from 4 raw RSSI values (different from ESP32 stats)
        stat_cols = ["rssi_mean", "rssi_std", "rssi_min", "rssi_max", "rssi_range"]
        features.extend([c for c in stat_cols if c in df.columns])

        # 6. Visibility feature (1 feature)
        if "visible_anchors" in df.columns:
            features.append("visible_anchors")

        return features

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        if not self.feature_names_:
            logger.warning("get_feature_names_out called before fitting/transforming")
            return np.array([], dtype=object)

        return np.array(self.feature_names_, dtype=object)

    def filter_low_sample_cells(self, df: pd.DataFrame) -> pd.DataFrame:
        if TARGET_COLUMN not in df.columns:
            return df

        counts = df.groupby(TARGET_COLUMN).size()
        valid_cells = counts[counts >= self.min_samples_per_cell].index

        before = len(df)
        df = df[df[TARGET_COLUMN].isin(valid_cells)]
        after = len(df)

        if before > after:
            removed = set(counts.index) - set(valid_cells)
            logger.info(
                f"Removed {before - after} samples from {len(removed)} "
                f"low-sample cells: {removed}"
            )

        return df

    def save(self, path: Optional[Path] = None) -> Path:
        if path is None:
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            path = MODELS_DIR / ENCODER_FILENAME

        state = {
            "encoder": self.encoder,
            "is_fitted": self.is_fitted,
            "feature_names_": self.feature_names_,
            "n_classes_": self.n_classes_,
            "feature_mode": self.feature_mode,
            "smoothing": self.smoothing,
            "smoothing_window": self.smoothing_window,
            "min_samples_per_cell": self.min_samples_per_cell,
            "generate_diffs": self.generate_diffs,
            "generate_ratios": self.generate_ratios,
            "generate_stats": self.generate_stats,
            "use_conservative_stats": self.use_conservative_stats,
        }

        joblib.dump(state, path)
        logger.info(f"Saved preprocessor to {path}")

        return path

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "BLEDataPreprocessor":
        if path is None:
            path = MODELS_DIR / ENCODER_FILENAME

        try:
            logger.info(f"Loading preprocessor from {path}")
            state = joblib.load(path)

            feature_mode = state.get("feature_mode", "standard")
            smoothing = state.get("smoothing", "ewm")

            preprocessor = cls(
                feature_mode=feature_mode,
                smoothing=smoothing,
                smoothing_window=state.get("smoothing_window", SMOOTHING_WINDOW_SIZE),
                min_samples_per_cell=state.get(
                    "min_samples_per_cell", MIN_SAMPLES_PER_CELL
                ),
            )

            preprocessor.encoder = state["encoder"]
            preprocessor.is_fitted = state["is_fitted"]
            preprocessor.feature_names_ = state.get("feature_names_", [])
            preprocessor.n_classes_ = state.get("n_classes_", 0)

            logger.info(
                f"Loaded preprocessor: {preprocessor.n_classes_} classes, "
                f"{len(preprocessor.feature_names_)} features"
            )

            return preprocessor

        except Exception as e:
            logger.error(f"Failed to load preprocessor: {e}")
            raise PreprocessingError(f"Failed to load preprocessor: {e}") from e

def preprocess_for_inference(rssi_values: dict) -> pd.DataFrame:
    data = {}
    for col in ANCHOR_COLUMNS:
        anchor_key = "ANCHOR_" + col.split("_")[1]
        value = rssi_values.get(anchor_key, MISSING_RSSI_VALUE)
        data[col] = [value]

    df = pd.DataFrame(data)

    preprocessor = BLEDataPreprocessor(feature_mode="standard", smoothing="none")

    processed = preprocessor.transform(df)

    return processed

if __name__ == "__main__":
    print("=" * 60)
    print("BLE Preprocessor - Demo")
    print("=" * 60)

    sample_data = {
        "timestamp": ["2025-01-01 10:00:00"] * 10,
        "ground_truth_cell": ["A1"] * 5 + ["B2"] * 5,
        "rssi_A": [-65, -67, -64, -66, -68, -75, -77, -74, -76, -78],
        "rssi_B": [-70, -72, -69, -71, -73, -65, -67, -64, -66, -68],
        "rssi_C": [-80, -82, -79, -81, -83, -70, -72, -69, -71, -73],
        "rssi_D": [-85, -87, -84, -86, -88, -80, -82, -79, -81, -83],
    }

    df = pd.DataFrame(sample_data)
    print(f"\nOriginal data: {len(df)} samples")
    print(df.head())

    print("\n" + "=" * 60)
    print("Testing Feature Modes:")
    print("=" * 60)

    for mode in ["basic", "standard", "advanced"]:
        print(f"\nMode: {mode}")
        print(f"  Description: {FEATURE_MODES[mode]['description']}")

        preprocessor = BLEDataPreprocessor(feature_mode=mode, smoothing="ewm")
        preprocessor.fit(df)
        processed = preprocessor.transform(df)

        features = preprocessor.get_feature_names_out()
        print(f"  Features ({len(features)}): {list(features)[:5]}...")

    print("\n" + "=" * 60)
    print("Single Sample Inference:")
    print("=" * 60)
    rssi = {"ANCHOR_A": -65, "ANCHOR_B": -72, "ANCHOR_C": -68, "ANCHOR_D": -70}
    features = preprocess_for_inference(rssi)
    print(f"Input: {rssi}")
    print(f"Output features: {len(features.columns)} columns")
    print(features.head())

    print("\n" + "=" * 60)