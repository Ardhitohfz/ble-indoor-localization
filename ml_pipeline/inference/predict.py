import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from config_ml import (
    MODELS_DIR,
    MODEL_FILENAME,
    ENSEMBLE_MODEL_FILENAME,
    ANCHOR_COLUMNS,
    STATS_COLUMNS,
    MISSING_RSSI_VALUE,
)
from core.preprocessing import BLEDataPreprocessor
from core.errors import ModelError, PredictionError
from core.validation import validate_rssi_input
from core.logger import get_logger

logger = get_logger(__name__)

class BLEPredictor:

    def __init__(self, model_path: Optional[Path] = None, feature_mode: str = "conservative") -> None:
        self.model: Optional[lgb.Booster] = None
        self.feature_names: List[str] = []
        self.class_names: List[str] = []
        self.n_classes: int = 0
        self.preprocessor = BLEDataPreprocessor(
            feature_mode=feature_mode, smoothing="none", smoothing_window=1
        )
        self.is_loaded = False

        if model_path is None:
            model_path = MODELS_DIR / MODEL_FILENAME

        self._load_model(model_path)

    def _load_model(self, model_path: Path) -> None:
        try:
            logger.info(f"Loading model from {model_path}")

            if not model_path.exists():
                raise ModelError(
                    f"Model file not found at {model_path}. "
                    "Run train_model.py first to create the model."
                )

            model_data = joblib.load(model_path)

            required_keys = ["model", "feature_names", "class_names", "n_classes"]
            missing_keys = [k for k in required_keys if k not in model_data]
            if missing_keys:
                raise ModelError(
                    f"Invalid model file format. Missing keys: {missing_keys}"
                )

            self.model = model_data["model"]
            self.feature_names = model_data["feature_names"]
            self.class_names = model_data["class_names"]
            self.n_classes = model_data["n_classes"]

            self.preprocessor.feature_names_ = self.feature_names

            self.is_loaded = self.model is not None

            logger.info(
                f"Model loaded successfully: {self.n_classes} classes, "
                f"{len(self.feature_names)} features"
            )
            logger.info(f"Model accuracy: {model_data['metrics']['accuracy']:.4f}")

        except FileNotFoundError as e:
            logger.error(f"Model file not found: {e}")
            raise ModelError(f"Model file not found at {model_path}") from e

        except KeyError as e:
            logger.error(f"Invalid model format: missing {e}")
            raise ModelError(f"Invalid model format: missing key {e}") from e

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise ModelError(f"Failed to load model: {e}") from e

    @validate_rssi_input
    def predict(self, rssi_values: Dict[str, int]) -> Tuple[str, float]:
        if not self.is_loaded or self.model is None:
            raise PredictionError(
                "Model not loaded. Check model file path and try reloading."
            )

        try:
            X = self._prepare_input(rssi_values)

            if len(X) != 1:
                raise PredictionError(
                    f"Expected exactly 1 sample, got {len(X)}. "
                    "This method only supports single-sample prediction."
                )

            best_iter = getattr(self.model, "best_iteration", None)
            if best_iter is None:
                logger.warning("best_iteration not found, using all trees")
                best_iter = self.model.num_trees()

            probs_raw = self.model.predict(X, num_iteration=best_iter)
            probs = np.asarray(probs_raw)[0]

            max_idx = int(np.argmax(probs))
            confidence = float(probs[max_idx])
            label = self.class_names[max_idx]

            logger.debug(f"Predicted {label} with confidence {confidence:.3f}")

            return label, confidence

        except PredictionError:
            raise

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise PredictionError(f"Prediction computation failed: {e}") from e

    @validate_rssi_input
    def predict_top_k(
        self, rssi_values: Dict[str, int], k: int = 3
    ) -> List[Tuple[str, float]]:
        if not self.is_loaded or self.model is None:
            raise PredictionError("Model not loaded")

        if k < 1 or k > self.n_classes:
            raise ValueError(f"k must be between 1 and {self.n_classes}, got {k}")

        try:
            X = self._prepare_input(rssi_values)

            if len(X) != 1:
                raise PredictionError(f"Expected exactly 1 sample, got {len(X)}")

            best_iter = getattr(self.model, "best_iteration", None)
            if best_iter is None:
                best_iter = self.model.num_trees()

            probs_raw = self.model.predict(X, num_iteration=best_iter)
            probs = np.asarray(probs_raw)[0]

            top_indices = np.argsort(probs)[::-1][:k]

            results = [
                (self.class_names[idx], float(probs[idx])) for idx in top_indices
            ]

            return results

        except PredictionError:
            raise

        except Exception as e:
            logger.error(f"Top-K prediction failed: {e}")
            raise PredictionError(f"Top-K prediction failed: {e}") from e

    def _prepare_input(self, rssi_values: Dict[str, int]) -> pd.DataFrame:
        """Prepare input DataFrame from raw RSSI and statistical features.
        
        Args:
            rssi_values: Dictionary with keys like 'ANCHOR_A', 'ANCHOR_B', etc.
                        Can also contain statistical feature keys like 'rssiA_mean', etc.
        
        Returns:
            DataFrame with all features ready for prediction
        """
        input_data = {}
        
        # 1. Raw RSSI values (4 features)
        for col in ANCHOR_COLUMNS:
            anchor_key = "ANCHOR_" + col.split("_")[1]
            value = rssi_values.get(anchor_key, MISSING_RSSI_VALUE)
            input_data[col] = [value]
        
        # 2. Statistical features from ESP32 (32 features)
        # These come from the anchor's CH_STATS_UUID characteristic
        for col in STATS_COLUMNS:
            # Try to get from input, otherwise set to empty (will be handled by model)
            value = rssi_values.get(col, None)
            if value is not None:
                input_data[col] = [value]

        df = pd.DataFrame(input_data)

        # Preprocessor will add engineered features (diffs, ratios, etc.)
        processed_df = self.preprocessor.transform(df)

        # Select only features that model was trained on
        X = processed_df[self.feature_names]

        return X

    def get_model_info(self) -> Dict[str, Any]:
        if not self.is_loaded:
            return {"loaded": False}

        return {
            "loaded": True,
            "n_classes": self.n_classes,
            "n_features": len(self.feature_names),
            "class_names": self.class_names,
            "feature_names": self.feature_names,
        }

class BLEEnsembleRegressor:

    def __init__(
        self, model_path: Optional[Path] = None, enable_parallel: bool = None
    ) -> None:
        from ..config_ml import PARALLEL_INFERENCE, PARALLEL_MAX_WORKERS

        self.model_path = model_path or (MODELS_DIR / ENSEMBLE_MODEL_FILENAME)
        self.weight: float = 0.5
        self.feature_names: List[str] = []
        self.preprocessor: BLEDataPreprocessor = BLEDataPreprocessor(
            feature_mode="standard", smoothing="none", smoothing_window=1
        )
        self.models_loaded = False

        self.lgbm_x: Optional[Any] = None
        self.lgbm_y: Optional[Any] = None
        self.et_x: Optional[Any] = None
        self.et_y: Optional[Any] = None

        self._use_parallel = (
            enable_parallel if enable_parallel is not None else PARALLEL_INFERENCE
        )
        self._executor: Optional[ThreadPoolExecutor] = None
        if self._use_parallel:
            self._executor = ThreadPoolExecutor(max_workers=PARALLEL_MAX_WORKERS)
            logger.info(f"Parallel inference ENABLED ({PARALLEL_MAX_WORKERS} workers)")
        else:
            logger.info("Parallel inference DISABLED (sequential mode)")

        self._load_model()

    def _load_model(self) -> None:
        try:
            logger.info(f"Loading ensemble model from {self.model_path}")

            if not self.model_path.exists():
                raise ModelError(f"Ensemble model not found at {self.model_path}")

            data = joblib.load(self.model_path)

            self.lgbm_x = data.get("lgbm_x")
            self.lgbm_y = data.get("lgbm_y")
            self.et_x = data.get("et_x")
            self.et_y = data.get("et_y")
            self.weight = data.get("weight", 0.5)
            self.feature_names = data["feature_names"]

            self.preprocessor.feature_names_ = self.feature_names
            self.models_loaded = all([self.lgbm_x, self.lgbm_y, self.et_x, self.et_y])

            if self.models_loaded:
                logger.info(f"Ensemble loaded successfully (weight={self.weight:.2f})")
            else:
                raise ModelError("Ensemble model incomplete (missing components)")

        except FileNotFoundError as e:
            logger.error(f"Ensemble model file not found: {e}")
            raise ModelError(f"Ensemble model not found") from e

        except Exception as e:
            logger.error(f"Failed to load ensemble model: {e}")
            raise ModelError(f"Failed to load ensemble model: {e}") from e

    def _prepare_input(self, rssi_values: Dict[str, int]) -> pd.DataFrame:
        input_data = {}
        for col in ANCHOR_COLUMNS:
            anchor_key = "ANCHOR_" + col.split("_")[1]
            input_data[col] = [rssi_values.get(anchor_key, MISSING_RSSI_VALUE)]
        df = pd.DataFrame(input_data)
        processed = self.preprocessor.transform(df)
        return processed[self.feature_names]

    def _predict_parallel(self, X: pd.DataFrame) -> Tuple[float, float]:
        from concurrent.futures import as_completed
        from ..config_ml import PARALLEL_TIMEOUT_SEC

        assert self.lgbm_x is not None
        assert self.lgbm_y is not None
        assert self.et_x is not None
        assert self.et_y is not None
        assert self._executor is not None

        futures = {
            "lgbm_x": self._executor.submit(self.lgbm_x.predict, X),
            "lgbm_y": self._executor.submit(self.lgbm_y.predict, X),
            "et_x": self._executor.submit(self.et_x.predict, X),
            "et_y": self._executor.submit(self.et_y.predict, X),
        }

        results = {}
        try:
            for name, future in futures.items():
                results[name] = float(future.result(timeout=PARALLEL_TIMEOUT_SEC)[0])
        except Exception as e:
            logger.error(f"Parallel prediction failed: {e}")
            raise PredictionError(f"Parallel prediction failed: {e}") from e

        x = self.weight * results["lgbm_x"] + (1 - self.weight) * results["et_x"]
        y = self.weight * results["lgbm_y"] + (1 - self.weight) * results["et_y"]

        return x, y

    def _predict_sequential(self, X: pd.DataFrame) -> Tuple[float, float]:
        assert self.lgbm_x is not None
        assert self.lgbm_y is not None
        assert self.et_x is not None
        assert self.et_y is not None

        lgbm_x_pred = float(self.lgbm_x.predict(X)[0])
        lgbm_y_pred = float(self.lgbm_y.predict(X)[0])
        et_x_pred = float(self.et_x.predict(X)[0])
        et_y_pred = float(self.et_y.predict(X)[0])

        x = self.weight * lgbm_x_pred + (1 - self.weight) * et_x_pred
        y = self.weight * lgbm_y_pred + (1 - self.weight) * et_y_pred

        return x, y

    @validate_rssi_input
    def predict_coordinates(self, rssi_values: Dict[str, int]) -> Tuple[float, float]:
        if not self.models_loaded:
            raise PredictionError("Ensemble models not loaded")

        try:
            X = self._prepare_input(rssi_values)

            if len(X) != 1:
                raise PredictionError(f"Expected 1 sample, got {len(X)}")

            if self._use_parallel and self._executor is not None:
                x, y = self._predict_parallel(X)
            else:
                x, y = self._predict_sequential(X)

            logger.debug(f"Predicted coordinates: ({x:.2f}, {y:.2f})")
            return x, y

        except PredictionError:
            raise

        except Exception as e:
            logger.error(f"Coordinate prediction failed: {e}")
            raise PredictionError(f"Coordinate prediction failed: {e}") from e

    def __del__(self):
        if hasattr(self, "_executor") and self._executor is not None:
            self._executor.shutdown(wait=False)
            logger.debug("Thread pool shutdown")

def predict_single(rssi_values: Dict[str, int]) -> Tuple[str, float]:
    predictor = BLEPredictor()
    return predictor.predict(rssi_values)

if __name__ == "__main__":
    print("=" * 60)
    print("BLE PREDICTOR - Demo")
    print("=" * 60)

    try:
        predictor = BLEPredictor()

        if not predictor.is_loaded:
            print("\nModel not found. Please run train_model.py first.")
            exit(1)

        info = predictor.get_model_info()
        print(f"\nModel Info:")
        print(f"  Classes: {info['n_classes']}")
        print(f"  Features: {info['n_features']}")

        test_samples = [
            {"ANCHOR_A": -60, "ANCHOR_B": -75, "ANCHOR_C": -80, "ANCHOR_D": -90},
            {"ANCHOR_A": -80, "ANCHOR_B": -60, "ANCHOR_C": -70, "ANCHOR_D": -85},
            {"ANCHOR_A": -70, "ANCHOR_B": -70, "ANCHOR_C": -60, "ANCHOR_D": -75},
        ]

        print("\nTest Predictions:")
        for i, rssi in enumerate(test_samples):
            label, conf = predictor.predict(rssi)
            print(f"\n  Sample {i+1}: {rssi}")
            print(f"  Predicted: {label} ({conf:.1%})")

            top3 = predictor.predict_top_k(rssi, k=3)
            print(f"  Top 3: {[(l, f'{c:.1%}') for l, c in top3]}")

        print("\n" + "=" * 60)

    except (ModelError, PredictionError) as e:
        print(f"\nError: {e}")
        exit(1)