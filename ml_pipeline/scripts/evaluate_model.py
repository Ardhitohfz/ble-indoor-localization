import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config_ml import (
    EVAL_CELL_SIZE_METERS,
    EVAL_ADJACENT_MANHATTAN_DISTANCE,
    EVAL_CONFUSION_MATRIX_DPI,
    EVAL_HEATMAP_VMIN,
    EVAL_HEATMAP_VMAX,
)

def cell_to_coords(cell: str) -> Tuple[int, int]:
    col = ord(cell[0]) - ord("A")
    row = int(cell[1]) - 1
    return (col, row)

def cell_to_meters(
    cell: str, cell_size: float = EVAL_CELL_SIZE_METERS
) -> Tuple[float, float]:
    col, row = cell_to_coords(cell)
    x = (col * cell_size) + (cell_size / 2)
    y = (row * cell_size) + (cell_size / 2)
    return (x, y)

def _calculate_euclidean_distance(cell1: str, cell2: str) -> float:
    coords1 = cell_to_meters(cell1)
    coords2 = cell_to_meters(cell2)

    distance = np.sqrt((coords1[0] - coords2[0]) ** 2 + (coords1[1] - coords2[1]) ** 2)

    return distance

def calculate_adjacent_tolerance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    correct = 0
    for true_cell, pred_cell in zip(y_true, y_pred):
        true_coords = cell_to_coords(true_cell)
        pred_coords = cell_to_coords(pred_cell)

        distance = abs(true_coords[0] - pred_coords[0]) + abs(
            true_coords[1] - pred_coords[1]
        )

        if distance <= EVAL_ADJACENT_MANHATTAN_DISTANCE:
            correct += 1

    return correct / len(y_true)

def calculate_spatial_error(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    errors = []
    for true_cell, pred_cell in zip(y_true, y_pred):
        error = _calculate_euclidean_distance(true_cell, pred_cell)
        errors.append(error)

    errors = np.array(errors)

    return {
        "mean": float(np.mean(errors)),
        "median": float(np.median(errors)),
        "std": float(np.std(errors)),
        "max": float(np.max(errors)),
        "min": float(np.min(errors)),
        "p95": float(np.percentile(errors, 95)),
    }

def calculate_per_cell_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    cells = np.unique(y_true)
    results = []

    for cell in cells:
        mask = y_true == cell
        support = mask.sum()

        if support > 0:
            cell_accuracy = (y_pred[mask] == cell).sum() / support

            cell_true_binary = (y_true == cell).astype(int)
            cell_pred_binary = (y_pred == cell).astype(int)

            tp = ((cell_true_binary == 1) & (cell_pred_binary == 1)).sum()
            fp = ((cell_true_binary == 0) & (cell_pred_binary == 1)).sum()
            fn = ((cell_true_binary == 1) & (cell_pred_binary == 0)).sum()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0
            )

            results.append(
                {
                    "cell": cell,
                    "support": int(support),
                    "accuracy": cell_accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "row": cell[0],
                    "col": int(cell[1]),
                }
            )

    df = pd.DataFrame(results)
    return df.sort_values("cell")

def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, output_path: Path, normalize: bool = False
):
    cells = sorted(np.unique(np.concatenate([y_true, y_pred])))

    cm = confusion_matrix(y_true, y_pred, labels=cells)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        fmt = ".2f"
        cmap = "Blues"
    else:
        fmt = "d"
        cmap = "YlOrRd"

    plt.figure(figsize=(14, 12))
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        xticklabels=cells,
        yticklabels=cells,
        cbar_kws={"label": "Normalized Count" if normalize else "Count"},
    )

    plt.title("Confusion Matrix (25×25)" + (" - Normalized" if normalize else ""))
    plt.ylabel("True Cell")
    plt.xlabel("Predicted Cell")
    plt.tight_layout()
    plt.savefig(output_path, dpi=EVAL_CONFUSION_MATRIX_DPI, bbox_inches="tight")
    plt.close()

    print(f"[INFO] Saved confusion matrix to {output_path}")

def plot_per_cell_heatmap(per_cell_df: pd.DataFrame, output_path: Path):
    accuracy_grid = np.zeros((5, 5))

    for _, row in per_cell_df.iterrows():
        r = ord(row["row"]) - ord("A")
        c = row["col"] - 1
        accuracy_grid[r, c] = row["accuracy"]

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        accuracy_grid,
        annot=True,
        fmt=".2%",
        cmap="RdYlGn",
        vmin=EVAL_HEATMAP_VMIN,
        vmax=EVAL_HEATMAP_VMAX,
        xticklabels=["1", "2", "3", "4", "5"],
        yticklabels=["A", "B", "C", "D", "E"],
        cbar_kws={"label": "Accuracy"},
    )

    plt.title("Per-Cell Accuracy Heatmap")
    plt.xlabel("Column")
    plt.ylabel("Row")
    plt.tight_layout()
    plt.savefig(output_path, dpi=EVAL_CONFUSION_MATRIX_DPI, bbox_inches="tight")
    plt.close()

    print(f"[INFO] Saved per-cell heatmap to {output_path}")

def plot_spatial_error_distribution(
    y_true: np.ndarray, y_pred: np.ndarray, output_path: Path
):
    errors = []
    for true_cell, pred_cell in zip(y_true, y_pred):
        error = _calculate_euclidean_distance(true_cell, pred_cell)
        errors.append(error)

    errors = np.array(errors)

    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=30, edgecolor="black", alpha=0.7)
    mean_error = float(errors.mean())
    median_error = float(np.median(errors))
    plt.axvline(mean_error, color="r", linestyle="--", label=f"Mean: {mean_error:.2f}m")
    plt.axvline(
        median_error, color="g", linestyle="--", label=f"Median: {median_error:.2f}m"
    )
    plt.axvline(
        EVAL_CELL_SIZE_METERS,
        color="orange",
        linestyle="--",
        label=f"1 Cell Width ({EVAL_CELL_SIZE_METERS}m)",
    )

    plt.xlabel("Spatial Error (meters)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Spatial Prediction Errors")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=EVAL_CONFUSION_MATRIX_DPI, bbox_inches="tight")
    plt.close()

    print(f"[INFO] Saved spatial error distribution to {output_path}")

def evaluate_model(model_path: Path, test_data_path: Path, output_dir: Path) -> Dict:
    print("=" * 70)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("=" * 70)

    print(f"\n[INFO] Loading model from {model_path}")

    if model_path.suffix == ".txt":
        model = lgb.Booster(model_file=str(model_path))
        print(f"[INFO] Loaded LightGBM model from text format")

        feature_info_path = model_path.parent / "feature_info.json"
        if feature_info_path.exists():
            with open(feature_info_path, "r") as f:
                feature_info = json.load(f)
            feature_names = feature_info["feature_columns"]
            print(
                f"[INFO] Loaded {len(feature_names)} features from {feature_info_path.name}"
            )
        else:
            raise FileNotFoundError(f"Feature info not found: {feature_info_path}")
    else:
        import joblib

        model_data = joblib.load(model_path)
        model = model_data["model"]
        class_names = model_data["class_names"]
        feature_names = model_data["feature_names"]
        print(f"[INFO] Loaded model from pickle format")

    print(f"[INFO] Loading test data from {test_data_path}")
    test_df = pd.read_csv(test_data_path)
    X_test = test_df[feature_names].to_numpy()
    y_true = test_df["ground_truth_cell"].to_numpy()

    if model_path.suffix == ".txt":
        class_names = sorted(test_df["ground_truth_cell"].unique())
        print(f"[INFO] Generated {len(class_names)} class names from test data")

    print(f"[INFO] Running predictions on {len(test_df)} samples...")
    y_pred_proba = model.predict(X_test)
    y_pred_proba = np.asarray(y_pred_proba)
    y_pred_encoded = np.argmax(y_pred_proba, axis=1)
    y_pred = np.array([class_names[i] for i in y_pred_encoded])

    print("\n[INFO] Calculating metrics...")

    metrics = {}

    metrics["accuracy"] = accuracy_score(y_true, y_pred)

    f1_weighted = f1_score(y_true, y_pred, average="weighted")
    f1_macro = f1_score(y_true, y_pred, average="macro")
    metrics["f1_weighted"] = f1_weighted
    metrics["f1_macro"] = f1_macro

    metrics["cohen_kappa"] = cohen_kappa_score(y_true, y_pred)

    metrics["adjacent_tolerance"] = calculate_adjacent_tolerance(y_true, y_pred)

    spatial_metrics = calculate_spatial_error(y_true, y_pred)
    metrics["spatial_error"] = spatial_metrics

    per_cell_df = calculate_per_cell_accuracy(y_true, y_pred)

    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(
        f"\nOverall Accuracy:       {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)"
    )
    print(f"F1 Score (weighted):    {metrics['f1_weighted']:.4f}")
    print(f"F1 Score (macro):       {metrics['f1_macro']:.4f}")
    print(f"Cohen's Kappa:          {metrics['cohen_kappa']:.4f}")
    print(
        f"Adjacent Tolerance:     {metrics['adjacent_tolerance']:.4f} ({metrics['adjacent_tolerance']*100:.2f}%)"
    )

    print(f"\nSpatial Error (meters):")
    print(f"  Mean:   {spatial_metrics['mean']:.3f}m")
    print(f"  Median: {spatial_metrics['median']:.3f}m")
    print(f"  Std:    {spatial_metrics['std']:.3f}m")
    print(f"  Max:    {spatial_metrics['max']:.3f}m")
    print(f"  P95:    {spatial_metrics['p95']:.3f}m")

    print(f"\nPer-Cell Accuracy:")
    print(f"  Mean:   {per_cell_df['accuracy'].mean():.4f}")
    print(f"  Std:    {per_cell_df['accuracy'].std():.4f}")
    print(
        f"  Min:    {per_cell_df['accuracy'].min():.4f} (cell: {per_cell_df.loc[per_cell_df['accuracy'].idxmin(), 'cell']})"
    )
    print(
        f"  Max:    {per_cell_df['accuracy'].max():.4f} (cell: {per_cell_df.loc[per_cell_df['accuracy'].idxmax(), 'cell']})"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[INFO] Saved metrics to {metrics_path}")

    per_cell_path = output_dir / "per_cell_accuracy.csv"
    per_cell_df.to_csv(per_cell_path, index=False)
    print(f"[INFO] Saved per-cell metrics to {per_cell_path}")

    spatial_errors = []
    for true_cell, pred_cell in zip(y_true, y_pred):
        error = _calculate_euclidean_distance(true_cell, pred_cell)
        spatial_errors.append(error)

    predictions_df = pd.DataFrame(
        {
            "ground_truth_cell": y_true,
            "predicted_cell": y_pred,
            "correct": y_true == y_pred,
            "spatial_error_m": spatial_errors,
            **{f"prob_{cls}": y_pred_proba[:, i] for i, cls in enumerate(class_names)},
        }
    )
    predictions_path = output_dir / "predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)
    print(f"[INFO] Saved predictions to {predictions_path}")

    cm_path = output_dir / "confusion_matrix.png"
    plot_confusion_matrix(y_true, y_pred, cm_path, normalize=False)

    cm_norm_path = output_dir / "confusion_matrix_normalized.png"
    plot_confusion_matrix(y_true, y_pred, cm_norm_path, normalize=True)

    heatmap_path = output_dir / "per_cell_accuracy_heatmap.png"
    plot_per_cell_heatmap(per_cell_df, heatmap_path)

    spatial_path = output_dir / "spatial_error_distribution.png"
    plot_spatial_error_distribution(y_true, y_pred, spatial_path)

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print(f"\nAll results saved to: {output_dir}")

    return metrics

def main():
    parser = argparse.ArgumentParser(description="Evaluate BLE localization model")
    parser.add_argument(
        "--model", type=Path, required=True, help="Path to trained model (.pkl or .txt)"
    )
    parser.add_argument(
        "--test-data", type=Path, required=True, help="Path to test CSV file"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/evaluation"),
        help="Directory to save evaluation results",
    )

    args = parser.parse_args()

    if not args.model.exists():
        print(f"[ERROR] Model file not found: {args.model}")
        return 1

    if not args.test_data.exists():
        print(f"[ERROR] Test data not found: {args.test_data}")
        return 1

    try:
        evaluate_model(args.model, args.test_data, args.output_dir)
        return 0

    except Exception as e:
        print(f"\n[ERROR] Evaluation failed: {e}")
        import traceback

        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())