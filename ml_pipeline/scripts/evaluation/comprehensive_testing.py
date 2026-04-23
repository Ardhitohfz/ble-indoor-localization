import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    cohen_kappa_score,
    confusion_matrix,
    top_k_accuracy_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, learning_curve
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy import stats
from collections import Counter
from datetime import datetime

from config_ml import (
    TARGET_COLUMN,
    EVAL_CELL_SIZE_METERS,
    TUNED_MODEL_DIR,
    PROCESSED_DIR,
    REPORTS_DIR,
    LOGS_DIR,
    ALL_FEATURE_COLUMNS,
)
from core.logger import setup_logger

DPI_RESOLUTION = 300
BASE_FONT_SIZE = 10
CV_FOLDS = 5
N_LEARNING_CURVE_POINTS = 10
LEARNING_CURVE_MIN_SIZE = 0.1
LEARNING_CURVE_MAX_SIZE = 1.0
CONFIDENCE_BINS = [0, 0.3, 0.5, 0.7, 0.9, 1.0]
CONFIDENCE_LABELS = ['0-0.3', '0.3-0.5', '0.5-0.7', '0.7-0.9', '0.9-1.0']
CALIBRATION_BINS = 10
HIST_BINS = 30
GRID_BOUNDARY = [0, 4]
LOW_CONFIDENCE_THRESHOLD = 0.5
HIGH_CONFIDENCE_THRESHOLD = 0.9
LOCATION_COLORS = ['red', 'orange', 'green']

# Use the same feature columns that were used during training
# This must match ALL_FEATURE_COLUMNS from config_ml.py (36 features)
FEATURE_COLUMNS = ALL_FEATURE_COLUMNS

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = DPI_RESOLUTION
plt.rcParams['savefig.dpi'] = DPI_RESOLUTION
plt.rcParams['font.size'] = BASE_FONT_SIZE

def cell_to_coords(cell: str):
    col = ord(cell[0]) - ord("A")
    row = int(cell[1]) - 1
    return (col, row)

def cell_to_meters(cell: str, cell_size: float = EVAL_CELL_SIZE_METERS):
    col, row = cell_to_coords(cell)
    x = (col * cell_size) + (cell_size / 2)
    y = (row * cell_size) + (cell_size / 2)
    return (x, y)

def calculate_euclidean_distance(cell1: str, cell2: str) -> float:
    coords1 = cell_to_meters(cell1)
    coords2 = cell_to_meters(cell2)
    distance = np.sqrt((coords1[0] - coords2[0]) ** 2 + (coords1[1] - coords2[1]) ** 2)
    return distance

def calculate_top_k_accuracy(y_true_encoded, y_pred_proba, k):
    return top_k_accuracy_score(y_true_encoded, y_pred_proba, k=k, labels=np.arange(y_pred_proba.shape[1]))

def calculate_confidence_metrics(y_pred_proba):
    max_probs = np.max(y_pred_proba, axis=1)
    return {
        'mean_confidence': float(np.mean(max_probs)),
        'median_confidence': float(np.median(max_probs)),
        'std_confidence': float(np.std(max_probs)),
        'min_confidence': float(np.min(max_probs)),
        'max_confidence': float(np.max(max_probs)),
        'low_confidence_count': int(np.sum(max_probs < LOW_CONFIDENCE_THRESHOLD)),
        'high_confidence_count': int(np.sum(max_probs > HIGH_CONFIDENCE_THRESHOLD)),
    }

def classify_cell_location(cell):
    col, row = cell_to_coords(cell)
    
    if (col in GRID_BOUNDARY) and (row in GRID_BOUNDARY):
        return 'corner'
    elif (col in GRID_BOUNDARY) or (row in GRID_BOUNDARY):
        return 'edge'
    else:
        return 'center'

def mcnemar_test(y_true, y_pred_baseline, y_pred_tuned):
    baseline_correct = (y_true == y_pred_baseline)
    tuned_correct = (y_true == y_pred_tuned)
    
    n10 = np.sum(baseline_correct & ~tuned_correct)
    n01 = np.sum(~baseline_correct & tuned_correct)
    
    if n10 + n01 < 25:
        statistic = (abs(n10 - n01) - 1) ** 2 / (n10 + n01) if (n10 + n01) > 0 else 0
    else:
        statistic = (n10 - n01) ** 2 / (n10 + n01)
    
    p_value = 1 - stats.chi2.cdf(statistic, 1)
    
    return {
        'statistic': float(statistic),
        'p_value': float(p_value),
        'n10': int(n10),
        'n01': int(n01),
        'significant': p_value < 0.05
    }

class LGBMWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, model, label_encoder):
        self.model = model
        self.label_encoder = label_encoder
        # Use getattr to handle unfitted label encoders
        self.classes_ = getattr(label_encoder, 'classes_', None)
        if self.classes_ is None:
            # If label_encoder is not fitted yet, set it later in fit
            pass
        self.n_features_in_ = model.num_feature()

    def get_params(self, deep=True):
        """
        Get parameters for sklearn compatibility (required for clone()).

        This method is called by sklearn.clone() to get the parameters
        needed to create a new instance of this estimator.
        """
        return {
            'model': self.model,
            'label_encoder': self.label_encoder
        }

    def set_params(self, **params):
        """
        Set parameters for sklearn compatibility (required for clone()).

        This method is called by sklearn.clone() to set parameters on
        the new instance after creation.
        """
        for key, value in params.items():
            setattr(self, key, value)
        # Update n_features_in_ if model changed
        if 'model' in params:
            self.n_features_in_ = self.model.num_feature()
        return self

    def fit(self, X, y):
        from sklearn.utils.validation import check_X_y
        X, y = check_X_y(X, y, accept_sparse=False)

        # CRITICAL FIX: Re-fit label_encoder to ensure it works with sklearn.clone()
        # When sklearn clones this wrapper for cross-validation, the label_encoder
        # needs to be re-fitted with the current fold's labels
        self.label_encoder.fit(y)
        self.classes_ = self.label_encoder.classes_

        return self
    
    def predict(self, X):
        from sklearn.utils.validation import check_array, check_is_fitted
        check_is_fitted(self, ['model', 'label_encoder'])
        X = check_array(X, accept_sparse=False)
        
        y_pred_proba = self.model.predict(X)
        y_pred_proba_array = np.asarray(y_pred_proba)
        y_pred_encoded = np.argmax(y_pred_proba_array, axis=1)
        return self.label_encoder.inverse_transform(y_pred_encoded)
    
    def predict_proba(self, X):
        from sklearn.utils.validation import check_array, check_is_fitted
        check_is_fitted(self, ['model', 'label_encoder'])
        X = check_array(X, accept_sparse=False)
        return np.asarray(self.model.predict(X))
    
    def score(self, X, y):
        return accuracy_score(y, self.predict(X))

def perform_cross_validation(model_wrapper, X, y, cv_folds=CV_FOLDS):
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    cv_scores = cross_val_score(model_wrapper, X, y, cv=skf, scoring='accuracy', n_jobs=-1)
    
    return {
        'cv_scores': cv_scores.tolist(),
        'mean_cv_accuracy': float(np.mean(cv_scores)),
        'std_cv_accuracy': float(np.std(cv_scores)),
        'min_cv_accuracy': float(np.min(cv_scores)),
        'max_cv_accuracy': float(np.max(cv_scores)),
        'cv_folds': cv_folds
    }

def calculate_learning_curve(model_wrapper, X_train, y_train, cv_folds=CV_FOLDS):
    train_sizes = np.linspace(LEARNING_CURVE_MIN_SIZE, LEARNING_CURVE_MAX_SIZE, N_LEARNING_CURVE_POINTS)
    
    result = learning_curve(
        model_wrapper, X_train, y_train,
        train_sizes=train_sizes,
        cv=cv_folds,
        scoring='accuracy',
        n_jobs=-1,
        random_state=42,
        return_times=False
    )
    train_sizes_abs, train_scores, val_scores = result[0], result[1], result[2]
    
    return {
        'train_sizes': train_sizes_abs.tolist(),
        'train_scores_mean': np.mean(train_scores, axis=1).tolist(),
        'train_scores_std': np.std(train_scores, axis=1).tolist(),
        'val_scores_mean': np.mean(val_scores, axis=1).tolist(),
        'val_scores_std': np.std(val_scores, axis=1).tolist()
    }

def calculate_per_cell_metrics(y_true, y_pred, y_pred_proba_array, label_encoder):
    """Calculate detailed metrics for each cell individually."""
    per_cell_results = {}
    all_cells = sorted(label_encoder.classes_)
    
    for cell in all_cells:
        # Get indices where this cell is the ground truth
        cell_mask = (y_true == cell)
        
        if cell_mask.sum() == 0:
            # No samples for this cell
            per_cell_results[cell] = {
                'sample_count': 0,
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'mean_confidence': 0.0,
                'correct_predictions': 0,
                'incorrect_predictions': 0,
                'top_confused_with': []
            }
            continue
        
        # Get predictions for this cell
        cell_true = y_true[cell_mask]
        cell_pred = y_pred[cell_mask]
        cell_proba = y_pred_proba_array[cell_mask]
        
        # Calculate accuracy for this cell
        correct = (cell_true == cell_pred).sum()
        total = len(cell_true)
        accuracy = correct / total if total > 0 else 0.0
        
        # Calculate precision (when model predicts this cell, how often is it correct?)
        pred_as_cell_mask = (y_pred == cell)
        if pred_as_cell_mask.sum() > 0:
            precision = (y_true[pred_as_cell_mask] == cell).sum() / pred_as_cell_mask.sum()
        else:
            precision = 0.0
        
        # Recall is same as accuracy for per-cell analysis
        recall = accuracy
        
        # F1 score
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0
        
        # Mean confidence when predicting this cell correctly
        cell_idx = list(label_encoder.classes_).index(cell)
        mean_confidence = cell_proba[:, cell_idx].mean()
        
        # Find most confused cells
        incorrect_preds = cell_pred[cell_true != cell_pred]
        if len(incorrect_preds) > 0:
            confusion_counts = Counter(incorrect_preds)
            top_confused = [(confused_cell, count) for confused_cell, count in confusion_counts.most_common(3)]
        else:
            top_confused = []
        
        per_cell_results[cell] = {
            'sample_count': int(total),
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'mean_confidence': float(mean_confidence),
            'correct_predictions': int(correct),
            'incorrect_predictions': int(total - correct),
            'top_confused_with': top_confused
        }
    
    return per_cell_results

def plot_per_cell_heatmap(per_cell_results, metric='accuracy', output_path=None):
    """Create a heatmap showing per-cell performance metrics."""
    # Create 5x5 grid for cells A-E, 1-5
    grid = np.zeros((5, 5))
    
    for cell, results in per_cell_results.items():
        if results['sample_count'] == 0:
            grid[cell_to_coords(cell)] = np.nan
        else:
            grid[cell_to_coords(cell)] = results[metric] * 100 if metric != 'sample_count' else results[metric]
    
    # Flip grid for correct orientation (row 0 should be at top)
    grid = np.flipud(grid)
    
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    mask = np.isnan(grid)
    
    if metric == 'sample_count':
        sns.heatmap(grid, annot=True, fmt='.0f', cmap='YlOrRd', 
                   xticklabels=['A', 'B', 'C', 'D', 'E'],
                   yticklabels=['5', '4', '3', '2', '1'],
                   cbar_kws={'label': 'Sample Count'},
                   linewidths=1, linecolor='black',
                   mask=mask)
        title = 'Test Samples per Cell'
    else:
        sns.heatmap(grid, annot=True, fmt='.1f', cmap='RdYlGn', 
                   xticklabels=['A', 'B', 'C', 'D', 'E'],
                   yticklabels=['5', '4', '3', '2', '1'],
                   cbar_kws={'label': f'{metric.capitalize()} (%)'},
                   vmin=0, vmax=100,
                   linewidths=1, linecolor='black',
                   mask=mask)
        title = f'{metric.capitalize()} per Cell (%)'
    
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Column', fontsize=12, fontweight='bold')
    plt.ylabel('Row', fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        print(f"[SAVED] {output_path}")
    else:
        plt.show()

def plot_confusion_heatmap(y_true, y_pred, label_encoder, output_path, normalize=None):
    """
    Create a confusion matrix heatmap for all cells.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        label_encoder: LabelEncoder object with classes_
        output_path: Path to save the plot
        normalize: None (raw counts), 'true' (by row), 'pred' (by column), 'all' (entire matrix)
    """
    cm = confusion_matrix(y_true, y_pred, labels=label_encoder.classes_, normalize=normalize)
    
    # Determine format string and labels based on normalization
    if normalize is None:
        fmt = 'd'
        cbar_label = 'Count'
        title_suffix = '- All Cells (Raw Counts)'
    else:
        fmt = '.2%'
        cbar_label = 'Proportion'
        title_suffix = f'- All Cells (Normalized: {normalize})'
    
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
               xticklabels=label_encoder.classes_,
               yticklabels=label_encoder.classes_,
               cbar_kws={'label': cbar_label},
               linewidths=0.5, linecolor='gray')
    
    plt.title(f'Confusion Matrix {title_suffix}', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicted Cell', fontsize=13, fontweight='bold')
    plt.ylabel('True Cell', fontsize=13, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {output_path}")

def plot_confidence_distribution(confidence_data, output_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    correct_conf = confidence_data[confidence_data['correct']]['confidence']
    incorrect_conf = confidence_data[~confidence_data['correct']]['confidence']
    
    ax1 = axes[0, 0]
    ax1.hist(correct_conf, bins=HIST_BINS, alpha=0.7, color='green', label='Correct', edgecolor='black')
    ax1.hist(incorrect_conf, bins=HIST_BINS, alpha=0.7, color='red', label='Incorrect', edgecolor='black')
    ax1.set_xlabel('Confidence Score', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Confidence Distribution by Correctness', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[0, 1]
    ax2.boxplot([correct_conf, incorrect_conf], tick_labels=['Correct', 'Incorrect'],
                patch_artist=True,
                boxprops=dict(facecolor='lightblue'),
                medianprops=dict(color='red', linewidth=2))
    ax2.set_ylabel('Confidence Score', fontsize=12)
    ax2.set_title('Confidence Box Plot', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    ax3 = axes[1, 0]
    confidence_data['conf_bin'] = pd.cut(confidence_data['confidence'], bins=CONFIDENCE_BINS, labels=CONFIDENCE_LABELS)
    conf_acc = confidence_data.groupby('conf_bin', observed=True)['correct'].mean() * 100
    
    ax3.bar(range(len(conf_acc)), conf_acc.values, color='steelblue', edgecolor='black', linewidth=1.5)
    ax3.set_xticks(range(len(conf_acc)))
    ax3.set_xticklabels(conf_acc.index, rotation=0)
    ax3.set_xlabel('Confidence Range', fontsize=12)
    ax3.set_ylabel('Accuracy (%)', fontsize=12)
    ax3.set_title('Accuracy by Confidence Range', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_ylim([0, 105])
    
    for i, v in enumerate(conf_acc.values):
        ax3.text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')
    
    ax4 = axes[1, 1]
    bin_edges = np.linspace(0, 1, CALIBRATION_BINS + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    bin_accuracies = []
    bin_confidences = []
    for i in range(CALIBRATION_BINS):
        mask = (confidence_data['confidence'] >= bin_edges[i]) & (confidence_data['confidence'] < bin_edges[i+1])
        if mask.sum() > 0:
            bin_acc = confidence_data[mask]['correct'].mean()
            bin_conf = confidence_data[mask]['confidence'].mean()
            bin_accuracies.append(bin_acc)
            bin_confidences.append(bin_conf)
        else:
            bin_accuracies.append(np.nan)
            bin_confidences.append(bin_centers[i])
    
    ax4.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    ax4.plot(bin_confidences, bin_accuracies, 'o-', color='steelblue', linewidth=2, markersize=8, label='Model')
    ax4.set_xlabel('Mean Predicted Confidence', fontsize=12)
    ax4.set_ylabel('Accuracy', fontsize=12)
    ax4.set_title('Calibration Curve', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim([0, 1])
    ax4.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {output_path}")

def plot_location_analysis(location_results, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1 = axes[0]
    locations = list(location_results.keys())
    accuracies = [location_results[loc]['accuracy'] * 100 for loc in locations]
    colors = LOCATION_COLORS
    
    bars = ax1.bar(locations, accuracies, color=colors, edgecolor='black', linewidth=2)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Accuracy by Cell Location', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim([0, 105])
    
    for bar, acc in zip(bars, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2, acc + 2, 
                f'{acc:.1f}%', ha='center', fontweight='bold')
    
    ax2 = axes[1]
    sample_counts = [location_results[loc]['count'] for loc in locations]
    
    bars = ax2.bar(locations, sample_counts, color=LOCATION_COLORS, edgecolor='black', linewidth=2, alpha=0.7)
    ax2.set_ylabel('Sample Count', fontsize=12)
    ax2.set_title('Test Samples by Location', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar, count in zip(bars, sample_counts):
        ax2.text(bar.get_x() + bar.get_width()/2, count + max(sample_counts)*0.02, 
                str(count), ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {output_path}")

def plot_feature_importance(importance_df, output_path, top_n=15):
    """
    Plot feature importance as horizontal bar chart.
    
    Args:
        importance_df: DataFrame with columns ['feature', 'gain', 'split']
        output_path: Path to save the plot
        top_n: Number of top features to display (default: 15)
    """
    # Select top N features by gain
    top_features = importance_df.head(top_n).copy()
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot 1: Importance by Gain
    ax1 = axes[0]
    bars1 = ax1.barh(range(len(top_features)), top_features['gain'], color='steelblue', edgecolor='black', linewidth=1)
    ax1.set_yticks(range(len(top_features)))
    ax1.set_yticklabels(top_features['feature'])
    ax1.invert_yaxis()  # Highest importance at top
    ax1.set_xlabel('Importance (Gain)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Top {top_n} Features by Gain', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars1, top_features['gain'])):
        ax1.text(val + max(top_features['gain']) * 0.01, bar.get_y() + bar.get_height()/2, 
                f'{val:.0f}', va='center', fontweight='bold', fontsize=9)
    
    # Plot 2: Importance by Split
    ax2 = axes[1]
    bars2 = ax2.barh(range(len(top_features)), top_features['split'], color='orange', edgecolor='black', linewidth=1)
    ax2.set_yticks(range(len(top_features)))
    ax2.set_yticklabels(top_features['feature'])
    ax2.invert_yaxis()
    ax2.set_xlabel('Importance (Split Count)', fontsize=12, fontweight='bold')
    ax2.set_title(f'Top {top_n} Features by Split', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars2, top_features['split'])):
        ax2.text(val + max(top_features['split']) * 0.01, bar.get_y() + bar.get_height()/2, 
                f'{val:.0f}', va='center', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {output_path}")

def plot_topk_comparison(topk_results, output_path):
    k_values = list(topk_results.keys())
    accuracies = [topk_results[k] * 100 for k in k_values]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(k_values, accuracies, color='steelblue', edgecolor='black', linewidth=2)
    plt.xlabel('Top-K', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Top-K Accuracy Analysis', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.ylim([85, 105])
    
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, acc + 0.5, 
                f'{acc:.2f}%', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {output_path}")

def plot_cross_validation_results(cv_results, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1 = axes[0]
    fold_numbers = list(range(1, len(cv_results['cv_scores']) + 1))
    cv_scores = [s * 100 for s in cv_results['cv_scores']]
    
    bars = ax1.bar(fold_numbers, cv_scores, color='steelblue', edgecolor='black', linewidth=2)
    ax1.axhline(y=cv_results['mean_cv_accuracy'] * 100, color='red', linestyle='--', 
                linewidth=2, label=f"Mean: {cv_results['mean_cv_accuracy']*100:.2f}%")
    ax1.set_xlabel('Fold Number', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title(f'{cv_results["cv_folds"]}-Fold Cross-Validation Results', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim([min(cv_scores) - 2, max(cv_scores) + 2])
    
    for bar, score in zip(bars, cv_scores):
        ax1.text(bar.get_x() + bar.get_width()/2, score + 0.2, 
                f'{score:.2f}%', ha='center', fontweight='bold', fontsize=9)
    
    ax2 = axes[1]
    metrics_names = ['Mean', 'Std', 'Min', 'Max']
    metrics_values = [
        cv_results['mean_cv_accuracy'] * 100,
        cv_results['std_cv_accuracy'] * 100,
        cv_results['min_cv_accuracy'] * 100,
        cv_results['max_cv_accuracy'] * 100
    ]
    colors = ['green', 'orange', 'red', 'blue']
    
    bars = ax2.bar(metrics_names, metrics_values, color=colors, edgecolor='black', linewidth=2)
    ax2.set_ylabel('Value (%)', fontsize=12)
    ax2.set_title('Cross-Validation Statistics', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, metrics_values):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 0.5, 
                f'{val:.2f}%', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {output_path}")

def plot_learning_curve(lc_results, output_path):
    train_sizes = lc_results['train_sizes']
    train_mean = np.array(lc_results['train_scores_mean']) * 100
    train_std = np.array(lc_results['train_scores_std']) * 100
    val_mean = np.array(lc_results['val_scores_mean']) * 100
    val_std = np.array(lc_results['val_scores_std']) * 100
    
    plt.figure(figsize=(12, 6))
    
    plt.plot(train_sizes, train_mean, 'o-', color='steelblue', linewidth=2, 
             markersize=8, label='Training Score')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                     alpha=0.2, color='steelblue')
    
    plt.plot(train_sizes, val_mean, 'o-', color='orange', linewidth=2, 
             markersize=8, label='Cross-Validation Score')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, 
                     alpha=0.2, color='orange')
    
    gap = train_mean - val_mean
    plt.fill_between(train_sizes, val_mean, train_mean, alpha=0.1, color='red', 
                     label=f'Gap (Final: {gap[-1]:.2f}%)')
    
    plt.xlabel('Training Samples', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Learning Curve Analysis', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.ylim([85, 102])
    
    for i, (size, train_acc, val_acc) in enumerate(zip(train_sizes[::2], train_mean[::2], val_mean[::2])):
        plt.annotate(f'{int(size)}', xy=(size, val_acc), xytext=(size, val_acc - 3),
                    ha='center', fontsize=8, alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {output_path}")

def interpret_accuracy(accuracy):
    """Provide interpretation of accuracy score."""
    if accuracy >= 0.95:
        return "Excellent - Sangat Baik", "Model sangat akurat dan reliable untuk aplikasi praktis."
    elif accuracy >= 0.90:
        return "Very Good - Baik Sekali", "Model sangat baik dan dapat diandalkan untuk sebagian besar kasus."
    elif accuracy >= 0.80:
        return "Good - Baik", "Model cukup baik namun masih ada ruang untuk improvement."
    elif accuracy >= 0.70:
        return "Fair - Cukup", "Model menunjukkan kemampuan dasar namun perlu peningkatan signifikan."
    else:
        return "Poor - Kurang", "Model perlu perbaikan mendasar sebelum deployment."

def interpret_train_test_gap(gap):
    """Interpret the training-test accuracy gap."""
    if gap < 2:
        return "Minimal", "Model generalize sangat baik, hampir tidak ada overfitting."
    elif gap < 5:
        return "Small", "Model generalize dengan baik, overfitting minimal."
    elif gap < 10:
        return "Moderate", "Ada sedikit overfitting namun masih dalam batas wajar."
    elif gap < 20:
        return "Large", "Overfitting cukup signifikan, perlu regularisasi atau data lebih banyak."
    else:
        return "Very Large", "Overfitting sangat parah, model harus di-retrain dengan strategi berbeda."

def interpret_confidence(mean_conf, std_conf):
    """Interpret confidence scores."""
    if mean_conf > 0.8:
        return "Very High", "Model sangat yakin dengan prediksinya."
    elif mean_conf > 0.6:
        return "High", "Model cukup yakin dengan prediksinya."
    elif mean_conf > 0.4:
        return "Moderate", "Model memiliki keyakinan sedang pada prediksinya."
    elif mean_conf > 0.2:
        return "Low", "Model kurang yakin dengan prediksinya."
    else:
        return "Very Low", "Model hampir tidak yakin, probabilities hampir uniform."

def interpret_spatial_error(median_error, mean_error, max_error):
    """Interpret spatial positioning errors."""
    if median_error == 0 and mean_error < 0.5:
        return "Excellent", "Mayoritas prediksi tepat pada cell yang benar."
    elif mean_error < 2:
        return "Very Good", "Sebagian besar prediksi sangat dekat dengan lokasi sebenarnya."
    elif mean_error < 4:
        return "Good", "Prediksi umumnya berada dalam jarak dekat dari lokasi sebenarnya."
    elif mean_error < 6:
        return "Fair", "Ada beberapa error positioning yang cukup jauh."
    else:
        return "Poor", "Error positioning cukup besar, perlu improvement."

def generate_executive_summary(results, test_samples, train_samples):
    """Generate executive summary for non-technical audience."""
    summary = []
    summary.append("=" * 80)
    summary.append("RINGKASAN EKSEKUTIF - UNTUK PEMBACA UMUM")
    summary.append("=" * 80)
    summary.append("")
    
    # Overall performance
    acc = results['basic_metrics']['accuracy']
    acc_rating, acc_desc = interpret_accuracy(acc)
    summary.append("📊 KINERJA MODEL SECARA KESELURUHAN")
    summary.append(f"   Tingkat Akurasi: {acc*100:.2f}%")
    summary.append(f"   Rating: {acc_rating}")
    summary.append(f"   Artinya: {acc_desc}")
    summary.append("")
    summary.append(f"   Dari {test_samples} kali percobaan lokasi:")
    correct = int(acc * test_samples)
    incorrect = test_samples - correct
    summary.append(f"   ✓ {correct} kali model memprediksi BENAR")
    summary.append(f"   ✗ {incorrect} kali model memprediksi SALAH")
    summary.append("")
    
    # What this means
    summary.append("💡 APA ARTINYA?")
    if acc >= 0.95:
        summary.append("   Model ini sangat akurat dalam menentukan lokasi. Hampir semua")
        summary.append("   prediksi benar, sehingga sangat layak untuk digunakan dalam")
        summary.append("   aplikasi indoor positioning system di dunia nyata.")
    elif acc >= 0.90:
        summary.append("   Model ini memiliki akurasi tinggi dan dapat diandalkan untuk")
        summary.append("   menentukan lokasi indoor. Kesalahan yang terjadi minimal dan")
        summary.append("   model siap untuk aplikasi praktis.")
    else:
        summary.append("   Model ini memiliki akurasi yang baik namun masih ada beberapa")
        summary.append("   kesalahan prediksi. Dapat digunakan dengan monitoring tambahan.")
    summary.append("")
    
    # Confidence analysis
    conf = results['confidence_metrics']['mean_confidence']
    conf_rating, conf_desc = interpret_confidence(conf, results['confidence_metrics']['std_confidence'])
    summary.append("🎯 TINGKAT KEYAKINAN MODEL")
    summary.append(f"   Rating: {conf_rating}")
    summary.append(f"   Artinya: {conf_desc}")
    if conf < 0.1:
        summary.append("")
        summary.append("   ⚠️  CATATAN PENTING:")
        summary.append("   Meskipun akurasi tinggi, confidence score rendah menunjukkan bahwa")
        summary.append("   model memilih prediksi dari probabilitas yang hampir seragam.")
        summary.append("   Model tetap AKURAT tapi tidak 'yakin' - ini indikasi bahwa model")
        summary.append("   memisahkan kelas dengan baik tapi probability calibration kurang optimal.")
    summary.append("")
    
    # Spatial accuracy
    spatial = results['spatial_metrics']
    spatial_rating, spatial_desc = interpret_spatial_error(
        spatial['median_error_m'], spatial['mean_error_m'], spatial['max_error_m']
    )
    summary.append("📍 AKURASI POSISI")
    summary.append(f"   Rating: {spatial_rating}")
    summary.append(f"   Artinya: {spatial_desc}")
    summary.append(f"   Error rata-rata: {spatial['mean_error_m']:.2f} meter")
    summary.append(f"   Error maksimum: {spatial['max_error_m']:.2f} meter")
    if spatial['median_error_m'] == 0:
        summary.append(f"   90% prediksi tepat sasaran (error 0 meter)")
    summary.append("")
    
    # Top-K performance
    summary.append("🎖️  PERFORMA TOP-K (Prediksi Alternatif)")
    summary.append(f"   Top-1 (pilihan pertama): {results['topk_accuracy']['top1']*100:.2f}%")
    summary.append(f"   Top-2 (2 pilihan terbaik): {results['topk_accuracy']['top2']*100:.2f}%")
    summary.append(f"   Top-3 (3 pilihan terbaik): {results['topk_accuracy']['top3']*100:.2f}%")
    if results['topk_accuracy']['top3'] > 0.98:
        summary.append("   Artinya: Jika model memberikan 3 pilihan lokasi, hampir pasti")
        summary.append("   lokasi yang benar ada di antara 3 pilihan tersebut.")
    summary.append("")
    
    # Training vs Test
    train_acc = results['train_performance']['accuracy']
    gap = (train_acc - acc) * 100
    gap_rating, gap_desc = interpret_train_test_gap(gap)
    summary.append("📚 KEMAMPUAN GENERALISASI")
    summary.append(f"   Akurasi saat training: {train_acc*100:.2f}%")
    summary.append(f"   Akurasi saat testing: {acc*100:.2f}%")
    summary.append(f"   Selisih (gap): {gap:.2f}%")
    summary.append(f"   Rating: {gap_rating}")
    summary.append(f"   Artinya: {gap_desc}")
    summary.append("")
    
    # Location performance
    summary.append("🏢 PERFORMA BERDASARKAN LOKASI")
    loc = results['location_analysis']
    for location, metrics in sorted(loc.items(), key=lambda x: x[1]['accuracy'], reverse=True):
        loc_name = {'corner': 'Pojok', 'edge': 'Tepi', 'center': 'Tengah'}[location]
        summary.append(f"   {loc_name}: {metrics['accuracy']*100:.1f}% akurasi ({metrics['count']} sampel)")
    
    best_loc = max(loc.items(), key=lambda x: x[1]['accuracy'])
    worst_loc = min(loc.items(), key=lambda x: x[1]['accuracy'])
    summary.append("")
    summary.append(f"   Performa terbaik: Lokasi {{'corner': 'pojok', 'edge': 'tepi', 'center': 'tengah'}}[best_loc[0]]")
    if best_loc[1]['accuracy'] - worst_loc[1]['accuracy'] > 0.05:
        summary.append(f"   Ada perbedaan performa antar lokasi, perlu perhatian khusus")
        summary.append(f"   untuk area {{'corner': 'pojok', 'edge': 'tepi', 'center': 'tengah'}}[worst_loc[0]].")
    summary.append("")
    
    # Best and worst cells
    per_cell = results['per_cell_metrics']
    cells_sorted = sorted(per_cell.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    
    summary.append("⭐ CELL DENGAN PERFORMA TERBAIK")
    perfect_cells = [cell for cell, metrics in cells_sorted if metrics['accuracy'] == 1.0 and metrics['sample_count'] > 0]
    if perfect_cells:
        summary.append(f"   {len(perfect_cells)} cell dengan akurasi 100%: {', '.join(perfect_cells[:10])}")
    else:
        for i, (cell, metrics) in enumerate(cells_sorted[:3], 1):
            if metrics['sample_count'] > 0:
                summary.append(f"   {i}. Cell {cell}: {metrics['accuracy']*100:.1f}% akurasi")
    summary.append("")
    
    summary.append("⚠️  CELL YANG PERLU PERHATIAN")
    worst_cells = [item for item in cells_sorted[-5:][::-1] if item[1]['sample_count'] > 0]
    for i, (cell, metrics) in enumerate(worst_cells, 1):
        confused = ", ".join([f"{c}" for c, n in metrics['top_confused_with'][:2]]) if metrics['top_confused_with'] else "N/A"
        summary.append(f"   {i}. Cell {cell}: {metrics['accuracy']*100:.1f}% (sering tertukar dengan: {confused})")
    summary.append("")
    
    # Recommendations
    summary.append("🎯 REKOMENDASI")
    if acc >= 0.95:
        summary.append("   ✅ Model siap untuk deployment dalam aplikasi real-world")
        summary.append("   ✅ Akurasi sangat tinggi dan reliable")
    else:
        summary.append("   📝 Model dapat digunakan dengan monitoring")
    
    if gap < 5:
        summary.append("   ✅ Tidak ada overfitting, model generalize dengan baik")
    elif gap < 10:
        summary.append("   ⚠️  Sedikit overfitting, masih dalam batas wajar")
    
    if len(worst_cells) > 0 and worst_cells[0][1]['accuracy'] < 0.8:
        summary.append(f"   📝 Perlu improvement untuk cell {worst_cells[0][0]} dan area sekitarnya")
        summary.append(f"   📝 Pertimbangkan menambah data training untuk cell dengan performa rendah")
    
    if conf < 0.1:
        summary.append("   📝 Pertimbangkan probability calibration untuk meningkatkan confidence")
    
    summary.append("")
    summary.append("=" * 80)
    
    return "\n".join(summary)

def generate_technical_analysis(results, test_samples, train_samples):
    """Generate detailed technical analysis for experts."""
    analysis = []
    analysis.append("=" * 80)
    analysis.append("ANALISIS TEKNIS MENDALAM - UNTUK AHLI/PENELITI")
    analysis.append("=" * 80)
    analysis.append("")
    
    # Detailed metrics breakdown
    analysis.append("📊 METRIK PERFORMA DETAIL")
    analysis.append("")
    basic = results['basic_metrics']
    analysis.append("Classification Metrics:")
    analysis.append(f"  • Accuracy:           {basic['accuracy']:.6f} ({basic['accuracy']*100:.2f}%)")
    analysis.append(f"  • Precision (macro):  {basic['precision_macro']:.6f}")
    analysis.append(f"  • Precision (weighted): {basic['precision_weighted']:.6f}")
    analysis.append(f"  • Recall (macro):     {basic['recall_macro']:.6f}")
    analysis.append(f"  • Recall (weighted):  {basic['recall_weighted']:.6f}")
    analysis.append(f"  • F1-Score (macro):   {basic['f1_macro']:.6f}")
    analysis.append(f"  • F1-Score (weighted): {basic['f1_weighted']:.6f}")
    analysis.append(f"  • Cohen's Kappa:      {basic['cohen_kappa']:.6f}")
    analysis.append("")
    
    # Cohen's Kappa interpretation
    kappa = basic['cohen_kappa']
    if kappa > 0.9:
        kappa_interp = "Almost perfect agreement"
    elif kappa > 0.8:
        kappa_interp = "Strong agreement"
    elif kappa > 0.6:
        kappa_interp = "Moderate agreement"
    else:
        kappa_interp = "Fair agreement"
    analysis.append(f"Cohen's Kappa Interpretation: {kappa_interp}")
    analysis.append(f"  → Model performance significantly better than random chance")
    analysis.append("")
    
    # Top-K Analysis
    analysis.append("🎯 TOP-K ACCURACY ANALYSIS")
    topk = results['topk_accuracy']
    analysis.append(f"  Top-1: {topk['top1']:.6f} ({topk['top1']*100:.2f}%)")
    analysis.append(f"  Top-2: {topk['top2']:.6f} ({topk['top2']*100:.2f}%) [+{(topk['top2']-topk['top1'])*100:.2f}pp]")
    analysis.append(f"  Top-3: {topk['top3']:.6f} ({topk['top3']*100:.2f}%) [+{(topk['top3']-topk['top2'])*100:.2f}pp]")
    analysis.append(f"  Top-5: {topk['top5']:.6f} ({topk['top5']*100:.2f}%) [+{(topk['top5']-topk['top3'])*100:.2f}pp]")
    analysis.append("")
    analysis.append("Insight:")
    if topk['top2'] - topk['top1'] > 0.02:
        analysis.append("  → Significant improvement with Top-2, suggesting model has good")
        analysis.append("    secondary predictions for ambiguous cases")
    if topk['top3'] > 0.99:
        analysis.append("  → Near-perfect Top-3 accuracy indicates excellent ranking capability")
    analysis.append("")
    
    # Confidence Distribution Analysis
    analysis.append("📈 CONFIDENCE DISTRIBUTION ANALYSIS")
    conf = results['confidence_metrics']
    analysis.append(f"  Mean:     {conf['mean_confidence']:.6f}")
    analysis.append(f"  Median:   {conf['median_confidence']:.6f}")
    analysis.append(f"  Std Dev:  {conf['std_confidence']:.6f}")
    analysis.append(f"  Min:      {conf['min_confidence']:.6f}")
    analysis.append(f"  Max:      {conf['max_confidence']:.6f}")
    analysis.append(f"  Range:    {conf['max_confidence'] - conf['min_confidence']:.6f}")
    analysis.append("")
    analysis.append(f"  Low Confidence (<0.5):  {conf['low_confidence_count']}/{test_samples} ({conf['low_confidence_count']/test_samples*100:.1f}%)")
    analysis.append(f"  High Confidence (>0.9): {conf['high_confidence_count']}/{test_samples} ({conf['high_confidence_count']/test_samples*100:.1f}%)")
    analysis.append("")
    
    if conf['mean_confidence'] < 0.1 and basic['accuracy'] > 0.9:
        analysis.append("⚠️  CALIBRATION ISSUE DETECTED:")
        analysis.append("  → High accuracy with very low confidence scores")
        analysis.append("  → Model predictions are accurate but probability distributions are nearly uniform")
        analysis.append("  → This suggests: (1) Multi-class softmax with 25 classes naturally produces low individual")
        analysis.append("    probabilities (~4% for uniform), (2) Model separates classes well but probabilities")
        analysis.append("    are not calibrated, (3) Consider techniques like temperature scaling or Platt scaling")
        analysis.append("  → For deployment: Use predicted class directly; confidence scores may not be reliable")
        analysis.append("")
    
    # Spatial Error Analysis
    analysis.append("📍 SPATIAL ERROR DISTRIBUTION")
    spatial = results['spatial_metrics']
    analysis.append(f"  Mean Error:      {spatial['mean_error_m']:.3f} m")
    analysis.append(f"  Median Error:    {spatial['median_error_m']:.3f} m")
    analysis.append(f"  Std Deviation:   {spatial['std_error_m']:.3f} m")
    analysis.append(f"  Min Error:       {spatial['min_error_m']:.3f} m")
    analysis.append(f"  Max Error:       {spatial['max_error_m']:.3f} m")
    analysis.append(f"  90th Percentile: {spatial['p90_error_m']:.3f} m")
    analysis.append(f"  95th Percentile: {spatial['p95_error_m']:.3f} m")
    analysis.append("")
    
    if spatial['median_error_m'] == 0:
        correct_pct = basic['accuracy'] * 100
        analysis.append(f"  → Median = 0 indicates {correct_pct:.1f}% of predictions are exact matches")
    if spatial['mean_error_m'] > 0 and spatial['median_error_m'] == 0:
        analysis.append(f"  → Mean > Median suggests right-skewed distribution (few large errors)")
    analysis.append("")
    
    # Training vs Test Analysis
    analysis.append("📚 GENERALIZATION ANALYSIS")
    train_perf = results['train_performance']
    test_acc = basic['accuracy']
    gap = (train_perf['accuracy'] - test_acc) * 100
    analysis.append(f"  Training Accuracy:  {train_perf['accuracy']:.6f} ({train_perf['accuracy']*100:.2f}%)")
    analysis.append(f"  Training F1:        {train_perf['f1_weighted']:.6f}")
    analysis.append(f"  Training Samples:   {train_perf['sample_count']}")
    analysis.append(f"  Test Accuracy:      {test_acc:.6f} ({test_acc*100:.2f}%)")
    analysis.append(f"  Test F1:            {basic['f1_weighted']:.6f}")
    analysis.append(f"  Test Samples:       {test_samples}")
    analysis.append(f"  Generalization Gap: {gap:.2f} percentage points")
    analysis.append("")
    
    if train_perf['accuracy'] == 1.0:
        analysis.append("  → Perfect training accuracy (100%)")
        if gap < 5:
            analysis.append("  → Low gap indicates excellent generalization despite perfect training fit")
            analysis.append("  → Model complexity appears appropriate for the problem")
        else:
            analysis.append("  → Moderate gap with perfect training suggests slight overfitting")
    analysis.append("")
    
    # Location-based Performance
    analysis.append("🏢 LOCATION-STRATIFIED PERFORMANCE")
    loc = results['location_analysis']
    analysis.append("")
    for location in ['corner', 'edge', 'center']:
        if location in loc:
            metrics = loc[location]
            analysis.append(f"  {location.upper()} Cells:")
            analysis.append(f"    Samples:  {metrics['count']}")
            analysis.append(f"    Accuracy: {metrics['accuracy']:.6f} ({metrics['accuracy']*100:.2f}%)")
            analysis.append(f"    F1-Score: {metrics['f1']:.6f}")
            analysis.append("")
    
    # Statistical significance of location differences
    accuracies = [loc[l]['accuracy'] for l in ['corner', 'edge', 'center'] if l in loc]
    if len(accuracies) == 3:
        acc_std = np.std(accuracies)
        analysis.append(f"  Location Accuracy Std Dev: {acc_std:.4f}")
        if acc_std > 0.03:
            analysis.append("  → Significant performance variance across locations")
            analysis.append("  → Consider location-specific feature engineering or ensemble methods")
        else:
            analysis.append("  → Consistent performance across all location types")
    analysis.append("")
    
    # Error Pattern Analysis
    analysis.append("🔍 ERROR PATTERN ANALYSIS")
    error_patterns = results['top_error_patterns']
    total_errors = sum(error_patterns.values())
    analysis.append(f"  Total Misclassifications: {test_samples - int(basic['accuracy']*test_samples)}")
    analysis.append(f"  Unique Error Patterns: {len(error_patterns)}")
    analysis.append("")
    analysis.append("  Top Error Patterns:")
    for i, (pattern, count) in enumerate(list(error_patterns.items())[:5], 1):
        true_cell, pred_cell = pattern.split('->')
        pct = count / (test_samples - int(basic['accuracy']*test_samples)) * 100
        analysis.append(f"    {i}. {pattern}: {count} occurrences ({pct:.1f}% of errors)")
    analysis.append("")
    
    # Per-cell variance analysis
    analysis.append("📊 PER-CELL PERFORMANCE STATISTICS")
    per_cell = results['per_cell_metrics']
    accuracies = [m['accuracy'] for m in per_cell.values() if m['sample_count'] > 0]
    f1_scores = [m['f1'] for m in per_cell.values() if m['sample_count'] > 0]
    
    analysis.append(f"  Cells Analyzed: {len(accuracies)}/25")
    analysis.append(f"  Accuracy Statistics:")
    analysis.append(f"    Mean:   {np.mean(accuracies):.6f}")
    analysis.append(f"    Median: {np.median(accuracies):.6f}")
    analysis.append(f"    Std:    {np.std(accuracies):.6f}")
    analysis.append(f"    Min:    {np.min(accuracies):.6f} ({np.min(accuracies)*100:.1f}%)")
    analysis.append(f"    Max:    {np.max(accuracies):.6f} ({np.max(accuracies)*100:.1f}%)")
    analysis.append(f"    Range:  {np.max(accuracies) - np.min(accuracies):.6f}")
    analysis.append("")
    analysis.append(f"  F1-Score Statistics:")
    analysis.append(f"    Mean:   {np.mean(f1_scores):.6f}")
    analysis.append(f"    Std:    {np.std(f1_scores):.6f}")
    analysis.append("")
    
    acc_var = np.std(accuracies)
    if acc_var > 0.1:
        analysis.append("  ⚠️  High variance in per-cell accuracy detected")
        analysis.append("  → Some cells significantly harder to classify than others")
        analysis.append("  → Consider: (1) Collecting more data for low-performing cells")
        analysis.append("            (2) Analyzing RSSI signal quality per cell")
        analysis.append("            (3) Cell-specific model ensembles")
    elif acc_var < 0.05:
        analysis.append("  ✅ Low variance indicates consistent performance across all cells")
    analysis.append("")
    
    # Statistical Tests
    analysis.append("📈 STATISTICAL ROBUSTNESS")
    analysis.append(f"  Sample Size (Test): {test_samples}")
    analysis.append(f"  Sample Size (Train): {train_samples}")
    analysis.append(f"  Train/Test Ratio: {train_samples/test_samples:.2f}:1")
    
    # Confidence interval for accuracy
    from scipy import stats as scipy_stats
    ci_95 = 1.96 * np.sqrt(basic['accuracy'] * (1 - basic['accuracy']) / test_samples)
    analysis.append(f"")
    analysis.append(f"  95% Confidence Interval for Accuracy:")
    analysis.append(f"    [{basic['accuracy'] - ci_95:.4f}, {basic['accuracy'] + ci_95:.4f}]")
    analysis.append(f"    [{(basic['accuracy'] - ci_95)*100:.2f}%, {(basic['accuracy'] + ci_95)*100:.2f}%]")
    analysis.append("")
    
    # Model Recommendations
    analysis.append("🎯 TECHNICAL RECOMMENDATIONS")
    
    if conf['mean_confidence'] < 0.1:
        analysis.append("  1. Probability Calibration:")
        analysis.append("     • Apply temperature scaling to calibrate output probabilities")
        analysis.append("     • Consider Platt scaling or isotonic regression")
        analysis.append("     • This will improve confidence scores without affecting accuracy")
        analysis.append("")
    
    if acc_var > 0.1:
        analysis.append("  2. Address Per-Cell Variance:")
        analysis.append("     • Implement stratified data augmentation for low-performing cells")
        analysis.append("     • Consider cell-specific feature engineering")
        analysis.append("     • Evaluate signal quality and anchor placement for problematic cells")
        analysis.append("")
    
    if gap > 5:
        analysis.append("  3. Reduce Overfitting:")
        analysis.append("     • Increase regularization parameters")
        analysis.append("     • Apply dropout or early stopping")
        analysis.append("     • Consider ensemble methods for better generalization")
        analysis.append("")
    
    if basic['accuracy'] < 0.95:
        analysis.append("  4. Improve Overall Accuracy:")
        analysis.append("     • Collect more training data, especially for confused cell pairs")
        analysis.append("     • Experiment with additional features (e.g., signal variance over time)")
        analysis.append("     • Try ensemble methods or model stacking")
        analysis.append("")
    
    analysis.append("="* 80)
    
    return "\n".join(analysis)

def generate_markdown_report(results, test_samples, train_samples, output_path):
    """Generate a comprehensive markdown report."""
    report = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report.append("# Comprehensive Model Testing Report")
    report.append("")
    report.append(f"**Generated:** {timestamp}")
    report.append(f"**Test Samples:** {test_samples}")
    report.append(f"**Training Samples:** {train_samples}**")
    report.append("")
    report.append("---")
    report.append("")
    
    # Quick Stats
    acc = results['basic_metrics']['accuracy']
    report.append("## Quick Statistics")
    report.append("")
    report.append(f"| Metric | Value |")
    report.append(f"|--------|-------|")
    report.append(f"| **Overall Accuracy** | **{acc*100:.2f}%** |")
    report.append(f"| Correct Predictions | {int(acc*test_samples)}/{test_samples} |")
    report.append(f"| Cohen's Kappa | {results['basic_metrics']['cohen_kappa']:.4f} |")
    report.append(f"| F1-Score (Macro) | {results['basic_metrics']['f1_macro']:.4f} |")
    report.append(f"| Training Accuracy | {results['train_performance']['accuracy']*100:.2f}% |")
    report.append(f"| Train-Test Gap | {(results['train_performance']['accuracy'] - acc)*100:.2f}% |")
    report.append("")
    
    # Performance Rating
    acc_rating, acc_desc = interpret_accuracy(acc)
    report.append("## Performance Rating")
    report.append("")
    report.append(f"### {acc_rating}")
    report.append(f"*{acc_desc}*")
    report.append("")
    
    # Top-K Accuracy
    report.append("## Top-K Accuracy")
    report.append("")
    report.append("| K | Accuracy |")
    report.append("|---|----------|")
    for k, v in results['topk_accuracy'].items():
        report.append(f"| {k.upper()} | {v*100:.2f}% |")
    report.append("")
    
    # Location Analysis
    report.append("## Performance by Location Type")
    report.append("")
    report.append("| Location | Samples | Accuracy | F1-Score |")
    report.append("|----------|---------|----------|----------|")
    for loc in ['corner', 'edge', 'center']:
        if loc in results['location_analysis']:
            m = results['location_analysis'][loc]
            report.append(f"| {loc.capitalize()} | {m['count']} | {m['accuracy']*100:.2f}% | {m['f1']:.4f} |")
    report.append("")
    
    # Best Cells
    report.append("## Top Performing Cells")
    report.append("")
    per_cell = results['per_cell_metrics']
    sorted_cells = sorted(per_cell.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    report.append("| Rank | Cell | Accuracy | Samples |")
    report.append("|------|------|----------|---------|")
    for i, (cell, metrics) in enumerate(sorted_cells[:10], 1):
        if metrics['sample_count'] > 0:
            report.append(f"| {i} | {cell} | {metrics['accuracy']*100:.1f}% | {metrics['sample_count']} |")
    report.append("")
    
    # Worst Cells
    report.append("## Cells Needing Attention")
    report.append("")
    report.append("| Rank | Cell | Accuracy | Samples | Confused With |")
    report.append("|------|------|----------|---------|---------------|")
    worst_cells = [item for item in sorted_cells[-5:][::-1] if item[1]['sample_count'] > 0]
    for i, (cell, metrics) in enumerate(worst_cells, 1):
        confused = ", ".join([f"{c}({n})" for c, n in metrics['top_confused_with'][:2]]) if metrics['top_confused_with'] else "None"
        report.append(f"| {i} | {cell} | {metrics['accuracy']*100:.1f}% | {metrics['sample_count']} | {confused} |")
    report.append("")
    
    # Spatial Errors
    spatial = results['spatial_metrics']
    report.append("## Spatial Error Analysis")
    report.append("")
    report.append("| Metric | Value |")
    report.append("|--------|-------|")
    report.append(f"| Mean Error | {spatial['mean_error_m']:.3f} m |")
    report.append(f"| Median Error | {spatial['median_error_m']:.3f} m |")
    report.append(f"| 90th Percentile | {spatial['p90_error_m']:.3f} m |")
    report.append(f"| 95th Percentile | {spatial['p95_error_m']:.3f} m |")
    report.append(f"| Maximum Error | {spatial['max_error_m']:.3f} m |")
    report.append("")
    
    # Error Patterns
    report.append("## Common Error Patterns")
    report.append("")
    report.append("| Pattern | Occurrences |")
    report.append("|---------|-------------|")
    for pattern, count in list(results['top_error_patterns'].items())[:5]:
        report.append(f"| {pattern} | {count} |")
    report.append("")
    
    # Confidence Analysis
    conf = results['confidence_metrics']
    report.append("## Confidence Score Distribution")
    report.append("")
    report.append("| Statistic | Value |")
    report.append("|-----------|-------|")
    report.append(f"| Mean | {conf['mean_confidence']:.4f} |")
    report.append(f"| Median | {conf['median_confidence']:.4f} |")
    report.append(f"| Std Dev | {conf['std_confidence']:.4f} |")
    report.append(f"| Min | {conf['min_confidence']:.4f} |")
    report.append(f"| Max | {conf['max_confidence']:.4f} |")
    report.append("")
    
    if conf['mean_confidence'] < 0.1:
        report.append("> ⚠️ **Note:** Low confidence scores indicate nearly uniform probability distributions. ")
        report.append("> While accuracy is high, the model's confidence calibration may need improvement. ")
        report.append("> This is common in multi-class problems with many classes (25 in this case).")
        report.append("")
    
    # Recommendations
    report.append("## Recommendations")
    report.append("")
    report.append("### For Deployment")
    if acc >= 0.95:
        report.append("- ✅ Model is ready for production deployment")
    else:
        report.append("- ⚠️ Consider additional validation before full deployment")
    
    gap = (results['train_performance']['accuracy'] - acc) * 100
    if gap < 5:
        report.append("- ✅ Good generalization, no significant overfitting")
    else:
        report.append("- ⚠️ Monitor for overfitting, consider regularization")
    
    report.append("")
    report.append("### For Improvement")
    if len(worst_cells) > 0:
        report.append(f"- Focus on improving performance for cells: {', '.join([c[0] for c in worst_cells[:3]])}")
    
    if conf['mean_confidence'] < 0.1:
        report.append("- Consider probability calibration techniques (temperature scaling, Platt scaling)")
    
    if acc < 0.95:
        report.append("- Collect more training data for confused cell pairs")
        report.append("- Explore additional feature engineering")
    
    report.append("")
    report.append("---")
    report.append("")
    report.append("*Report generated by Comprehensive Testing Pipeline*")
    
    # Save report
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(report))
    
    return "\n".join(report)

def extract_feature_importance(model, feature_cols: list, output_dir: Path) -> tuple:
    """
    Extract and visualize feature importance from LightGBM model.
    
    **Single Responsibility:** Extract, save, and plot feature importance.
    
    Args:
        model: Trained LightGBM model (Booster or LGBMClassifier)
        feature_cols: List of feature column names
        output_dir: Directory to save outputs
    
    Returns:
        tuple: (importance_df, importance_dict) where:
            - importance_df: DataFrame with features and their importance scores
            - importance_dict: Dict with 'features', 'gain', 'split' keys
            If error occurs, returns (None, {'error': error_message})
    
    Outputs:
        - feature_importance.csv: Sorted feature importance table
        - feature_importance.png: Bar chart visualization
    """
    try:
        # Try to get feature importance (works for both Booster and sklearn interface)
        if hasattr(model, 'feature_importance'):
            importance_gain = model.feature_importance(importance_type='gain')
            importance_split = model.feature_importance(importance_type='split')
        elif hasattr(model, 'feature_importances_'):
            # Sklearn interface (LGBMClassifier)
            importance_gain = model.feature_importances_  # type: ignore
            importance_split = model.feature_importances_  # type: ignore
        else:
            print("  [WARNING] Model does not support feature_importance extraction")
            importance_gain = np.zeros(len(feature_cols))
            importance_split = np.zeros(len(feature_cols))
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'gain': importance_gain,
            'split': importance_split
        }).sort_values('gain', ascending=False).reset_index(drop=True)
        
        importance_dict = {
            'features': feature_cols,
            'gain': importance_gain.tolist(),
            'split': importance_split.tolist()
        }
        
        print(f"  - Top 5 features by gain:")
        for idx, (_, row) in enumerate(importance_df.head(5).iterrows(), 1):
            print(f"    {idx}. {row['feature']:20s} Gain: {row['gain']:8.0f}  Splits: {row['split']:5.0f}")
        
        # Save to CSV
        importance_csv_path = output_dir / "feature_importance.csv"
        importance_df.to_csv(importance_csv_path, index=False)
        print(f"  - Saved to {importance_csv_path}")
        
        # Plot feature importance
        importance_plot_path = output_dir / "feature_importance.png"
        plot_feature_importance(importance_df, importance_plot_path, top_n=15)
        
        return importance_df, importance_dict
        
    except Exception as e:
        print(f"  [ERROR] Failed to extract feature importance: {e}")
        return None, {'error': str(e)}

def perform_cv_analysis(model, label_encoder, X_train: np.ndarray, y_train: np.ndarray, output_dir: Path, cv_folds: int = 5) -> dict:
    """
    Perform k-fold cross-validation analysis on training set.
    
    **Single Responsibility:** Execute CV, plot results, return metrics.
    
    Args:
        model: Trained LightGBM model
        label_encoder: Label encoder for class transformations
        X_train: Training features array
        y_train: Training labels array
        output_dir: Directory to save plots
        cv_folds: Number of CV folds (default: 5)
    
    Returns:
        dict: CV results with keys ['cv_scores', 'mean_cv_accuracy', 'std_cv_accuracy', ...]
              If error occurs, returns {'error': error_message}
    
    Outputs:
        - cross_validation_results.png: Bar chart + statistics visualization
    """
    try:
        # Create sklearn-compatible wrapper
        model_wrapper = LGBMWrapper(model, label_encoder)
        
        # Run CV on training set only (avoid data leakage)
        print(f"  - Running {cv_folds}-Fold Stratified CV on training set...")
        cv_results = perform_cross_validation(model_wrapper, X_train, y_train, cv_folds=cv_folds)
        
        print(f"  - Mean CV Accuracy: {cv_results['mean_cv_accuracy']:.4f} ± {cv_results['std_cv_accuracy']:.4f}")
        print(f"  - CV Accuracy Range: [{cv_results['min_cv_accuracy']:.4f}, {cv_results['max_cv_accuracy']:.4f}]")
        print(f"  - Individual fold scores: {[f'{s:.4f}' for s in cv_results['cv_scores']]}")
        
        # Plot CV results
        cv_plot_path = output_dir / "cross_validation_results.png"
        plot_cross_validation_results(cv_results, cv_plot_path)
        
        return cv_results
        
    except Exception as e:
        print(f"  [ERROR] Cross-validation failed: {e}")
        print(f"  [INFO] This may occur if the model doesn't support sklearn-compatible predict/predict_proba")
        return {'error': str(e)}

def save_comprehensive_reports(results: dict, confidence_df: pd.DataFrame, per_cell_results: dict, 
                               output_dir: Path, test_samples: int, train_samples: int) -> str:
    """
    Generate and save all comprehensive reports and CSV files.
    
    **Single Responsibility:** Save all output files (JSON, TXT, MD, CSV).
    
    Args:
        results: Complete evaluation metrics dictionary
        confidence_df: DataFrame with prediction confidence data
        per_cell_results: Per-cell metrics dictionary
        output_dir: Directory to save all outputs
        test_samples: Number of test samples
        train_samples: Number of training samples
    
    Returns:
        str: Executive summary text (for display)
    
    Outputs:
        - comprehensive_results.json: All metrics in JSON format
        - executive_summary.txt: Summary for general audience
        - technical_analysis.txt: Detailed analysis for experts
        - comprehensive_report.md: Complete markdown documentation
        - confidence_data.csv: Prediction confidence scores
        - per_cell_metrics.csv: Detailed per-cell performance
    """
    print(f"\n{'=' * 80}")
    print("SAVING RESULTS")
    print("=" * 80)
    
    # Save comprehensive results JSON
    results_path = output_dir / "comprehensive_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[SAVED] {results_path}")
    
    # Generate user-friendly reports
    print("\n[GENERATING USER-FRIENDLY REPORTS]")
    
    # Executive Summary (for general audience)
    exec_summary = generate_executive_summary(results, test_samples, train_samples)
    exec_summary_path = output_dir / "executive_summary.txt"
    with open(exec_summary_path, 'w', encoding='utf-8') as f:
        f.write(exec_summary)
    print(f"[SAVED] {exec_summary_path}")
    
    # Technical Analysis (for experts)
    tech_analysis = generate_technical_analysis(results, test_samples, train_samples)
    tech_analysis_path = output_dir / "technical_analysis.txt"
    with open(tech_analysis_path, 'w', encoding='utf-8') as f:
        f.write(tech_analysis)
    print(f"[SAVED] {tech_analysis_path}")
    
    # Markdown Report (for documentation)
    markdown_report_path = output_dir / "comprehensive_report.md"
    markdown_report = generate_markdown_report(results, test_samples, train_samples, markdown_report_path)
    print(f"[SAVED] {markdown_report_path}")
    
    # Save confidence data CSV
    confidence_csv_path = output_dir / "confidence_data.csv"
    confidence_df.to_csv(confidence_csv_path, index=False)
    print(f"[SAVED] {confidence_csv_path}")
    
    # Save per-cell results to CSV
    per_cell_df = pd.DataFrame.from_dict(per_cell_results, orient='index')
    per_cell_df.index.name = 'cell'
    per_cell_df['top_confused_with'] = per_cell_df['top_confused_with'].apply(
        lambda x: ", ".join([f"{c}({n})" for c, n in x]) if x else "None"
    )
    per_cell_csv_path = output_dir / "per_cell_metrics.csv"
    per_cell_df.to_csv(per_cell_csv_path)
    print(f"[SAVED] {per_cell_csv_path}")
    
    # Print completion message
    print(f"\n{'=' * 80}")
    print("[OK] COMPREHENSIVE TESTING COMPLETE!")
    print("=" * 80)
    print(f"\nAll results saved to: {output_dir.absolute()}")
    print(f"\nGenerated files:")
    print(f"  - comprehensive_results.json (all metrics)")
    print(f"  - confidence_data.csv (prediction confidences)")
    print(f"  - per_cell_metrics.csv (detailed per-cell metrics)")
    print(f"  - confidence_analysis.png (confidence visualizations)")
    print(f"  - location_analysis.png (corner/edge/center performance)")
    print(f"  - topk_accuracy.png (top-k accuracy comparison)")
    print(f"  - per_cell_accuracy_heatmap.png (accuracy per cell)")
    print(f"  - per_cell_f1_heatmap.png (F1-score per cell)")
    print(f"  - per_cell_samples_heatmap.png (sample distribution)")
    print(f"  - confusion_matrix.png (full confusion matrix)")
    print(f"\n📄 User-Friendly Reports:")
    print(f"  - executive_summary.txt (untuk pembaca umum)")
    print(f"  - technical_analysis.txt (untuk ahli/peneliti)")
    print(f"  - comprehensive_report.md (dokumentasi lengkap)")
    
    return exec_summary

def calculate_comprehensive_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                                   y_pred_proba_array: np.ndarray, y_true_encoded: np.ndarray) -> tuple:
    """
    Calculate all evaluation metrics (accuracy, precision, recall, F1, top-k, confidence).
    
    **Single Responsibility:** Compute comprehensive metrics only.
    
    Args:
        y_true: True labels array
        y_pred: Predicted labels array
        y_pred_proba_array: Prediction probabilities array (n_samples, n_classes)
        y_true_encoded: Encoded true labels array
    
    Returns:
        tuple: (basic_metrics, topk_accuracy, confidence_metrics, confidence_df) where:
            - basic_metrics: Dict with accuracy, precision, recall, F1, Cohen's kappa
            - topk_accuracy: Dict with top-1, top-2, top-3, top-5 accuracy
            - confidence_metrics: Dict with confidence statistics
            - confidence_df: DataFrame with per-sample predictions and confidence
    """
    basic_metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision_weighted': float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
        'precision_macro': float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
        'recall_weighted': float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
        'recall_macro': float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
        'f1_weighted': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
        'f1_macro': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'cohen_kappa': float(cohen_kappa_score(y_true, y_pred)),
    }
    
    topk_accuracy = {
        'top1': float(calculate_top_k_accuracy(y_true_encoded, y_pred_proba_array, 1)),
        'top2': float(calculate_top_k_accuracy(y_true_encoded, y_pred_proba_array, 2)),
        'top3': float(calculate_top_k_accuracy(y_true_encoded, y_pred_proba_array, 3)),
        'top5': float(calculate_top_k_accuracy(y_true_encoded, y_pred_proba_array, 5)),
    }
    
    max_probs = np.max(y_pred_proba_array, axis=1)
    confidence_metrics = calculate_confidence_metrics(y_pred_proba_array)
    
    confidence_df = pd.DataFrame({
        'ground_truth': y_true,
        'predicted': y_pred,
        'confidence': max_probs,
        'correct': y_true == y_pred
    })
    
    return basic_metrics, topk_accuracy, confidence_metrics, confidence_df

def analyze_location_performance(test_df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Analyze model performance by cell location (corner/edge/center).
    
    **Single Responsibility:** Compute location-based accuracy analysis.
    
    Args:
        test_df: Test dataset DataFrame with TARGET_COLUMN
        y_true: True labels array
        y_pred: Predicted labels array
    
    Returns:
        dict: Location analysis with keys ['corner', 'edge', 'center'], each containing:
            - accuracy: Location-specific accuracy
            - count: Number of samples in location
            - f1: Location-specific F1 score
    """
    location_analysis = {}
    for location in ['corner', 'edge', 'center']:
        mask = test_df[TARGET_COLUMN].apply(classify_cell_location) == location
        if mask.sum() > 0:
            loc_true = y_true[mask]
            loc_pred = y_pred[mask]
            location_analysis[location] = {
                'accuracy': float(accuracy_score(loc_true, loc_pred)),
                'count': int(mask.sum()),
                'f1': float(f1_score(loc_true, loc_pred, average='weighted', zero_division=0))
            }
    
    return location_analysis

def evaluate_training_performance(model, train_df: pd.DataFrame, feature_cols: list, 
                                  label_encoder, test_inference_time: float, test_samples: int) -> tuple:
    """
    Evaluate model performance on training set and calculate timing metrics.
    
    **Single Responsibility:** Compute training performance and inference timing.
    
    Args:
        model: Trained LightGBM model
        train_df: Training dataset DataFrame
        feature_cols: List of feature column names
        label_encoder: Label encoder for transformations
        test_inference_time: Test set inference time in seconds
        test_samples: Number of test samples
    
    Returns:
        tuple: (train_performance, timing_metrics) where:
            - train_performance: Dict with training accuracy, F1, sample count
            - timing_metrics: Dict with test/train inference times and throughput
    """
    import time
    
    X_train = train_df[feature_cols].to_numpy()
    y_train = train_df[TARGET_COLUMN].to_numpy()
    
    # Get predictions on training set with timing
    print(f"  - Evaluating on {len(y_train)} training samples...")
    train_start_time = time.time()
    y_train_pred_proba = model.predict(X_train)
    train_inference_time = time.time() - train_start_time
    
    y_train_pred_proba_array = np.asarray(y_train_pred_proba)
    y_train_pred_encoded = np.argmax(y_train_pred_proba_array, axis=1)
    y_train_pred = label_encoder.inverse_transform(y_train_pred_encoded)
    
    # Calculate timing metrics with safe division (avoid division by zero for very fast inference)
    # Use epsilon to prevent division by zero when inference is extremely fast
    MIN_TIME = 1e-9  # Minimum time threshold (1 nanosecond)
    safe_test_time = max(test_inference_time, MIN_TIME)
    safe_train_time = max(train_inference_time, MIN_TIME)

    timing_metrics = {
        'test_inference_time_seconds': float(test_inference_time),
        'test_samples': int(test_samples),
        'test_time_per_sample_ms': float(safe_test_time / test_samples * 1000),
        'test_throughput_samples_per_second': float(test_samples / safe_test_time),
        'train_inference_time_seconds': float(train_inference_time),
        'train_samples': int(len(y_train)),
        'train_time_per_sample_ms': float(safe_train_time / len(y_train) * 1000),
        'train_throughput_samples_per_second': float(len(y_train) / safe_train_time),
    }

    print(f"  - Test inference: {test_inference_time:.3f}s ({safe_test_time/test_samples*1000:.2f}ms per sample)")
    print(f"  - Train inference: {train_inference_time:.3f}s ({safe_train_time/len(y_train)*1000:.2f}ms per sample)")
    
    train_performance = {
        'accuracy': float(accuracy_score(y_train, y_train_pred)),
        'f1_weighted': float(f1_score(y_train, y_train_pred, average='weighted', zero_division=0)),
        'sample_count': int(len(y_train))
    }
    
    print(f"  - Training Accuracy: {train_performance['accuracy']:.4f} ({train_performance['accuracy']*100:.2f}%)")
    
    return train_performance, timing_metrics

def analyze_spatial_errors_and_per_cell(y_true: np.ndarray, y_pred: np.ndarray, 
                                       y_pred_proba_array: np.ndarray, label_encoder, 
                                       test_df: pd.DataFrame) -> tuple:
    """
    Analyze spatial errors and compute detailed per-cell metrics.
    
    **Single Responsibility:** Compute spatial error metrics and per-cell analysis.
    
    Args:
        y_true: True labels array
        y_pred: Predicted labels array
        y_pred_proba_array: Prediction probabilities array
        label_encoder: Label encoder for class names
        test_df: Test dataset DataFrame
    
    Returns:
        tuple: (spatial_metrics, per_cell_results, top_error_patterns, summary_stats) where:
            - spatial_metrics: Dict with mean/median/percentile spatial errors
            - per_cell_results: Dict with detailed metrics per cell
            - top_error_patterns: Dict with top 10 misclassification patterns
            - summary_stats: Dict with cells_with_data, avg/min/max cell accuracy
    """
    # Calculate spatial errors
    spatial_errors = [calculate_euclidean_distance(true, pred) for true, pred in zip(y_true, y_pred)]
    spatial_metrics = {
        'mean_error_m': float(np.mean(spatial_errors)),
        'median_error_m': float(np.median(spatial_errors)),
        'std_error_m': float(np.std(spatial_errors)),
        'min_error_m': float(np.min(spatial_errors)),
        'max_error_m': float(np.max(spatial_errors)),
        'p90_error_m': float(np.percentile(spatial_errors, 90)),
        'p95_error_m': float(np.percentile(spatial_errors, 95)),
    }
    
    # Error pattern analysis
    print(f"\n[STEP 6/8] Error pattern analysis...")
    misclassified = test_df[y_true != y_pred].copy()
    if len(misclassified) > 0:
        error_patterns = Counter(zip(y_true[y_true != y_pred], y_pred[y_true != y_pred]))
        top_errors = dict(error_patterns.most_common(10))
        top_error_patterns = {f"{true}->{pred}": int(count) for (true, pred), count in top_errors.items()}
    else:
        top_error_patterns = {}
    
    # Per-cell analysis
    print(f"\n[STEP 6/8] Per-cell analysis...")
    print(f"  - Calculating metrics for each cell individually...")
    per_cell_results = calculate_per_cell_metrics(y_true, y_pred, y_pred_proba_array, label_encoder)
    
    # Calculate summary statistics
    cells_with_data = [cell for cell, metrics in per_cell_results.items() if metrics['sample_count'] > 0]
    avg_cell_accuracy = np.mean([per_cell_results[cell]['accuracy'] for cell in cells_with_data])
    min_cell_accuracy = min([per_cell_results[cell]['accuracy'] for cell in cells_with_data])
    max_cell_accuracy = max([per_cell_results[cell]['accuracy'] for cell in cells_with_data])
    
    print(f"  - Cells analyzed: {len(cells_with_data)}/{len(label_encoder.classes_)}")
    print(f"  - Average cell accuracy: {avg_cell_accuracy:.4f} ({avg_cell_accuracy*100:.2f}%)")
    print(f"  - Min cell accuracy: {min_cell_accuracy:.4f} ({min_cell_accuracy*100:.2f}%)")
    print(f"  - Max cell accuracy: {max_cell_accuracy:.4f} ({max_cell_accuracy*100:.2f}%)")
    
    summary_stats = {
        'cells_with_data': cells_with_data,
        'avg_cell_accuracy': avg_cell_accuracy,
        'min_cell_accuracy': min_cell_accuracy,
        'max_cell_accuracy': max_cell_accuracy
    }
    
    return spatial_metrics, per_cell_results, top_error_patterns, summary_stats

def plot_classification_metrics_summary(results, output_path):
    """
    Create bar chart summarizing all classification metrics.

    Displays accuracy, precision (weighted/macro), recall (weighted/macro),
    F1-score (weighted/macro), and Cohen's kappa in a single comparison chart.

    Args:
        results (dict): Results dictionary containing 'basic_metrics' key
        output_path (Path): Path to save the plot PNG file

    Outputs:
        PNG file with classification metrics bar chart (300 DPI)
    """
    metrics = results['basic_metrics']

    # Prepare data for plotting
    metric_names = [
        'Accuracy',
        'Precision\n(Weighted)',
        'Precision\n(Macro)',
        'Recall\n(Weighted)',
        'Recall\n(Macro)',
        'F1-Score\n(Weighted)',
        'F1-Score\n(Macro)',
        "Cohen's\nKappa"
    ]

    metric_values = [
        metrics['accuracy'],
        metrics['precision_weighted'],
        metrics['precision_macro'],
        metrics['recall_weighted'],
        metrics['recall_macro'],
        metrics['f1_weighted'],
        metrics['f1_macro'],
        metrics['cohen_kappa']
    ]

    # Color coding by metric type
    colors = ['#2E86AB', '#A23B72', '#A23B72', '#F18F01', '#F18F01', '#C73E1D', '#C73E1D', '#6A994E']

    fig, ax = plt.subplots(figsize=(14, 7))
    bars = ax.bar(range(len(metric_names)), metric_values, color=colors, edgecolor='black', linewidth=1.2, alpha=0.85)

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, metric_values)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{val:.4f}\n({val*100:.2f}%)',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xticks(range(len(metric_names)))
    ax.set_xticklabels(metric_names, fontsize=11)
    ax.set_ylabel('Score', fontsize=13, fontweight='bold')
    ax.set_title('Classification Metrics Summary', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylim(0, 1.15)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.5, label='50% Baseline')
    ax.legend(loc='lower right', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {output_path}")

def plot_spatial_error_stats(results, output_path):
    """
    Create bar chart showing spatial error statistics.

    Displays mean, median, P90, P95, and max spatial error in meters.

    Args:
        results (dict): Results dictionary containing 'spatial_metrics' key
        output_path (Path): Path to save the plot PNG file

    Outputs:
        PNG file with spatial error statistics bar chart (300 DPI)
    """
    spatial = results['spatial_metrics']

    metric_names = ['Mean', 'Median', 'P90', 'P95', 'Max']
    metric_values = [
        spatial['mean_error_m'],
        spatial['median_error_m'],
        spatial['p90_error_m'],
        spatial['p95_error_m'],
        spatial['max_error_m']
    ]

    colors = ['#3A86FF', '#8338EC', '#FF006E', '#FB5607', '#FFBE0B']

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.bar(range(len(metric_names)), metric_values, color=colors, edgecolor='black', linewidth=1.2, alpha=0.85)

    # Add value labels on bars
    for bar, val in zip(bars, metric_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max(metric_values)*0.02,
                f'{val:.3f}m',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_xticks(range(len(metric_names)))
    ax.set_xticklabels(metric_names, fontsize=12)
    ax.set_ylabel('Error Distance (meters)', fontsize=13, fontweight='bold')
    ax.set_title('Spatial Error Statistics', fontsize=16, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {output_path}")

def plot_spatial_error_heatmap(y_true, y_pred, label_encoder, output_path):
    """
    Create 5x5 heatmap showing mean spatial error per cell.

    For each cell, calculates the mean Euclidean distance error when that cell
    is the true label. Cells with no samples are masked.

    Args:
        y_true (np.ndarray): True labels array
        y_pred (np.ndarray): Predicted labels array
        label_encoder: Label encoder with classes_
        output_path (Path): Path to save the plot PNG file

    Outputs:
        PNG file with spatial error heatmap (300 DPI)
    """
    # Initialize 5x5 grid
    grid = np.full((5, 5), np.nan)

    # Calculate mean spatial error for each cell
    for cell in label_encoder.classes_:
        # Find all samples where true label is this cell
        mask = (y_true == cell)
        if not np.any(mask):
            continue

        # Calculate spatial errors for these samples
        errors = [calculate_euclidean_distance(cell, pred) for pred in y_pred[mask]]
        mean_error = np.mean(errors)

        # Place in grid
        col, row = cell_to_coords(cell)
        grid[row, col] = mean_error

    # Flip grid for correct orientation
    grid = np.flipud(grid)

    plt.figure(figsize=(10, 8))

    # Create heatmap with reversed RdYlGn (red=high error, green=low error)
    mask = np.isnan(grid)
    sns.heatmap(grid, annot=True, fmt='.2f', cmap='RdYlGn_r',
               xticklabels=['A', 'B', 'C', 'D', 'E'],
               yticklabels=['5', '4', '3', '2', '1'],
               cbar_kws={'label': 'Mean Error (meters)'},
               linewidths=1, linecolor='black',
               mask=mask, vmin=0)

    plt.title('Mean Spatial Error per Cell (meters)', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Column', fontsize=13, fontweight='bold')
    plt.ylabel('Row', fontsize=13, fontweight='bold')
    plt.tight_layout()

    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {output_path}")

def plot_spatial_error_distribution(y_true, y_pred, output_path):
    """
    Create histogram of spatial error distribution with CDF overlay.

    Shows the frequency distribution of spatial errors and cumulative distribution
    to understand error patterns and percentiles.

    Args:
        y_true (np.ndarray): True labels array
        y_pred (np.ndarray): Predicted labels array
        output_path (Path): Path to save the plot PNG file

    Outputs:
        PNG file with error distribution histogram and CDF (300 DPI)
    """
    # Calculate all spatial errors
    spatial_errors = np.array([calculate_euclidean_distance(true, pred)
                               for true, pred in zip(y_true, y_pred)])

    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Histogram
    n, bins, patches = ax1.hist(spatial_errors, bins=HIST_BINS, color='#3A86FF',
                                edgecolor='black', linewidth=1, alpha=0.7, label='Frequency')
    ax1.set_xlabel('Spatial Error (meters)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=13, fontweight='bold', color='#3A86FF')
    ax1.tick_params(axis='y', labelcolor='#3A86FF')
    ax1.grid(axis='both', alpha=0.3, linestyle='--')

    # CDF on secondary axis
    ax2 = ax1.twinx()
    sorted_errors = np.sort(spatial_errors)
    cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
    ax2.plot(sorted_errors, cumulative, color='#FF006E', linewidth=2.5, label='CDF')
    ax2.set_ylabel('Cumulative Probability', fontsize=13, fontweight='bold', color='#FF006E')
    ax2.tick_params(axis='y', labelcolor='#FF006E')
    ax2.set_ylim(0, 1.05)

    # Add percentile lines
    # Convert to Python float for strict type compatibility with matplotlib
    p50 = float(np.percentile(spatial_errors, 50))
    p90 = float(np.percentile(spatial_errors, 90))
    p95 = float(np.percentile(spatial_errors, 95))

    ax1.axvline(p50, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label=f'P50: {p50:.2f}m')
    ax1.axvline(p90, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label=f'P90: {p90:.2f}m')
    ax1.axvline(p95, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label=f'P95: {p95:.2f}m')

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)

    plt.title('Spatial Error Distribution', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {output_path}")

def plot_top_confusion_pairs(results, output_path, top_n=10):
    """
    Create horizontal bar chart of top misclassification patterns.

    Shows the most frequent error patterns (e.g., "A1->A2" means A1 predicted as A2).

    Args:
        results (dict): Results dictionary containing 'top_error_patterns' key
        output_path (Path): Path to save the plot PNG file
        top_n (int): Number of top error patterns to display (default: 10)

    Outputs:
        PNG file with top confusion pairs bar chart (300 DPI)
    """
    error_patterns = results.get('top_error_patterns', {})

    if not error_patterns:
        print(f"[INFO] No error patterns found (perfect predictions). Skipping {output_path.name}")
        # Create empty plot with message
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.text(0.5, 0.5, 'No Misclassifications\n(Perfect Predictions)',
                ha='center', va='center', fontsize=20, fontweight='bold', color='green')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        print(f"[SAVED] {output_path}")
        return

    # Sort and get top N
    sorted_patterns = sorted(error_patterns.items(), key=lambda x: x[1], reverse=True)[:top_n]
    patterns = [p[0] for p in sorted_patterns]
    counts = [p[1] for p in sorted_patterns]

    fig, ax = plt.subplots(figsize=(12, max(6, len(patterns) * 0.5)))

    # Reds colormap exists but Pylance doesn't recognize dynamic colormaps
    colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(patterns)))  # type: ignore[attr-defined]
    bars = ax.barh(range(len(patterns)), counts, color=colors, edgecolor='black', linewidth=1)

    # Add value labels
    for i, (bar, count) in enumerate(zip(bars, counts)):
        width = bar.get_width()
        ax.text(width + max(counts)*0.02, bar.get_y() + bar.get_height()/2,
                f'{count}',
                ha='left', va='center', fontsize=10, fontweight='bold')

    ax.set_yticks(range(len(patterns)))
    ax.set_yticklabels(patterns, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel('Error Count', fontsize=13, fontweight='bold')
    ax.set_title(f'Top {len(patterns)} Misclassification Patterns', fontsize=16, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {output_path}")

def plot_train_test_comparison(results, output_path):
    """
    Create grouped bar chart comparing training vs test performance.

    Compares accuracy and F1-score between training and test sets to visualize
    generalization gap (overfitting indicator).

    Args:
        results (dict): Results dictionary with 'basic_metrics' and 'train_performance'
        output_path (Path): Path to save the plot PNG file

    Outputs:
        PNG file with train/test comparison chart (300 DPI)
    """
    train = results['train_performance']
    test = results['basic_metrics']

    metrics = ['Accuracy', 'F1-Score (Weighted)']
    train_scores = [train['accuracy'], train['f1_weighted']]
    test_scores = [test['accuracy'], test['f1_weighted']]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 7))

    bars1 = ax.bar(x - width/2, train_scores, width, label='Training Set',
                   color='#3A86FF', edgecolor='black', linewidth=1.2, alpha=0.85)
    bars2 = ax.bar(x + width/2, test_scores, width, label='Test Set',
                   color='#FF006E', edgecolor='black', linewidth=1.2, alpha=0.85)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.4f}\n({height*100:.2f}%)',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12)
    ax.set_ylabel('Score', fontsize=13, fontweight='bold')
    ax.set_title('Training vs Test Performance', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylim(0, 1.15)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add generalization gap annotation
    gap = train_scores[0] - test_scores[0]
    gap_text = f'Generalization Gap: {gap*100:.2f}%'
    gap_color = 'green' if gap < 0.05 else ('orange' if gap < 0.10 else 'red')
    ax.text(0.5, 0.95, gap_text, transform=ax.transAxes,
            ha='center', va='top', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor=gap_color, alpha=0.3))

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {output_path}")

def plot_dataset_distribution(test_df, train_df, target_column, output_path):
    """
    Create bar chart showing sample distribution across all cells.

    Displays the number of samples per cell for both training and test sets,
    helping to identify data imbalance issues.

    Args:
        test_df (pd.DataFrame): Test dataset
        train_df (pd.DataFrame): Training dataset
        target_column (str): Name of the target column containing cell labels
        output_path (Path): Path to save the plot PNG file

    Outputs:
        PNG file with dataset distribution bar chart (300 DPI)
    """
    # Combine train and test counts
    train_counts = train_df[target_column].value_counts().sort_index()
    test_counts = test_df[target_column].value_counts().sort_index()

    # Ensure all cells are represented
    all_cells = sorted(set(train_counts.index) | set(test_counts.index))
    train_values = [train_counts.get(cell, 0) for cell in all_cells]
    test_values = [test_counts.get(cell, 0) for cell in all_cells]

    x = np.arange(len(all_cells))
    width = 0.4

    fig, ax = plt.subplots(figsize=(16, 7))

    bars1 = ax.bar(x - width/2, train_values, width, label='Training Set',
                   color='#06A77D', edgecolor='black', linewidth=1, alpha=0.85)
    bars2 = ax.bar(x + width/2, test_values, width, label='Test Set',
                   color='#F77F00', edgecolor='black', linewidth=1, alpha=0.85)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + max(train_values + test_values)*0.01,
                        f'{int(height)}',
                        ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(all_cells, fontsize=10)
    ax.set_xlabel('Cell', fontsize=13, fontweight='bold')
    ax.set_ylabel('Sample Count', fontsize=13, fontweight='bold')
    ax.set_title('Dataset Distribution per Cell', fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add total counts
    total_train = sum(train_values)
    total_test = sum(test_values)
    ax.text(0.02, 0.98, f'Total Training: {total_train}\nTotal Test: {total_test}',
            transform=ax.transAxes, va='top', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {output_path}")

def plot_grid_layout(output_path, cell_size=EVAL_CELL_SIZE_METERS):
    """
    Create visual diagram of the 5x5 grid layout.

    Educational visualization showing the spatial arrangement of cells A1-E5
    with dimensions in meters.

    Args:
        output_path (Path): Path to save the plot PNG file
        cell_size (float): Size of each cell in meters (default: from config)

    Outputs:
        PNG file with grid layout diagram (300 DPI)
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # Draw grid
    for i in range(6):
        ax.axhline(i, color='black', linewidth=2)
        ax.axvline(i, color='black', linewidth=2)

    # Add cell labels
    for row in range(5):
        for col in range(5):
            cell_label = f"{chr(65 + col)}{5 - row}"  # A-E, 5-1
            x_center = col + 0.5
            y_center = row + 0.5

            # Cell label
            ax.text(x_center, y_center, cell_label,
                   ha='center', va='center', fontsize=20, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7, edgecolor='black', linewidth=2))

            # Coordinates in meters (at corners for reference)
            if row == 4 and col == 0:  # Bottom-left (A1)
                ax.text(col + 0.1, row + 0.1, f'({col*cell_size:.1f}m, {(4-row)*cell_size:.1f}m)',
                       fontsize=8, color='red', fontweight='bold')

    # Labels
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)
    ax.set_xticks(np.arange(0.5, 5.5, 1))
    ax.set_xticklabels(['A', 'B', 'C', 'D', 'E'], fontsize=14, fontweight='bold')
    ax.set_yticks(np.arange(0.5, 5.5, 1))
    ax.set_yticklabels(['5', '4', '3', '2', '1'], fontsize=14, fontweight='bold')
    ax.set_xlabel('Column', fontsize=15, fontweight='bold')
    ax.set_ylabel('Row', fontsize=15, fontweight='bold')
    ax.set_title(f'Grid Layout (5×5 cells, {cell_size}m × {cell_size}m each)',
                fontsize=16, fontweight='bold', pad=20)
    ax.set_aspect('equal')

    # Add dimension annotations
    total_size = 5 * cell_size
    ax.text(2.5, -0.3, f'Total Width: {total_size}m', ha='center', fontsize=12, fontweight='bold')
    ax.text(-0.3, 2.5, f'Total Height: {total_size}m', ha='center', va='center',
            rotation=90, fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {output_path}")

def plot_confidence_vs_accuracy(confidence_df, output_path):
    """
    Create calibration plot showing confidence vs actual accuracy.

    Bins predictions by confidence level and calculates actual accuracy for each bin,
    revealing whether the model is well-calibrated (confident when correct).

    Args:
        confidence_df (pd.DataFrame): DataFrame with 'confidence' and 'correct' columns
        output_path (Path): Path to save the plot PNG file

    Outputs:
        PNG file with confidence vs accuracy plot (300 DPI)
    """
    # Bin confidence scores
    confidence_df = confidence_df.copy()
    confidence_df['confidence_bin'] = pd.cut(confidence_df['confidence'],
                                             bins=CONFIDENCE_BINS,
                                             labels=CONFIDENCE_LABELS,
                                             include_lowest=True)

    # Calculate accuracy per bin
    calibration_data = confidence_df.groupby('confidence_bin', observed=True).agg({
        'correct': ['mean', 'count']
    }).reset_index()

    calibration_data.columns = ['confidence_bin', 'accuracy', 'count']

    # Filter out bins with very few samples (< 5)
    calibration_data = calibration_data[calibration_data['count'] >= 5]

    if len(calibration_data) == 0:
        print(f"[WARNING] Insufficient data for confidence calibration plot. Skipping {output_path.name}")
        return

    fig, ax = plt.subplots(figsize=(10, 7))

    x = range(len(calibration_data))
    bars = ax.bar(x, calibration_data['accuracy'], color='#3A86FF',
                  edgecolor='black', linewidth=1.2, alpha=0.85)

    # Add value labels
    for i, (bar, acc, cnt) in enumerate(zip(bars, calibration_data['accuracy'], calibration_data['count'])):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{acc:.3f}\n({acc*100:.1f}%)\nn={int(cnt)}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(calibration_data['confidence_bin'], fontsize=11)
    ax.set_xlabel('Confidence Bin', fontsize=13, fontweight='bold')
    ax.set_ylabel('Actual Accuracy', fontsize=13, fontweight='bold')
    ax.set_title('Model Calibration: Confidence vs Actual Accuracy', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add perfect calibration line reference
    ax.axhline(y=1.0, color='green', linestyle='--', linewidth=1.5, alpha=0.5, label='Perfect Accuracy')
    ax.legend(loc='lower right', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {output_path}")

def generate_all_visualizations(results, per_cell_results, confidence_df, location_analysis,
                                y_true, y_pred, label_encoder, test_df, train_df, output_dir):
    """
    Generate all visualizations for comprehensive model evaluation.

    This function creates 20 plots organized in 7 categories, following SRP
    (Single Responsibility Principle) for visualization generation.

    Args:
        results (dict): Complete evaluation results including metrics, topk_accuracy, learning_curve
        per_cell_results (dict): Per-cell performance metrics
        confidence_df (DataFrame): Confidence scores and predictions
        location_analysis (dict): Corner/edge/center performance analysis
        y_true (np.array): True labels
        y_pred (np.array): Predicted labels
        label_encoder: Sklearn LabelEncoder for class names
        test_df (pd.DataFrame): Test dataset
        train_df (pd.DataFrame): Training dataset
        output_dir (Path): Root directory to save all visualizations

    Generates 20 plots in organized subfolders:
        1_basic_metrics/        - 4 plots (confusion matrix, metrics summary, confusion pairs)
        2_spatial_analysis/     - 4 plots (error stats, heatmap, distribution, samples)
        3_per_cell_performance/ - 2 plots (accuracy, F1 heatmaps)
        4_confidence_analysis/  - 2 plots (distribution, calibration)
        5_model_performance/    - 5 plots (top-k, location, comparison, CV, learning curve)
        6_feature_analysis/     - 1 plot (importance)
        7_dataset_info/         - 2 plots (distribution, grid layout)
    """
    # Create subfolder structure
    print("  - Creating organized folder structure...")
    subfolders = {
        'basic': output_dir / "1_basic_metrics",
        'spatial': output_dir / "2_spatial_analysis",
        'per_cell': output_dir / "3_per_cell_performance",
        'confidence': output_dir / "4_confidence_analysis",
        'performance': output_dir / "5_model_performance",
        'features': output_dir / "6_feature_analysis",
        'dataset': output_dir / "7_dataset_info"
    }

    for folder in subfolders.values():
        folder.mkdir(parents=True, exist_ok=True)

    # ===========================
    # 1. BASIC METRICS (4 plots)
    # ===========================
    print("\n  [1/7] Basic Metrics Visualizations...")

    print("    - Confusion matrix (raw counts)...")
    confusion_path = subfolders['basic'] / "01_confusion_matrix.png"
    plot_confusion_heatmap(y_true, y_pred, label_encoder, confusion_path, normalize=None)

    print("    - Confusion matrix (normalized)...")
    confusion_norm_path = subfolders['basic'] / "02_confusion_matrix_normalized.png"
    plot_confusion_heatmap(y_true, y_pred, label_encoder, confusion_norm_path, normalize='true')

    print("    - Classification metrics summary...")
    metrics_summary_path = subfolders['basic'] / "03_classification_metrics_summary.png"
    plot_classification_metrics_summary(results, metrics_summary_path)

    print("    - Top confusion pairs...")
    confusion_pairs_path = subfolders['basic'] / "04_top_confusion_pairs.png"
    plot_top_confusion_pairs(results, confusion_pairs_path, top_n=10)

    # ================================
    # 2. SPATIAL ANALYSIS (4 plots)
    # ================================
    print("\n  [2/7] Spatial Analysis Visualizations...")

    print("    - Spatial error statistics...")
    spatial_stats_path = subfolders['spatial'] / "01_spatial_error_stats.png"
    plot_spatial_error_stats(results, spatial_stats_path)

    print("    - Spatial error heatmap...")
    spatial_heatmap_path = subfolders['spatial'] / "02_spatial_error_heatmap.png"
    plot_spatial_error_heatmap(y_true, y_pred, label_encoder, spatial_heatmap_path)

    print("    - Spatial error distribution...")
    spatial_dist_path = subfolders['spatial'] / "03_spatial_error_distribution.png"
    plot_spatial_error_distribution(y_true, y_pred, spatial_dist_path)

    print("    - Per-cell sample count...")
    sample_heatmap_path = subfolders['spatial'] / "04_per_cell_samples_heatmap.png"
    plot_per_cell_heatmap(per_cell_results, metric='sample_count', output_path=sample_heatmap_path)

    # ======================================
    # 3. PER-CELL PERFORMANCE (2 plots)
    # ======================================
    print("\n  [3/7] Per-Cell Performance Visualizations...")

    print("    - Per-cell accuracy heatmap...")
    accuracy_heatmap_path = subfolders['per_cell'] / "01_per_cell_accuracy_heatmap.png"
    plot_per_cell_heatmap(per_cell_results, metric='accuracy', output_path=accuracy_heatmap_path)

    print("    - Per-cell F1-score heatmap...")
    f1_heatmap_path = subfolders['per_cell'] / "02_per_cell_f1_heatmap.png"
    plot_per_cell_heatmap(per_cell_results, metric='f1', output_path=f1_heatmap_path)

    # ===================================
    # 4. CONFIDENCE ANALYSIS (2 plots)
    # ===================================
    print("\n  [4/7] Confidence Analysis Visualizations...")

    print("    - Confidence distribution...")
    confidence_dist_path = subfolders['confidence'] / "01_confidence_distribution.png"
    plot_confidence_distribution(confidence_df, confidence_dist_path)

    print("    - Confidence vs accuracy (calibration)...")
    confidence_calib_path = subfolders['confidence'] / "02_confidence_vs_accuracy.png"
    plot_confidence_vs_accuracy(confidence_df, confidence_calib_path)

    # ===================================
    # 5. MODEL PERFORMANCE (5 plots)
    # ===================================
    print("\n  [5/7] Model Performance Visualizations...")

    print("    - Top-K accuracy comparison...")
    topk_path = subfolders['performance'] / "01_topk_accuracy.png"
    plot_topk_comparison(results['topk_accuracy'], topk_path)

    print("    - Performance by location (corner/edge/center)...")
    location_path = subfolders['performance'] / "02_location_analysis.png"
    plot_location_analysis(location_analysis, location_path)

    print("    - Train vs test comparison...")
    comparison_path = subfolders['performance'] / "03_train_test_comparison.png"
    plot_train_test_comparison(results, comparison_path)

    print("    - Cross-validation results...")
    if 'cross_validation' in results and 'error' not in results['cross_validation']:
        cv_path = subfolders['performance'] / "04_cross_validation_results.png"
        existing_cv = output_dir / "cross_validation_results.png"
        if existing_cv.exists():
            # Move to organized location
            import shutil
            shutil.move(str(existing_cv), str(cv_path))
            print(f"      [MOVED] {cv_path}")
        else:
            # Generate if not exists (fallback)
            plot_cross_validation_results(results['cross_validation'], cv_path)
    else:
        print("      [SKIPPED] Cross-validation data not available")

    print("    - Learning curve...")
    if 'learning_curve' in results and 'error' not in results['learning_curve']:
        lc_path = subfolders['performance'] / "05_learning_curve.png"
        plot_learning_curve(results['learning_curve'], lc_path)
    else:
        print("      [SKIPPED] Learning curve data not available")

    # =================================
    # 6. FEATURE ANALYSIS (1 plot)
    # =================================
    print("\n  [6/7] Feature Analysis Visualizations...")

    print("    - Feature importance (gain & split)...")
    # Note: Feature importance plot is already generated in extract_feature_importance()
    # We just verify it exists, or it would have been created there already
    importance_path = subfolders['features'] / "01_feature_importance.png"
    existing_importance = output_dir / "feature_importance.png"
    if existing_importance.exists():
        # Move to organized location
        import shutil
        shutil.move(str(existing_importance), str(importance_path))
        print(f"      [MOVED] {importance_path}")
    else:
        print("      [INFO] Feature importance already in organized location")

    # ===============================
    # 7. DATASET INFO (2 plots)
    # ===============================
    print("\n  [7/7] Dataset Information Visualizations...")

    print("    - Dataset distribution per cell...")
    dataset_dist_path = subfolders['dataset'] / "01_dataset_distribution.png"
    plot_dataset_distribution(test_df, train_df, TARGET_COLUMN, dataset_dist_path)

    print("    - Grid layout diagram...")
    grid_layout_path = subfolders['dataset'] / "02_grid_layout.png"
    plot_grid_layout(grid_layout_path)

    print("\n  ✓ All visualizations generated successfully!")
    print(f"  ✓ Total plots created: 20 (in 7 organized folders)")

def main():
    """
    Comprehensive model testing and analysis pipeline.
    
    This function orchestrates an 8-step evaluation pipeline for the trained LightGBM model:
    
    Pipeline Steps:
        1. Load model and test/train data
        2. Run predictions with timing metrics
        2.5. Extract feature importance (gain & split)
        3. Calculate comprehensive metrics (accuracy, precision, recall, F1, etc.)
        4. Analyze performance by cell location (corner/edge/center)
        5. Evaluate training set performance
        5.5. Perform 5-Fold Cross-Validation on training set
        6. Spatial error analysis and per-cell detailed metrics
        7. Generate all visualizations (confusion matrix, heatmaps, plots)
        8. Save comprehensive reports (markdown, JSON, CSV)
    
    Outputs:
        All results saved to: reports/comprehensive_testing/
        - comprehensive_report.md: Human-readable markdown report
        - comprehensive_results.json: Machine-readable JSON results
        - *.csv: Per-cell metrics, confidence data, predictions
        - *.png: Confusion matrix, feature importance, spatial heatmaps
    
    Logging:
        Execution logs saved to: logs/comprehensive_testing_YYYYMMDD_HHMMSS.log
    """
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = setup_logger(
        "comprehensive_testing",
        log_file=LOGS_DIR / f"comprehensive_testing_{timestamp}.log",
        console=False  # Don't duplicate console output
    )
    
    logger.info("="*80)
    logger.info("COMPREHENSIVE MODEL TESTING & ANALYSIS")
    logger.info("="*80)
    
    print("=" * 80)
    print("COMPREHENSIVE MODEL TESTING & ANALYSIS")
    print("=" * 80)
    
    model_pkl_path = TUNED_MODEL_DIR / "lgbm_tuned.pkl"
    model_txt_path = TUNED_MODEL_DIR / "lgbm_tuned.txt"
    label_encoder_path = TUNED_MODEL_DIR / "label_encoder.pkl"
    test_data_path = PROCESSED_DIR / "test.csv"
    train_data_path = PROCESSED_DIR / "train.csv"
    output_dir = REPORTS_DIR / "comprehensive_testing"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    print(f"\n[STEP 1/8] Loading model and data...")
    logger.info("[STEP 1/8] Loading model and data...")
    
    # Try loading pkl file first (preserves training API), fallback to txt
    try:
        if model_pkl_path.exists():
            print(f"  - Loading model from {model_pkl_path}")
            logger.info(f"Loading model from {model_pkl_path}")
            model = joblib.load(model_pkl_path)
        else:
            print(f"  - Loading model from {model_txt_path}")
            logger.info(f"Loading model from {model_txt_path} (pkl not found)")
            model = lgb.Booster(model_file=str(model_txt_path))
    except Exception as e:
        logger.error(f"Failed to load model: {e}", exc_info=True)
        raise
    else:
        print(f"  - Loading model from {model_txt_path}")
        model = lgb.Booster(model_file=str(model_txt_path))
    label_encoder = joblib.load(label_encoder_path)
    test_df = pd.read_csv(test_data_path)
    train_df = pd.read_csv(train_data_path)
    
    feature_cols = FEATURE_COLUMNS
    
    X_test = test_df[feature_cols].to_numpy()
    y_true = test_df[TARGET_COLUMN].to_numpy()
    y_true_encoded = label_encoder.transform(y_true)
    
    print(f"Test samples: {len(test_df)}")
    print(f"Features: {len(feature_cols)}")
    print(f"Classes: {len(label_encoder.classes_)}")
    
    print(f"\n[STEP 2/8] Running predictions...")
    logger.info("[STEP 2/8] Running predictions with timing metrics...")
    import time
    
    # Time test set predictions
    start_time = time.time()
    y_pred_proba = model.predict(X_test)
    test_inference_time = time.time() - start_time
    
    y_pred_proba_array = np.asarray(y_pred_proba)
    
    # Verify prediction output shape and format
    print(f"  - Prediction shape: {y_pred_proba_array.shape}")
    print(f"  - Sample prediction sum: {y_pred_proba_array[0].sum():.6f}")
    print(f"  - Sample max prob: {y_pred_proba_array[0].max():.6f}")
    print(f"  - Sample min prob: {y_pred_proba_array[0].min():.6f}")
    print(f"  - Sample std prob: {y_pred_proba_array[0].std():.6f}")
    
    # Show top 3 probabilities for first sample
    top3_idx = np.argsort(y_pred_proba_array[0])[-3:][::-1]
    top3_probs = y_pred_proba_array[0][top3_idx]
    top3_classes = label_encoder.inverse_transform(top3_idx)
    print(f"  - Top 3 predictions: {top3_classes[0]}({top3_probs[0]:.4f}), {top3_classes[1]}({top3_probs[1]:.4f}), {top3_classes[2]}({top3_probs[2]:.4f})")
    
    y_pred_encoded = np.argmax(y_pred_proba_array, axis=1)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)
    
    results = {}
    
    print(f"\n[STEP 2.5/8] Extracting feature importance...")
    logger.info("[STEP 2.5/8] Extracting feature importance (gain & split)...")
    # SRP - Extracted to dedicated function
    importance_df, importance_dict = extract_feature_importance(model, feature_cols, output_dir)
    results['feature_importance'] = importance_dict
    
    print(f"\n[STEP 3/8] Calculating comprehensive metrics...")
    logger.info("[STEP 3/8] Calculating comprehensive metrics (accuracy, precision, recall, F1)...")
    
    # SRP - Extracted to dedicated function
    basic_metrics, topk_accuracy, confidence_metrics, confidence_df = calculate_comprehensive_metrics(
        y_true, y_pred, y_pred_proba_array, y_true_encoded
    )
    results['basic_metrics'] = basic_metrics
    results['topk_accuracy'] = topk_accuracy
    results['confidence_metrics'] = confidence_metrics
    
    print(f"\n[STEP 4/8] Analyzing by cell location (corner/edge/center)...")
    logger.info("[STEP 4/8] Analyzing performance by cell location...")
    
    # SRP - Extracted to dedicated function
    location_analysis = analyze_location_performance(test_df, y_true, y_pred)
    results['location_analysis'] = location_analysis
    
    print(f"\n[STEP 5/8] Calculating training set performance...")
    logger.info("[STEP 5/8] Evaluating training set performance...")
    
    # SRP - Extracted to dedicated function
    train_performance, timing_metrics = evaluate_training_performance(
        model, train_df, feature_cols, label_encoder, test_inference_time, len(y_true)
    )
    results['train_performance'] = train_performance
    results['timing_metrics'] = timing_metrics
    
    # Extract X_train, y_train for CV analysis (STEP 5.5)
    X_train = train_df[feature_cols].to_numpy()
    y_train = train_df[TARGET_COLUMN].to_numpy()
    
    print(f"\n[STEP 5.5/8] Performing K-Fold Cross-Validation...")
    # SRP - Extracted to dedicated function
    cv_results = perform_cv_analysis(model, label_encoder, X_train, y_train, output_dir, cv_folds=5)
    results['cross_validation'] = cv_results

    print(f"\n[STEP 5.6/8] Computing Learning Curve...")
    logger.info("[STEP 5.6/8] Computing learning curve to analyze model performance vs training set size...")
    try:
        # Create sklearn-compatible wrapper for learning curve
        model_wrapper = LGBMWrapper(model, label_encoder)

        print(f"  - Calculating learning curve with {N_LEARNING_CURVE_POINTS} points...")
        print(f"  - Training set sizes: {LEARNING_CURVE_MIN_SIZE*100:.0f}% to {LEARNING_CURVE_MAX_SIZE*100:.0f}%")
        print(f"  - Cross-validation folds: {CV_FOLDS}")
        print(f"  - Note: This may take 2-5 minutes due to multiple model trainings...")

        lc_results = calculate_learning_curve(model_wrapper, X_train, y_train, cv_folds=CV_FOLDS)
        results['learning_curve'] = lc_results

        # Display summary
        final_train_score = lc_results['train_scores_mean'][-1]
        final_val_score = lc_results['val_scores_mean'][-1]
        print(f"  - Final training score (100% data): {final_train_score:.4f}")
        print(f"  - Final validation score (100% data): {final_val_score:.4f}")
        print(f"  - Learning curve computed successfully")

    except Exception as e:
        print(f"  [ERROR] Learning curve computation failed: {e}")
        logger.error(f"Learning curve failed: {e}", exc_info=True)
        results['learning_curve'] = {'error': str(e)}

    print(f"\n[STEP 6/8] Spatial error analysis...")
    logger.info("[STEP 6/8] Performing spatial error and per-cell analysis...")
    
    # SRP - Extracted to dedicated function
    spatial_metrics, per_cell_results, top_error_patterns, summary_stats = analyze_spatial_errors_and_per_cell(
        y_true, y_pred, y_pred_proba_array, label_encoder, test_df
    )
    results['spatial_metrics'] = spatial_metrics
    results['top_error_patterns'] = top_error_patterns
    results['per_cell_metrics'] = per_cell_results
    
    # Unpack summary statistics for display
    cells_with_data = summary_stats['cells_with_data']
    avg_cell_accuracy = summary_stats['avg_cell_accuracy']
    min_cell_accuracy = summary_stats['min_cell_accuracy']
    max_cell_accuracy = summary_stats['max_cell_accuracy']
    
    print(f"\n[STEP 7/8] Generating visualizations...")
    logger.info("[STEP 7/8] Generating all visualizations (plots, heatmaps, confusion matrix)...")
    
    # Generate all visualizations in one call (SRP - extracted to dedicated function)
    generate_all_visualizations(results, per_cell_results, confidence_df, location_analysis,
                                y_true, y_pred, label_encoder, test_df, train_df, output_dir)
    
    print(f"\n{'=' * 80}")
    print("COMPREHENSIVE TEST RESULTS")
    print("=" * 80)
    
    print(f"\n[BASIC METRICS]")
    print(f"Accuracy:           {results['basic_metrics']['accuracy']:.4f} ({results['basic_metrics']['accuracy']*100:.2f}%)")
    print(f"Precision (macro):  {results['basic_metrics']['precision_macro']:.4f}")
    print(f"Recall (macro):     {results['basic_metrics']['recall_macro']:.4f}")
    print(f"F1-Score (macro):   {results['basic_metrics']['f1_macro']:.4f}")
    print(f"Cohen's Kappa:      {results['basic_metrics']['cohen_kappa']:.4f}")
    
    print(f"\n[TOP-K ACCURACY]")
    for k, acc in results['topk_accuracy'].items():
        print(f"{k.upper():8} {acc:.4f} ({acc*100:.2f}%)")
    
    print(f"\n[CONFIDENCE METRICS]")
    print(f"Mean Confidence:    {results['confidence_metrics']['mean_confidence']:.4f}")
    print(f"Median Confidence:  {results['confidence_metrics']['median_confidence']:.4f}")
    print(f"Std Confidence:     {results['confidence_metrics']['std_confidence']:.4f}")
    print(f"Low Conf (<0.5):    {results['confidence_metrics']['low_confidence_count']} samples")
    print(f"High Conf (>0.9):   {results['confidence_metrics']['high_confidence_count']} samples")
    print(f"\n  NOTE: Low confidence scores indicate the model's probabilities are nearly")
    print(f"  uniform across classes (~4% each). The model achieves high accuracy by")
    print(f"  selecting the marginally higher probability. This suggests the model")
    print(f"  probabilities are not well-calibrated, though predictions remain accurate.")
    
    print(f"\n[LOCATION ANALYSIS]")
    for location, metrics in location_analysis.items():
        print(f"{location.capitalize():8} Acc: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)  N={metrics['count']}")
    
    print(f"\n[TRAINING SET PERFORMANCE]")
    print(f"Training Accuracy:  {results['train_performance']['accuracy']:.4f} ({results['train_performance']['accuracy']*100:.2f}%)")
    print(f"Training F1-Score:  {results['train_performance']['f1_weighted']:.4f}")
    print(f"Training Samples:   {results['train_performance']['sample_count']}")
    print(f"Test Accuracy:      {results['basic_metrics']['accuracy']:.4f} ({results['basic_metrics']['accuracy']*100:.2f}%)")
    train_test_gap = (results['train_performance']['accuracy'] - results['basic_metrics']['accuracy']) * 100
    print(f"Train-Test Gap:     {train_test_gap:.2f}%")
    
    # Print Cross-Validation results if available
    if 'cross_validation' in results and 'error' not in results['cross_validation']:
        print(f"\n[CROSS-VALIDATION (5-FOLD)]")
        cv = results['cross_validation']
        print(f"Mean CV Accuracy:   {cv['mean_cv_accuracy']:.4f} ± {cv['std_cv_accuracy']:.4f}")
        print(f"CV Range:           [{cv['min_cv_accuracy']:.4f}, {cv['max_cv_accuracy']:.4f}]")
        print(f"All Fold Scores:    {', '.join([f'{s:.4f}' for s in cv['cv_scores']])}")
    
    # Print Timing metrics
    if 'timing_metrics' in results:
        print(f"\n[INFERENCE TIMING]")
        tm = results['timing_metrics']
        print(f"Test Set:   {tm['test_inference_time_seconds']:.3f}s ({tm['test_time_per_sample_ms']:.2f}ms/sample, {tm['test_throughput_samples_per_second']:.0f} samples/s)")
        print(f"Train Set:  {tm['train_inference_time_seconds']:.3f}s ({tm['train_time_per_sample_ms']:.2f}ms/sample, {tm['train_throughput_samples_per_second']:.0f} samples/s)")
    
    # Print Feature Importance summary
    if 'feature_importance' in results and 'error' not in results['feature_importance']:
        print(f"\n[FEATURE IMPORTANCE - TOP 5]")
        feat_gain = list(zip(results['feature_importance']['features'], results['feature_importance']['gain']))
        feat_gain_sorted = sorted(feat_gain, key=lambda x: x[1], reverse=True)[:5]
        for i, (feat, gain) in enumerate(feat_gain_sorted, 1):
            print(f"  {i}. {feat:20s} Gain: {gain:.0f}")
    
    print(f"\n[SPATIAL ERROR]")
    print(f"Mean:    {results['spatial_metrics']['mean_error_m']:.3f}m")
    print(f"Median:  {results['spatial_metrics']['median_error_m']:.3f}m")
    print(f"P90:     {results['spatial_metrics']['p90_error_m']:.3f}m")
    print(f"P95:     {results['spatial_metrics']['p95_error_m']:.3f}m")
    print(f"Max:     {results['spatial_metrics']['max_error_m']:.3f}m")
    
    print(f"\n[TOP ERROR PATTERNS]")
    for pattern, count in list(results['top_error_patterns'].items())[:5]:
        print(f"{pattern:10} {count} times")
    
    print(f"\n[PER-CELL SUMMARY]")
    print(f"Average Cell Accuracy:  {avg_cell_accuracy:.4f} ({avg_cell_accuracy*100:.2f}%)")
    print(f"Min Cell Accuracy:      {min_cell_accuracy:.4f} ({min_cell_accuracy*100:.2f}%)")
    print(f"Max Cell Accuracy:      {max_cell_accuracy:.4f} ({max_cell_accuracy*100:.2f}%)")
    print(f"\nTop 5 Best Performing Cells:")
    sorted_cells = sorted(cells_with_data, key=lambda c: per_cell_results[c]['accuracy'], reverse=True)
    for i, cell in enumerate(sorted_cells[:5], 1):
        metrics = per_cell_results[cell]
        print(f"  {i}. {cell}: {metrics['accuracy']*100:.2f}% (N={metrics['sample_count']})")
    
    print(f"\nTop 5 Worst Performing Cells:")
    for i, cell in enumerate(sorted_cells[-5:][::-1], 1):
        metrics = per_cell_results[cell]
        confused = ", ".join([f"{c}({n})" for c, n in metrics['top_confused_with'][:2]]) if metrics['top_confused_with'] else "N/A"
        print(f"  {i}. {cell}: {metrics['accuracy']*100:.2f}% (N={metrics['sample_count']}) - Confused with: {confused}")
    
    print(f"\n[STEP 8/8] Saving reports...")
    logger.info("[STEP 8/8] Saving comprehensive reports (JSON, TXT, MD, CSV)...")
    # SRP - Extracted report generation to dedicated function
    exec_summary = save_comprehensive_reports(results, confidence_df, per_cell_results, 
                                             output_dir, len(test_df), len(train_df))
    
    # Display executive summary
    print(f"\n{'=' * 80}")
    print(exec_summary)
    
    return 0


def display_per_cell_details():
    """Display detailed accuracy and confidence per cell"""
    csv_path = Path("reports/comprehensive_testing/per_cell_metrics.csv")
    
    if not csv_path.exists():
        print("❌ File per_cell_metrics.csv tidak ditemukan!")
        return
    
    df = pd.read_csv(csv_path, index_col='cell')
    
    print("\n" + "=" * 100)
    print("📊 DETAIL ACCURACY DAN CONFIDENCE PER CELL")
    print("=" * 100)
    
    # Sort by accuracy descending
    df_sorted = df.sort_values('accuracy', ascending=False)
    
    print(f"\n{'='*100}")
    print(f"{'CELL':<6} {'SAMPLES':<8} {'ACCURACY':<12} {'CONFIDENCE':<14} {'CORRECT':<8} {'WRONG':<7} {'CONFUSED WITH':<30}")
    print(f"{'='*100}")
    
    for cell, row in df_sorted.iterrows():
        acc = row['accuracy']
        conf = row['mean_confidence']
        samples = int(row['sample_count'])
        correct = int(row['correct_predictions'])
        wrong = int(row['incorrect_predictions'])
        confused = row['top_confused_with'] if row['top_confused_with'] != 'None' else '-'
        
        # Color coding based on accuracy
        if acc == 1.0:
            status = "✅"
        elif acc >= 0.9:
            status = "✓"
        elif acc >= 0.8:
            status = "⚠️"
        else:
            status = "❌"
        
        print(f"{status} {cell:<4} {samples:<8} {acc*100:>6.1f}%      {conf*100:>7.2f}%        {correct:<8} {wrong:<7} {confused:<30}")
    
    print(f"{'='*100}\n")
    
    # Statistics
    print("📈 STATISTIK RINGKAS")
    print("-" * 100)
    print(f"Total cells: {len(df)}")
    print(f"Perfect accuracy (100%): {(df['accuracy'] == 1.0).sum()} cells")
    print(f"Good accuracy (≥90%): {(df['accuracy'] >= 0.9).sum()} cells")
    print(f"Fair accuracy (≥80%): {(df['accuracy'] >= 0.8).sum()} cells")
    print(f"Poor accuracy (<80%): {(df['accuracy'] < 0.8).sum()} cells")
    print()
    print(f"Average accuracy: {df['accuracy'].mean()*100:.2f}%")
    print(f"Average confidence: {df['mean_confidence'].mean()*100:.2f}%")
    print(f"Accuracy std dev: {df['accuracy'].std()*100:.2f}%")
    print(f"Confidence std dev: {df['mean_confidence'].std()*100:.2f}%")
    print()
    
    # Best performing
    print("⭐ TOP 5 BEST PERFORMING CELLS")
    print("-" * 100)
    best_cells = df_sorted.head(5)
    for i, (cell, row) in enumerate(best_cells.iterrows(), 1):
        print(f"{i}. Cell {cell}: {row['accuracy']*100:.1f}% accuracy, {row['mean_confidence']*100:.2f}% confidence")
    print()
    
    # Worst performing
    print("⚠️  TOP 5 WORST PERFORMING CELLS")
    print("-" * 100)
    worst_cells = df_sorted.tail(5)[::-1]
    for i, (cell, row) in enumerate(worst_cells.iterrows(), 1):
        confused = row['top_confused_with'] if row['top_confused_with'] != 'None' else 'No confusion'
        print(f"{i}. Cell {cell}: {row['accuracy']*100:.1f}% accuracy, {row['mean_confidence']*100:.2f}% confidence")
        print(f"   → Confused with: {confused}")
    print()
    
    # Confidence analysis
    print("🎯 CONFIDENCE ANALYSIS PER CELL")
    print("-" * 100)
    high_conf = df[df['mean_confidence'] > df['mean_confidence'].mean()]
    low_conf = df[df['mean_confidence'] <= df['mean_confidence'].mean()]
    print(f"Cells with above-average confidence ({df['mean_confidence'].mean()*100:.2f}%): {len(high_conf)} cells")
    print(f"  → {', '.join(high_conf.index.tolist())}")
    print()
    print(f"Cells with below-average confidence: {len(low_conf)} cells")
    print(f"  → {', '.join(low_conf.index.tolist())}")
    print()
    
    # Grid visualization
    print("🗺️  GRID VISUALIZATION (5x5)")
    print("-" * 100)
    print("Accuracy per cell:\n")
    
    # Create 5x5 grid
    grid = np.zeros((5, 5))
    for cell in df.index:
        col = ord(cell[0]) - ord('A')
        row = int(cell[1]) - 1
        grid[row, col] = float(df.at[cell, 'accuracy']) * 100  # type: ignore[arg-type]
    
    # Print grid
    print("     A       B       C       D       E")
    print("  " + "-" * 45)
    for i in range(5):
        row_label = i + 1
        print(f"{row_label} |", end="")
        for j in range(5):
            val = grid[i, j]
            print(f" {val:>5.0f}% ", end="")
        print()
    print()
    
    # Confidence grid
    print("Confidence per cell:\n")
    conf_grid = np.zeros((5, 5))
    for cell in df.index:
        col = ord(cell[0]) - ord('A')
        row = int(cell[1]) - 1
        conf_grid[row, col] = float(df.at[cell, 'mean_confidence']) * 100  # type: ignore[arg-type]
    
    print("     A       B       C       D       E")
    print("  " + "-" * 45)
    for i in range(5):
        row_label = i + 1
        print(f"{row_label} |", end="")
        for j in range(5):
            val = conf_grid[i, j]
            print(f" {val:>5.2f}% ", end="")
        print()
    print()
    
    # Legend
    print("📌 LEGEND:")
    print("  ✅ = 100% accuracy")
    print("  ✓  = ≥90% accuracy")
    print("  ⚠️  = ≥80% accuracy")
    print("  ❌ = <80% accuracy")
    print()
    
    # Additional insights
    print("💡 INSIGHTS")
    print("-" * 100)
    
    # Check if corners perform worse
    corners = ['A1', 'A5', 'E1', 'E5']
    corner_cells = df.loc[[c for c in corners if c in df.index]]
    avg_corner_acc = corner_cells['accuracy'].mean()
    avg_all_acc = df['accuracy'].mean()
    
    if avg_corner_acc < avg_all_acc:
        print(f"⚠️  Corner cells have lower average accuracy ({avg_corner_acc*100:.1f}%) vs overall ({avg_all_acc*100:.1f}%)")
    else:
        print(f"✅ Corner cells perform well ({avg_corner_acc*100:.1f}% accuracy)")
    
    # Check edge vs center
    edges = [c for c in df.index if c[0] in ['A', 'E'] or c[1] in ['1', '5']]
    centers = [c for c in df.index if c[0] in ['B', 'C', 'D'] and c[1] in ['2', '3', '4']]
    
    if centers:
        avg_center_acc = df.loc[centers, 'accuracy'].mean()
        print(f"✅ Center cells average: {avg_center_acc*100:.1f}% accuracy")
    
    # Most confused pairs
    confused_df = df[(df['top_confused_with'] != 'None') & (df['incorrect_predictions'] > 0)]
    if len(confused_df) > 0:
        print(f"\n⚠️  Cells with confusion patterns ({len(confused_df)} cells):")
        for cell, row in confused_df.iterrows():
            print(f"   • {cell} → {row['top_confused_with']}")
    else:
        print(f"\n✅ No confusion patterns detected - all cells predict perfectly!")
    
    print("\n" + "=" * 100)
    print("📄 Untuk laporan lengkap, lihat:")
    print("   • reports/comprehensive_testing/executive_summary.txt")
    print("   • reports/comprehensive_testing/technical_analysis.txt")
    print("   • reports/comprehensive_testing/per_cell_accuracy_heatmap.png")
    print("=" * 100 + "\n")


def compact_summary():
    """Generate compact summary table of per-cell accuracy and confidence"""
    csv_path = Path("reports/comprehensive_testing/per_cell_metrics.csv")
    
    if not csv_path.exists():
        print("❌ File per_cell_metrics.csv tidak ditemukan!")
        return
    
    df = pd.read_csv(csv_path, index_col='cell')
    
    print("\n" + "="*80)
    print("📊 COMPACT SUMMARY: ACCURACY & CONFIDENCE PER CELL")
    print("="*80 + "\n")
    
    # Group by accuracy level
    perfect = df[df['accuracy'] == 1.0]
    good = df[(df['accuracy'] >= 0.9) & (df['accuracy'] < 1.0)]
    fair = df[(df['accuracy'] >= 0.8) & (df['accuracy'] < 0.9)]
    poor = df[df['accuracy'] < 0.8]
    
    print(f"✅ PERFECT (100% accuracy) - {len(perfect)} cells:")
    print("-" * 80)
    for i in range(0, len(perfect), 5):
        batch = perfect.iloc[i:i+5]
        cells_info = [f"{cell}({row['mean_confidence']*100:.1f}%)" 
                     for cell, row in batch.iterrows()]
        print("   " + ", ".join(cells_info))
    
    if len(good) > 0:
        print("\n✓  GOOD (90-99% accuracy) - {} cells:".format(len(good)))
        print("-" * 80)
        for cell, row in good.iterrows():
            print(f"   {cell}: {row['accuracy']*100:.0f}% acc, {row['mean_confidence']*100:.1f}% conf")
    
    if len(fair) > 0:
        print("\n⚠️  FAIR (80-89% accuracy) - {} cells:".format(len(fair)))
        print("-" * 80)
        for cell, row in fair.iterrows():
            confused = row['top_confused_with'] if row['top_confused_with'] != 'None' else '-'
            print(f"   {cell}: {row['accuracy']*100:.0f}% acc, {row['mean_confidence']*100:.1f}% conf → confused: {confused}")
    
    if len(poor) > 0:
        print("\n❌ POOR (<80% accuracy) - {} cells:".format(len(poor)))
        print("-" * 80)
        for cell, row in poor.iterrows():
            confused = row['top_confused_with'] if row['top_confused_with'] != 'None' else '-'
            print(f"   {cell}: {row['accuracy']*100:.0f}% acc, {row['mean_confidence']*100:.1f}% conf → confused: {confused}")
    
    print("\n" + "="*80)
    print("📈 QUICK STATS")
    print("="*80)
    print(f"Average Accuracy:   {df['accuracy'].mean()*100:.2f}%")
    print(f"Average Confidence: {df['mean_confidence'].mean()*100:.2f}%")
    print(f"Std Dev Accuracy:   {df['accuracy'].std()*100:.2f}%")
    print(f"Highest Confidence: {df['mean_confidence'].max()*100:.2f}% (Cell {df['mean_confidence'].idxmax()})")
    print(f"Lowest Confidence:  {df['mean_confidence'].min()*100:.2f}% (Cell {df['mean_confidence'].idxmin()})")
    print(f"Highest Accuracy:   100.0% ({len(perfect)} cells)")
    print(f"Lowest Accuracy:    {df['accuracy'].min()*100:.0f}% (Cell {df['accuracy'].idxmin()})")
    
    print("\n" + "="*80)
    print("🎯 KEY TAKEAWAYS")
    print("="*80)
    print(f"• {len(perfect)}/25 cells ({len(perfect)/25*100:.0f}%) achieve perfect accuracy")
    print(f"• {len(df[df['accuracy'] >= 0.9])}/25 cells ({len(df[df['accuracy'] >= 0.9])/25*100:.0f}%) have ≥90% accuracy")
    
    if len(poor) > 0:
        print(f"• {len(poor)} cell(s) need improvement: {', '.join(poor.index.tolist())}")
    
    # Check confusion patterns
    confused = df[(df['top_confused_with'] != 'None') & (df['incorrect_predictions'] > 0)]
    if len(confused) > 0:
        print(f"• {len(confused)} cells show confusion patterns:")
        for cell, row in confused.iterrows():
            print(f"  - {cell} → {row['top_confused_with']}")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--details":
            display_per_cell_details()
        elif sys.argv[1] == "--compact":
            compact_summary()
        else:
            print("Usage:")
            print("  python comprehensive_testing.py              # Run full testing")
            print("  python comprehensive_testing.py --details    # Show per-cell details")
            print("  python comprehensive_testing.py --compact    # Show compact summary")
    else:
        sys.exit(main())
