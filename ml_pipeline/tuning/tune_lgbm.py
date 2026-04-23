"""
LightGBM Hyperparameter Tuning with Optuna

This script performs comprehensive hyperparameter optimization for BLE indoor
localization using LightGBM and Optuna with 5-Fold Cross-Validation.

Pipeline:
    1. Load prepared datasets (train, val, test)
    2. Optuna optimization with StratifiedKFold CV (100 trials)
    3. Train final model on merged train+val data
    4. Evaluate on all sets (train, val, test)
    5. Save model, metrics, and results

Features:
    - GPU auto-detection and configuration
    - Training time tracking (solves evaluation gap)
    - Feature importance extraction
    - Comprehensive metrics reporting
    - MedianPruner for early stopping of poor trials

Author: IoT Engineering Expert
Date: December 24, 2025
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from optuna.pruners import MedianPruner
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config_ml import (
    ALL_FEATURE_COLUMNS,
    LGBM_PARAMS,
    PROCESSED_DIR,
    RANDOM_STATE,
    TARGET_COLUMN,
    TUNED_MODEL_DIR,
)
from core.logger import setup_logger
from training.gpu_config import get_gpu_params

# Constants
N_FOLDS = 5
N_TRIALS = 100
EARLY_STOPPING_ROUNDS = 50
MAX_BOOST_ROUNDS = 1000  # Reduced: with lr=0.05, 1000 is sufficient


def load_datasets() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load prepared datasets from processed directory (80/20 Split).
    
    Returns:
        tuple: (train_df, test_df) DataFrames
    
    Raises:
        FileNotFoundError: If dataset files are missing
        ValueError: If datasets have incorrect structure
    """
    train_path = PROCESSED_DIR / "train.csv"
    test_path = PROCESSED_DIR / "test.csv"
    
    # Validate files exist
    for path in [train_path, test_path]:
        if not path.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {path}. "
                "Please run 'python training/prepare_dataset.py' first."
            )
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Validate required columns
    required_cols = ALL_FEATURE_COLUMNS + [TARGET_COLUMN]
    for df_name, df in [("train", train_df), ("test", test_df)]:
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(
                f"{df_name}.csv missing required columns: {missing}"
            )
    
    return train_df, test_df


def create_objective_function(
    X: np.ndarray,
    y: np.ndarray,
    label_encoder: LabelEncoder,
    base_params: Dict,
    logger
) -> Tuple[Callable[[optuna.Trial], float], Dict]:
    """
    Create Optuna objective function with 5-Fold Cross-Validation.
    
    Args:
        X: Feature matrix (train only)
        y: Target labels (train only)
        label_encoder: Fitted LabelEncoder
        base_params: Base LightGBM parameters
        logger: Logger instance
    
    Returns:
        tuple: (objective_function, shared_data_dict)
    """
    
    # Shared dictionary to store best_iteration across folds
    shared_data = {'best_iterations': []}
    
    def objective(trial: optuna.Trial) -> float:
        """
        Optuna objective function.
        
        Suggests hyperparameters and returns mean CV accuracy.
        """
        # Suggest hyperparameters
        params = base_params.copy()
        params.update({
            'num_leaves': trial.suggest_int('num_leaves', 15, 255),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 100),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        })
        
        # 5-Fold Stratified Cross-Validation
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        cv_scores = []
        fold_iterations = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_fold_train, X_fold_val = X[train_idx], X[val_idx]
            y_fold_train, y_fold_val = y[train_idx], y[val_idx]
            
            # Create LightGBM datasets
            train_data = lgb.Dataset(X_fold_train, label=y_fold_train)
            val_data = lgb.Dataset(X_fold_val, label=y_fold_val, reference=train_data)
            
            # Train model
            model = lgb.train(
                params,
                train_data,
                num_boost_round=500,
                valid_sets=[val_data],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=30),
                    lgb.log_evaluation(0)  # Suppress output
                ]
            )
            
            # Track best_iteration from this fold
            fold_iterations.append(model.best_iteration)
            
            # Evaluate on validation fold
            y_pred = model.predict(X_fold_val)
            y_pred_class = np.argmax(np.asarray(y_pred), axis=1)
            fold_accuracy = float(accuracy_score(y_fold_val, y_pred_class))
            cv_scores.append(fold_accuracy)
            
            # Prune trial if performing poorly
            trial.report(fold_accuracy, fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        # Store best_iterations for best trial (will be overwritten each trial)
        shared_data['best_iterations'] = fold_iterations
        
        mean_accuracy = float(np.mean(cv_scores))
        return mean_accuracy
    
    return objective, shared_data


def optimize_hyperparameters(
    X_train,
    y_train,
    label_encoder: LabelEncoder,
    base_params: Dict,
    logger,
    n_trials: int = N_TRIALS
) -> Tuple[Dict, optuna.Study, int]:
    """
    Optimize hyperparameters using Optuna with MedianPruner.
    
    Args:
        X_train: Training feature matrix
        y_train: Training labels (encoded)
        label_encoder: Fitted LabelEncoder
        base_params: Base LightGBM parameters
        logger: Logger instance
        n_trials: Number of Optuna trials
    
    Returns:
        tuple: (best_params, study, mean_best_iteration)
    """
    logger.info("=" * 70)
    logger.info("STEP 2: OPTUNA HYPERPARAMETER OPTIMIZATION")
    logger.info("=" * 70)
    logger.info(f"Trials: {n_trials}")
    logger.info(f"Cross-Validation: {N_FOLDS}-Fold Stratified")
    logger.info("Optimization started...")
    
    # Create study with MedianPruner
    study = optuna.create_study(
        direction='maximize',
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=3),
        study_name='lightgbm_tuning'
    )
    
    # Create objective function with shared data tracking
    objective, shared_data = create_objective_function(
        X_train, y_train, label_encoder, base_params, logger
    )
    
    # Optimize
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # Get mean best_iteration from best trial's CV folds
    mean_best_iteration = int(np.mean(shared_data['best_iterations']))
    
    # Results
    logger.info("=" * 70)
    logger.info("OPTIMIZATION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Best trial: #{study.best_trial.number}")
    logger.info(f"Best CV accuracy: {study.best_value:.4f} ({study.best_value*100:.2f}%)")
    logger.info(f"Mean best_iteration from CV: {mean_best_iteration}")
    logger.info("Best hyperparameters:")
    for key, value in study.best_params.items():
        logger.info(f"  {key}: {value}")
    
    completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
    logger.info(f"Trial statistics: {completed} completed, {pruned} pruned")
    
    return study.best_params, study, mean_best_iteration


def train_final_model(
    X_train,
    y_train,
    best_params: Dict,
    base_params: Dict,
    mean_best_iteration: int,
    logger
) -> Tuple[lgb.Booster, float]:
    """
    Train final model on full training data with CV-derived best_iteration.
    
    Uses num_boost_round from CV (no early stopping, no validation set).
    This is the conservative 80/20 approach: tune with CV, train on train only.
    
    Args:
        X_train: Training features (1,000 samples)
        y_train: Training labels
        best_params: Best hyperparameters from Optuna
        base_params: Base LightGBM parameters
        mean_best_iteration: Mean best_iteration from CV folds
        logger: Logger instance
    
    Returns:
        tuple: (trained_model, training_time_seconds)
    """
    logger.info("=" * 70)
    logger.info("STEP 3: TRAIN FINAL MODEL WITH BEST PARAMS")
    logger.info("=" * 70)
    logger.info(f"Training on: {len(X_train)} samples (80% of data)")
    logger.info(f"Using num_boost_round={mean_best_iteration} from CV")
    
    # Combine parameters
    final_params = base_params.copy()
    final_params.update(best_params)
    
    # Create dataset
    train_data = lgb.Dataset(X_train, label=y_train)
    
    # Track training time
    logger.info("Training model...")
    training_start_time = time.time()
    
    # Train with fixed iterations (no early stopping)
    model = lgb.train(
        final_params,
        train_data,
        num_boost_round=mean_best_iteration,
        callbacks=[lgb.log_evaluation(100)]
    )
    
    training_end_time = time.time()
    training_time_seconds = training_end_time - training_start_time
    
    logger.info(f"Training completed in {training_time_seconds:.2f} seconds")
    logger.info(f"Total boosting rounds: {model.current_iteration()}")
    
    return model, training_time_seconds


def evaluate_model(
    model: lgb.Booster,
    X_train,
    y_train,
    X_test,
    y_test,
    logger
) -> Dict:
    """
    Evaluate model on train and test sets.
    
    Args:
        model: Trained LightGBM model
        X_train, y_train: Training data
        X_test, y_test: Test data
        logger: Logger instance
    
    Returns:
        dict: Comprehensive evaluation metrics
    """
    logger.info("=" * 70)
    logger.info("STEP 4: EVALUATE MODEL ON TRAIN AND TEST SETS")
    logger.info("=" * 70)
    
    metrics = {}
    
    for set_name, X, y in [
        ('train', X_train, y_train),
        ('test', X_test, y_test)
    ]:
        y_pred = model.predict(X)
        y_pred_class = np.argmax(np.asarray(y_pred), axis=1)
        
        accuracy = accuracy_score(y, y_pred_class)
        f1_weighted = f1_score(y, y_pred_class, average='weighted')
        f1_macro = f1_score(y, y_pred_class, average='macro')
        
        metrics[f'{set_name}_accuracy'] = float(accuracy)
        metrics[f'{set_name}_f1_weighted'] = float(f1_weighted)
        metrics[f'{set_name}_f1_macro'] = float(f1_macro)
        metrics[f'{set_name}_samples'] = len(y)
        
        logger.info(f"{set_name.upper()} SET:")
        logger.info(f"  Accuracy:    {accuracy:.4f} ({accuracy*100:.2f}%)")
        logger.info(f"  F1-weighted: {f1_weighted:.4f}")
        logger.info(f"  F1-macro:    {f1_macro:.4f}")
        logger.info(f"  Samples:     {len(y)}")
    
    # Overfitting check
    train_test_gap = metrics['train_accuracy'] - metrics['test_accuracy']
    
    logger.info("=" * 70)
    logger.info("GENERALIZATION ANALYSIS:")
    logger.info(f"  Train-Test gap: {train_test_gap:.4f} ({train_test_gap*100:.2f}%)")
    
    if train_test_gap > 0.10:
        logger.warning("  [WARNING] High overfitting detected")
    elif train_test_gap > 0.05:
        logger.info("  [INFO] Moderate overfitting")
    else:
        logger.info("  [OK] Good generalization")
    
    return metrics


def extract_feature_importance(
    model: lgb.Booster,
    feature_names: list,
    logger
) -> Dict:
    """
    Extract feature importance (gain and split count).
    
    Args:
        model: Trained LightGBM model
        feature_names: List of feature names
        logger: Logger instance
    
    Returns:
        dict: Feature importance data
    """
    importance_gain = model.feature_importance(importance_type='gain')
    importance_split = model.feature_importance(importance_type='split')
    
    feature_info = {
        'features': feature_names,
        'importance_gain': importance_gain.tolist(),
        'importance_split': importance_split.tolist()
    }
    
    # Log top 10 features
    gain_sorted = sorted(
        zip(feature_names, importance_gain),
        key=lambda x: x[1],
        reverse=True
    )
    
    logger.info("=" * 70)
    logger.info("TOP 10 FEATURES (by gain):")
    for rank, (feat, gain) in enumerate(gain_sorted[:10], 1):
        logger.info(f"  {rank:2d}. {feat:20s} : {gain:.2f}")
    
    return feature_info


def save_results(
    model: lgb.Booster,
    label_encoder: LabelEncoder,
    best_params: Dict,
    metrics: Dict,
    training_time: float,
    feature_info: Dict,
    study: optuna.Study,
    mean_best_iteration: int,
    logger
) -> None:
    """
    Save all results to TUNED_MODEL_DIR.
    
    Args:
        model: Trained model
        label_encoder: Fitted LabelEncoder
        best_params: Best hyperparameters
        metrics: Evaluation metrics
        training_time: Training time in seconds
        feature_info: Feature importance data
        study: Optuna study object
        mean_best_iteration: Mean best_iteration from CV
        logger: Logger instance
    """
    logger.info("=" * 70)
    logger.info("STEP 5: SAVE RESULTS")
    logger.info("=" * 70)
    
    TUNED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Save model (text format - primary)
    model_txt_path = TUNED_MODEL_DIR / "lgbm_tuned.txt"
    model.save_model(str(model_txt_path))
    logger.info(f"  Model (txt):  {model_txt_path}")
    
    # 2. Save model (pickle - backup)
    model_pkl_path = TUNED_MODEL_DIR / "lgbm_tuned.pkl"
    joblib.dump(model, model_pkl_path)
    logger.info(f"  Model (pkl):  {model_pkl_path}")
    
    # 3. Save label encoder
    encoder_path = TUNED_MODEL_DIR / "label_encoder.pkl"
    joblib.dump(label_encoder, encoder_path)
    logger.info(f"  Label encoder: {encoder_path}")
    
    # 4. Save best parameters
    params_path = TUNED_MODEL_DIR / "best_params.json"
    with open(params_path, 'w') as f:
        json.dump(best_params, f, indent=2)
    logger.info(f"  Best params:   {params_path}")
    
    # 5. Save evaluation results (includes training_time, mean_best_iteration, and CV results!)
    results_path = TUNED_MODEL_DIR / "evaluation_results.json"
    evaluation_results = {
        **metrics,
        'best_cv_accuracy': float(study.best_value),
        'best_trial_number': study.best_trial.number,
        'n_trials_completed': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        'n_trials_pruned': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
        'training_time_seconds': training_time,
        'mean_best_iteration_cv': mean_best_iteration,
        'num_boost_rounds_used': model.current_iteration(),
        'n_features': len(feature_info['features']),
        'n_classes': len(label_encoder.classes_),
        'split_strategy': '80/20 (train/test)',
        'cv_folds': N_FOLDS,
        'timestamp': datetime.now().isoformat()
    }
    with open(results_path, 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    logger.info(f"  Eval results:  {results_path}")
    
    # 6. Save feature info
    feature_path = TUNED_MODEL_DIR / "feature_info.json"
    with open(feature_path, 'w') as f:
        json.dump(feature_info, f, indent=2)
    logger.info(f"  Feature info:  {feature_path}")
    
    # 7. Save Optuna study
    study_path = TUNED_MODEL_DIR / "optuna_study.pkl"
    joblib.dump(study, study_path)
    logger.info(f"  Optuna study:  {study_path}")
    
    logger.info("=" * 70)
    logger.info("[OK] All results saved successfully!")
    logger.info("=" * 70)


def main() -> int:
    """
    Main orchestration function.
    
    Returns:
        int: Exit code (0 = success, 1 = error)
    """
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path(__file__).parent.parent / "logs" / f"tune_lgbm_{timestamp}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logger(
        "tune_lgbm",
        log_file=log_path,
        level=20  # INFO
    )
    
    try:
        logger.info("=" * 70)
        logger.info("LIGHTGBM HYPERPARAMETER TUNING WITH OPTUNA (80/20 Split)")
        logger.info("=" * 70)
        logger.info(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Log file: {log_path}")
        logger.info("")
        
        # ===== STEP 1: Load datasets =====
        logger.info("=" * 70)
        logger.info("STEP 1: LOAD PREPARED DATASETS")
        logger.info("=" * 70)
        
        train_df, test_df = load_datasets()
        
        logger.info("Loaded datasets (80/20 split):")
        logger.info(f"  Train: {len(train_df)} samples")
        logger.info(f"  Test:  {len(test_df)} samples")
        logger.info(f"  Features: {len(ALL_FEATURE_COLUMNS)}")
        
        # Prepare data
        X_train = train_df[ALL_FEATURE_COLUMNS].to_numpy()
        X_test = test_df[ALL_FEATURE_COLUMNS].to_numpy()
        
        y_train_raw = train_df[TARGET_COLUMN].to_numpy()
        y_test_raw = test_df[TARGET_COLUMN].to_numpy()
        
        # Encode labels
        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(y_train_raw)
        y_test = label_encoder.transform(y_test_raw)
        
        n_classes = len(label_encoder.classes_)
        logger.info(f"Classes: {n_classes}")
        logger.info("")
        
        # ===== GPU Configuration =====
        base_params = LGBM_PARAMS.copy()
        base_params['objective'] = 'multiclass'
        base_params['num_class'] = n_classes
        base_params['metric'] = 'multi_error'
        base_params['verbose'] = -1
        
        # Apply GPU config
        base_params = get_gpu_params(base_params)
        logger.info(f"Device: {base_params.get('device', 'cpu')}")
        logger.info("")
        
        # ===== STEP 2: Hyperparameter optimization =====
        best_params, study, mean_best_iteration = optimize_hyperparameters(
            X_train, y_train, label_encoder, base_params, logger, n_trials=N_TRIALS
        )
        logger.info("")
        
        # ===== STEP 3: Train final model =====
        model, training_time = train_final_model(
            X_train, y_train, best_params, base_params, mean_best_iteration, logger
        )
        logger.info("")
        
        # ===== STEP 4: Evaluate model =====
        metrics = evaluate_model(
            model, X_train, y_train, X_test, y_test, logger
        )
        logger.info("")
        
        # ===== Feature importance =====
        feature_info = extract_feature_importance(
            model, ALL_FEATURE_COLUMNS, logger
        )
        logger.info("")
        
        # ===== STEP 5: Save results =====
        save_results(
            model, label_encoder, best_params, metrics,
            training_time, feature_info, study, mean_best_iteration, logger
        )
        
        # ===== Final summary =====
        logger.info("")
        logger.info("=" * 70)
        logger.info("TUNING COMPLETE! (80/20 Split Strategy)")
        logger.info("=" * 70)
        logger.info(f"Best CV accuracy:     {study.best_value*100:.2f}%")
        logger.info(f"Test accuracy:        {metrics['test_accuracy']*100:.2f}%")
        logger.info(f"Training time:        {training_time:.2f}s")
        logger.info(f"Mean best_iteration:  {mean_best_iteration}")
        logger.info(f"Saved to:             {TUNED_MODEL_DIR}")
        logger.info("=" * 70)
        
        return 0
        
    except Exception as e:
        logger.error(f"[ERROR] Tuning failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
