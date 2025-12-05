"""
Model evaluation utilities
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)
from sklearn.model_selection import cross_validate, StratifiedKFold
from scipy import stats
from typing import Dict, List, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

from .config import CV_FOLDS, RANDOM_STATE


def evaluate_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """
    Evaluate model on test set
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
    
    Returns:
        Dictionary of evaluation metrics
    """
    # Predictions
    y_pred = model.predict(X_test)
    
    # Probabilities (for AUC-ROC)
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = y_pred
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='binary'),
        'recall': recall_score(y_test, y_pred, average='binary'),
        'f1': f1_score(y_test, y_pred, average='binary'),
        'precision_macro': precision_score(y_test, y_pred, average='macro'),
        'recall_macro': recall_score(y_test, y_pred, average='macro'),
        'f1_macro': f1_score(y_test, y_pred, average='macro'),
    }
    
    # AUC-ROC (only if probabilities available)
    try:
        metrics['roc_auc'] = roc_auc_score(y_test, y_proba)
    except:
        metrics['roc_auc'] = np.nan
    
    # Confusion matrix elements
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    metrics['true_negative'] = int(tn)
    metrics['false_positive'] = int(fp)
    metrics['false_negative'] = int(fn)
    metrics['true_positive'] = int(tp)
    
    # Clinical metrics
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    return metrics


def cross_validate_model(model: Any, X: pd.DataFrame, y: pd.Series, 
                        cv_folds: int = CV_FOLDS) -> Dict[str, Any]:
    """
    Perform cross-validation on model
    
    Args:
        model: Model to evaluate
        X: Features
        y: Labels
        cv_folds: Number of CV folds
    
    Returns:
        Dictionary with CV results
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision_macro',
        'recall': 'recall_macro',
        'f1': 'f1_macro',
        'roc_auc': 'roc_auc'
    }
    
    cv_results = cross_validate(
        model, X, y,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
        n_jobs=-1
    )
    
    # Compute mean and std for each metric
    results = {}
    for metric in scoring.keys():
        scores = cv_results[f'test_{metric}']
        results[f'{metric}_mean'] = np.mean(scores)
        results[f'{metric}_std'] = np.std(scores)
        results[f'{metric}_scores'] = scores
    
    return results


def evaluate_all_models(models_dict: Dict[str, Dict], 
                       X_test: pd.DataFrame, 
                       y_test: pd.Series) -> pd.DataFrame:
    """
    Evaluate all trained models on test set
    
    Args:
        models_dict: Dictionary of trained models
        X_test: Test features
        y_test: Test labels
    
    Returns:
        DataFrame with evaluation results
    """
    print("="*80)
    print("EVALUATING ALL MODELS ON TEST SET")
    print("="*80)
    
    results = []
    
    for experiment_id, model_data in models_dict.items():
        if model_data['status'] != 'success':
            print(f"Skipping {experiment_id} (training failed)")
            continue
        
        print(f"\nEvaluating {experiment_id}...")
        
        model = model_data['model']
        metrics = evaluate_model(model, X_test, y_test)
        
        # Add experiment info
        metrics['experiment_id'] = experiment_id
        metrics['model_name'] = model_data['info']['model_name']
        metrics['imbalance_strategy'] = model_data['info']['imbalance_strategy']
        metrics['cv_f1_macro'] = model_data['info']['best_score']
        
        results.append(metrics)
        
        # Print summary
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1-macro: {metrics['f1_macro']:.4f}")
        print(f"  AUC-ROC: {metrics['roc_auc']:.4f}")
    
    results_df = pd.DataFrame(results)
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    
    return results_df


def compare_models_statistical(results_df: pd.DataFrame, metric: str = 'f1_macro') -> pd.DataFrame:
    """
    Statistical comparison of models
    
    Args:
        results_df: DataFrame with evaluation results
        metric: Metric to compare
    
    Returns:
        DataFrame with statistical test results
    """
    # Sort by metric
    sorted_df = results_df.sort_values(metric, ascending=False)
    
    print(f"\n{'='*80}")
    print(f"MODEL RANKING BY {metric.upper()}")
    print(f"{'='*80}")
    
    for i, row in sorted_df.iterrows():
        print(f"{row['experiment_id']:40s} | {metric}: {row[metric]:.4f}")
    
    return sorted_df


def get_best_model(results_df: pd.DataFrame, 
                  models_dict: Dict[str, Dict],
                  metric: str = 'f1_macro') -> Tuple[str, Any, Dict]:
    """
    Get the best performing model
    
    Args:
        results_df: DataFrame with evaluation results
        models_dict: Dictionary of trained models
        metric: Metric to use for selection
    
    Returns:
        Tuple of (experiment_id, model, metrics)
    """
    best_row = results_df.loc[results_df[metric].idxmax()]
    best_experiment_id = best_row['experiment_id']
    best_model = models_dict[best_experiment_id]['model']
    best_metrics = best_row.to_dict()
    
    print(f"\n{'='*80}")
    print(f"BEST MODEL: {best_experiment_id}")
    print(f"{'='*80}")
    print(f"Metric used for selection: {metric}")
    print(f"Best {metric}: {best_metrics[metric]:.4f}")
    print(f"\nAll metrics:")
    for key, value in best_metrics.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            print(f"  {key}: {value:.4f}")
    
    return best_experiment_id, best_model, best_metrics


def print_confusion_matrix(y_true: pd.Series, y_pred: np.ndarray, 
                          labels: List[str] = ['Control', 'Condition']):
    """
    Print confusion matrix in readable format
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Class labels
    """
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"\n{'='*80}")
    print("CONFUSION MATRIX")
    print(f"{'='*80}")
    print(f"\n                Predicted")
    print(f"              {labels[0]:>10s}  {labels[1]:>10s}")
    print(f"Actual")
    print(f"  {labels[0]:>10s}  {cm[0,0]:>10d}  {cm[0,1]:>10d}")
    print(f"  {labels[1]:>10s}  {cm[1,0]:>10d}  {cm[1,1]:>10d}")
    print()


def generate_classification_report(y_true: pd.Series, y_pred: np.ndarray,
                                  labels: List[str] = ['Control', 'Condition']) -> str:
    """
    Generate detailed classification report
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Class labels
    
    Returns:
        Classification report string
    """
    report = classification_report(y_true, y_pred, target_names=labels)
    
    print(f"\n{'='*80}")
    print("CLASSIFICATION REPORT")
    print(f"{'='*80}")
    print(report)
    
    return report


if __name__ == '__main__':
    # Test with dummy data
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import GaussianNB
    from sklearn.preprocessing import StandardScaler
    
    # Generate dummy data
    X, y = make_classification(
        n_samples=100,
        n_features=30,
        n_classes=2,
        weights=[0.6, 0.4],
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    X_train = pd.DataFrame(X_train, columns=[f'feature_{i}' for i in range(X.shape[1])])
    X_test = pd.DataFrame(X_test, columns=[f'feature_{i}' for i in range(X.shape[1])])
    y_train = pd.Series(y_train)
    y_test = pd.Series(y_test)
    
    # Train a simple model
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = GaussianNB()
    model.fit(X_train_scaled, y_train)
    
    # Wrap in a simple pipeline-like object for testing
    class SimpleModel:
        def __init__(self, scaler, model):
            self.scaler = scaler
            self.model = model
        
        def predict(self, X):
            X_scaled = self.scaler.transform(X)
            return self.model.predict(X_scaled)
        
        def predict_proba(self, X):
            X_scaled = self.scaler.transform(X)
            return self.model.predict_proba(X_scaled)
    
    wrapped_model = SimpleModel(scaler, model)
    
    # Evaluate
    print("Testing evaluation functions...")
    metrics = evaluate_model(wrapped_model, X_test, y_test)
    
    print("\nMetrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    # Confusion matrix
    y_pred = wrapped_model.predict(X_test)
    print_confusion_matrix(y_test, y_pred)
    generate_classification_report(y_test, y_pred)
