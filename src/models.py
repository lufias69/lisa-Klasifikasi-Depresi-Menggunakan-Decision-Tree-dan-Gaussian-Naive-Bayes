"""
Model training utilities
Includes: Gaussian NB, Decision Tree
With imbalanced data handling: SMOTE, ADASYN, Class Weights
"""

import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.pipeline import Pipeline as ImbPipeline
from typing import Dict, Tuple, Any
import joblib
import warnings
warnings.filterwarnings('ignore')

from .config import (
    GAUSSIAN_NB_PARAMS,
    DECISION_TREE_PARAMS,
    CV_FOLDS,
    RANDOM_STATE
)


def get_model_and_params(model_name: str) -> Tuple[Any, Dict]:
    """
    Get model instance and hyperparameter grid
    
    Args:
        model_name: Name of the model ('gaussian_nb', 'decision_tree')
    
    Returns:
        Tuple of (model instance, parameter grid)
    """
    if model_name == 'gaussian_nb':
        model = GaussianNB()
        params = {'model__var_smoothing': GAUSSIAN_NB_PARAMS['var_smoothing']}
    elif model_name == 'decision_tree':
        model = DecisionTreeClassifier(random_state=RANDOM_STATE)
        params = {
            'model__max_depth': DECISION_TREE_PARAMS['max_depth'],
            'model__min_samples_split': DECISION_TREE_PARAMS['min_samples_split'],
            'model__min_samples_leaf': DECISION_TREE_PARAMS['min_samples_leaf'],
            'model__criterion': DECISION_TREE_PARAMS['criterion']
        }
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model, params


def create_pipeline(model_name: str, imbalance_strategy: str = 'original') -> ImbPipeline:
    """
    Create training pipeline with optional imbalanced data handling
    
    Args:
        model_name: Name of the model
        imbalance_strategy: Strategy for handling imbalanced data
                           ('original', 'smote', 'adasyn', 'class_weight', 'smote_weight')
    
    Returns:
        Pipeline with scaler, sampler (optional), and model
    """
    steps = []
    
    # Standard scaling for all models
    steps.append(('scaler', StandardScaler()))
    
    # Add sampling technique
    if imbalance_strategy == 'smote':
        steps.append(('sampler', SMOTE(random_state=RANDOM_STATE)))
    elif imbalance_strategy == 'adasyn':
        steps.append(('sampler', ADASYN(random_state=RANDOM_STATE)))
    
    # Add model
    model, _ = get_model_and_params(model_name)
    
    # Apply class weight if specified
    if imbalance_strategy in ['class_weight', 'smote_weight']:
        if model_name == 'decision_tree':
            model.set_params(class_weight='balanced')
    
    steps.append(('model', model))
    
    pipeline = ImbPipeline(steps)
    return pipeline


def train_model(X_train: pd.DataFrame, y_train: pd.Series,
                model_name: str, imbalance_strategy: str = 'original',
                verbose: int = 1) -> Tuple[Any, Dict]:
    """
    Train a single model with hyperparameter tuning
    
    Args:
        X_train: Training features
        y_train: Training labels
        model_name: Name of the model
        imbalance_strategy: Imbalanced data handling strategy
        verbose: Verbosity level
    
    Returns:
        Tuple of (best model, training info)
    """
    print(f"\n{'='*80}")
    print(f"Training: {model_name.upper()} with {imbalance_strategy} strategy")
    print(f"{'='*80}")
    
    # Create pipeline
    pipeline = create_pipeline(model_name, imbalance_strategy)
    
    # Get parameter grid
    _, params = get_model_and_params(model_name)
    
    # Setup cross-validation
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    # Grid search
    grid_search = GridSearchCV(
        pipeline,
        params,
        cv=cv,
        scoring='f1_macro',
        n_jobs=-1,
        verbose=verbose
    )
    
    # Fit
    grid_search.fit(X_train, y_train)
    
    # Training info
    training_info = {
        'model_name': model_name,
        'imbalance_strategy': imbalance_strategy,
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'cv_results': grid_search.cv_results_
    }
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV F1-macro score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, training_info


def train_all_models(X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Dict]:
    """
    Train all models with all imbalanced strategies
    
    Args:
        X_train: Training features
        y_train: Training labels
    
    Returns:
        Dictionary with all trained models and their info
    """
    models = ['gaussian_nb', 'decision_tree']
    strategies = ['original', 'smote', 'adasyn', 'class_weight', 'smote_weight']
    
    results = {}
    
    total_experiments = len(models) * len(strategies)
    experiment_num = 0
    
    for model_name in models:
        for strategy in strategies:
            experiment_num += 1
            experiment_id = f"{model_name}_{strategy}"
            
            print(f"\n\n{'#'*80}")
            print(f"EXPERIMENT {experiment_num}/{total_experiments}: {experiment_id}")
            print(f"{'#'*80}")
            
            try:
                best_model, training_info = train_model(
                    X_train, y_train,
                    model_name, strategy,
                    verbose=0
                )
                
                results[experiment_id] = {
                    'model': best_model,
                    'info': training_info,
                    'status': 'success'
                }
                
            except Exception as e:
                print(f"ERROR in {experiment_id}: {str(e)}")
                results[experiment_id] = {
                    'model': None,
                    'info': None,
                    'status': 'failed',
                    'error': str(e)
                }
    
    return results


def save_model(model: Any, filepath: str):
    """
    Save trained model to disk
    
    Args:
        model: Trained model
        filepath: Path to save the model
    """
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")


def load_model(filepath: str) -> Any:
    """
    Load trained model from disk
    
    Args:
        filepath: Path to the saved model
    
    Returns:
        Loaded model
    """
    model = joblib.load(filepath)
    print(f"Model loaded from {filepath}")
    return model


if __name__ == '__main__':
    # Test with dummy data
    from sklearn.datasets import make_classification
    
    X, y = make_classification(
        n_samples=100,
        n_features=30,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        weights=[0.6, 0.4],
        random_state=RANDOM_STATE
    )
    
    X_train = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    y_train = pd.Series(y)
    
    print("Training set shape:", X_train.shape)
    print("Label distribution:", y_train.value_counts())
    
    # Train a single model
    model, info = train_model(X_train, y_train, 'gaussian_nb', 'smote', verbose=1)
    
    print("\nModel trained successfully!")
    print("Best parameters:", info['best_params'])
    print("Best CV score:", info['best_score'])
