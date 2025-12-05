"""
Feature selection utilities
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import (
    VarianceThreshold,
    SelectKBest,
    mutual_info_classif,
    RFE
)
from sklearn.tree import DecisionTreeClassifier
from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')


def remove_low_variance_features(X: pd.DataFrame, threshold: float = 0.01) -> Tuple[pd.DataFrame, List[str]]:
    """
    Remove features with low variance
    
    Args:
        X: Feature DataFrame
        threshold: Variance threshold
    
    Returns:
        Tuple of (filtered DataFrame, removed feature names)
    """
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(X)
    
    selected_features = X.columns[selector.get_support()].tolist()
    removed_features = X.columns[~selector.get_support()].tolist()
    
    print(f"Variance threshold filter: Kept {len(selected_features)} features, removed {len(removed_features)} features")
    
    return X[selected_features], removed_features


def remove_correlated_features(X: pd.DataFrame, threshold: float = 0.95) -> Tuple[pd.DataFrame, List[str]]:
    """
    Remove highly correlated features
    
    Args:
        X: Feature DataFrame
        threshold: Correlation threshold
    
    Returns:
        Tuple of (filtered DataFrame, removed feature names)
    """
    corr_matrix = X.corr().abs()
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]
    
    print(f"Correlation filter: Kept {len(X.columns) - len(to_drop)} features, removed {len(to_drop)} features")
    
    return X.drop(columns=to_drop), to_drop


def select_k_best_features(X: pd.DataFrame, y: pd.Series, k: int = 40) -> Tuple[pd.DataFrame, List[str], np.ndarray]:
    """
    Select K best features using mutual information
    
    Args:
        X: Feature DataFrame
        y: Target labels
        k: Number of features to select
    
    Returns:
        Tuple of (selected DataFrame, selected feature names, feature scores)
    """
    k = min(k, X.shape[1])  # Ensure k doesn't exceed number of features
    
    selector = SelectKBest(mutual_info_classif, k=k)
    selector.fit(X, y)
    
    selected_features = X.columns[selector.get_support()].tolist()
    feature_scores = selector.scores_
    
    print(f"SelectKBest (mutual information): Selected {len(selected_features)} features")
    
    return X[selected_features], selected_features, feature_scores


def select_features_rfe(X: pd.DataFrame, y: pd.Series, n_features: int = 30) -> Tuple[pd.DataFrame, List[str], np.ndarray]:
    """
    Select features using Recursive Feature Elimination
    
    Args:
        X: Feature DataFrame
        y: Target labels
        n_features: Number of features to select
    
    Returns:
        Tuple of (selected DataFrame, selected feature names, feature ranking)
    """
    n_features = min(n_features, X.shape[1])
    
    estimator = DecisionTreeClassifier(max_depth=5, random_state=42)
    selector = RFE(estimator, n_features_to_select=n_features, step=1)
    selector.fit(X, y)
    
    selected_features = X.columns[selector.get_support()].tolist()
    feature_ranking = selector.ranking_
    
    print(f"RFE: Selected {len(selected_features)} features")
    
    return X[selected_features], selected_features, feature_ranking


def feature_selection_pipeline(X: pd.DataFrame, y: pd.Series,
                               variance_threshold: float = 0.01,
                               correlation_threshold: float = 0.95,
                               k_best: int = 40,
                               n_features_rfe: int = 30) -> Tuple[pd.DataFrame, dict]:
    """
    Complete feature selection pipeline
    
    Args:
        X: Feature DataFrame
        y: Target labels
        variance_threshold: Threshold for low variance removal
        correlation_threshold: Threshold for correlation removal
        k_best: Number of features for SelectKBest
        n_features_rfe: Number of features for RFE
    
    Returns:
        Tuple of (selected features DataFrame, selection info dictionary)
    """
    print("="*80)
    print("FEATURE SELECTION PIPELINE")
    print("="*80)
    print(f"Initial number of features: {X.shape[1]}")
    
    selection_info = {
        'initial_features': X.shape[1],
        'initial_feature_names': X.columns.tolist()
    }
    
    # Step 1: Remove low variance features
    print("\n[Step 1] Removing low variance features...")
    X, removed_variance = remove_low_variance_features(X, threshold=variance_threshold)
    selection_info['removed_variance'] = removed_variance
    selection_info['after_variance'] = X.shape[1]
    
    # Step 2: Remove highly correlated features
    print("\n[Step 2] Removing highly correlated features...")
    X, removed_corr = remove_correlated_features(X, threshold=correlation_threshold)
    selection_info['removed_correlation'] = removed_corr
    selection_info['after_correlation'] = X.shape[1]
    
    # Step 3: SelectKBest with mutual information
    print("\n[Step 3] Selecting K best features...")
    X, selected_kbest, mi_scores = select_k_best_features(X, y, k=k_best)
    selection_info['selected_kbest'] = selected_kbest
    selection_info['mi_scores'] = mi_scores
    selection_info['after_kbest'] = X.shape[1]
    
    # Step 4: RFE for final selection
    print("\n[Step 4] Recursive Feature Elimination...")
    X, selected_rfe, rfe_ranking = select_features_rfe(X, y, n_features=n_features_rfe)
    selection_info['selected_rfe'] = selected_rfe
    selection_info['rfe_ranking'] = rfe_ranking
    selection_info['final_features'] = X.shape[1]
    
    print("\n" + "="*80)
    print(f"Feature selection complete!")
    print(f"Final number of features: {X.shape[1]}")
    print(f"Selected features: {X.columns.tolist()}")
    print("="*80)
    
    return X, selection_info


if __name__ == '__main__':
    # Test with dummy data
    np.random.seed(42)
    
    # Create dummy feature DataFrame
    n_samples = 50
    n_features = 60
    
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    y = pd.Series(np.random.randint(0, 2, n_samples))
    
    # Add some low variance features
    X['low_var_1'] = 0.001
    X['low_var_2'] = np.random.randn(n_samples) * 0.001
    
    # Add some correlated features
    X['corr_1'] = X['feature_0'] + np.random.randn(n_samples) * 0.01
    X['corr_2'] = X['feature_0'] + np.random.randn(n_samples) * 0.01
    
    print("Original shape:", X.shape)
    
    # Apply feature selection
    X_selected, info = feature_selection_pipeline(X, y, n_features_rfe=20)
    
    print("\nFinal shape:", X_selected.shape)
    print("\nSelected features:")
    print(X_selected.columns.tolist())
