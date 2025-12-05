"""
Main pipeline for Depression Classification Research
Runs complete workflow from data loading to model evaluation
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.config import (
    RESULTS_PATH,
    MODELS_PATH,
    RANDOM_STATE,
    TEST_SIZE
)
from src.data_loader import load_all_time_series, load_scores
from src.preprocessing import preprocess_all_subjects
from src.feature_extraction import extract_features_from_all_subjects
from src.feature_selection import feature_selection_pipeline
from src.models import train_all_models, save_model
from src.evaluation import (
    evaluate_all_models,
    compare_models_statistical,
    get_best_model,
    print_confusion_matrix,
    generate_classification_report
)

from sklearn.model_selection import train_test_split
import joblib


def main():
    """
    Main execution pipeline
    """
    print("\n" + "="*80)
    print("DEPRESSION CLASSIFICATION RESEARCH PIPELINE")
    print("Gaussian Naive Bayes & Decision Tree for Imbalanced Data")
    print("="*80)
    
    # ========================================================================
    # PHASE 1: DATA LOADING
    # ========================================================================
    print("\n[PHASE 1] LOADING DATA")
    print("-"*80)
    
    condition_dfs, control_dfs = load_all_time_series()
    scores_df = load_scores()
    
    all_dfs = condition_dfs + control_dfs
    print(f"Total subjects loaded: {len(all_dfs)}")
    
    # ========================================================================
    # PHASE 2: PREPROCESSING
    # ========================================================================
    print("\n[PHASE 2] PREPROCESSING TIME SERIES DATA")
    print("-"*80)
    
    preprocessed_dfs = preprocess_all_subjects(all_dfs)
    print(f"Preprocessing complete for {len(preprocessed_dfs)} subjects")
    
    # ========================================================================
    # PHASE 3: FEATURE EXTRACTION
    # ========================================================================
    print("\n[PHASE 3] FEATURE EXTRACTION")
    print("-"*80)
    
    features_df = extract_features_from_all_subjects(preprocessed_dfs)
    
    # Save raw features
    features_path = RESULTS_PATH / 'features_raw.csv'
    features_df.to_csv(features_path, index=False)
    print(f"\nRaw features saved to: {features_path}")
    
    # ========================================================================
    # PHASE 4: FEATURE SELECTION
    # ========================================================================
    print("\n[PHASE 4] FEATURE SELECTION")
    print("-"*80)
    
    # Separate features and labels
    X = features_df.drop(columns=['subject_id', 'label'])
    y = features_df['label']
    
    print(f"Initial features: {X.shape[1]}")
    print(f"Samples: {X.shape[0]} (Class 0: {(y==0).sum()}, Class 1: {(y==1).sum()})")
    
    # Feature selection
    X_selected, selection_info = feature_selection_pipeline(
        X, y,
        variance_threshold=0.01,
        correlation_threshold=0.95,
        k_best=40,
        n_features_rfe=30
    )
    
    # Save selected features
    selected_features_df = pd.concat([
        features_df[['subject_id', 'label']],
        X_selected
    ], axis=1)
    
    selected_features_path = RESULTS_PATH / 'features_selected.csv'
    selected_features_df.to_csv(selected_features_path, index=False)
    print(f"\nSelected features saved to: {selected_features_path}")
    
    # Save selection info
    selection_info_path = RESULTS_PATH / 'feature_selection_info.pkl'
    joblib.dump(selection_info, selection_info_path)
    print(f"Feature selection info saved to: {selection_info_path}")
    
    # ========================================================================
    # PHASE 5: TRAIN-TEST SPLIT
    # ========================================================================
    print("\n[PHASE 5] TRAIN-TEST SPLIT")
    print("-"*80)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"  Class 0 (Control): {(y_train==0).sum()}")
    print(f"  Class 1 (Condition): {(y_train==1).sum()}")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"  Class 0 (Control): {(y_test==0).sum()}")
    print(f"  Class 1 (Condition): {(y_test==1).sum()}")
    
    # ========================================================================
    # PHASE 6: MODEL TRAINING
    # ========================================================================
    print("\n[PHASE 6] MODEL TRAINING (10 EXPERIMENTS)")
    print("-"*80)
    print("Training 2 models × 5 imbalance strategies = 10 experiments")
    print("Models: Gaussian NB, Decision Tree")
    print("Strategies: original, SMOTE, ADASYN, class_weight, SMOTE+weight")
    
    models_dict = train_all_models(X_train, y_train)
    
    # Save all models
    for experiment_id, model_data in models_dict.items():
        if model_data['status'] == 'success':
            model_path = MODELS_PATH / f'{experiment_id}.pkl'
            save_model(model_data['model'], model_path)
    
    print(f"\nAll models saved to: {MODELS_PATH}")
    
    # ========================================================================
    # PHASE 7: MODEL EVALUATION
    # ========================================================================
    print("\n[PHASE 7] MODEL EVALUATION ON TEST SET")
    print("-"*80)
    
    results_df = evaluate_all_models(models_dict, X_test, y_test)
    
    # Save results
    results_path = RESULTS_PATH / 'evaluation_results.csv'
    results_df.to_csv(results_path, index=False)
    print(f"\nEvaluation results saved to: {results_path}")
    
    # ========================================================================
    # PHASE 8: MODEL COMPARISON & BEST MODEL SELECTION
    # ========================================================================
    print("\n[PHASE 8] MODEL COMPARISON & ANALYSIS")
    print("-"*80)
    
    # Compare models by F1-macro
    sorted_results = compare_models_statistical(results_df, metric='f1_macro')
    
    # Get best model
    best_experiment_id, best_model, best_metrics = get_best_model(
        results_df, models_dict, metric='f1_macro'
    )
    
    # Detailed evaluation of best model
    print("\n" + "="*80)
    print("DETAILED EVALUATION OF BEST MODEL")
    print("="*80)
    
    y_pred = best_model.predict(X_test)
    print_confusion_matrix(y_test, y_pred)
    report = generate_classification_report(y_test, y_pred)
    
    # Save best model separately
    best_model_path = MODELS_PATH / 'best_model.pkl'
    save_model(best_model, best_model_path)
    
    # Save best model info
    best_model_info = {
        'experiment_id': best_experiment_id,
        'metrics': best_metrics,
        'selected_features': X_selected.columns.tolist(),
        'classification_report': report
    }
    
    best_model_info_path = RESULTS_PATH / 'best_model_info.pkl'
    joblib.dump(best_model_info, best_model_info_path)
    print(f"\nBest model info saved to: {best_model_info_path}")
    
    # ========================================================================
    # PHASE 9: SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("PIPELINE EXECUTION COMPLETE!")
    print("="*80)
    
    print("\n📊 SUMMARY:")
    print(f"  Total subjects: {len(features_df)}")
    print(f"  Features extracted: {len(X.columns)}")
    print(f"  Features selected: {len(X_selected.columns)}")
    print(f"  Models trained: {len([v for v in models_dict.values() if v['status'] == 'success'])}")
    print(f"  Best model: {best_experiment_id}")
    print(f"  Best F1-macro: {best_metrics['f1_macro']:.4f}")
    print(f"  Best Accuracy: {best_metrics['accuracy']:.4f}")
    print(f"  Best AUC-ROC: {best_metrics['roc_auc']:.4f}")
    
    print("\n📁 OUTPUT FILES:")
    print(f"  Features (raw): {features_path}")
    print(f"  Features (selected): {selected_features_path}")
    print(f"  Evaluation results: {results_path}")
    print(f"  All models: {MODELS_PATH}")
    print(f"  Best model: {best_model_path}")
    print(f"  Best model info: {best_model_info_path}")
    
    print("\n✅ Research pipeline completed successfully!")
    print("="*80 + "\n")
    
    return {
        'features_df': features_df,
        'X_selected': X_selected,
        'y': y,
        'models_dict': models_dict,
        'results_df': results_df,
        'best_model': best_model,
        'best_metrics': best_metrics
    }


if __name__ == '__main__':
    results = main()
