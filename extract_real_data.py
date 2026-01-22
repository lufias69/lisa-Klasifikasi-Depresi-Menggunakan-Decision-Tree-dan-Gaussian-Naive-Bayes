"""
Extract real data from saved models and results for BAB 4 validation
"""

import pickle
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Load evaluation results
print("="*80)
print("EXTRACTING REAL DATA FROM EXPERIMENTS")
print("="*80)

# 1. Load evaluation results
eval_df = pd.read_csv('experiments/results/evaluation_results.csv')
print("\n1. EVALUATION RESULTS:")
print(eval_df[['experiment_id', 'model_name', 'imbalance_strategy', 
              'accuracy', 'f1_macro', 'roc_auc', 'cv_f1_macro']].to_string(index=False))

# 2. Load best model info
with open('experiments/results/best_model_info.pkl', 'rb') as f:
    best_model_info = pickle.load(f)

print("\n2. BEST MODEL INFO:")
print(f"   Experiment ID: {best_model_info['metrics']['experiment_id']}")
print(f"   Model Name: {best_model_info['metrics']['model_name']}")
print(f"   Strategy: {best_model_info['metrics']['imbalance_strategy']}")
print(f"   Test Accuracy: {best_model_info['metrics']['accuracy']}")
print(f"   Test F1-Macro: {best_model_info['metrics']['f1_macro']}")
print(f"   CV F1-Macro: {best_model_info['metrics']['cv_f1_macro']}")

# 3. Load actual best model to get feature importances
print("\n3. LOADING BEST MODEL FOR FEATURE IMPORTANCE...")
best_model_path = f"experiments/models/{best_model_info['metrics']['experiment_id']}.pkl"
best_model = joblib.load(best_model_path)

# Get the actual model from pipeline
if hasattr(best_model, 'named_steps'):
    actual_model = best_model.named_steps['model']
else:
    actual_model = best_model

if hasattr(actual_model, 'feature_importances_'):
    importances = actual_model.feature_importances_
    feature_names = best_model_info['selected_features']
    
    # Create feature importance dataframe
    fi_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False).reset_index(drop=True)
    
    print("\n4. FEATURE IMPORTANCES (TOP 30):")
    fi_df['rank'] = range(1, len(fi_df) + 1)
    print(fi_df[['rank', 'feature', 'importance']].to_string(index=False))
    
    # Save to CSV
    fi_df.to_csv('experiments/results/feature_importance_real.csv', index=False)
    print("\n✅ Feature importances saved to: experiments/results/feature_importance_real.csv")
else:
    print("\n⚠️ Model doesn't have feature_importances_ attribute")

# 4. Selected features list
print("\n5. SELECTED FEATURES (30 total):")
for i, feat in enumerate(best_model_info['selected_features'], 1):
    print(f"   {i:2d}. {feat}")

# 5. Classification report
print("\n6. CLASSIFICATION REPORT:")
print(best_model_info['classification_report'])

# 6. Confusion matrix from metrics
print("\n7. CONFUSION MATRIX:")
metrics = best_model_info['metrics']
print("                Predicted")
print("                Control  Condition")
print("Actual")
print(f"  Control         {metrics['true_negative']:5.0f}      {metrics['false_positive']:5.0f}")
print(f"  Condition       {metrics['false_negative']:5.0f}      {metrics['true_positive']:5.0f}")

print("\n" + "="*80)
print("DATA EXTRACTION COMPLETE!")
print("="*80)
