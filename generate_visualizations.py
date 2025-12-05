"""
Generate all visualizations for the research
Run this after main_pipeline.py completes
"""

import pandas as pd
import sys
from pathlib import Path
import joblib

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.config import RESULTS_PATH, MODELS_PATH, FIGURES_PATH
from src.visualization import generate_all_visualizations

def main():
    """
    Load results and generate visualizations
    """
    print("="*80)
    print("GENERATING RESEARCH VISUALIZATIONS")
    print("="*80)
    
    # Load data
    print("\n[1] Loading data...")
    features_df = pd.read_csv(RESULTS_PATH / 'features_selected.csv')
    results_df = pd.read_csv(RESULTS_PATH / 'evaluation_results.csv')
    best_model_info = joblib.load(RESULTS_PATH / 'best_model_info.pkl')
    best_model = joblib.load(MODELS_PATH / 'best_model.pkl')
    
    # Load all models
    models_dict = {}
    for model_path in MODELS_PATH.glob('*.pkl'):
        if model_path.stem != 'best_model':
            experiment_id = model_path.stem
            try:
                model = joblib.load(model_path)
                models_dict[experiment_id] = {
                    'model': model,
                    'status': 'success'
                }
            except:
                pass
    
    print(f"Loaded {len(models_dict)} models")
    print(f"Best model: {best_model_info['experiment_id']}")
    
    # Prepare data
    X = features_df.drop(columns=['subject_id', 'label'])
    y = features_df['label']
    
    # Generate visualizations
    print("\n[2] Generating visualizations...")
    generate_all_visualizations(
        models_dict=models_dict,
        results_df=results_df,
        features_df=features_df,
        X_test=X,  # Using all data for visualization
        y_test=y,
        best_model=best_model,
        best_experiment_id=best_model_info['experiment_id'],
        figures_path=FIGURES_PATH
    )
    
    print("\n✅ Visualization generation complete!")
    print(f"All figures saved to: {FIGURES_PATH}")

if __name__ == '__main__':
    main()
