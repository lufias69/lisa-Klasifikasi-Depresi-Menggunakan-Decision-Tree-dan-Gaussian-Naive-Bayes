"""
Visualization utilities for research analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.tree import plot_tree
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def plot_confusion_matrix(y_true, y_pred, labels=['Control', 'Condition'], 
                         save_path=None, title='Confusion Matrix'):
    """
    Plot confusion matrix heatmap
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Class labels
        save_path: Path to save figure
        title: Plot title
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Count'})
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()


def plot_roc_curves(models_dict, results_df, X_test, y_test, save_path=None):
    """
    Plot ROC curves for multiple models
    
    Args:
        models_dict: Dictionary of trained models
        results_df: Results DataFrame
        X_test: Test features
        y_test: Test labels
        save_path: Path to save figure
    """
    plt.figure(figsize=(10, 8))
    
    for _, row in results_df.iterrows():
        experiment_id = row['experiment_id']
        model_data = models_dict.get(experiment_id)
        
        if model_data and model_data['status'] == 'success':
            model = model_data['model']
            
            # Get probabilities
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                roc_auc = auc(fpr, tpr)
                
                plt.plot(fpr, tpr, lw=2, 
                        label=f'{experiment_id} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random (AUC = 0.500)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curves saved to {save_path}")
    
    plt.show()


def plot_model_comparison(results_df, metric='f1_macro', save_path=None):
    """
    Plot bar chart comparing model performances
    
    Args:
        results_df: Results DataFrame
        metric: Metric to compare
        save_path: Path to save figure
    """
    # Sort by metric
    sorted_df = results_df.sort_values(metric, ascending=False)
    
    # Create color map
    colors = []
    for exp_id in sorted_df['experiment_id']:
        if 'gaussian' in exp_id:
            colors.append('#4ECDC4')
        elif 'decision_tree' in exp_id:
            colors.append('#95E1D3')
    
    plt.figure(figsize=(14, 8))
    bars = plt.barh(range(len(sorted_df)), sorted_df[metric], color=colors)
    
    plt.yticks(range(len(sorted_df)), sorted_df['experiment_id'], fontsize=10)
    plt.xlabel(f'{metric.upper()}', fontsize=12, fontweight='bold')
    plt.title(f'Model Performance Comparison - {metric.upper()}', 
              fontsize=14, fontweight='bold')
    plt.xlim([0, 1.05])
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, sorted_df[metric])):
        plt.text(value + 0.01, i, f'{value:.4f}', 
                va='center', fontsize=9)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#4ECDC4', label='Gaussian NB'),
        Patch(facecolor='#95E1D3', label='Decision Tree')
    ]
    plt.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Model comparison saved to {save_path}")
    
    plt.show()


def plot_feature_importance(model, feature_names, top_n=20, save_path=None):
    """
    Plot feature importance (for tree-based models)
    
    Args:
        model: Trained model
        feature_names: List of feature names
        top_n: Number of top features to show
        save_path: Path to save figure
    """
    # Get the actual model from pipeline
    if hasattr(model, 'named_steps'):
        actual_model = model.named_steps['model']
    else:
        actual_model = model
    
    if hasattr(actual_model, 'feature_importances_'):
        importances = actual_model.feature_importances_
        
        # Create DataFrame
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).head(top_n)
        
        plt.figure(figsize=(10, 8))
        bars = plt.barh(range(len(feature_importance_df)), 
                       feature_importance_df['importance'],
                       color='skyblue')
        
        plt.yticks(range(len(feature_importance_df)), 
                  feature_importance_df['feature'], fontsize=10)
        plt.xlabel('Feature Importance', fontsize=12, fontweight='bold')
        plt.title(f'Top {top_n} Most Important Features', 
                 fontsize=14, fontweight='bold')
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, feature_importance_df['importance'])):
            plt.text(value + 0.001, i, f'{value:.4f}', 
                    va='center', fontsize=9)
        
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature importance saved to {save_path}")
        
        plt.show()
        
        return feature_importance_df
    else:
        print("Model does not have feature_importances_ attribute")
        return None


def plot_decision_tree(model, feature_names, class_names=['Control', 'Condition'], 
                       save_path=None, max_depth=3):
    """
    Visualize decision tree structure
    
    Args:
        model: Trained decision tree model
        feature_names: List of feature names
        class_names: Class labels
        save_path: Path to save figure
        max_depth: Maximum depth to display
    """
    # Get the actual model from pipeline
    if hasattr(model, 'named_steps'):
        actual_model = model.named_steps['model']
    else:
        actual_model = model
    
    if hasattr(actual_model, 'tree_'):
        plt.figure(figsize=(20, 10))
        plot_tree(actual_model, 
                 feature_names=feature_names,
                 class_names=class_names,
                 filled=True,
                 rounded=True,
                 fontsize=10,
                 max_depth=max_depth)
        plt.title('Decision Tree Visualization', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Decision tree saved to {save_path}")
        
        plt.show()
    else:
        print("Model is not a decision tree")


def plot_activity_patterns_comparison(features_df, save_path=None):
    """
    Compare hourly activity patterns between condition and control
    
    Args:
        features_df: DataFrame with features
        save_path: Path to save figure
    """
    # Extract hourly features - only columns like activity_hour_XX where XX is a number
    hourly_cols = [col for col in features_df.columns if col.startswith('activity_hour_') and col.split('_')[-1].isdigit()]
    
    if len(hourly_cols) > 0:
        condition_data = features_df[features_df['label'] == 1][hourly_cols].mean()
        control_data = features_df[features_df['label'] == 0][hourly_cols].mean()
        
        hours = [int(col.split('_')[-1]) for col in hourly_cols]
        
        plt.figure(figsize=(14, 6))
        plt.plot(hours, condition_data.values, 'r-o', linewidth=2, 
                label='Condition (Depression)', markersize=6)
        plt.plot(hours, control_data.values, 'b-s', linewidth=2, 
                label='Control (Healthy)', markersize=6)
        
        plt.xlabel('Hour of Day', fontsize=12, fontweight='bold')
        plt.ylabel('Average Activity Level', fontsize=12, fontweight='bold')
        plt.title('24-Hour Activity Patterns: Condition vs Control', 
                 fontsize=14, fontweight='bold')
        plt.xticks(range(0, 24, 2))
        plt.legend(fontsize=11, loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Activity patterns saved to {save_path}")
        
        plt.show()
    else:
        print("No hourly activity features found")


def plot_metrics_heatmap(results_df, save_path=None):
    """
    Plot heatmap of metrics across all models
    
    Args:
        results_df: Results DataFrame
        save_path: Path to save figure
    """
    metrics_cols = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'roc_auc']
    
    # Create pivot table
    heatmap_data = results_df[['experiment_id'] + metrics_cols].set_index('experiment_id')
    
    plt.figure(figsize=(10, 12))
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlGnBu',
                cbar_kws={'label': 'Score'}, linewidths=0.5)
    plt.title('Performance Metrics Heatmap - All Models', 
             fontsize=14, fontweight='bold')
    plt.xlabel('Metrics', fontsize=12, fontweight='bold')
    plt.ylabel('Models', fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Metrics heatmap saved to {save_path}")
    
    plt.show()


def generate_all_visualizations(models_dict, results_df, features_df, 
                               X_test, y_test, best_model, best_experiment_id,
                               figures_path):
    """
    Generate all visualizations for the research
    
    Args:
        models_dict: Dictionary of trained models
        results_df: Results DataFrame
        features_df: Features DataFrame
        X_test: Test features
        y_test: Test labels
        best_model: Best trained model
        best_experiment_id: ID of best model
        figures_path: Path to save figures
    """
    figures_path = Path(figures_path)
    figures_path.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    # 1. Confusion Matrix for best model
    print("\n[1/7] Confusion matrix...")
    y_pred = best_model.predict(X_test)
    plot_confusion_matrix(y_test, y_pred, 
                         save_path=figures_path / 'confusion_matrix_best.png',
                         title=f'Confusion Matrix - {best_experiment_id}')
    
    # 2. ROC Curves
    print("\n[2/7] ROC curves...")
    plot_roc_curves(models_dict, results_df, X_test, y_test,
                   save_path=figures_path / 'roc_curves.png')
    
    # 3. Model Comparison
    print("\n[3/7] Model comparison...")
    plot_model_comparison(results_df, metric='f1_macro',
                         save_path=figures_path / 'model_comparison_f1.png')
    plot_model_comparison(results_df, metric='accuracy',
                         save_path=figures_path / 'model_comparison_accuracy.png')
    
    # 4. Feature Importance (if tree-based)
    print("\n[4/7] Feature importance...")
    feature_names = X_test.columns.tolist()
    plot_feature_importance(best_model, feature_names, top_n=20,
                          save_path=figures_path / 'feature_importance.png')
    
    # 5. Decision Tree Visualization
    print("\n[5/7] Decision tree...")
    if 'decision_tree' in best_experiment_id:
        plot_decision_tree(best_model, feature_names,
                         save_path=figures_path / 'decision_tree_viz.png',
                         max_depth=3)
    
    # 6. Activity Patterns
    print("\n[6/7] Activity patterns...")
    plot_activity_patterns_comparison(features_df,
                                    save_path=figures_path / 'activity_patterns_24h.png')
    
    # 7. Metrics Heatmap
    print("\n[7/7] Metrics heatmap...")
    plot_metrics_heatmap(results_df,
                        save_path=figures_path / 'metrics_heatmap.png')
    
    print("\n" + "="*80)
    print(f"All visualizations saved to: {figures_path}")
    print("="*80)


if __name__ == '__main__':
    # Test with dummy data
    print("Testing visualization functions...")
    
    # Dummy data
    y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0])
    y_pred = np.array([0, 1, 1, 1, 0, 0, 1, 0])
    
    plot_confusion_matrix(y_true, y_pred)
