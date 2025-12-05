"""
Configuration file for the research project
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / 'data'
CONDITION_PATH = DATA_PATH / 'condition'
CONTROL_PATH = DATA_PATH / 'control'
SCORES_PATH = DATA_PATH / 'scores.csv'

EXPERIMENTS_PATH = PROJECT_ROOT / 'experiments'
RESULTS_PATH = EXPERIMENTS_PATH / 'results'
MODELS_PATH = EXPERIMENTS_PATH / 'models'
FIGURES_PATH = EXPERIMENTS_PATH / 'figures'

# Create directories if they don't exist
for path in [RESULTS_PATH, MODELS_PATH, FIGURES_PATH]:
    path.mkdir(parents=True, exist_ok=True)

# Random seed for reproducibility
RANDOM_STATE = 42

# Data split configuration
TEST_SIZE = 0.2
CV_FOLDS = 5

# Feature engineering parameters
SLEEP_THRESHOLD = 0  # Activity value considered as sleep
MIN_SLEEP_DURATION = 240  # Minimum sleep duration in minutes (4 hours)

# Circadian rhythm parameters
CIRCADIAN_PERIOD = 24  # hours

# Model hyperparameters
GAUSSIAN_NB_PARAMS = {
    'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
}

DECISION_TREE_PARAMS = {
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 5, 10],
    'criterion': ['gini', 'entropy']
}

# Imbalanced data handling scenarios
IMBALANCE_STRATEGIES = ['original', 'smote', 'adasyn', 'class_weight', 'smote_weight']

# Evaluation metrics
METRICS = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
