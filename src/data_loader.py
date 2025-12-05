"""
Data loading utilities
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')

from .config import CONDITION_PATH, CONTROL_PATH, SCORES_PATH


def load_time_series_file(filepath: Path, label: int) -> pd.DataFrame:
    """
    Load a single time series CSV file
    
    Args:
        filepath: Path to CSV file
        label: Class label (0 for control, 1 for condition)
    
    Returns:
        DataFrame with timestamp, date, activity, and label
    """
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['label'] = label
    df['subject_id'] = filepath.stem  # e.g., 'condition_1' or 'control_1'
    return df


def load_all_time_series() -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
    """
    Load all time series data from condition and control folders
    
    Returns:
        Tuple of (condition_dataframes, control_dataframes)
    """
    condition_files = sorted(CONDITION_PATH.glob('condition_*.csv'))
    control_files = sorted(CONTROL_PATH.glob('control_*.csv'))
    
    print(f"Loading {len(condition_files)} condition files...")
    condition_dfs = []
    for filepath in condition_files:
        df = load_time_series_file(filepath, label=1)
        condition_dfs.append(df)
    
    print(f"Loading {len(control_files)} control files...")
    control_dfs = []
    for filepath in control_files:
        df = load_time_series_file(filepath, label=0)
        control_dfs.append(df)
    
    print(f"Total loaded: {len(condition_dfs)} condition + {len(control_dfs)} control = {len(condition_dfs) + len(control_dfs)} subjects")
    
    return condition_dfs, control_dfs


def load_scores() -> pd.DataFrame:
    """
    Load scores.csv containing demographics and MADRS scores
    
    Returns:
        DataFrame with scores data
    """
    df = pd.read_csv(SCORES_PATH)
    return df


def get_subject_info(subject_id: str, scores_df: pd.DataFrame) -> dict:
    """
    Get demographic and MADRS information for a subject
    
    Args:
        subject_id: Subject identifier (e.g., 'condition_1')
        scores_df: Scores DataFrame
    
    Returns:
        Dictionary with subject information
    """
    row = scores_df[scores_df['number'] == subject_id]
    if len(row) == 0:
        return {}
    
    return row.iloc[0].to_dict()


def create_subject_summary(time_series_dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Create a summary DataFrame with one row per subject
    
    Args:
        time_series_dfs: List of time series DataFrames
    
    Returns:
        DataFrame with subject_id and label
    """
    summary = []
    for df in time_series_dfs:
        subject_id = df['subject_id'].iloc[0]
        label = df['label'].iloc[0]
        n_records = len(df)
        duration_days = (df['timestamp'].max() - df['timestamp'].min()).days + 1
        
        summary.append({
            'subject_id': subject_id,
            'label': label,
            'n_records': n_records,
            'duration_days': duration_days
        })
    
    return pd.DataFrame(summary)


if __name__ == '__main__':
    # Test loading
    condition_dfs, control_dfs = load_all_time_series()
    scores_df = load_scores()
    
    print("\nScores shape:", scores_df.shape)
    print("\nSample condition data:")
    print(condition_dfs[0].head())
    
    print("\nSample control data:")
    print(control_dfs[0].head())
    
    # Create summary
    all_dfs = condition_dfs + control_dfs
    summary = create_subject_summary(all_dfs)
    print("\nSubject summary:")
    print(summary.head(10))
    print(f"\nLabel distribution:\n{summary['label'].value_counts()}")
