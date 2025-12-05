"""
Data preprocessing utilities
"""

import pandas as pd
import numpy as np
from typing import List, Tuple
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


def detect_outliers_zscore(df: pd.DataFrame, column: str = 'activity', threshold: float = 3.0) -> pd.Series:
    """
    Detect outliers using Z-score method
    
    Args:
        df: DataFrame with activity data
        column: Column name to check for outliers
        threshold: Z-score threshold (default: 3.0)
    
    Returns:
        Boolean series indicating outliers
    """
    z_scores = np.abs(stats.zscore(df[column], nan_policy='omit'))
    return z_scores > threshold


def detect_outliers_iqr(df: pd.DataFrame, column: str = 'activity', multiplier: float = 1.5) -> pd.Series:
    """
    Detect outliers using IQR method
    
    Args:
        df: DataFrame with activity data
        column: Column name to check for outliers
        multiplier: IQR multiplier (default: 1.5)
    
    Returns:
        Boolean series indicating outliers
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    return (df[column] < lower_bound) | (df[column] > upper_bound)


def handle_outliers(df: pd.DataFrame, method: str = 'cap', outlier_method: str = 'iqr') -> pd.DataFrame:
    """
    Handle outliers in activity data
    
    Args:
        df: DataFrame with activity data
        method: How to handle outliers ('cap', 'remove', 'keep')
        outlier_method: Method to detect outliers ('zscore', 'iqr')
    
    Returns:
        DataFrame with outliers handled
    """
    df = df.copy()
    
    if outlier_method == 'zscore':
        outliers = detect_outliers_zscore(df)
    elif outlier_method == 'iqr':
        outliers = detect_outliers_iqr(df)
    else:
        raise ValueError(f"Unknown outlier method: {outlier_method}")
    
    n_outliers = outliers.sum()
    print(f"  Detected {n_outliers} outliers ({n_outliers/len(df)*100:.2f}%)")
    
    if method == 'cap':
        # Cap outliers to 95th percentile
        upper_cap = df['activity'].quantile(0.95)
        df.loc[outliers, 'activity'] = upper_cap
        print(f"  Capped outliers to {upper_cap:.2f}")
    elif method == 'remove':
        df = df[~outliers]
        print(f"  Removed {n_outliers} outlier records")
    elif method == 'keep':
        print(f"  Keeping outliers")
    
    return df


def handle_missing_values(df: pd.DataFrame, strategy: str = 'forward_fill') -> pd.DataFrame:
    """
    Handle missing values in time series data
    
    Args:
        df: DataFrame with potential missing values
        strategy: Strategy to handle missing values ('forward_fill', 'interpolate', 'drop')
    
    Returns:
        DataFrame with missing values handled
    """
    df = df.copy()
    
    missing_count = df['activity'].isnull().sum()
    if missing_count > 0:
        print(f"  Found {missing_count} missing values ({missing_count/len(df)*100:.2f}%)")
        
        if strategy == 'forward_fill':
            df['activity'] = df['activity'].fillna(method='ffill')
            df['activity'] = df['activity'].fillna(method='bfill')  # Handle leading NaNs
        elif strategy == 'interpolate':
            df['activity'] = df['activity'].interpolate(method='linear')
        elif strategy == 'drop':
            df = df.dropna(subset=['activity'])
        
        print(f"  Applied {strategy} strategy")
    
    return df


def preprocess_time_series(df: pd.DataFrame, 
                           handle_missing: str = 'forward_fill',
                           handle_outlier_method: str = 'cap') -> pd.DataFrame:
    """
    Complete preprocessing pipeline for a single time series
    
    Args:
        df: DataFrame with time series data
        handle_missing: Missing value strategy
        handle_outlier_method: Outlier handling method
    
    Returns:
        Preprocessed DataFrame
    """
    subject_id = df['subject_id'].iloc[0]
    print(f"\nPreprocessing {subject_id}...")
    
    # Sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Handle missing values
    df = handle_missing_values(df, strategy=handle_missing)
    
    # Handle outliers
    df = handle_outliers(df, method=handle_outlier_method, outlier_method='iqr')
    
    # Ensure non-negative activity (actigraphy should be >= 0)
    if (df['activity'] < 0).any():
        print(f"  Found {(df['activity'] < 0).sum()} negative values, setting to 0")
        df.loc[df['activity'] < 0, 'activity'] = 0
    
    return df


def preprocess_all_subjects(time_series_list: List[pd.DataFrame]) -> List[pd.DataFrame]:
    """
    Preprocess all subjects' time series data
    
    Args:
        time_series_list: List of time series DataFrames
    
    Returns:
        List of preprocessed DataFrames
    """
    print("="*80)
    print("PREPROCESSING TIME SERIES DATA")
    print("="*80)
    
    preprocessed = []
    for df in time_series_list:
        preprocessed_df = preprocess_time_series(df)
        preprocessed.append(preprocessed_df)
    
    print("\n" + "="*80)
    print(f"Preprocessing complete for {len(preprocessed)} subjects")
    print("="*80)
    
    return preprocessed


if __name__ == '__main__':
    from data_loader import load_all_time_series
    
    # Load data
    condition_dfs, control_dfs = load_all_time_series()
    
    # Preprocess a sample
    sample_df = condition_dfs[0]
    print("\nOriginal data shape:", sample_df.shape)
    print("Activity stats before:")
    print(sample_df['activity'].describe())
    
    preprocessed = preprocess_time_series(sample_df)
    print("\nPreprocessed data shape:", preprocessed.shape)
    print("Activity stats after:")
    print(preprocessed['activity'].describe())
