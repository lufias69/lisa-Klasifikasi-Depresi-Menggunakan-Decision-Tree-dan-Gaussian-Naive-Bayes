"""
Feature Extraction from Time Series Activity Data
Includes: Statistical, Temporal, Circadian Rhythm, and Derived Features
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from scipy import stats, signal
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

from .config import SLEEP_THRESHOLD, MIN_SLEEP_DURATION, CIRCADIAN_PERIOD


# ============================================================================
# STATISTICAL FEATURES
# ============================================================================

def extract_statistical_features(df: pd.DataFrame) -> Dict[str, float]:
    """
    Extract basic statistical features from activity data
    
    Returns:
        Dictionary of statistical features
    """
    activity = df['activity'].values
    
    features = {
        # Basic statistics
        'activity_mean': np.mean(activity),
        'activity_median': np.median(activity),
        'activity_std': np.std(activity),
        'activity_min': np.min(activity),
        'activity_max': np.max(activity),
        'activity_range': np.max(activity) - np.min(activity),
        
        # Distribution shape
        'activity_skewness': stats.skew(activity),
        'activity_kurtosis': stats.kurtosis(activity),
        
        # Percentiles
        'activity_q25': np.percentile(activity, 25),
        'activity_q50': np.percentile(activity, 50),
        'activity_q75': np.percentile(activity, 75),
        'activity_q95': np.percentile(activity, 95),
        
        # Variability
        'activity_cv': np.std(activity) / (np.mean(activity) + 1e-10),  # Coefficient of Variation
        'activity_iqr': np.percentile(activity, 75) - np.percentile(activity, 25),
        
        # Zero activity
        'zero_activity_count': np.sum(activity == 0),
        'zero_activity_pct': np.sum(activity == 0) / len(activity) * 100,
    }
    
    return features


# ============================================================================
# TEMPORAL FEATURES (NOVELTY)
# ============================================================================

def extract_temporal_features(df: pd.DataFrame) -> Dict[str, float]:
    """
    Extract temporal features: hourly patterns, day/night ratio
    
    Returns:
        Dictionary of temporal features
    """
    df = df.copy()
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    
    features = {}
    
    # Hourly activity patterns (24 features)
    for hour in range(24):
        hour_data = df[df['hour'] == hour]['activity']
        if len(hour_data) > 0:
            features[f'activity_hour_{hour:02d}'] = hour_data.mean()
        else:
            features[f'activity_hour_{hour:02d}'] = 0.0
    
    # Day vs Night activity
    day_hours = (df['hour'] >= 6) & (df['hour'] < 22)
    night_hours = ~day_hours
    
    day_activity = df[day_hours]['activity'].mean() if day_hours.sum() > 0 else 0
    night_activity = df[night_hours]['activity'].mean() if night_hours.sum() > 0 else 0
    
    features['day_activity_mean'] = day_activity
    features['night_activity_mean'] = night_activity
    features['day_night_ratio'] = day_activity / (night_activity + 1e-10)
    
    # Peak activity time
    hourly_means = df.groupby('hour')['activity'].mean()
    features['peak_activity_hour'] = hourly_means.idxmax() if len(hourly_means) > 0 else 12
    features['peak_activity_value'] = hourly_means.max() if len(hourly_means) > 0 else 0
    
    # Activity regularity (autocorrelation)
    if len(df) > 24:
        # Lag-24 autocorrelation (daily pattern)
        activity_values = df['activity'].values
        if len(activity_values) > 24:
            lag_24_autocorr = np.corrcoef(activity_values[:-24], activity_values[24:])[0, 1]
            features['autocorr_lag24'] = lag_24_autocorr if not np.isnan(lag_24_autocorr) else 0
        else:
            features['autocorr_lag24'] = 0
    else:
        features['autocorr_lag24'] = 0
    
    # Weekday vs Weekend (if data spans multiple weeks)
    weekday_activity = df[df['day_of_week'] < 5]['activity'].mean()
    weekend_activity = df[df['day_of_week'] >= 5]['activity'].mean()
    features['weekday_activity_mean'] = weekday_activity if not np.isnan(weekday_activity) else 0
    features['weekend_activity_mean'] = weekend_activity if not np.isnan(weekend_activity) else 0
    
    return features


# ============================================================================
# CIRCADIAN RHYTHM FEATURES (MAIN NOVELTY)
# ============================================================================

def cosinor_model(t, amplitude, acrophase, mesor):
    """
    Cosinor model for circadian rhythm
    
    Args:
        t: Time in hours (0-24)
        amplitude: Amplitude of rhythm
        acrophase: Peak time (phase)
        mesor: Midline estimating statistic of rhythm (mean)
    
    Returns:
        Predicted activity value
    """
    omega = 2 * np.pi / 24  # 24-hour period
    return mesor + amplitude * np.cos(omega * (t - acrophase))


def extract_circadian_features(df: pd.DataFrame) -> Dict[str, float]:
    """
    Extract circadian rhythm features using Cosinor analysis
    
    Returns:
        Dictionary of circadian features
    """
    df = df.copy()
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    df['time_decimal'] = df['hour'] + df['minute'] / 60.0
    
    features = {}
    
    # Cosinor analysis
    try:
        # Aggregate to hourly means for cleaner signal
        hourly_activity = df.groupby(df['time_decimal'].astype(int))['activity'].mean().values
        hours = np.arange(len(hourly_activity))
        
        # Initial parameters
        mesor_init = np.mean(hourly_activity)
        amplitude_init = (np.max(hourly_activity) - np.min(hourly_activity)) / 2
        acrophase_init = hours[np.argmax(hourly_activity)]
        
        # Fit cosinor model
        params, _ = curve_fit(
            cosinor_model,
            hours,
            hourly_activity,
            p0=[amplitude_init, acrophase_init, mesor_init],
            bounds=([0, 0, 0], [np.inf, 24, np.inf]),
            maxfev=5000
        )
        
        amplitude, acrophase, mesor = params
        
        features['circadian_amplitude'] = amplitude
        features['circadian_acrophase'] = acrophase
        features['circadian_mesor'] = mesor
        features['circadian_rhythm_strength'] = amplitude / (mesor + 1e-10)
        
    except Exception as e:
        # If fitting fails, use fallback
        features['circadian_amplitude'] = 0
        features['circadian_acrophase'] = 12
        features['circadian_mesor'] = df['activity'].mean()
        features['circadian_rhythm_strength'] = 0
    
    # Interdaily Stability (IS) and Intradaily Variability (IV)
    try:
        # IS: Consistency of rhythm across days
        hourly_means = df.groupby('hour')['activity'].mean()
        overall_mean = df['activity'].mean()
        
        is_numerator = np.sum((hourly_means - overall_mean) ** 2) * len(df)
        is_denominator = np.sum((df['activity'] - overall_mean) ** 2) * 24
        is_value = is_numerator / (is_denominator + 1e-10)
        
        features['interdaily_stability'] = np.clip(is_value, 0, 1)
        
        # IV: Fragmentation of rhythm
        activity_diff = np.diff(df['activity'].values)
        iv_numerator = np.sum(activity_diff ** 2) * len(df)
        iv_denominator = np.sum((df['activity'] - overall_mean) ** 2) * (len(df) - 1)
        iv_value = iv_numerator / (iv_denominator + 1e-10)
        
        features['intradaily_variability'] = iv_value
        
    except Exception:
        features['interdaily_stability'] = 0
        features['intradaily_variability'] = 0
    
    return features


def extract_sleep_features(df: pd.DataFrame) -> Dict[str, float]:
    """
    Extract sleep-related features from activity data
    
    Returns:
        Dictionary of sleep features
    """
    df = df.copy()
    df['hour'] = df['timestamp'].dt.hour
    
    features = {}
    
    # Detect sleep periods (consecutive zeros or low activity)
    is_sleep = df['activity'] <= SLEEP_THRESHOLD
    
    # Find consecutive sleep periods
    sleep_changes = is_sleep.astype(int).diff()
    sleep_starts = df[sleep_changes == 1].index.tolist()
    sleep_ends = df[sleep_changes == -1].index.tolist()
    
    # Ensure pairing
    if len(sleep_ends) > 0 and len(sleep_starts) > 0:
        if sleep_ends[0] < sleep_starts[0]:
            sleep_ends = sleep_ends[1:]
        if len(sleep_starts) > len(sleep_ends):
            sleep_starts = sleep_starts[:len(sleep_ends)]
    
    sleep_durations = []
    sleep_onset_times = []
    sleep_wake_times = []
    
    for start_idx, end_idx in zip(sleep_starts, sleep_ends):
        duration = end_idx - start_idx  # Duration in minutes
        if duration >= MIN_SLEEP_DURATION:  # Minimum 4 hours
            sleep_durations.append(duration)
            sleep_onset_times.append(df.loc[start_idx, 'hour'])
            sleep_wake_times.append(df.loc[end_idx, 'hour'])
    
    if len(sleep_durations) > 0:
        features['avg_sleep_duration'] = np.mean(sleep_durations)
        features['total_sleep_time'] = np.sum(sleep_durations)
        features['sleep_efficiency'] = np.sum(sleep_durations) / len(df) * 100
        features['num_sleep_periods'] = len(sleep_durations)
        features['avg_sleep_onset_hour'] = np.mean(sleep_onset_times)
        features['avg_wake_time_hour'] = np.mean(sleep_wake_times)
    else:
        features['avg_sleep_duration'] = 0
        features['total_sleep_time'] = 0
        features['sleep_efficiency'] = 0
        features['num_sleep_periods'] = 0
        features['avg_sleep_onset_hour'] = 0
        features['avg_wake_time_hour'] = 0
    
    # Longest inactive period
    inactive_periods = []
    if len(sleep_starts) > 0 and len(sleep_ends) > 0:
        for start_idx, end_idx in zip(sleep_starts, sleep_ends):
            inactive_periods.append(end_idx - start_idx)
        features['longest_inactive_period'] = max(inactive_periods) if inactive_periods else 0
    else:
        features['longest_inactive_period'] = 0
    
    return features


# ============================================================================
# DERIVED FEATURES
# ============================================================================

def extract_derived_features(df: pd.DataFrame) -> Dict[str, float]:
    """
    Extract derived features: change rate, moving averages, entropy
    
    Returns:
        Dictionary of derived features
    """
    activity = df['activity'].values
    features = {}
    
    # Activity change rate
    if len(activity) > 1:
        activity_diff = np.diff(activity)
        features['activity_change_mean'] = np.mean(activity_diff)
        features['activity_change_std'] = np.std(activity_diff)
        features['activity_change_abs_mean'] = np.mean(np.abs(activity_diff))
    else:
        features['activity_change_mean'] = 0
        features['activity_change_std'] = 0
        features['activity_change_abs_mean'] = 0
    
    # Moving averages
    if len(activity) >= 60:  # At least 1 hour
        ma_1h = np.convolve(activity, np.ones(60)/60, mode='valid')
        features['moving_avg_1h_mean'] = np.mean(ma_1h)
        features['moving_avg_1h_std'] = np.std(ma_1h)
    else:
        features['moving_avg_1h_mean'] = np.mean(activity)
        features['moving_avg_1h_std'] = np.std(activity)
    
    if len(activity) >= 240:  # At least 4 hours
        ma_4h = np.convolve(activity, np.ones(240)/240, mode='valid')
        features['moving_avg_4h_mean'] = np.mean(ma_4h)
        features['moving_avg_4h_std'] = np.std(ma_4h)
    else:
        features['moving_avg_4h_mean'] = np.mean(activity)
        features['moving_avg_4h_std'] = np.std(activity)
    
    # Entropy (activity predictability)
    # Discretize activity into bins
    if np.max(activity) > 0:
        n_bins = 20
        hist, _ = np.histogram(activity, bins=n_bins, density=True)
        hist = hist[hist > 0]  # Remove zero probabilities
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        features['activity_entropy'] = entropy
    else:
        features['activity_entropy'] = 0
    
    # Activity transitions (zero to non-zero and vice versa)
    is_active = activity > 0
    transitions = np.sum(np.abs(np.diff(is_active.astype(int))))
    features['activity_transitions'] = transitions
    features['activity_transitions_per_hour'] = transitions / (len(activity) / 60)
    
    return features


# ============================================================================
# MAIN FEATURE EXTRACTION
# ============================================================================

def extract_features_from_subject(df: pd.DataFrame) -> Dict[str, float]:
    """
    Extract all features from a single subject's time series
    
    Args:
        df: DataFrame with time series data for one subject
    
    Returns:
        Dictionary with all extracted features
    """
    subject_id = df['subject_id'].iloc[0]
    label = df['label'].iloc[0]
    
    features = {
        'subject_id': subject_id,
        'label': label,
        'n_records': len(df),
        'duration_days': (df['timestamp'].max() - df['timestamp'].min()).days + 1
    }
    
    # Extract all feature types
    features.update(extract_statistical_features(df))
    features.update(extract_temporal_features(df))
    features.update(extract_circadian_features(df))
    features.update(extract_sleep_features(df))
    features.update(extract_derived_features(df))
    
    return features


def extract_features_from_all_subjects(time_series_list: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Extract features from all subjects
    
    Args:
        time_series_list: List of time series DataFrames
    
    Returns:
        DataFrame with features (one row per subject)
    """
    print("="*80)
    print("FEATURE EXTRACTION")
    print("="*80)
    
    all_features = []
    
    for i, df in enumerate(time_series_list):
        subject_id = df['subject_id'].iloc[0]
        print(f"[{i+1}/{len(time_series_list)}] Extracting features for {subject_id}...")
        
        features = extract_features_from_subject(df)
        all_features.append(features)
    
    features_df = pd.DataFrame(all_features)
    
    print("\n" + "="*80)
    print(f"Feature extraction complete!")
    print(f"Total subjects: {len(features_df)}")
    print(f"Total features: {len(features_df.columns) - 2}")  # Exclude subject_id and label
    print("="*80)
    
    return features_df


if __name__ == '__main__':
    from data_loader import load_all_time_series
    from preprocessing import preprocess_all_subjects
    
    # Load and preprocess
    condition_dfs, control_dfs = load_all_time_series()
    all_dfs = condition_dfs + control_dfs
    preprocessed_dfs = preprocess_all_subjects(all_dfs)
    
    # Extract features
    features_df = extract_features_from_all_subjects(preprocessed_dfs)
    
    print("\nFeature DataFrame shape:", features_df.shape)
    print("\nFeature columns:")
    for i, col in enumerate(features_df.columns):
        print(f"{i+1}. {col}")
    
    print("\nSample features:")
    print(features_df.head())
    
    print("\nLabel distribution:")
    print(features_df['label'].value_counts())
