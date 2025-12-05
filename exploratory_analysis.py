"""
Exploratory Data Analysis (EDA) untuk Depression Dataset
Tujuan: Analisis mendalam untuk publikasi Sinta 1
"""

import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("EXPLORATORY DATA ANALYSIS - DEPRESSION DATASET")
print("="*80)

# ============================================================================
# 1. ANALISIS STRUKTUR DATA
# ============================================================================
print("\n[1] ANALISIS STRUKTUR DATA")
print("-"*80)

# Path data
data_path = Path('data')
condition_path = data_path / 'condition'
control_path = data_path / 'control'

# Hitung jumlah file
condition_files = list(condition_path.glob('condition_*.csv'))
control_files = list(control_path.glob('control_*.csv'))

n_condition = len(condition_files)
n_control = len(control_files)
total_samples = n_condition + n_control

print(f"Jumlah sampel CONDITION (Depresi): {n_condition}")
print(f"Jumlah sampel CONTROL (Sehat): {n_control}")
print(f"Total sampel: {total_samples}")
print(f"\nIMBALANCE RATIO: {n_control}/{n_condition} = {n_control/n_condition:.2f}:1")
print(f"Persentase CONDITION: {(n_condition/total_samples)*100:.2f}%")
print(f"Persentase CONTROL: {(n_control/total_samples)*100:.2f}%")

# ============================================================================
# 2. ANALISIS SCORES.CSV (Data Demografi & MADRS)
# ============================================================================
print("\n\n[2] ANALISIS SCORES.CSV")
print("-"*80)

scores_df = pd.read_csv(data_path / 'scores.csv')
print(f"Shape: {scores_df.shape}")
print(f"\nKolom: {list(scores_df.columns)}")

# Missing values
print("\n--- Missing Values ---")
missing = scores_df.isnull().sum()
missing_pct = (missing / len(scores_df) * 100).round(2)
missing_df = pd.DataFrame({
    'Missing Count': missing,
    'Percentage': missing_pct
})
print(missing_df[missing_df['Missing Count'] > 0])

# Statistik deskriptif
print("\n--- Statistik Deskriptif Variabel Numerik ---")
print(scores_df[['days', 'madrs1', 'madrs2']].describe())

# Distribusi kategorikal
print("\n--- Distribusi Variabel Kategorikal ---")

categorical_vars = ['gender', 'age', 'afftype', 'melanch', 'inpatient', 
                    'edu', 'marriage', 'work']

for var in categorical_vars:
    if var in scores_df.columns:
        print(f"\n{var.upper()}:")
        print(scores_df[var].value_counts().sort_index())

# Analisis MADRS (severity indicator)
print("\n--- Analisis Skor MADRS ---")
print("MADRS1 (Saat mulai):")
print(f"  Mean: {scores_df['madrs1'].mean():.2f}")
print(f"  Median: {scores_df['madrs1'].median():.2f}")
print(f"  Std: {scores_df['madrs1'].std():.2f}")
print(f"  Range: {scores_df['madrs1'].min():.0f} - {scores_df['madrs1'].max():.0f}")

print("\nMADRS2 (Saat selesai):")
print(f"  Mean: {scores_df['madrs2'].mean():.2f}")
print(f"  Median: {scores_df['madrs2'].median():.2f}")
print(f"  Std: {scores_df['madrs2'].std():.2f}")
print(f"  Range: {scores_df['madrs2'].min():.0f} - {scores_df['madrs2'].max():.0f}")

# Perubahan MADRS
madrs_change = scores_df['madrs2'] - scores_df['madrs1']
print(f"\nPerubahan MADRS (madrs2 - madrs1):")
print(f"  Mean change: {madrs_change.mean():.2f}")
print(f"  Improved (negative): {(madrs_change < 0).sum()} pasien")
print(f"  Worsened (positive): {(madrs_change > 0).sum()} pasien")
print(f"  No change: {(madrs_change == 0).sum()} pasien")

# ============================================================================
# 3. ANALISIS TIME SERIES DATA (Sample dari beberapa file)
# ============================================================================
print("\n\n[3] ANALISIS TIME SERIES DATA")
print("-"*80)

# Fungsi untuk analisis sample file
def analyze_activity_file(filepath, label):
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    stats = {
        'label': label,
        'n_records': len(df),
        'duration_days': (df['timestamp'].max() - df['timestamp'].min()).days + 1,
        'activity_mean': df['activity'].mean(),
        'activity_median': df['activity'].median(),
        'activity_std': df['activity'].std(),
        'activity_min': df['activity'].min(),
        'activity_max': df['activity'].max(),
        'activity_zeros': (df['activity'] == 0).sum(),
        'zeros_pct': ((df['activity'] == 0).sum() / len(df) * 100)
    }
    return stats

# Sample analysis: 5 file dari masing-masing kelompok
print("\n--- Sample Analysis: 5 CONDITION files ---")
condition_stats = []
for i, filepath in enumerate(condition_files[:5]):
    stats = analyze_activity_file(filepath, 'condition')
    condition_stats.append(stats)
    print(f"\n{filepath.name}:")
    print(f"  Records: {stats['n_records']:,}")
    print(f"  Duration: {stats['duration_days']} days")
    print(f"  Activity - Mean: {stats['activity_mean']:.2f}, Std: {stats['activity_std']:.2f}")
    print(f"  Activity - Range: {stats['activity_min']:.0f} - {stats['activity_max']:.0f}")
    print(f"  Zero activity: {stats['zeros_pct']:.2f}%")

print("\n--- Sample Analysis: 5 CONTROL files ---")
control_stats = []
for i, filepath in enumerate(control_files[:5]):
    stats = analyze_activity_file(filepath, 'control')
    control_stats.append(stats)
    print(f"\n{filepath.name}:")
    print(f"  Records: {stats['n_records']:,}")
    print(f"  Duration: {stats['duration_days']} days")
    print(f"  Activity - Mean: {stats['activity_mean']:.2f}, Std: {stats['activity_std']:.2f}")
    print(f"  Activity - Range: {stats['activity_min']:.0f} - {stats['activity_max']:.0f}")
    print(f"  Zero activity: {stats['zeros_pct']:.2f}%")

# Perbandingan agregat
print("\n--- PERBANDINGAN CONDITION vs CONTROL (dari sample) ---")
condition_df = pd.DataFrame(condition_stats)
control_df = pd.DataFrame(control_stats)

print("\nCONDITION (n=5):")
print(f"  Avg records: {condition_df['n_records'].mean():,.0f}")
print(f"  Avg duration: {condition_df['duration_days'].mean():.1f} days")
print(f"  Avg activity mean: {condition_df['activity_mean'].mean():.2f}")
print(f"  Avg activity std: {condition_df['activity_std'].mean():.2f}")
print(f"  Avg zero activity: {condition_df['zeros_pct'].mean():.2f}%")

print("\nCONTROL (n=5):")
print(f"  Avg records: {control_df['n_records'].mean():,.0f}")
print(f"  Avg duration: {control_df['duration_days'].mean():.1f} days")
print(f"  Avg activity mean: {control_df['activity_mean'].mean():.2f}")
print(f"  Avg activity std: {control_df['activity_std'].mean():.2f}")
print(f"  Avg zero activity: {control_df['zeros_pct'].mean():.2f}%")

# ============================================================================
# 4. FULL ANALYSIS - SEMUA FILE (lebih detail)
# ============================================================================
print("\n\n[4] FULL ANALYSIS - SEMUA FILE")
print("-"*80)
print("Menganalisis semua file time series...")

all_condition_stats = []
for filepath in condition_files:
    stats = analyze_activity_file(filepath, 'condition')
    all_condition_stats.append(stats)

all_control_stats = []
for filepath in control_files:
    stats = analyze_activity_file(filepath, 'control')
    all_control_stats.append(stats)

all_condition_df = pd.DataFrame(all_condition_stats)
all_control_df = pd.DataFrame(all_control_stats)

print("\n--- STATISTIK LENGKAP ---")
print("\nCONDITION (n=23):")
print(all_condition_df[['n_records', 'duration_days', 'activity_mean', 
                         'activity_std', 'zeros_pct']].describe())

print("\nCONTROL (n=32):")
print(all_control_df[['n_records', 'duration_days', 'activity_mean', 
                       'activity_std', 'zeros_pct']].describe())

# ============================================================================
# 5. KEY FINDINGS & INSIGHTS
# ============================================================================
print("\n\n[5] KEY FINDINGS & INSIGHTS")
print("="*80)

print("\n📊 TEMUAN PENTING UNTUK PENELITIAN:")
print("-"*80)

print("\n1. IMBALANCED DATA:")
print(f"   - Rasio imbalance: {n_control/n_condition:.2f}:1 (Control lebih banyak)")
   print(f"   - Ini MEMERLUKAN teknik handling: SMOTE, ADASYN, atau class weighting")print("\n2. TIME SERIES CHARACTERISTICS:")
print(f"   - Condition: Rata-rata {all_condition_df['n_records'].mean():,.0f} records/pasien")
print(f"   - Control: Rata-rata {all_control_df['n_records'].mean():,.0f} records/pasien")
print(f"   - Variasi durasi monitoring yang berbeda antar pasien")

print("\n3. ACTIVITY PATTERNS:")
cond_activity = all_condition_df['activity_mean'].mean()
ctrl_activity = all_control_df['activity_mean'].mean()
print(f"   - Condition activity mean: {cond_activity:.2f}")
print(f"   - Control activity mean: {ctrl_activity:.2f}")
print(f"   - Perbedaan: {abs(ctrl_activity - cond_activity):.2f}")
if ctrl_activity > cond_activity:
    print(f"   - Control group menunjukkan aktivitas motorik LEBIH TINGGI")
else:
    print(f"   - Condition group menunjukkan aktivitas motorik LEBIH TINGGI")

print("\n4. MISSING VALUES:")
if missing.sum() > 0:
    print(f"   - Terdapat missing values di: {list(missing[missing > 0].index)}")
    print(f"   - Perlu strategi handling (imputation atau removal)")
else:
    print(f"   - Tidak ada missing values di scores.csv")

print("\n5. MADRS SCORE DISTRIBUTION:")
print(f"   - Range MADRS1: {scores_df['madrs1'].min():.0f} - {scores_df['madrs1'].max():.0f}")
print(f"   - Mean MADRS1: {scores_df['madrs1'].mean():.2f}")
print(f"   - MADRS dapat dijadikan feature atau target tambahan")

print("\n\n" + "="*80)
print("ANALISIS SELESAI - Siap untuk tahap feature engineering & modeling")
print("="*80)
