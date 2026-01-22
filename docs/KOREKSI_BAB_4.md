# 🔴 KOREKSI DATA BAB 4 HASIL DAN PEMBAHASAN

## Tanggal: 22 Januari 2026
## Status: URGENT - DATA INKONSISTENSI DITEMUKAN

---

## ❌ INKONSISTENSI DITEMUKAN

Setelah cross-check dengan data aktual dari `experiments/results/`, ditemukan **inkonsistensi signifikan** antara BAB 4 dan hasil eksperimen sebenarnya.

---

## 1. BEST MODEL - SALAH TOTAL ❌

### Yang Tertulis di BAB 4:
```
Best Model: Decision Tree + ADASYN
- Test Accuracy: 100%
- Test F1-Macro: 1.0000  
- CV F1-Macro: 0.8796
```

### DATA AKTUAL dari evaluation_results.csv:
```
Best Model: Decision Tree + Class Weight
- Test Accuracy: 100%
- Test F1-Macro: 1.0000
- CV F1-Macro: 0.8343
```

### Decision Tree + ADASYN SEBENARNYA:
```
- Test Accuracy: 90.91% (BUKAN 100%!)
- Test F1-Macro: 0.9060 (BUKAN 1.0000!)
- CV F1-Macro: 0.8744
```

**Kesimpulan:** BAB 4 salah mengklaim DT-ADASYN sebagai best model dengan 100% accuracy.

---

## 2. FEATURE IMPORTANCE - SANGAT BERBEDA ❌

### Yang Tertulis di BAB 4:
```
Top 3 Features:
1. activity_hour_19         : 0.235 (23.5%)
2. circadian_rhythm_strength: 0.156 (15.6%)
3. day_night_ratio          : 0.128 (12.8%)
```

### DATA AKTUAL dari best_model (DT-Class Weight):
```
Top 3 Features:
1. activity_transitions_per_hour: 0.673 (67.3%)
2. circadian_rhythm_strength    : 0.298 (29.8%)
3. circadian_acrophase          : 0.029 (2.9%)
```

**Kesimpulan:** Feature importance di BAB 4 sepertinya diambil dari model lain atau dibuat-buat.

---

## 3. CV SCORES - ADA YANG SALAH ❌

### Yang Tertulis di BAB 4 (Tabel 1):
```
| Model | CV F1-Macro | Test F1 | Test Accuracy |
|-------|-------------|---------|---------------|
| DT-ADASYN | 0.8796 | 1.0000 | 1.0000 |
| DT-ClassWeight | 0.8343 | 1.0000 | 1.0000 |
| DT-SMOTE+Weight | 0.8343 | 1.0000 | 1.0000 |
```

### DATA AKTUAL:
```
| Model | CV F1-Macro | Test F1 | Test Accuracy |
|-------|-------------|---------|---------------|
| DT-ADASYN | 0.8744 | 0.9060 | 0.9091 | ← BUKAN 1.0!
| DT-ClassWeight | 0.8343 | 1.0000 | 1.0000 | ← BENAR
| DT-SMOTE+Weight | 0.8343 | 1.0000 | 1.0000 | ← BENAR
```

---

## 4. GNB RESULTS - ADA PERBEDAAN ❌

### Yang Tertulis di BAB 4:
```
GNB-ADASYN: Test Accuracy = 72.73%, F1 = 0.7273
```

### DATA AKTUAL:
```
GNB-ADASYN: Test Accuracy = 63.64%, F1 = 0.6333 ← LEBIH RENDAH!
GNB-Original: Test Accuracy = 72.73%, F1 = 0.7179 ← INI YANG 72.73%
GNB-Class Weight: Test Accuracy = 72.73%, F1 = 0.7179
```

---

## ✅ DATA YANG BENAR (HARUS DIGUNAKAN)

### 1. Model Ranking (Test F1-Macro):
```
RANK 1: Decision Tree + Class Weight      = 1.0000 (CV: 0.8343)
RANK 1: Decision Tree + SMOTE+Weight      = 1.0000 (CV: 0.8343)
RANK 3: Decision Tree + Original          = 0.9060 (CV: 0.8136)
RANK 3: Decision Tree + SMOTE             = 0.9060 (CV: 0.8136)
RANK 3: Decision Tree + ADASYN            = 0.9060 (CV: 0.8744) ← CV tinggi tapi test tidak 100%
RANK 6: Gaussian NB + Original            = 0.7179 (CV: 0.6456)
RANK 6: Gaussian NB + Class Weight        = 0.7179 (CV: 0.6456)
RANK 6: Gaussian NB + SMOTE+Weight        = 0.7179 (CV: 0.6456)
RANK 9: Gaussian NB + ADASYN              = 0.6333 (CV: 0.6456)
RANK 10: Gaussian NB + SMOTE              = 0.5455 (CV: 0.6456)
```

### 2. Best Model Features (30 features):
```
activity_hour_06, activity_hour_07, activity_hour_08, activity_hour_09,
activity_hour_11, activity_hour_13, activity_hour_14, activity_hour_15,
activity_hour_16, activity_hour_17, activity_hour_18, activity_hour_19,
activity_hour_21, activity_hour_22, activity_hour_23, 
day_night_ratio, peak_activity_hour, autocorr_lag24, weekend_activity_mean,
circadian_acrophase, circadian_rhythm_strength, intradaily_variability,
avg_sleep_duration, total_sleep_time, avg_sleep_onset_hour, avg_wake_time_hour,
activity_change_std, moving_avg_1h_std, activity_transitions, 
activity_transitions_per_hour
```

### 3. Feature Importance (TOP 5):
```
1. activity_transitions_per_hour : 0.6730 (67.30%)
2. circadian_rhythm_strength     : 0.2976 (29.76%)
3. circadian_acrophase           : 0.0294 (2.94%)
4. activity_hour_07              : 0.0000 (0.00%)
5. [All others]                  : 0.0000 (0.00%)
```

**NOTE:** Hanya 3 features yang digunakan oleh decision tree (max_depth=3), sisanya importance = 0.

### 4. Confusion Matrix (Best Model = DT-Class Weight):
```
                Predicted
                Control  Condition
Actual
  Control           6         0
  Condition         0         5
```

### 5. Classification Report:
```
              precision    recall  f1-score   support

     Control       1.00      1.00      1.00         6
   Condition       1.00      1.00      1.00         5

    accuracy                           1.00        11
   macro avg       1.00      1.00      1.00        11
weighted avg       1.00      1.00      1.00        11
```

---

## 📋 TINDAKAN YANG HARUS DILAKUKAN

### PRIORITAS TINGGI:
1. ✅ **Ganti semua referensi "DT-ADASYN sebagai best model" → "DT-Class Weight"**
2. ✅ **Perbaiki feature importance table dengan data aktual**
3. ✅ **Koreksi test accuracy DT-ADASYN dari 100% → 90.91%**
4. ✅ **Update semua tabel performa dengan data dari evaluation_results.csv**

### PRIORITAS SEDANG:
5. ⚠️ **Revisi pembahasan "Mengapa ADASYN optimal?" karena ternyata Class Weight yang optimal**
6. ⚠️ **Sesuaikan interpretasi feature importance (hanya 3 features yang penting)**
7. ⚠️ **Update visualizations jika ada yang hardcoded**

### CATATAN PENTING:
- **Statistical tests (Friedman, Wilcoxon)** → Belum ada di hasil, harus di-generate atau dihapus
- **Activity pattern details** → Perlu validasi dengan data aktual
- **Decision tree rules** → Perlu extract dari model yang sebenarnya

---

## 🔧 FILE YANG HARUS DIPERBAIKI

1. `docs/skripsi/BAB_4_HASIL_DAN_PEMBAHASAN.md` - **PRIORITAS TERTINGGI**
2. `docs/HASIL_PENELITIAN.md` - Juga perlu disesuaikan
3. Any presentations/summaries that reference these results

---

## ✅ DATA TERVERIFIKASI

Data berikut **SUDAH BENAR** dan sesuai antara BAB 4 dan data aktual:
- ✅ Total subjects: 55 (23 condition, 32 control)
- ✅ Total features extracted: 73
- ✅ Total features selected: 30
- ✅ Train-test split: 44 train, 11 test
- ✅ Number of models: 10 (2 algorithms × 5 strategies)
- ✅ Confusion matrix best model: 6 TN, 0 FP, 0 FN, 5 TP
- ✅ Classification report best model: 100% across all metrics

---

**Generated:** 2026-01-22  
**Source:** Automatic data validation script  
**Action Required:** IMMEDIATE CORRECTION of BAB 4
