# RENCANA PENELITIAN: Depression Classification using Naive Bayes & Decision Tree

---

## 📋 INFORMASI PENELITIAN

**Tujuan**: Publikasi Skripsi di Jurnal Sinta 1  
**Dataset**: Depression Motor Activity Database (Depresjon)  
**Metode Utama**: Gaussian Naive Bayes & Decision Tree  
**Fokus**: Handling Imbalanced Data + Feature Engineering dari Time Series

---

## 🔍 HASIL ANALISIS DATA (EDA)

### 1. Karakteristik Dataset
- **Total Sampel**: 55 subjek
  - Condition (Depresi): 23 subjek (41.82%)
  - Control (Sehat): 32 subjek (58.18%)
  - **Imbalance Ratio**: 1.39:1 ✅ (Mild imbalance)

### 2. Data Time Series
- **Condition Group**:
  - Rata-rata 23,988 records/pasien (~17 hari monitoring)
  - Activity mean: 163.72 (STD: 300.62)
  - Zero activity: 43.62%
  
- **Control Group**:
  - Rata-rata 31,875 records/pasien (~23 hari monitoring)
  - Activity mean: 208.65 (STD: 369.27)
  - Zero activity: 38.11%

- **KEY INSIGHT**: 🎯 **Control group memiliki aktivitas motorik LEBIH TINGGI dibanding Condition** (perbedaan: 44.93 unit)

### 3. Missing Values
- Variabel dengan missing > 50%: `afftype`, `melanch`, `inpatient`, `marriage`, `work`, `madrs1`, `madrs2`
- **Strategi**: Hanya gunakan data yang lengkap untuk analisis demografi, fokus pada time series features

### 4. MADRS Score
- Mean MADRS1: 22.74 (Range: 13-29) - Severity indicator
- Mean MADRS2: 20.00 - Ada perbaikan rata-rata -2.74 poin
- 15 pasien membaik, 4 memburuk, 4 tidak berubah

---

## 💡 NOVELTY & RESEARCH GAP

### 1. **Feature Engineering dari Circadian Rhythm** ⭐ NOVELTY UTAMA
   - **Gap**: Penelitian existing fokus pada statistical features sederhana
   - **Novelty**: 
     - Ekstraksi fitur berbasis ritme sirkadian (24-jam patterns)
     - Sleep-wake cycle detection dari actigraphy
     - Hourly activity distribution analysis
     - Temporal irregularity features

### 2. **Comprehensive Comparison dengan Imbalanced Techniques** ⭐
   - **Gap**: Penelitian sebelumnya tidak secara khusus address imbalanced problem
   - **Novelty**: 
     - Perbandingan Gaussian NB vs Decision Tree
     - Testing dengan berbagai teknik: SMOTE, ADASYN, class weights
     - Cross-validation dengan stratified sampling

### 3. **Clinical Interpretability Analysis** ⭐
   - **Gap**: Kebanyakan penelitian fokus pada akurasi, kurang pada interpretability
   - **Novelty**: 
     - Feature importance analysis untuk clinical insight
     - Decision tree visualization untuk pattern discovery
     - Correlation analysis antara activity patterns dan MADRS scores

---

## 🎯 RESEARCH QUESTIONS

1. **RQ1**: Apakah Gaussian Naive Bayes atau Decision Tree lebih efektif untuk klasifikasi depresi pada imbalanced data?

2. **RQ2**: Fitur time series apa yang paling signifikan membedakan individu dengan depresi vs kontrol?

3. **RQ3**: Bagaimana pengaruh berbagai teknik handling imbalanced data (SMOTE, ADASYN, class weight) terhadap performa model?

4. **RQ4**: Apakah pola circadian rhythm dapat menjadi biomarker untuk deteksi depresi?

---

## 🔬 METODOLOGI PENELITIAN

### Phase 1: Data Preprocessing & Feature Engineering

#### 1.1 Data Cleaning
- Handle missing values (fokus pada data lengkap)
- Outlier detection (Z-score > 3 atau IQR method)
- Timestamp parsing dan validation

#### 1.2 Feature Extraction (TIME SERIES → TABULAR)

**A. Statistical Features (per subjek)**
- Mean, Median, Std, Min, Max activity
- Skewness, Kurtosis (distribusi shape)
- Percentile (25th, 50th, 75th, 95th)
- Zero activity percentage
- Coefficient of Variation (CV)

**B. Temporal Features** ⭐ NOVELTY
- Hourly activity patterns (24 features)
- Day vs Night activity ratio
- Peak activity time
- Activity regularity (autocorrelation)

**C. Circadian Rhythm Features** ⭐⭐ NOVELTY
- Cosinor analysis (amplitude, acrophase, MESOR)
- Sleep detection (consecutive zero activity periods)
- Sleep onset time, wake time, sleep duration
- Sleep efficiency
- Interdaily stability (IS) & Intradaily variability (IV)

**D. Derived Features**
- Activity change rate (consecutive differences)
- Moving averages (1hr, 4hr, 8hr windows)
- Entropy (activity predictability)
- Longest inactive period

**E. Demographic Features (jika tersedia)**
- Gender, Age, Days of monitoring

**Total Expected Features**: ~60-80 features

#### 1.3 Feature Selection
- Variance Threshold (remove low-variance features)
- Correlation analysis (remove highly correlated features > 0.95)
- Mutual Information (select top-K features)
- Recursive Feature Elimination (RFE)

### Phase 2: Model Development

#### 2.1 Train-Test Split
- Stratified split: 80% train, 20% test
- Random state fixed untuk reproducibility

#### 2.2 Models to Compare

**1. Gaussian Naive Bayes**
   - sklearn.naive_bayes.GaussianNB
   - Hyperparameters: var_smoothing
   - Probabilistic baseline model

**2. Decision Tree** (INTERPRETABILITY FOCUS)
   - sklearn.tree.DecisionTreeClassifier
   - Hyperparameters: max_depth, min_samples_split, criterion (gini/entropy)
   - White-box model untuk clinical insights

#### 2.3 Imbalanced Data Handling Strategies

**Scenario A: Original Data** (baseline)
**Scenario B: SMOTE** (Synthetic Minority Over-sampling)
**Scenario C: ADASYN** (Adaptive Synthetic Sampling)
**Scenario D: Class Weights** (weighted loss function)
**Scenario E: Combination** (SMOTE + Class Weight)

Total experiments: 2 models × 5 scenarios = **10 eksperimen**

#### 2.4 Hyperparameter Tuning
- GridSearchCV dengan Stratified K-Fold (k=5)
- Scoring: F1-macro (cocok untuk imbalanced)

### Phase 3: Evaluation

#### 3.1 Metrics
- **Accuracy** (overall correctness)
- **Precision, Recall, F1-Score** (per class dan macro average)
- **AUC-ROC** (discrimination ability)
- **Confusion Matrix** (true/false positives/negatives)
- **Specificity & Sensitivity** (clinical perspective)

#### 3.2 Cross-Validation
- Stratified 5-Fold Cross-Validation
- Report mean ± std untuk setiap metric

#### 3.3 Statistical Testing
- Paired t-test atau Wilcoxon test (compare model performance)
- p-value < 0.05 untuk significance

#### 3.4 Feature Importance Analysis
- For Decision Tree: feature_importances_
- For Naive Bayes: log probability analysis
- Top-10 most discriminative features

### Phase 4: Interpretability & Clinical Insights

#### 4.1 Decision Tree Visualization
- Tree structure visualization
- Decision rules extraction

#### 4.2 Activity Pattern Analysis
- Compare hourly patterns: Condition vs Control
- Circadian rhythm visualization
- Sleep pattern comparison

#### 4.3 Correlation with MADRS
- Correlation between features dan MADRS scores
- Regression analysis (optional)

---

## 📁 STRUKTUR PROJECT

```
lisa/
│
├── data/                          # Dataset (sudah ada)
│   ├── condition/
│   ├── control/
│   └── scores.csv
│
├── notebooks/                     # Jupyter notebooks untuk eksplorasi
│   ├── 01_EDA.ipynb
│   ├── 02_Feature_Engineering.ipynb
│   ├── 03_Model_Training.ipynb
│   └── 04_Results_Visualization.ipynb
│
├── src/                           # Source code modular
│   ├── __init__.py
│   ├── data_loader.py            # Load dan parse data
│   ├── preprocessing.py          # Cleaning, handling missing values
│   ├── feature_extraction.py     # Extract features dari time series
│   ├── feature_selection.py      # Feature selection methods
│   ├── models.py                 # Model training functions
│   ├── evaluation.py             # Metrics dan evaluation
│   └── visualization.py          # Plotting functions
│
├── experiments/                   # Hasil eksperimen
│   ├── results/                  # Model results (CSV, JSON)
│   ├── models/                   # Saved models (.pkl)
│   └── figures/                  # Plots dan visualisasi
│
├── exploratory_analysis.py        # Script EDA (sudah dibuat)
├── main_pipeline.py              # Main execution script
├── requirements.txt              # Dependencies
├── config.yaml                   # Configuration file
│
├── RESEARCH_PLAN.md              # Dokumen ini
└── README.md                     # Project documentation
```

---

## 📦 DEPENDENCIES

```txt
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
imbalanced-learn>=0.11.0
matplotlib>=3.7.0
seaborn>=0.12.0
scipy>=1.10.0
pyyaml>=6.0
joblib>=1.3.0
```

---

## 🎯 DELIVERABLES

### 1. **Paper Structure (untuk Jurnal Sinta 1)**

**Abstract**
- Background: Depression detection challenge, importance of objective measures
- Objective: Compare Gaussian NB vs Decision Tree for depression classification
- Method: Time series feature extraction + ML classification with imbalanced techniques
- Results: Highlight best performance
- Conclusion: Effectiveness of Decision Tree with ADASYN for imbalanced depression data

**Introduction**
- Problem: Subjective depression diagnosis, need for objective biomarkers
- Importance: Actigraphy as non-invasive monitoring
- Gap: Limited circadian rhythm features, inadequate imbalanced data handling
- Contribution: Novel circadian features + comprehensive comparison of imbalanced techniques

**Related Work**
- Literature review (5-10 papers)
- Comparison table: methods, datasets, results

**Methodology**
- Dataset description
- Feature engineering (detail + novelty)
- Models + imbalanced handling techniques
- Evaluation protocol

**Results**
- Performance comparison table
- Confusion matrices
- ROC curves
- Feature importance
- Statistical significance tests

**Discussion**
- Why Decision Tree performs best
- Clinical implications of top features
- Comparison with existing methods
- Limitations
- Future work

**Conclusion**
- Summary of contributions
- Practical implications

### 2. **Code Repository** (GitHub)
- Well-documented, modular code
- Reproducible experiments
- README with instructions

### 3. **Visualization & Tables**
- Performance comparison tables
- Confusion matrices heatmaps
- ROC curves
- Feature importance bar charts
- Activity pattern comparisons
- Circadian rhythm plots

---

## ⏱️ TIMELINE ESTIMASI

| Phase | Tasks | Estimasi |
|-------|-------|----------|
| **Week 1-2** | Data Preprocessing & Feature Engineering | 2 minggu |
| **Week 3** | Feature Selection & Data Preparation | 1 minggu |
| **Week 4-5** | Model Training & Hyperparameter Tuning | 2 minggu |
| **Week 6** | Evaluation & Statistical Analysis | 1 minggu |
| **Week 7** | Interpretability & Visualization | 1 minggu |
| **Week 8-10** | Paper Writing & Revision | 3 minggu |

**Total**: ~10 minggu (2.5 bulan)

---

## 🎓 TARGET JURNAL SINTA 1

**Rekomendasi Jurnal:**

1. **Jurnal Nasional:**
   - Jurnal Teknologi dan Sistem Komputer (JTSISKOM)
   - Jurnal Ilmu Komputer dan Informasi (JIKI)
   - Register: Jurnal Ilmiah Teknologi Sistem Informasi

2. **Fokus Submission:**
   - Healthcare Informatics
   - Machine Learning Applications
   - Mental Health Technology

**Kriteria:**
- Novelty: ✅ Complement NB + Circadian features
- Methodology: ✅ Rigorous evaluation
- Results: ✅ Comprehensive comparison
- Writing: ✅ Clear, structured, scientifically sound

---

## 🚀 NEXT STEPS

1. ✅ **EDA SELESAI** - Pahami karakteristik data
2. ⏭️ **Feature Engineering** - Implementasi ekstraksi fitur
3. ⏭️ **Model Pipeline** - Build training pipeline
4. ⏭️ **Experiments** - Run all scenarios
5. ⏭️ **Analysis** - Evaluate & compare results
6. ⏭️ **Paper Writing** - Document findings

---

## 📝 CATATAN PENTING

### Kekuatan Penelitian Ini:
- ✅ Dataset real-world dari publikasi ilmiah
- ✅ Novelty jelas: Complement NB + Circadian features
- ✅ Metodologi rigorous dengan multiple comparisons
- ✅ Clinical relevance tinggi
- ✅ Imbalanced data handling (sesuai realita)

### Tantangan:
- ⚠️ Dataset relatif kecil (55 subjek) - gunakan cross-validation
- ⚠️ Missing values signifikan - fokus pada available data
- ⚠️ Computational cost - optimize feature extraction

### Tips untuk Publikasi:
1. **Highlight novelty** di abstract & introduction
2. **Rigorous evaluation** dengan statistical tests
3. **Clinical interpretation** untuk relevance
4. **Compare dengan SOTA** (State-of-the-art baselines)
5. **Clear visualization** untuk easy understanding

---

**Dibuat**: Desember 2025  
**Status**: READY FOR IMPLEMENTATION  
**Persetujuan**: Pending
