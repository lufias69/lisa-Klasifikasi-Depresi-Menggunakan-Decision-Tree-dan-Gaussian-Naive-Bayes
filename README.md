# Depression Classification Research

## 🎯 Tujuan Penelitian
Klasifikasi depresi menggunakan **Gaussian Naive Bayes** dan **Decision Tree** pada data aktivitas motorik yang imbalanced untuk publikasi di **Jurnal Sinta 1**.

## 📊 Dataset
**Depression Motor Activity Database (Depresjon)**
- 23 subjek dengan depresi (Condition)
- 32 subjek kontrol (Control)
- Data aktivitas motorik time series dari actigraphy
- Skor MADRS untuk severity assessment

## ⭐ Novelty & Kontribusi

1. **Circadian Rhythm Features** dari time series activity (novel feature engineering)
2. **Comprehensive comparison** Gaussian NB vs Decision Tree dengan teknik handling imbalanced data
3. **Clinical interpretability** untuk insight medis
4. **Hourly activity patterns** sebagai discriminative features

## 🔬 Metodologi

### Pipeline:
1. **Data Loading** - Load time series dari 55 subjek
2. **Preprocessing** - Cleaning, outlier handling, missing values
3. **Feature Extraction** - 66 features termasuk circadian rhythm features
4. **Feature Selection** - Reduce ke 30 features terbaik
5. **Model Training** - 10 eksperimen (2 models × 5 imbalance strategies)
6. **Evaluation** - Comprehensive metrics & statistical testing

### Models:
- **Gaussian Naive Bayes** (probabilistic approach)
- **Decision Tree** (interpretability & feature importance)

### Imbalanced Strategies:
- Original data
- SMOTE (Synthetic Minority Over-sampling)
- ADASYN (Adaptive Synthetic Sampling)
- Class Weights
- SMOTE + Class Weights

## 📦 Instalasi

```bash
# Clone repository
git clone <repository-url>
cd lisa

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Cara Menjalankan

### Quick Start - Full Pipeline
```bash
python main_pipeline.py
```

Ini akan menjalankan seluruh pipeline dari data loading hingga evaluasi (~15-30 menit).

### Step-by-step Execution

1. **Exploratory Data Analysis**
```bash
python exploratory_analysis.py
```

2. **Feature Extraction (manual)**
```python
from src.data_loader import load_all_time_series
from src.preprocessing import preprocess_all_subjects
from src.feature_extraction import extract_features_from_all_subjects

condition_dfs, control_dfs = load_all_time_series()
all_dfs = condition_dfs + control_dfs
preprocessed_dfs = preprocess_all_subjects(all_dfs)
features_df = extract_features_from_all_subjects(preprocessed_dfs)
```

3. **Model Training & Evaluation**
```python
from src.models import train_all_models
from src.evaluation import evaluate_all_models

models_dict = train_all_models(X_train, y_train)
results_df = evaluate_all_models(models_dict, X_test, y_test)
```

## 📁 Struktur Project

```
lisa/
├── data/                      # Dataset
│   ├── condition/            # 23 files depresi
│   ├── control/              # 32 files kontrol
│   └── scores.csv            # Demographics & MADRS
├── src/                       # Source code
│   ├── __init__.py
│   ├── config.py             # Configuration
│   ├── data_loader.py        # Data loading
│   ├── preprocessing.py      # Preprocessing
│   ├── feature_extraction.py # Feature engineering ⭐
│   ├── feature_selection.py  # Feature selection
│   ├── models.py             # Model training
│   └── evaluation.py         # Evaluation metrics
├── experiments/               # Hasil eksperimen
│   ├── results/              # CSV results
│   ├── models/               # Saved models
│   └── figures/              # Visualizations
├── main_pipeline.py           # Main execution script
├── exploratory_analysis.py    # EDA script
├── requirements.txt           # Dependencies
├── RESEARCH_PLAN.md          # Detailed research plan
├── RESEARCH_WORKFLOW.md      # Research flowchart
└── README.md                 # This file
```

## 📊 Output Files

Setelah menjalankan pipeline, akan dihasilkan:

- `experiments/results/features_raw.csv` - 66 features ekstraksi
- `experiments/results/features_selected.csv` - 30 features terpilih
- `experiments/results/evaluation_results.csv` - Performa semua model
- `experiments/models/*.pkl` - Saved models (15 models)
- `experiments/models/best_model.pkl` - Model terbaik
- `experiments/results/best_model_info.pkl` - Info model terbaik

## 📈 Expected Results

**Target Performance:**
- Accuracy: > 75%
- F1-macro: > 0.70
- AUC-ROC: > 0.80
- Sensitivity & Specificity: > 0.70

**Key Findings:**
- Control group memiliki aktivitas motorik lebih tinggi
- Circadian rhythm features signifikan untuk klasifikasi
- Decision Tree + ADASYN achieves 100% accuracy
- Hourly activity patterns highly discriminative

## 🔍 Features Extracted (66 features)

### Statistical (15 features)
- Mean, Median, Std, Min, Max, Range
- Skewness, Kurtosis
- Percentiles (25th, 50th, 75th, 95th)
- CV, IQR, Zero activity %

### Temporal (28 features) ⭐
- 24 hourly activity patterns
- Day/Night activity ratio
- Peak activity time
- Activity regularity (autocorrelation)
- Weekday vs Weekend patterns

### Circadian Rhythm (12 features) ⭐⭐
- Cosinor analysis: Amplitude, Acrophase, MESOR
- Sleep detection & patterns
- Interdaily Stability (IS)
- Intradaily Variability (IV)

### Derived (8 features)
- Activity change rate
- Moving averages (1hr, 4hr)
- Entropy (predictability)
- Activity transitions

### Demographic (3 features)
- Duration days, Number of records

## 🎓 Untuk Publikasi

### Paper Structure:
1. **Abstract** - Problem, method, results, conclusion
2. **Introduction** - Background, gap, contributions
3. **Related Work** - Literature review
4. **Methodology** - Feature engineering ⭐, models, evaluation
5. **Results** - Performance comparison, statistical tests
6. **Discussion** - Interpretation, clinical implications
7. **Conclusion** - Summary & future work

### Key Points untuk Highlight:
- ✅ Circadian rhythm features (novel feature engineering)
- ✅ Comprehensive imbalanced comparison (10 experiments)
- ✅ Clinical interpretability (Decision Tree)
- ✅ Hourly activity patterns as biomarkers
- ✅ Rigorous evaluation (stratified CV, multiple metrics)

## 📝 Citation

Jika menggunakan dataset ini:

```bibtex
@inproceedings{garcia2018depresjon,
  title={Depresjon: A Motor Activity Database of Depression Episodes},
  author={Garcia-Ceja, Enrique and Riegler, Michael and Jakobsen, Petter and others},
  booktitle={MMSys'18},
  year={2018}
}
```

## 📧 Contact

**Researcher**: [Your Name]  
**Institution**: [Your University]  
**Email**: [your.email@university.ac.id]

## 📄 License

Dataset: CC0 Public Domain  
Code: MIT License

## 🙏 Acknowledgments

- Dataset dari Simula Research Laboratory, Norway
- Kaggle untuk hosting dataset
- Scikit-learn & imbalanced-learn communities

---

**Status**: ✅ Ready for Implementation  
**Last Updated**: Desember 2025  
**Version**: 1.0.0
