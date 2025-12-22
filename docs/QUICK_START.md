# 🚀 QUICK START GUIDE

## Panduan Cepat Menjalankan Penelitian

---

## ⚡ TL;DR - Super Quick

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run full pipeline
python main_pipeline.py

# 3. Generate visualizations  
python generate_visualizations.py

# Done! ✅
```

**Durasi Total**: ~20-25 menit

---

## 📋 PREREQUISITES

### 1. Python Environment
```bash
python --version  # Harus >= 3.8
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

**Required packages:**
- pandas >= 2.0.0
- numpy >= 1.24.0
- scikit-learn >= 1.3.0
- imbalanced-learn >= 0.11.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- scipy >= 1.10.0
- joblib >= 1.3.0

### 3. Data Structure
Pastikan folder `data/` sudah ada dengan struktur:
```
data/
├── condition/       # 23 files
├── control/         # 32 files
└── scores.csv       # Demographics
```

---

## 🎯 STEP-BY-STEP EXECUTION

### **Step 1: Exploratory Data Analysis (Optional)**

```bash
python exploratory_analysis.py
```

**Output:**
- Karakteristik dataset
- Statistik deskriptif
- Imbalance ratio
- Missing values analysis
- Activity patterns summary

**Duration**: ~1 menit

---

### **Step 2: Run Full Pipeline (MAIN)**

```bash
python main_pipeline.py
```

**What it does:**
1. ✅ Load data (55 subjects)
2. ✅ Preprocess time series
3. ✅ Extract 73 features
4. ✅ Select 30 best features
5. ✅ Train 15 models (3 models × 5 strategies)
6. ✅ Evaluate all models
7. ✅ Save results & best model

**Output Files:**
```
experiments/
├── results/
│   ├── features_raw.csv              # 73 features
│   ├── features_selected.csv         # 30 features
│   ├── evaluation_results.csv        # Performance
│   ├── feature_selection_info.pkl
│   └── best_model_info.pkl
└── models/
    ├── best_model.pkl                # Decision Tree + ADASYN
    ├── gaussian_nb_*.pkl             # 5 models
    └── decision_tree_*.pkl           # 5 models
```

**Duration**: ~15-20 menit

**Expected Best Result:**
- Model: **Decision Tree + ADASYN**
- Accuracy: **100%**
- F1-macro: **100%**
- AUC-ROC: **100%**

---

### **Step 3: Generate Visualizations**

```bash
python generate_visualizations.py
```

**Output Figures:**
```
experiments/figures/
├── confusion_matrix_best.png
├── roc_curves.png
├── model_comparison_f1.png
├── model_comparison_accuracy.png
├── feature_importance.png
├── decision_tree_viz.png
├── activity_patterns_24h.png
└── metrics_heatmap.png
```

**Duration**: ~2-3 menit

---

## 📊 CHECKING RESULTS

### **1. View Performance**
```bash
# Open evaluation results
cat experiments/results/evaluation_results.csv
```

atau gunakan Python:
```python
import pandas as pd
results = pd.read_csv('experiments/results/evaluation_results.csv')
print(results.sort_values('f1_macro', ascending=False))
```

### **2. Load Best Model**
```python
import joblib

# Load best model
best_model = joblib.load('experiments/models/best_model.pkl')

# Load model info
best_info = joblib.load('experiments/results/best_model_info.pkl')

print(f"Best Model: {best_info['experiment_id']}")
print(f"F1-Macro: {best_info['metrics']['f1_macro']:.4f}")
print(f"Accuracy: {best_info['metrics']['accuracy']:.4f}")
```

### **3. View Features**
```python
import pandas as pd

# Selected features
features = pd.read_csv('experiments/results/features_selected.csv')
print(f"Total samples: {len(features)}")
print(f"Features: {features.columns.tolist()}")
print(f"\nLabel distribution:\n{features['label'].value_counts()}")
```

---

## 🔧 TROUBLESHOOTING

### **Issue 1: ModuleNotFoundError**
```bash
# Solution: Install missing package
pip install <package-name>

# Or reinstall all
pip install -r requirements.txt --upgrade
```

### **Issue 2: FileNotFoundError for data**
```bash
# Check data folder exists
ls data/

# Should show:
# - condition/ (with 23 CSV files)
# - control/ (with 32 CSV files)  
# - scores.csv
```

---

## 📖 UNDERSTANDING OUTPUT

### **evaluation_results.csv Columns:**
- `experiment_id`: Model + strategy name
- `model_name`: gaussian_nb, decision_tree
- `imbalance_strategy`: original, smote, adasyn, class_weight, smote_weight
- `accuracy`: Overall accuracy
- `precision`, `recall`, `f1`: Binary metrics
- `precision_macro`, `recall_macro`, `f1_macro`: Macro-averaged
- `roc_auc`: Area under ROC curve
- `true_negative`, `false_positive`, `false_negative`, `true_positive`: Confusion matrix
- `specificity`, `sensitivity`: Clinical metrics
- `cv_f1_macro`: Cross-validation F1-macro score

### **Best Model Metrics:**
```
Accuracy:     100%  ← Perfect classification
Precision:    100%  ← No false positives
Recall:       100%  ← No false negatives
F1-Score:     100%  ← Harmonic mean
Specificity:  100%  ← True negative rate
Sensitivity:  100%  ← True positive rate
AUC-ROC:      100%  ← Perfect discrimination
```

---

## 🎨 CUSTOMIZATION

### **Change Random Seed**
Edit `src/config.py`:
```python
RANDOM_STATE = 42  # Change to any integer
```

### **Adjust Train-Test Split**
Edit `src/config.py`:
```python
TEST_SIZE = 0.2  # Change to 0.3 for 70-30 split
```

### **Modify Feature Selection**
Edit `main_pipeline.py` line 79:
```python
X_selected, selection_info = feature_selection_pipeline(
    X, y,
    variance_threshold=0.01,      # Lower = more features kept
    correlation_threshold=0.95,   # Lower = more aggressive removal
    k_best=40,                    # Increase for more features
    n_features_rfe=30            # Final number of features
)
```

### **Add More Models**
Edit `src/models.py` - add to `train_all_models()`:
```python
models = ['gaussian_nb', 'decision_tree', 'random_forest']  # Add new
```

---

## 📚 FILE DESCRIPTIONS

### **Python Scripts:**
| File | Purpose | Duration |
|------|---------|----------|
| `exploratory_analysis.py` | EDA & data understanding | ~1 min |
| `main_pipeline.py` | Full training pipeline | ~15-20 min |
| `generate_visualizations.py` | Create all figures | ~2-3 min |

### **Source Modules:**
| Module | Purpose |
|--------|---------|
| `src/config.py` | Configuration & parameters |
| `src/data_loader.py` | Load CSV files |
| `src/preprocessing.py` | Clean & preprocess |
| `src/feature_extraction.py` | Extract 73 features ⭐ |
| `src/feature_selection.py` | Select 30 features |
| `src/models.py` | Train 15 experiments |
| `src/evaluation.py` | Compute metrics |
| `src/visualization.py` | Generate plots |

### **Documentation:**
| File | Content |
|------|---------|
| `README.md` | Project overview |
| `RESEARCH_PLAN.md` | Detailed planning |
| `RESEARCH_WORKFLOW.md` | Visual flowchart |
| `HASIL_PENELITIAN.md` | Results & analysis |
| `IMPLEMENTATION_COMPLETE.md` | Final summary |
| `QUICK_START.md` | This guide |

---

## ⏱️ TIMELINE

| Phase | Task | Duration |
|-------|------|----------|
| **Setup** | Install dependencies | 5 min |
| **Phase 1** | EDA (optional) | 1 min |
| **Phase 2-7** | Full pipeline | 15-20 min |
| **Phase 8** | Generate viz | 2-3 min |
| **Phase 9** | Review results | 5-10 min |
| **TOTAL** | | **~25-40 min** |

---

## ✅ SUCCESS INDICATORS

Jika pipeline berhasil, Anda akan melihat:

### **Console Output:**
```
================================================================================
PIPELINE EXECUTION COMPLETE!
================================================================================

📊 SUMMARY:
  Total subjects: 55
  Features extracted: 73
  Features selected: 30
  Models trained: 10
  Best model: decision_tree_adasyn
  Best F1-macro: 1.0000
  Best Accuracy: 1.0000
  Best AUC-ROC: 1.0000

✅ Research pipeline completed successfully!
```

### **Files Created:**
- ✅ `experiments/results/` - 5 files
- ✅ `experiments/models/` - 11 .pkl files
- ✅ `experiments/figures/` - 8 .png files (after viz)

### **Performance:**
- ✅ Best model F1-macro: **1.0000** (100%)
- ✅ Best model accuracy: **1.0000** (100%)
- ✅ All metrics: **1.0000** (perfect!)

---

## 🎓 NEXT STEPS

### **For Paper Writing:**
1. Open `HASIL_PENELITIAN.md` - read analysis
2. Review figures in `experiments/figures/`
3. Check tables in `experiments/results/`
4. Use `RESEARCH_PLAN.md` for paper structure

### **For Presentation:**
1. Use figures dari `experiments/figures/`
2. Highlight 100% accuracy result
3. Show feature importance
4. Explain decision tree

### **For Further Research:**
1. Re-run with different random seeds
2. Try ensemble methods
3. Validate on external dataset
4. Develop web interface

---

## 💡 TIPS

1. **Run pipeline pada malam hari** - biarkan berjalan overnight
2. **Check GPU availability** - untuk speed up (optional)
3. **Save intermediate results** - jika pipeline error di tengah
4. **Document changes** - jika modify code
5. **Backup results** - sebelum re-run

---

## 📞 NEED HELP?

1. **Check documentation** - README, RESEARCH_PLAN
2. **Review code comments** - semua module documented
3. **Examine error messages** - biasanya self-explanatory
4. **Check GitHub issues** - jika using repository


---

**Last Updated**: December 5, 2025  
**Version**: 1.0  
**Status**: ✅ Ready to Use
