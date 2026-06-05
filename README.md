# DiabetesAI Pro — Clinical Decision Support Dashboard

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train models (one-time setup)
```bash
python train_models.py
```
This downloads the Pima dataset, trains LR / KNN / SVM with GridSearchCV,
computes SHAP values, and saves everything to `./models/`.

### 3. Launch dashboard
```bash
streamlit run app.py
```

---

## Sidebar Pages

| Page | Description |
|---|---|
| 🏠 Overview | Project summary, leaderboard, risk stratification guide |
| 🔮 Prediction | Input patient data, select model, get risk probability + SHAP waterfall |
| 📊 Model Performance | Metrics table, comparison bar chart, confusion matrices |
| 📈 ROC Curves | All-model ROC overlay with AUC ranking |
| 🧠 XAI — SHAP | Feature importance, beeswarm, waterfall, dependence plots |
| 🗂️ Data Explorer | Browse test data, histograms, correlation heatmap |
| 📋 About & Methods | Dataset, preprocessing, model tuning, SHAP methodology |

## Risk Stratification
| Probability | Level | Action |
|---|---|---|
| 0–30% | 🟢 Low | Annual screening |
| 31–50% | 🟡 Mild | Lifestyle changes |
| 51–70% | 🟠 Moderate | Medical review |
| 71–100% | 🔴 High | Urgent referral |

## ⚠️ Disclaimer
For educational/research purposes only. Not a substitute for clinical judgement.
