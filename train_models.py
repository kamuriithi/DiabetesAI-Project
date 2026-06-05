"""
Train and save all models, scalers, SHAP explainers, and metrics.
Run this once before launching the dashboard.
"""

import pandas as pd
import numpy as np
import joblib
import warnings
import os
import json
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve, confusion_matrix)
import shap

os.makedirs("models", exist_ok=True)

# ── Data ──────────────────────────────────────────────────────────────────────
print("Loading data...")
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
cols = ["Pregnancies","Glucose","BloodPressure","SkinThickness",
        "Insulin","BMI","DiabetesPedigreeFunction","Age","Outcome"]
try:
    df = pd.read_csv(url, names=cols)
    print("  Loaded from GitHub.")
except Exception:
    df = pd.read_csv("diabetes.csv")
    print("  Loaded locally.")

# ── Preprocessing ─────────────────────────────────────────────────────────────
invalid_cols = ["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]
for col in invalid_cols:
    df[col] = df[col].replace(0, np.nan)
    df[col].fillna(df[col].median(), inplace=True)
df = df.dropna()

X = df.drop("Outcome", axis=1)
y = df["Outcome"]
feature_names = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42)

joblib.dump((X_test, y_test), "models/test_data.pkl")
joblib.dump(feature_names, "models/feature_names.pkl")
print("  Preprocessing done.")

# ── Model definitions ─────────────────────────────────────────────────────────
model_configs = {
    "Logistic Regression": {
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=500))
        ]),
        "params": {"model__C": [0.01, 0.1, 1, 10, 100], "model__penalty": ["l2"]},
    },
    "KNN": {
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier())
        ]),
        "params": {"model__n_neighbors": [3,5,7,9,11,13],
                   "model__weights": ["uniform","distance"]},
    },
    "SVM": {
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(probability=True))
        ]),
        "params": {"model__kernel": ["linear","rbf"],
                   "model__C": [0.1,1,10,100],
                   "model__gamma": ["scale",0.1,0.01]},
    },
}

metrics_all = {}
roc_data = {}

for name, cfg in model_configs.items():
    print(f"\nTraining {name}...")
    grid = GridSearchCV(cfg["pipeline"], cfg["params"], cv=5, scoring="roc_auc")
    grid.fit(X_train, y_train)
    best = grid.best_estimator_
    print(f"  Best params: {grid.best_params_}")

    safe_name = name.lower().replace(" ", "_")
    joblib.dump(best, f"models/{safe_name}.pkl")

    pred = best.predict(X_test)
    prob = best.predict_proba(X_test)[:, 1]

    metrics_all[name] = {
        "Accuracy":  round(accuracy_score(y_test, pred), 4),
        "Precision": round(precision_score(y_test, pred), 4),
        "Recall":    round(recall_score(y_test, pred), 4),
        "F1 Score":  round(f1_score(y_test, pred), 4),
        "ROC-AUC":   round(roc_auc_score(y_test, prob), 4),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist()
    }

    fpr, tpr, _ = roc_curve(y_test, prob)
    roc_data[name] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(),
                       "auc": metrics_all[name]["ROC-AUC"]}

    # ── SHAP ──────────────────────────────────────────────────────────────────
    print(f"  Computing SHAP for {name}...")
    background_arr = np.array(shap.sample(X_train, 80, random_state=42))
    X_test_arr = np.array(X_test[:80])

    # Use a module-level picklable wrapper
    class ModelWrapper:
        def __init__(self, m, cols):
            self.m = m
            self.cols = cols
        def __call__(self, X):
            return self.m.predict_proba(pd.DataFrame(X, columns=self.cols))

    wrapper = ModelWrapper(best, feature_names)
    explainer = shap.KernelExplainer(wrapper, background_arr)
    shap_vals = explainer.shap_values(X_test_arr, nsamples=100)

    if isinstance(shap_vals, list):
        sv_positive = shap_vals[1]
    else:
        sv_positive = shap_vals[:, :, 1]

    importance = np.abs(sv_positive).mean(axis=0)
    fi_df = pd.DataFrame({"Feature": feature_names, "Mean_SHAP": importance.tolist()})
    fi_df = fi_df.sort_values("Mean_SHAP", ascending=False)

    exp_val = explainer.expected_value
    ev = float(exp_val[1]) if hasattr(exp_val, '__len__') else float(exp_val)

    joblib.dump({
        "expected_value": ev,
        "shap_values_positive": sv_positive,
        "X_test_sample": pd.DataFrame(X_test_arr, columns=feature_names),
        "feature_importance": fi_df,
    }, f"models/{safe_name}_shap.pkl")

    print(f"  SHAP saved for {name}.")

with open("models/metrics.json", "w") as f:
    json.dump(metrics_all, f, indent=2)
with open("models/roc_data.json", "w") as f:
    json.dump(roc_data, f, indent=2)

print("\n✅ All models trained and saved to ./models/")
