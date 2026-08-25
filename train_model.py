"""
train_model.py
================
Trains:
  1. placement_classifier  -> RandomForestClassifier  (Placed: 0/1)
  2. package_regressor     -> RandomForestRegressor    (Package_LPA, placed students only)

Saves into models/:
  placement_classifier.pkl, package_regressor.pkl, scaler.pkl,
  label_encoders.pkl, feature_columns.pkl, metrics.json

Run:  python train_model.py
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_absolute_error, r2_score

BASE = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE, "data", "student_employability_dataset.csv")
MODEL_DIR = os.path.join(BASE, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

CATEGORICAL_FEATURES = ["Gender", "Branch", "Placement_Training"]
NUMERIC_FEATURES = [
    "CGPA", "Internships", "Projects", "Certifications", "Aptitude_Score",
    "Technical_Skill", "Coding_Skill", "Communication_Skill", "Soft_Skill",
    "Backlogs", "Extracurricular_Score", "Leadership_Score", "LinkedIn_GitHub_Activity",
]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# ---- encode categoricals ----------------------------------------------------
encoders = {}
X = df[FEATURE_COLUMNS].copy()
for col in CATEGORICAL_FEATURES:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le

y_clf = df["Placed"].values
y_reg = df["Package_LPA"].values

X_train, X_test, yclf_train, yclf_test, yreg_train, yreg_test = train_test_split(
    X, y_clf, y_reg, test_size=0.2, random_state=42, stratify=y_clf
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# ---- classifier --------------------------------------------------------------
clf = RandomForestClassifier(
    n_estimators=300, max_depth=8, min_samples_leaf=4,
    random_state=42, class_weight="balanced"
)
clf.fit(X_train_s, yclf_train)
clf_pred = clf.predict(X_test_s)
clf_proba = clf.predict_proba(X_test_s)[:, 1]

acc = accuracy_score(yclf_test, clf_pred)
f1 = f1_score(yclf_test, clf_pred)
auc = roc_auc_score(yclf_test, clf_proba)

# ---- regressor (trained only on placed students) -----------------------------
placed_mask_train = yclf_train == 1
placed_mask_test = yclf_test == 1

reg = RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=4, random_state=42)
reg.fit(X_train_s[placed_mask_train], yreg_train[placed_mask_train])
reg_pred = reg.predict(X_test_s[placed_mask_test])
mae = mean_absolute_error(yreg_test[placed_mask_test], reg_pred)
r2 = r2_score(yreg_test[placed_mask_test], reg_pred)

print(f"Classifier -> accuracy: {acc:.3f}  f1: {f1:.3f}  roc_auc: {auc:.3f}")
print(f"Regressor  -> MAE: {mae:.2f} LPA   R2: {r2:.3f}")

# ---- save artifacts ------------------------------------------------------------
joblib.dump(clf, os.path.join(MODEL_DIR, "placement_classifier.pkl"))
joblib.dump(reg, os.path.join(MODEL_DIR, "package_regressor.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
joblib.dump(encoders, os.path.join(MODEL_DIR, "label_encoders.pkl"))
joblib.dump(FEATURE_COLUMNS, os.path.join(MODEL_DIR, "feature_columns.pkl"))

feature_importances = dict(zip(FEATURE_COLUMNS, clf.feature_importances_.tolist()))

metrics = {
    "accuracy": round(float(acc), 4),
    "f1_score": round(float(f1), 4),
    "roc_auc": round(float(auc), 4),
    "package_mae": round(float(mae), 3),
    "package_r2": round(float(r2), 4),
    "feature_importances": feature_importances,
    "n_train": int(len(X_train)),
    "n_test": int(len(X_test)),
}
with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

print("Saved all artifacts to", MODEL_DIR)
