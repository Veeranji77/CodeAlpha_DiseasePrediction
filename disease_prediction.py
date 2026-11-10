# CodeAlpha ML Internship - Task 4: Disease Prediction from Medical Data
# Dataset: Breast Cancer Wisconsin (Diagnostic) - UCI ML Repository
# (comes bundled with scikit-learn, so no download is needed)

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, roc_curve)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=FutureWarning)  # SVC probability warning

OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------
# 1. Load the data
# ---------------------------------------------------------------
data = load_breast_cancer(as_frame=True)
df = data.frame.copy()

# In sklearn's version: 0 = malignant, 1 = benign.
# I flip it so that 1 = disease (malignant) which is easier to read.
df["target"] = 1 - df["target"]

print("Shape:", df.shape)
print("Missing values:", df.isnull().sum().sum())
print("Class counts (1 = malignant):")
print(df["target"].value_counts())

# ---------------------------------------------------------------
# 2. Quick EDA
# ---------------------------------------------------------------
plt.figure(figsize=(5, 4))
sns.countplot(x="target", data=df)
plt.xticks([0, 1], ["Benign", "Malignant"])
plt.title("Class distribution")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/class_distribution.png", dpi=120)
plt.close()

# correlation of each feature with the target
corr = df.corr()["target"].drop("target").sort_values()
plt.figure(figsize=(7, 8))
corr.plot(kind="barh")
plt.title("Correlation of features with malignant diagnosis")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/feature_correlation.png", dpi=120)
plt.close()

# ---------------------------------------------------------------
# 3. Train / test split
# ---------------------------------------------------------------
X = df.drop(columns="target")
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# ---------------------------------------------------------------
# 4. Models
# SVM and Logistic Regression need scaled features, so they are put
# inside a pipeline. Tree models don't need scaling.
# ---------------------------------------------------------------
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000))]),
    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", probability=True, random_state=42))]),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.1,
                             eval_metric="logloss", random_state=42),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []
fitted = {}

for name, model in models.items():
    cv_auc = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc").mean()
    model.fit(X_train, y_train)
    fitted[name] = model

    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]

    results.append({
        "Model": name,
        "CV ROC-AUC": round(cv_auc, 4),
        "Accuracy": round(accuracy_score(y_test, pred), 4),
        "Precision": round(precision_score(y_test, pred), 4),
        "Recall": round(recall_score(y_test, pred), 4),
        "F1": round(f1_score(y_test, pred), 4),
        "Test ROC-AUC": round(roc_auc_score(y_test, prob), 4),
    })

results_df = pd.DataFrame(results).sort_values("F1", ascending=False)
print("\nModel comparison:")
print(results_df.to_string(index=False))
results_df.to_csv(f"{OUT_DIR}/model_comparison.csv", index=False)

# ---------------------------------------------------------------
# 5. Look closer at the best model
# For disease prediction, recall matters a lot (missing a sick patient
# is worse than a false alarm), so I look at recall as well as F1.
# ---------------------------------------------------------------
best_name = results_df.iloc[0]["Model"]
best_model = fitted[best_name]
best_pred = best_model.predict(X_test)
print(f"\nBest model by F1: {best_name}")
print(classification_report(y_test, best_pred, target_names=["Benign", "Malignant"]))

cm = confusion_matrix(y_test, best_pred)
plt.figure(figsize=(4.5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Benign", "Malignant"],
            yticklabels=["Benign", "Malignant"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion matrix - {best_name}")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/confusion_matrix.png", dpi=120)
plt.close()

# ROC curves for all models
plt.figure(figsize=(6, 5))
for name, model in fitted.items():
    prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, prob)
    auc = roc_auc_score(y_test, prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
plt.plot([0, 1], [0, 1], "k--", label="Random guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC curves")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/roc_curves.png", dpi=120)
plt.close()

# Feature importance from Random Forest
rf = fitted["Random Forest"]
imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values().tail(10)
plt.figure(figsize=(7, 5))
imp.plot(kind="barh")
plt.title("Top 10 features - Random Forest")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/feature_importance.png", dpi=120)
plt.close()

print("\nTop 5 important features (Random Forest):")
print(imp.sort_values(ascending=False).head(5).round(4))
print(f"\nPlots and tables saved in '{OUT_DIR}/'")
