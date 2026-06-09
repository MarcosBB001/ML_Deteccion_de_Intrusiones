import sys
import json
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from pathlib import Path
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.data_loader import load_lazy
from src.data_selection import sample_balanced
from src.data_preprocessing import scale_features, encode_protocol_type
from src.config import CIC_TRAIN_PATH, RESULTS_DIR, EXPERIMENTS_DIR, PLOTS_DIR
from src.utils import load_config

PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
experiment_file = sys.argv[1] if len(sys.argv) > 1 else "default.yaml"
cfg = load_config(EXPERIMENTS_DIR / experiment_file)

TEST_SIZE = cfg["experiment"]["test_size"]
N_PER_CLASS = cfg["experiment"]["n_per_class"]
SEED = cfg["experiment"]["seed"]
COLUMNS_TO_DROP = cfg["features"]["to_drop"]
COLUMNS_TO_SCALE = cfg["features"]["to_scale"]
LABEL_COL = cfg["features"]["target"]

experiment_name = Path(experiment_file).stem

# MODEL = LGBMClassifier(n_estimators=100, n_jobs=-1, random_state=SEED, verbose=-1)
# MODEL_NAME = "LightGBM"

MODEL = XGBClassifier(n_estimators=100, eval_metric="mlogloss", n_jobs=-1, random_state=SEED)
MODEL_NAME = "XGBoost"

# ── Load & prepare data ───────────────────────────────────────────────────────
print("Loading and sampling data...")
train_df = load_lazy(CIC_TRAIN_PATH)
df_balanced = sample_balanced(train_df, label_col=LABEL_COL, n_per_class=N_PER_CLASS, seed=SEED)
df_balanced = df_balanced.drop([col for col in COLUMNS_TO_DROP if col in df_balanced.columns])

if "Protocol Type" in df_balanced.columns:
    df_balanced = encode_protocol_type(df_balanced)

feature_cols = [col for col in df_balanced.columns if col != LABEL_COL]
x = df_balanced.select(feature_cols).to_numpy()
y = df_balanced[LABEL_COL].to_numpy()

classes = sorted(set(y))
class_to_idx = {c: i for i, c in enumerate(classes)}
y_encoded = np.array([class_to_idx[c] for c in y])

x_train, x_test, y_train, y_test = train_test_split(
    x, y_encoded, test_size=TEST_SIZE, random_state=SEED, stratify=y_encoded
)

# ── Train ─────────────────────────────────────────────────────────────────────
print(f"Training {MODEL_NAME}...")
MODEL.fit(x_train, y_train)
y_pred = MODEL.predict(x_test)

# ── Classification report ─────────────────────────────────────────────────────
print("\nClassification Report:")
report = classification_report(y_test, y_pred, target_names=classes, zero_division=0)
print(report)

report_path = RESULTS_DIR / f"{experiment_name}_{MODEL_NAME}_report.txt"
with open(report_path, "w") as f:
    f.write(report)
print(f"Report saved to {report_path}")

# ── Confusion matrix ──────────────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(
    cm_normalized,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    xticklabels=classes,
    yticklabels=classes,
    linewidths=0.5,
    ax=ax,
    annot_kws={"size": 9}
)
ax.set_title(f"Confusion Matrix — {MODEL_NAME}", fontsize=14, fontweight="bold", pad=15)
ax.set_xlabel("Predicted", fontsize=11)
ax.set_ylabel("Actual", fontsize=11)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(PLOTS_DIR / f"{experiment_name}_{MODEL_NAME}_confusion_matrix.png", dpi=150)
plt.show()

# ── Feature importance ────────────────────────────────────────────────────────
importances = MODEL.feature_importances_
indices = np.argsort(importances)[::-1]
top_n = 20

fig, ax = plt.subplots(figsize=(10, 8))
sns.barplot(
    x=importances[indices[:top_n]],
    y=[feature_cols[i] for i in indices[:top_n]],
    palette="muted",
    ax=ax
)
ax.set_title(f"Top {top_n} Feature Importances — {MODEL_NAME}", fontsize=14, fontweight="bold", pad=15)
ax.set_xlabel("Importance", fontsize=11)
ax.set_ylabel("")
ax.xaxis.grid(True, linestyle="--", alpha=0.7)
ax.set_axisbelow(True)
sns.despine()
plt.tight_layout()
plt.savefig(PLOTS_DIR / f"{experiment_name}_{MODEL_NAME}_feature_importance.png", dpi=150)
plt.show()

# ── Learning curve ────────────────────────────────────────────────────────────

print("Computing learning curve (this may take a while)...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
train_sizes, train_scores, val_scores = learning_curve(MODEL, x, y_encoded, cv=cv, scoring="f1_macro", 
                                                       train_sizes=np.linspace(0.1, 1.0, 8), n_jobs=-1)

train_mean = train_scores.mean(axis=1)
train_std = train_scores.std(axis=1)
val_mean = val_scores.mean(axis=1)
val_std = val_scores.std(axis=1)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(train_sizes, train_mean, label="Train", marker="o")
ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15)
ax.plot(train_sizes, val_mean, label="Validation", marker="o")
ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15)
ax.set_title(f"Learning Curve — {MODEL_NAME}", fontsize=14, fontweight="bold", pad=15)
ax.set_xlabel("Training samples", fontsize=11)
ax.set_ylabel("F1 Macro", fontsize=11)
ax.legend()
ax.yaxis.grid(True, linestyle="--", alpha=0.7)
ax.set_axisbelow(True)
sns.despine()
plt.tight_layout()
plt.savefig(PLOTS_DIR / f"{experiment_name}_{MODEL_NAME}_learning_curve.png", dpi=150)
plt.show()