import sys
import json
import numpy as np
import polars as pl

from pathlib import Path
from rich.console import Console
from rich.table import Table
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

from src.data_loader import load_lazy
from src.data_selection import sample_balanced
from src.data_preprocessing import scale_features, encode_protocol_type
from src.config import CIC_TEST_PATH, RESULTS_DIR, EXPERIMENTS_DIR, PLOTS_DIR
from src.utils import load_config

# Run script
# py -m src.test_evaluation tuning_15k.yaml

console = Console()
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
experiment_file = sys.argv[1] if len(sys.argv) > 1 else "tuning_15k.yaml"
cfg = load_config(EXPERIMENTS_DIR / experiment_file)

N_PER_CLASS = cfg["experiment"]["n_per_class"]
SEED = cfg["experiment"]["seed"]
COLUMNS_TO_DROP = cfg["features"]["to_drop"]
COLUMNS_TO_SCALE = cfg["features"]["to_scale"]
LABEL_COL = cfg["features"]["target"]

experiment_name = Path(experiment_file).stem
RESULTS_PATH = RESULTS_DIR / f"{experiment_name}_test_evaluation.json"

BEST_PARAMS = {
    "colsample_bytree": 0.6,
    "learning_rate": 0.1,
    "max_depth": 7,
    "min_child_weight": 5,
    "n_estimators": 500,
    "subsample": 1.0,
}

# ── Load train data ───────────────────────────────────────────────────────────
console.rule("[bold blue]Loading Train Data")
with console.status("Loading and sampling train data..."):
    train_df = load_lazy(CIC_TEST_PATH.parent / "train" / CIC_TEST_PATH.name) 
    # use your actual CIC_TRAIN_PATH here
    from src.config import CIC_TRAIN_PATH
    train_df = load_lazy(CIC_TRAIN_PATH)
    df_train = sample_balanced(train_df, label_col=LABEL_COL, n_per_class=N_PER_CLASS, seed=SEED)
    df_train = df_train.drop([col for col in COLUMNS_TO_DROP if col in df_train.columns])
    if "Protocol Type" in df_train.columns:
        df_train = encode_protocol_type(df_train)

    feature_cols = [col for col in df_train.columns if col != LABEL_COL]
    x_train = df_train.select(feature_cols).to_numpy()
    y_train = df_train[LABEL_COL].to_numpy()

    classes = sorted(set(y_train))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_train_encoded = np.array([class_to_idx[c] for c in y_train])

console.print(f"[green]✓[/green] Train data ready — shape: {x_train.shape}")

# ── Load test data ────────────────────────────────────────────────────────────
console.rule("[bold blue]Loading Test Data")
with console.status("Loading and sampling test data..."):
    test_df = load_lazy(CIC_TEST_PATH)
    df_test = sample_balanced(test_df, label_col=LABEL_COL, n_per_class=N_PER_CLASS, seed=SEED)
    df_test = df_test.drop([col for col in COLUMNS_TO_DROP if col in df_test.columns])
    if "Protocol Type" in df_test.columns:
        df_test = encode_protocol_type(df_test)

    x_test = df_test.select(feature_cols).to_numpy()
    y_test = df_test[LABEL_COL].to_numpy()
    y_test_encoded = np.array([class_to_idx[c] for c in y_test])

console.print(f"[green]✓[/green] Test data ready — shape: {x_test.shape}")

# ── Scale ─────────────────────────────────────────────────────────────────────
x_train_scaled, x_test_scaled = scale_features(x_train, x_test, feature_cols, COLUMNS_TO_SCALE)

# ── Train on full train set ───────────────────────────────────────────────────
console.rule("[bold blue]Training")
with console.status("Training XGBoost with best parameters..."):
    model = XGBClassifier(
        **BEST_PARAMS,
        eval_metric="mlogloss",
        n_jobs=-1,
        random_state=SEED,
    )
    model.fit(x_train, y_train_encoded)

console.print(f"[green]✓[/green] Training complete")

# ── Evaluate ──────────────────────────────────────────────────────────────────
console.rule("[bold blue]Evaluation on Test Set")

y_pred = model.predict(x_test)

report = classification_report(y_test_encoded, y_pred, target_names=classes, zero_division=0)
console.print(report)

# ── Confusion matrix ──────────────────────────────────────────────────────────
cm = confusion_matrix(y_test_encoded, y_pred)
cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(
    cm_normalized, annot=True, fmt=".2f", cmap="Blues",
    xticklabels=classes, yticklabels=classes,
    linewidths=0.5, ax=ax, annot_kws={"size": 9}
)
ax.set_title("Confusion Matrix — XGBoost (Test Set)", fontsize=14, fontweight="bold", pad=15)
ax.set_xlabel("Predicted", fontsize=11)
ax.set_ylabel("Actual", fontsize=11)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(PLOTS_DIR / f"{experiment_name}_test_confusion_matrix.png", dpi=150)
plt.show()

# ── Save results ──────────────────────────────────────────────────────────────
summary = {
    "accuracy": round(accuracy_score(y_test_encoded, y_pred), 4),
    "precision": round(precision_score(y_test_encoded, y_pred, average="macro", zero_division=0), 4),
    "recall": round(recall_score(y_test_encoded, y_pred, average="macro", zero_division=0), 4),
    "f1_macro": round(f1_score(y_test_encoded, y_pred, average="macro", zero_division=0), 4),
    "per_class": {
        c: {
            "precision": round(precision_score(y_test_encoded, y_pred, labels=[i], average="macro", zero_division=0), 4),
            "recall": round(recall_score(y_test_encoded, y_pred, labels=[i], average="macro", zero_division=0), 4),
            "f1": round(f1_score(y_test_encoded, y_pred, labels=[i], average="macro", zero_division=0), 4),
        }
        for i, c in enumerate(classes)
    }
}

with open(RESULTS_PATH, "w") as f:
    json.dump(summary, f, indent=2)

# ── Print results table ───────────────────────────────────────────────────────
table = Table(show_header=True, header_style="bold cyan")
table.add_column("Metric")
table.add_column("Score", justify="right")
table.add_row("Accuracy", f"{summary['accuracy']:.4f}")
table.add_row("Precision", f"{summary['precision']:.4f}")
table.add_row("Recall", f"{summary['recall']:.4f}")
table.add_row("F1 Macro", f"{summary['f1_macro']:.4f}")

console.print(table)
console.print(f"\n[green]✓[/green] Results saved to [bold]{RESULTS_PATH}[/bold]")