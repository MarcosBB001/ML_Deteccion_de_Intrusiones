import json
import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from src.data_loader import load_lazy
from src.data_selection import sample_balanced
from src.data_preprocessing import scale_features, encode_protocol_type
from src.config import CIC_TRAIN_PATH, RESULTS_DIR

# ── Config ────────────────────────────────────────────────────────────────────
N_ITERATIONS = 20
TEST_SIZE = 0.25
N_PER_CLASS = 15_000
LABEL_COL = "attack_class"
RESULTS_PATH = RESULTS_DIR / "model_comparison.json"

COLUMNS_TO_DROP = ["Tot sum", "Min", "Max", "Tot size", "Variance", "IPv", "LLC", "TCP", "UDP", "ICMP", "fin_count", "syn_count", 
                   "rst_count", "ack_count", "Label", "label",]
COLUMNS_TO_SCALE = ["Header_Length", "Time_To_Live", "Rate", "AVG", "Std", "IAT", "Number"]

MODELS = {
    "LogisticRegression": LogisticRegression(max_iter=1000, solver="lbfgs", random_state=1),
    "RandomForest": RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=1),
    "XGBoost": XGBClassifier(n_estimators=100, eval_metric="mlogloss", n_jobs=-1, random_state=1),
    "LightGBM": LGBMClassifier(n_estimators=100, n_jobs=-1, random_state=1, verbose=-1),
}

# ── Load & prepare data ───────────────────────────────────────────────────────
print("Loading and sampling data...")

# Load data, remove redundant columns and apply one hot enconding to "Protocol Type"
train_df = load_lazy(CIC_TRAIN_PATH)
df_balanced = sample_balanced(train_df, label_col=LABEL_COL, n_per_class=N_PER_CLASS, seed=1)
df_balanced = df_balanced.drop([col for col in COLUMNS_TO_DROP if col in df_balanced.columns])
df_balanced = encode_protocol_type(df_balanced)

# Get column names (needed after the train/test split is performed to apply scaling)
feature_cols = [col for col in df_balanced.columns if col != LABEL_COL]
x = df_balanced.select(feature_cols).to_numpy()
y = df_balanced[LABEL_COL].to_numpy()

classes = sorted(set(y))
class_to_idx = {c: i for i, c in enumerate(classes)}
y_encoded = np.array([class_to_idx[c] for c in y])

print(f"Features: {feature_cols}")
print(f"Classes: {classes}")
print(f"Dataset shape: {x.shape}")

# ── Evaluation loop ───────────────────────────────────────────────────────────
results = {
    name: {"accuracy": [], "precision": [], "recall": [], "f1_macro": [],
        "per_class": {c: {"precision": [], "recall": [], "f1": []} for c in classes}
    }
    for name in MODELS
}

for i in range(N_ITERATIONS):
    print(f"\nIteration {i + 1}/{N_ITERATIONS}")

    # Scale features
    x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size=TEST_SIZE, random_state=i, stratify=y_encoded)
    x_train_scaled, x_test_scaled = scale_features(x_train, x_test, feature_cols, COLUMNS_TO_SCALE)

    # Fit and evaluate each model
    for model_name, model in MODELS.items():
        print(f"  Training {model_name}...")

        needs_scaling = model_name in ("LogisticRegression",)
        _x_train = x_train_scaled if needs_scaling else x_train
        _x_test = x_test_scaled if needs_scaling else x_test

        model.fit(_x_train, y_train)
        y_pred = model.predict(_x_test)

        results[model_name]["accuracy"].append(accuracy_score(y_test, y_pred))
        results[model_name]["precision"].append(precision_score(y_test, y_pred, average="macro", zero_division=0))
        results[model_name]["recall"].append(recall_score(y_test, y_pred, average="macro", zero_division=0))
        results[model_name]["f1_macro"].append(f1_score(y_test, y_pred, average="macro", zero_division=0))

        report = classification_report(y_test, y_pred, target_names=classes, output_dict=True, zero_division=0)
        for c in classes:
            results[model_name]["per_class"][c]["precision"].append(report[c]["precision"])
            results[model_name]["per_class"][c]["recall"].append(report[c]["recall"])
            results[model_name]["per_class"][c]["f1"].append(report[c]["f1-score"])

# ── Aggregate & save ──────────────────────────────────────────────────────────
def aggregate(values):
    return {"mean": round(float(np.mean(values)), 4), "std": round(float(np.std(values)), 4)}

summary = {}
for model_name, metrics in results.items():
    summary[model_name] = {
        "accuracy":  aggregate(metrics["accuracy"]),
        "precision": aggregate(metrics["precision"]),
        "recall":    aggregate(metrics["recall"]),
        "f1_macro":  aggregate(metrics["f1_macro"]),
        "per_class": {
            c: {
                "precision": aggregate(metrics["per_class"][c]["precision"]),
                "recall":    aggregate(metrics["per_class"][c]["recall"]),
                "f1":        aggregate(metrics["per_class"][c]["f1"]),
            }
            for c in classes
        }
    }

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
with open(RESULTS_PATH, "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nResults saved to {RESULTS_PATH}")
print(json.dumps({k: {m: v for m, v in v.items() if m != "per_class"} for k, v in summary.items()}, indent=2))

