import sys
import json
import numpy as np
import polars as pl
import logging
import warnings

from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich import print as rprint

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.data_loader import load_lazy
from src.data_selection import sample_balanced
from src.data_preprocessing import scale_features, encode_protocol_type
from src.config import CIC_TRAIN_PATH, RESULTS_DIR, EXPERIMENTS_DIR
from src.utils import load_config

# How to run
# py -m src.pipeline balanced_small.yaml

logging.getLogger("lightgbm").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message="X does not have valid feature names")
console = Console()

def run_pipeline(experiment_file):
    cfg = load_config(EXPERIMENTS_DIR / experiment_file)

    TEST_SIZE = cfg["experiment"]["test_size"]
    N_PER_CLASS = cfg["experiment"]["n_per_class"]
    N_ITERATIONS = cfg["experiment"]["n_iterations"]
    SEED = cfg["experiment"]["seed"]
    COLUMNS_TO_DROP = cfg["features"]["to_drop"]
    COLUMNS_TO_SCALE = cfg["features"]["to_scale"]
    LABEL_COL = cfg["features"]["target"]

    experiment_name = Path(experiment_file).stem
    RESULTS_PATH = RESULTS_DIR / f"{experiment_name}.json"

    MODELS = {
        "LogisticRegression": LogisticRegression(max_iter=1000, solver="lbfgs", random_state=SEED),
        "LinearSVC": LinearSVC(max_iter=1000, random_state=SEED),
        "RandomForest": RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=SEED),
        "XGBoost": XGBClassifier(n_estimators=100, eval_metric="mlogloss", n_jobs=-1, random_state=SEED),
        "LightGBM": LGBMClassifier(n_estimators=100, n_jobs=-1, random_state=SEED, verbose=-1),
        "KNN": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    }

    # ── Load & prepare data ───────────────────────────────────────────────────
    console.rule("[bold blue]Loading Data")
    with console.status("Loading and sampling data..."):
        train_df = load_lazy(CIC_TRAIN_PATH)
        df_balanced = sample_balanced(train_df, label_col=LABEL_COL, n_per_class=N_PER_CLASS, seed=SEED)
        df_balanced = df_balanced.drop([col for col in COLUMNS_TO_DROP if col in df_balanced.columns])
        df_balanced = encode_protocol_type(df_balanced)

        feature_cols = [col for col in df_balanced.columns if col != LABEL_COL]
        x = df_balanced.select(feature_cols).to_numpy()
        y = df_balanced[LABEL_COL].to_numpy()

        classes = sorted(set(y))
        class_to_idx = {c: i for i, c in enumerate(classes)}
        y_encoded = np.array([class_to_idx[c] for c in y])

    console.print(f"[green]✓[/green] Dataset ready — shape: {x.shape}, classes: {len(classes)}")

    # ── Evaluation loop ───────────────────────────────────────────────────────
    console.rule("[bold blue]Training")

    results = {
        name: {"accuracy": [], "precision": [], "recall": [], "f1_macro": [],
            "per_class": {c: {"precision": [], "recall": [], "f1": []} for c in classes}
        }
        for name in MODELS
    }

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), TextColumn("{task.completed}/{task.total}"), TimeElapsedColumn(), console=console) as progress:
        iter_task = progress.add_task("Iterations", total=N_ITERATIONS)
        model_task = progress.add_task("Models    ", total=len(MODELS))

        for i in range(N_ITERATIONS):
            progress.update(iter_task, completed=i + 1, description=f"Iteration {i + 1}/{N_ITERATIONS}")
            progress.reset(model_task)

            x_train, x_test, y_train, y_test = train_test_split(
                x, y_encoded, test_size=TEST_SIZE, random_state=SEED + i, stratify=y_encoded
            )
            x_train_scaled, x_test_scaled = scale_features(x_train, x_test, feature_cols, COLUMNS_TO_SCALE)

            for j, (model_name, model) in enumerate(MODELS.items()):
                progress.update(model_task, completed=j + 1, description=f"  {model_name:<25}")

                needs_scaling = model_name in ("LogisticRegression", "KNN", "LinearSVC")
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

    # ── Aggregate & save ──────────────────────────────────────────────────────
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

    # ── Print results table ───────────────────────────────────────────────────
    console.rule("[bold blue]Results")
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Model")
    table.add_column("Accuracy", justify="right")
    table.add_column("Precision", justify="right")
    table.add_column("Recall", justify="right")
    table.add_column("F1 Macro", justify="right")

    for model_name, metrics in summary.items():
        table.add_row(
            model_name,
            f"{metrics['accuracy']['mean']:.4f}",
            f"{metrics['precision']['mean']:.4f}",
            f"{metrics['recall']['mean']:.4f}",
            f"{metrics['f1_macro']['mean']:.4f}",
        )

    console.print(table)
    console.print(f"\n[green]✓[/green] Results saved to [bold]{RESULTS_PATH}[/bold]")

    return experiment_name

if __name__ == "__main__":
    experiment_file = sys.argv[1] if len(sys.argv) > 1 else "default.yaml"
    run_pipeline(experiment_file)   