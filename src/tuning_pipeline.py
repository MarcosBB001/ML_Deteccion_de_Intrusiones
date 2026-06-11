import sys
import json
import numpy as np
import polars as pl

from pathlib import Path
from rich.console import Console
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier

from src.data_loader import load_lazy
from src.data_selection import sample_balanced
from src.data_preprocessing import scale_features, encode_protocol_type
from src.config import CIC_TRAIN_PATH, RESULTS_DIR, EXPERIMENTS_DIR
from src.utils import load_config, make_range

# Run with
# py -m src.tuning_pipeline tuning_15k.yaml

console = Console()
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
experiment_file = sys.argv[1] if len(sys.argv) > 1 else "tuning_15k.yaml"
cfg = load_config(EXPERIMENTS_DIR / experiment_file)

TEST_SIZE = cfg["experiment"]["test_size"]
N_PER_CLASS = cfg["experiment"]["n_per_class"]
SEED = cfg["experiment"]["seed"]
RANDOM_SEARCH_ITER = cfg["experiment"]["random_search_iter"]
CV_FOLDS = cfg["experiment"]["cv_folds"]
COLUMNS_TO_DROP = cfg["features"]["to_drop"]
COLUMNS_TO_SCALE = cfg["features"]["to_scale"]
LABEL_COL = cfg["features"]["target"]

experiment_name = Path(experiment_file).stem
RESULTS_PATH = RESULTS_DIR / f"{experiment_name}_tuning_results.json"

# ── Load & prepare data ───────────────────────────────────────────────────────
console.rule("[bold blue]Loading Data")
with console.status("Loading and sampling data..."):
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

console.print(f"[green]✓[/green] Dataset ready — shape: {x.shape}, classes: {len(classes)}")

# ── Randomized Search ─────────────────────────────────────────────────────────
console.rule("[bold blue]Randomized Search")

cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)

random_param_grid = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [3, 4, 5, 6, 7, 8],
    "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5, 7, 10],
}

random_search = RandomizedSearchCV(
    XGBClassifier(eval_metric="mlogloss", n_jobs=-1, random_state=SEED),
    param_distributions=random_param_grid,
    n_iter=RANDOM_SEARCH_ITER,
    scoring="f1_macro",
    cv=cv,
    verbose=1,
    random_state=SEED,
    n_jobs=-1,
)

with console.status("Running Randomized Search..."):
    random_search.fit(x, y_encoded)

console.print(f"[green]✓[/green] Best score (RandomizedSearch): {random_search.best_score_:.4f}")
console.print(f"Best params: {random_search.best_params_}")

# ── Grid Search ───────────────────────────────────────────────────────────────
console.rule("[bold blue]Grid Search")

best = random_search.best_params_

# Create a grid search around the best parameters found in randomzied search
grid_param_grid = {
    "n_estimators": make_range(best["n_estimators"], [100, 200, 300, 500]),
    "max_depth": make_range(best["max_depth"], [3, 4, 5, 6, 7, 8]),
    "learning_rate": make_range(best["learning_rate"], [0.01, 0.05, 0.1, 0.2, 0.3]),
    "subsample": make_range(best["subsample"], [0.6, 0.7, 0.8, 0.9, 1.0]),
    "colsample_bytree": make_range(best["colsample_bytree"], [0.6, 0.7, 0.8, 0.9, 1.0]),
    "min_child_weight": make_range(best["min_child_weight"], [1, 3, 5, 7, 10]),
}

console.print(f"Grid search space: {grid_param_grid}")

grid_search = GridSearchCV(
    XGBClassifier(eval_metric="mlogloss", n_jobs=-1, random_state=SEED),
    param_grid=grid_param_grid,
    scoring="f1_macro",
    cv=cv,
    verbose=1,
    n_jobs=-1,
)

with console.status("Running Grid Search..."):
    grid_search.fit(x, y_encoded)

console.print(f"[green]✓[/green] Best score (GridSearch): {grid_search.best_score_:.4f}")
console.print(f"Best params: {grid_search.best_params_}")

# ── Save results ──────────────────────────────────────────────────────────────
results = {
    "random_search": {
        "best_score": round(grid_search.best_score_, 4),
        "best_params": random_search.best_params_,
    },
    "grid_search": {
        "best_score": round(grid_search.best_score_, 4),
        "best_params": grid_search.best_params_,
    },
}

with open(RESULTS_PATH, "w") as f:
    json.dump(results, f, indent=2)

console.print(f"\n[green]✓[/green] Results saved to [bold]{RESULTS_PATH}[/bold]")