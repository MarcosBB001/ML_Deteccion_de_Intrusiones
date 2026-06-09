import sys
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from src.config import RESULTS_DIR, PLOTS_DIR

# How to run
# py -m src.plots balanced_small.json

PLOTS_DIR.mkdir(parents=True, exist_ok=True)

def generate_plots(results_file):
    RESULTS_PATH = RESULTS_DIR / results_file
    experiment_name = Path(results_file).stem

    with open(RESULTS_PATH) as f:
        results = json.load(f)

    # ── Macro metrics ─────────────────────────────────────────────────────────
    macro_metrics = ["accuracy", "precision", "recall", "f1_macro"]
    macro_rows = []
    for model, metrics in results.items():
        for metric in macro_metrics:
            macro_rows.append({"model": model, "metric": metric, "value": metrics[metric]["mean"]})

    df_macro = pd.DataFrame(macro_rows)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=df_macro, x="metric", y="value", hue="model", palette="muted", ax=ax)

    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=8, padding=2)

    ax.set_title("Model Comparison — Macro Metrics", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("")
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim(0.55, 0.82)
    ax.legend(title="Model", framealpha=0.9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)
    sns.despine()

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"{experiment_name}_macro_metrics.png", dpi=150)
    plt.show()

    # ── Per class F1 heatmap ──────────────────────────────────────────────────
    class_data = {}
    for model, metrics in results.items():
        class_data[model] = {cls: cls_metrics["f1"]["mean"] for cls, cls_metrics in metrics["per_class"].items()}

    df_heatmap = pd.DataFrame(class_data).T

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.heatmap(
        df_heatmap, annot=True, fmt=".3f", cmap="RdYlGn",
        vmin=0.4, vmax=1.0, linewidths=0.5, ax=ax, annot_kws={"size": 10}
    )

    ax.set_title("Per Class F1 Score by Model", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"{experiment_name}_per_class_f1.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    results_file = sys.argv[1] if len(sys.argv) > 1 else "model_comparison.json"
    generate_plots(results_file)