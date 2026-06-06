import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from src.config import RESULTS_DIR, PLOTS_DIR

PLOTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = RESULTS_DIR / "model_comparison.json"

with open(RESULTS_PATH) as f:
    results = json.load(f)

# ── Macro metrics comparison ──────────────────────────────────────────────────
macro_metrics = ["accuracy", "precision", "recall", "f1_macro"]
macro_rows = []
for model, metrics in results.items():
    for metric in macro_metrics:
        macro_rows.append({
            "model": model,
            "metric": metric,
            "value": metrics[metric]["mean"]
        })

df_macro = pd.DataFrame(macro_rows)

fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(data=df_macro, x="metric", y="value", hue="model", palette="muted", ax=ax)

# Add value labels on top of each bar
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
plt.savefig(PLOTS_DIR / "macro_metrics.png", dpi=150)
plt.show()

# ── Per class F1 heatmap ──────────────────────────────────────────────────────
class_data = {}
for model, metrics in results.items():
    class_data[model] = {cls: cls_metrics["f1"]["mean"] for cls, cls_metrics in metrics["per_class"].items()}

df_heatmap = pd.DataFrame(class_data).T  # models as rows, classes as columns

fig, ax = plt.subplots(figsize=(12, 5))
sns.heatmap(
    df_heatmap,
    annot=True,
    fmt=".3f",
    cmap="RdYlGn",
    vmin=0.4,
    vmax=1.0,
    linewidths=0.5,
    ax=ax,
    annot_kws={"size": 10}
)
ax.set_title("Per Class F1 Score by Model", fontsize=14, fontweight="bold", pad=15)
ax.set_xlabel("")
ax.set_ylabel("")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "per_class_f1.png", dpi=150)
plt.show()