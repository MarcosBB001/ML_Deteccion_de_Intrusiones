import sys
import json
import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score, recall_score

from src.data_loader import load_lazy
from src.data_selection import sample_balanced
from src.data_preprocessing import scale_features, encode_protocol_type
from src.config import CIC_TRAIN_PATH, RESULTS_DIR, EXPERIMENTS_DIR
from src.utils import load_config

# Run script
# py -m src.mlp_pipeline mlp_15k.yaml

console = Console()
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
experiment_file = sys.argv[1] if len(sys.argv) > 1 else "mlp_15k.yaml"
cfg = load_config(EXPERIMENTS_DIR / experiment_file)

TEST_SIZE = cfg["experiment"]["test_size"]
N_PER_CLASS = cfg["experiment"]["n_per_class"]
SEED = cfg["experiment"]["seed"]
EPOCHS = cfg["experiment"]["epochs"]
BATCH_SIZE = cfg["experiment"]["batch_size"]
PATIENCE = cfg["experiment"]["patience"]
COLUMNS_TO_DROP = cfg["features"]["to_drop"]
COLUMNS_TO_SCALE = cfg["features"]["to_scale"]
LABEL_COL = cfg["features"]["target"]

experiment_name = Path(experiment_file).stem
RESULTS_PATH = RESULTS_DIR / f"{experiment_name}_results.json"

torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
console.print(f"[green]✓[/green] Using device: [bold]{device}[/bold]")

# ── MLP ───────────────────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.network(x)

# ── Load & prepare data ───────────────────────────────────────────────────────
console.rule("[bold blue]Loading Data")
with console.status("Loading and sampling data..."):
    train_df = load_lazy(CIC_TRAIN_PATH)
    df_balanced = sample_balanced(train_df, label_col=LABEL_COL, n_per_class=N_PER_CLASS, seed=SEED)
    df_balanced = df_balanced.drop([col for col in COLUMNS_TO_DROP if col in df_balanced.columns])

    if "Protocol Type" in df_balanced.columns:
        df_balanced = encode_protocol_type(df_balanced)

    feature_cols = [col for col in df_balanced.columns if col != LABEL_COL]
    x = df_balanced.select(feature_cols).to_numpy().astype(np.float32)
    y = df_balanced[LABEL_COL].to_numpy()

    classes = sorted(set(y))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_encoded = np.array([class_to_idx[c] for c in y], dtype=np.int64)

console.print(f"[green]✓[/green] Dataset ready — shape: {x.shape}, classes: {len(classes)}")

# ── Split & scale ─────────────────────────────────────────────────────────────
x_train, x_test, y_train, y_test = train_test_split(
    x, y_encoded, test_size=TEST_SIZE, random_state=SEED, stratify=y_encoded
)
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.15, random_state=SEED, stratify=y_train
)

x_train, x_test = scale_features(x_train, x_test, feature_cols, COLUMNS_TO_SCALE)
x_train, x_val = scale_features(x_train, x_val, feature_cols, COLUMNS_TO_SCALE)

# ── DataLoaders ───────────────────────────────────────────────────────────────
def make_loader(x, y, shuffle=False):
    dataset = TensorDataset(torch.tensor(x), torch.tensor(y))
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle)

train_loader = make_loader(x_train, y_train, shuffle=True)
val_loader = make_loader(x_val, y_val)
test_loader = make_loader(x_test, y_test)

# ── Train ─────────────────────────────────────────────────────────────────────
console.rule("[bold blue]Training")

model = MLP(input_dim=len(feature_cols), num_classes=len(classes)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

best_val_loss = float("inf")
patience_counter = 0
best_model_state = None
history = {"train_loss": [], "val_loss": [], "val_f1": []}

with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), TextColumn("{task.completed}/{task.total}"), TimeElapsedColumn(), console=console) as progress:
    epoch_task = progress.add_task("Epochs", total=EPOCHS)

    for epoch in range(EPOCHS):
        # Train
        model.train()
        train_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0
        val_preds, val_targets = [], []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                out = model(x_batch)
                val_loss += criterion(out, y_batch).item()
                val_preds.extend(out.argmax(dim=1).cpu().numpy())
                val_targets.extend(y_batch.cpu().numpy())
        val_loss /= len(val_loader)
        val_f1 = f1_score(val_targets, val_preds, average="macro", zero_division=0)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)

        progress.update(epoch_task, advance=1,
            description=f"Epoch {epoch + 1}/{EPOCHS} — val_loss: {val_loss:.4f} — val_f1: {val_f1:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                console.print(f"[yellow]Early stopping at epoch {epoch + 1}[/yellow]")
                break

# ── Evaluate ──────────────────────────────────────────────────────────────────
console.rule("[bold blue]Evaluation")

model.load_state_dict(best_model_state)
model.eval()

all_preds, all_targets = [], []
with torch.no_grad():
    for x_batch, y_batch in test_loader:
        x_batch = x_batch.to(device)
        all_preds.extend(model(x_batch).argmax(dim=1).cpu().numpy())
        all_targets.extend(y_batch.numpy())

report = classification_report(all_targets, all_preds, target_names=classes, zero_division=0)
console.print(report)

summary = {
    "accuracy": round(accuracy_score(all_targets, all_preds), 4),
    "precision": round(precision_score(all_targets, all_preds, average="macro", zero_division=0), 4),
    "recall": round(recall_score(all_targets, all_preds, average="macro", zero_division=0), 4),
    "f1_macro": round(f1_score(all_targets, all_preds, average="macro", zero_division=0), 4),
    "per_class": {
        c: {
            "precision": round(precision_score(all_targets, all_preds, labels=[i], average="macro", zero_division=0), 4),
            "recall": round(recall_score(all_targets, all_preds, labels=[i], average="macro", zero_division=0), 4),
            "f1": round(f1_score(all_targets, all_preds, labels=[i], average="macro", zero_division=0), 4),
        }
        for i, c in enumerate(classes)
    },
    "history": history,
}

with open(RESULTS_PATH, "w") as f:
    json.dump(summary, f, indent=2)

console.print(f"\n[green]✓[/green] Results saved to [bold]{RESULTS_PATH}[/bold]")