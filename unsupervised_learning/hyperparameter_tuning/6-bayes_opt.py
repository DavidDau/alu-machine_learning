#!/usr/bin/env python3
"""
Bayesian hyperparameter optimization with GPyOpt + PyTorch on MNIST.

Requirements (pip):
    pip install torch torchvision GPy GPyOpt matplotlib numpy

Run:
    python bayes_opt_gpyopt_pytorch.py
"""

import os
import math
import time
import json
import random
from functools import partial
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# GPyOpt
import GPyOpt

# ---------------------------
# Reproducibility & config
# ---------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = "bayes_opt_results"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------
# Dataset: MNIST (small subset for speed)
# ---------------------------
def get_dataloaders(batch_size, subset_frac=0.25):
    transform = transforms.Compose([transforms.ToTensor()])
    train_full = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    # Use a subset to speed up training iterations
    n_total = len(train_full)
    n_subset = max(1000, int(n_total * subset_frac))  # at least 1000
    subset_indices = list(range(n_subset))
    train_subset = Subset(train_full, subset_indices)

    # split into train and val
    val_frac = 0.2
    val_len = int(len(train_subset) * val_frac)
    train_len = len(train_subset) - val_len
    train_ds, val_ds = random_split(train_subset, [train_len, val_len])

    test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader

# ---------------------------
# Model: simple MLP with variable layers & units
# ---------------------------
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_units, n_layers, dropout, n_classes=10):
        super().__init__()
        layers = []
        in_dim = input_dim
        for i in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_units))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_units
        layers.append(nn.Linear(in_dim, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        return self.net(x)

# ---------------------------
# Training & eval utilities
# ---------------------------
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for xb, yb in dataloader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * xb.size(0)
        preds = out.argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += xb.size(0)
    return running_loss / total, correct / total

def eval_model(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            loss = criterion(out, yb)
            running_loss += loss.item() * xb.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += xb.size(0)
    return running_loss / total, correct / total

# ---------------------------
# Utility to make a safe filename encoding hyperparams
# ---------------------------
def hp_filename(prefix, hp_dict):
    parts = [f"{k}={str(v).replace('.', '_')}" for k, v in hp_dict.items()]
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = f"{prefix}__" + "__".join(parts) + f"__{stamp}.pt"
    # make safe
    name = name.replace(" ", "").replace("/", "_")
    return os.path.join(OUT_DIR, name)

# ---------------------------
# Single run: given hyperparams train & return 1 - val_acc
# ---------------------------
def single_run(hparams, max_epochs=30, patience=5, subset_frac=0.25):
    """
    hparams is a dict with:
        - lr: float
        - hidden_units: int
        - n_layers: int
        - dropout: float
        - weight_decay: float (L2)
        - batch_size: int
    Returns validation_loss_to_minimize (1 - val_accuracy), and metadata
    """
    # Unpack
    lr = float(hparams["lr"])
    hidden_units = int(hparams["hidden_units"])
    n_layers = int(hparams["n_layers"])
    dropout = float(hparams["dropout"])
    weight_decay = float(hparams["weight_decay"])
    batch_size = int(hparams["batch_size"])

    # Prepare data
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=batch_size, subset_frac=subset_frac)

    # Build model
    input_dim = 28 * 28
    model = SimpleMLP(input_dim=input_dim, hidden_units=hidden_units, n_layers=n_layers, dropout=dropout).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_acc = 0.0
    best_epoch = -1
    best_state = None
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    epochs_no_improve = 0

    for epoch in range(1, max_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_acc = eval_model(model, val_loader, criterion, DEVICE)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Checkpointing per-run: if best so far within this training session, keep state
        if val_acc > best_val_acc + 1e-12:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
                "hparams": hparams,
                "val_acc": val_acc,
                "train_acc": train_acc,
                "timestamp": datetime.now().isoformat(),
            }
            # Save checkpoint file naming hyperparams
            fname = hp_filename("checkpoint", hparams)
            try:
                torch.save(best_state, fname)
            except Exception as e:
                print("Warning: couldn't save checkpoint:", e)
        # Early stopping
        if val_acc <= best_val_acc + 1e-12:
            epochs_no_improve += 1
        else:
            epochs_no_improve = 0
        if epochs_no_improve >= patience:
            break

    # At end return the metric to minimize
    val_metric_to_minimize = 1.0 - best_val_acc  # lower is better
    result = {
        "val_metric": val_metric_to_minimize,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "history": history,
    }
    return result

# ---------------------------
# Objective wrapper for GPyOpt
# ---------------------------
def gpyopt_objective(x):
    """
    GPyOpt supplies x as a 2D numpy array of shape (1, d) for single-evaluation calls.
    We must return a 2D numpy array [[objective_value]] (minimization).
    """
    # x columns order must match domain below: lr, hidden_units, n_layers, dropout, weight_decay, batch_size_index
    values = x[0]
    lr = float(values[0])
    hidden_units = int(round(values[1]))
    n_layers = int(round(values[2]))
    dropout = float(values[3])
    weight_decay = float(values[4])
    batch_size = int(round(values[5]))  # we will encode as actual batch size from domain choices

    # Build hparams dict
    hparams = {
        "lr": lr,
        "hidden_units": hidden_units,
        "n_layers": n_layers,
        "dropout": float(dropout),
        "weight_decay": weight_decay,
        "batch_size": batch_size,
    }

    # Run single training
    start = time.time()
    res = single_run(hparams, max_epochs=30, patience=6, subset_frac=0.25)
    elapsed = time.time() - start

    # Logging
    entry = {
        "x": hparams,
        "val_metric": float(res["val_metric"]),
        "best_val_acc": float(res["best_val_acc"]),
        "best_epoch": int(res["best_epoch"]),
        "elapsed_sec": elapsed,
    }
    # append to global runs list
    runs_log.append(entry)
    print(f"[GPyOpt] tried: {hparams}, val_metric={entry['val_metric']:.4f}, best_acc={entry['best_val_acc']:.4f}, time={elapsed:.1f}s")
    # return objective in required shape
    return np.array([[entry["val_metric"]]], dtype=float)

# ---------------------------
# Main: set domain and run GPyOpt
# ---------------------------
if __name__ == "__main__":
    # Global log of runs
    runs_log = []

    # Domain: define search space for 6 hyperparameters
    # Note: GPyOpt works with continuous values; we round where needed.
    # lr: log-uniform between 1e-4 and 1e-1
    # hidden_units: between 32 and 512
    # n_layers: between 1 and 4
    # dropout: between 0.0 and 0.6
    # weight_decay: log-uniform between 1e-6 and 1e-2
    # batch_size: choose from common values -> we encode actual ints in domain with a categorical-like continuous representation
    domain = [
        {"name": "lr", "type": "continuous", "domain": (1e-4, 1e-1)},
        {"name": "hidden_units", "type": "continuous", "domain": (32, 512)},
        {"name": "n_layers", "type": "continuous", "domain": (1, 4)},
        {"name": "dropout", "type": "continuous", "domain": (0.0, 0.6)},
        {"name": "weight_decay", "type": "continuous", "domain": (1e-6, 1e-2)},
        # batch_size: we'll restrict to powers-of-two typical choices
        {"name": "batch_size", "type": "discrete", "domain": (32, 64, 128)},
    ]

    # Create Bayesian optimizer
    optimizer = GPyOpt.methods.BayesianOptimization(
        f=gpyopt_objective,
        domain=domain,
        acquisition_type="EI",   # expected improvement
        exact_feval=True,
        maximize=False,
        initial_design_numdata=5,
        verbosity=True,
    )

    # Run optimization: max 30 iterations (after initial points)
    max_iter = 30
    print("Starting Bayesian optimization (GPyOpt). This may take a while...")
    optimizer.run_optimization(max_iter=max_iter)

    # After optimization: collect best
    best_x = optimizer.x_opt
    best_val = optimizer.fx_opt
    best_hp = {
        "lr": float(best_x[0]),
        "hidden_units": int(round(best_x[1])),
        "n_layers": int(round(best_x[2])),
        "dropout": float(best_x[3]),
        "weight_decay": float(best_x[4]),
        "batch_size": int(round(best_x[5])),
    }

    summary = {
        "best_hyperparameters": best_hp,
        "best_val_metric (1 - val_acc)": float(best_val),
        "best_val_accuracy": float(1.0 - best_val),
        "num_evaluations": len(runs_log),
        "runs": runs_log,
    }

    # Save report
    report_path = os.path.join(OUT_DIR, "bayes_opt.txt")
    with open(report_path, "w") as f:
        f.write("Bayesian Optimization Report\n")
        f.write("============================\n")
        f.write(f"Date: {datetime.now().isoformat()}\n")
        f.write(json.dumps(summary, indent=2))
    print(f"Saved report to {report_path}")

    # Plot convergence (objective vs iteration)
    iters = list(range(1, len(optimizer.Y).flatten().shape[0] + 1))
    y_vals = np.array(optimizer.Y).flatten()  # objective (to minimize)

    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(y_vals) + 1), y_vals, marker="o")
    plt.xlabel("Iteration")
    plt.ylabel("Objective (1 - val_acc) to minimize")
    plt.title("Bayesian Optimization Convergence")
    plt.grid(True)
    conv_path = os.path.join(OUT_DIR, "convergence.png")
    plt.savefig(conv_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved convergence plot to {conv_path}")

    # Save the runs log as JSON
    runs_json = os.path.join(OUT_DIR, "runs_log.json")
    with open(runs_json, "w") as f:
        json.dump(runs_log, f, indent=2)
    print(f"Saved runs log to {runs_json}")

    # Print summary to console
    print("\nOptimization finished.")
    print("Best hyperparameters found:")
    for k, v in best_hp.items():
        print(f"  {k}: {v}")
    print(f"Best validation accuracy: {1.0 - best_val:.4f}")
    print(f"Report: {report_path}")
    print(f"Convergence: {conv_path}")
