"""Experiment 3: deliberate overfitting, then fixing it with dropout
and L2 weight decay, on 1000 Fashion-MNIST samples.

Reproduces the numbers in report section 4.3 and saves the CSV.
Run:  python3 experiments/overfitting_comparison.py
"""

import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from src.dataset import load_fashion_mnist
from src.nn import Dropout, Linear, ReLU, Sequential
from src.optimisers import Adam
from src.train import train

os.makedirs("results", exist_ok=True)


def run():
    print("Loading Fashion-MNIST...")
    X_train, y_train = load_fashion_mnist("train", max_n=1000)
    X_test, y_test = load_fashion_mnist("test", max_n=2000)
    print(f"Using {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")

    configs = [
        ("no_reg", "No regularisation", 0.0, 0.0),
        ("dropout", "Dropout (p=0.5)", 0.5, 0.0),
        ("l2", "L2 (lambda=0.005)", 0.0, 0.005),
    ]
    rows = []
    for key, label, drop_p, l2 in configs:
        print(f"\n=== {label} ===")
        layers = [Linear(784, 256, rng=np.random.default_rng(42)), ReLU()]
        if drop_p:
            layers += [Dropout(drop_p, seed=0),
                       Linear(256, 128, rng=np.random.default_rng(43)), ReLU(),
                       Dropout(drop_p, seed=1)]
        else:
            layers += [Linear(256, 128, rng=np.random.default_rng(43)), ReLU()]
        layers += [Linear(128, 10, rng=np.random.default_rng(44))]
        model = Sequential(*layers)
        history = train(model, X_train, y_train, epochs=50, batch_size=32,
                        optimiser=Adam(learning_rate=0.001, weight_decay=l2),
                        X_val=X_test, y_val=y_test, verbose=False)
        acc = float((model.predict(X_test) == y_test).mean())
        rows.append({
            "config": key,
            "label": label,
            "test_acc": round(acc, 4),
            "final_train_loss": round(history["train_loss"][-1], 4),
            "final_val_loss": round(history["val_loss"][-1], 4),
            "train_val_gap": round(history["val_loss"][-1] - history["train_loss"][-1], 4),
        })
        print(f"  test_acc = {acc:.4f}")
    with open("results/overfitting_comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print("\nSaved results/overfitting_comparison.csv")


if __name__ == "__main__":
    run()
