"""Experiment 2: optimisers (SGD vs SGD+Momentum vs Adam) on MNIST.

Reproduces the numbers in report section 4.2 and saves the CSV.
Run:  python3 experiments/optimiser_comparison.py
"""

import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from src.dataset import load_mnist
from src.nn import Linear, ReLU, Sequential
from src.optimisers import SGD, SGDMomentum, Adam
from src.train import train

os.makedirs("results", exist_ok=True)


def run():
    print("Loading MNIST...")
    X_train, y_train = load_mnist("train", max_n=12000)
    X_test, y_test = load_mnist("test", max_n=2000)
    configs = [
        ("SGD", SGD(learning_rate=0.1)),
        ("SGD+Momentum", SGDMomentum(learning_rate=0.1, momentum=0.9)),
        ("Adam", Adam(learning_rate=0.001)),
    ]
    rows = []
    for name, opt in configs:
        print(f"\n=== {name} ===")
        model = Sequential(Linear(784, 128, rng=np.random.default_rng(42)),
                           ReLU(),
                           Linear(128, 10, rng=np.random.default_rng(43)))
        history = train(model, X_train, y_train, epochs=10, batch_size=64,
                        optimiser=opt, X_val=X_test, y_val=y_test,
                        verbose=False)
        acc = float((model.predict(X_test) == y_test).mean())
        rows.append({
            "optimiser": name,
            "test_acc": round(acc, 4),
            "final_train_loss": round(history["train_loss"][-1], 4),
            "final_val_loss": round(history["val_loss"][-1], 4),
        })
        print(f"  test_acc = {acc:.4f}")
    with open("results/optimiser_comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print("\nSaved results/optimiser_comparison.csv")


if __name__ == "__main__":
    run()
