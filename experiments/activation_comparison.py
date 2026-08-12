"""Experiment 1: activation functions (ReLU vs Sigmoid vs Tanh) on MNIST.

Reproduces the numbers in report section 4.1 and saves the CSV.
Run:  python3 experiments/activation_comparison.py
"""

import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from src.dataset import load_mnist
from src.nn import Linear, ReLU, Sequential, Sigmoid, Tanh
from src.optimisers import Adam
from src.train import train

os.makedirs("results", exist_ok=True)


def run():
    print("Loading MNIST...")
    X_train, y_train = load_mnist("train", max_n=12000)
    X_test, y_test = load_mnist("test", max_n=2000)
    rows = []
    for name, act in [("ReLU", ReLU), ("Sigmoid", Sigmoid), ("Tanh", Tanh)]:
        print(f"\n=== {name} ===")
        model = Sequential(Linear(784, 128, rng=np.random.default_rng(42)),
                           act(),
                           Linear(128, 10, rng=np.random.default_rng(43)))
        history = train(model, X_train, y_train, epochs=10, batch_size=64,
                        optimiser=Adam(learning_rate=0.001),
                        X_val=X_test, y_val=y_test, verbose=False)
        acc = float((model.predict(X_test) == y_test).mean())
        rows.append({
            "activation": name,
            "test_acc": round(acc, 4),
            "final_train_loss": round(history["train_loss"][-1], 4),
            "final_val_loss": round(history["val_loss"][-1], 4),
        })
        print(f"  test_acc = {acc:.4f}")
    with open("results/activation_comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print("\nSaved results/activation_comparison.csv")


if __name__ == "__main__":
    run()
