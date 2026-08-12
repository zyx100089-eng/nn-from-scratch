"""Seed sweep: is the CNN's Fashion-MNIST edge real?

The single-seed MLP-vs-CNN run shows the CNN marginally ahead on
Fashion-MNIST.  This script re-runs both architectures across several
seeds to see whether that margin survives.  Run:

    SUBSET=12000 python3 experiments/cnn_seed_sweep.py
"""

import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from src.dataset import load_fashion_mnist, to_image_shape
from src.nn import (Conv2D, Dropout, Flatten, Linear, MaxPool2D, ReLU,
                    Sequential)
from src.optimisers import Adam
from src.train import train

os.makedirs("results", exist_ok=True)

SUBSET = int(os.environ.get("SUBSET", "0"))
SEEDS = [42, 7, 123]


def build_cnn(seed):
    return Sequential(Conv2D(1, 8, 3, stride=1, rng=np.random.default_rng(seed)),
                      ReLU(),
                      MaxPool2D(2),
                      Conv2D(8, 16, 3, stride=1, rng=np.random.default_rng(seed + 1)),
                      ReLU(),
                      MaxPool2D(2),
                      Flatten(),
                      Linear(16 * 5 * 5, 64, rng=np.random.default_rng(seed + 2)),
                      ReLU(),
                      Dropout(0.25, seed=seed),
                      Linear(64, 10, rng=np.random.default_rng(seed + 3)))


def build_mlp(seed):
    return Sequential(Linear(784, 256, rng=np.random.default_rng(seed)),
                      ReLU(),
                      Linear(256, 128, rng=np.random.default_rng(seed + 1)),
                      ReLU(),
                      Linear(128, 10, rng=np.random.default_rng(seed + 2)))


def run():
    print("Loading Fashion-MNIST...")
    X_train, y_train = load_fashion_mnist("train")
    X_test, y_test = load_fashion_mnist("test")
    if SUBSET:
        X_train, y_train = X_train[:SUBSET], y_train[:SUBSET]
    X_train_img = to_image_shape(X_train)
    X_test_img = to_image_shape(X_test)

    rows = []
    for seed in SEEDS:
        print(f"\n=== seed {seed} ===")
        mlp = build_mlp(seed)
        train(mlp, X_train, y_train, epochs=5, batch_size=64,
              optimiser=Adam(learning_rate=0.001),
              X_val=X_test, y_val=y_test, verbose=False)
        acc_mlp = float((mlp.predict(X_test) == y_test).mean())

        cnn = build_cnn(seed)
        train(cnn, X_train_img, y_train, epochs=5, batch_size=64,
              optimiser=Adam(learning_rate=0.001),
              X_val=X_test_img, y_val=y_test, verbose=False)
        acc_cnn = float((cnn.predict(X_test_img) == y_test).mean())
        print(f"  MLP {acc_mlp:.4f}   CNN {acc_cnn:.4f}   diff {acc_cnn - acc_mlp:+.4f}")
        rows.append({"seed": seed, "mlp": round(acc_mlp, 4),
                     "cnn": round(acc_cnn, 4)})

    with open("results/cnn_seed_sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "mlp", "cnn"])
        w.writeheader()
        w.writerows(rows)

    mlp_accs = [r["mlp"] for r in rows]
    cnn_accs = [r["cnn"] for r in rows]
    print(f"\nMLP: {np.mean(mlp_accs):.4f} ± {np.std(mlp_accs):.4f}")
    print(f"CNN: {np.mean(cnn_accs):.4f} ± {np.std(cnn_accs):.4f}")
    print("Saved results/cnn_seed_sweep.csv")


if __name__ == "__main__":
    run()
