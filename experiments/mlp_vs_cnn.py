"""Experiment 4: MLP vs CNN on MNIST and Fashion-MNIST.

Both architectures run on the SAME autodiff engine — the CNN's conv
and pooling layers are graph ops like everything else.  The question:
does spatial weight sharing beat a dense network on images, and by how
much per parameter?

Pure-NumPy convolution is CPU-bound.  Set SUBSET to limit the training
set for a quick run; leave unset for the full dataset.

Run:  python3 experiments/mlp_vs_cnn.py
"""

import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from src.dataset import load_fashion_mnist, load_mnist, to_image_shape
from src.nn import (Conv2D, Dropout, Flatten, Linear, MaxPool2D, ReLU,
                    Sequential)
from src.optimisers import Adam
from src.train import train

os.makedirs("results", exist_ok=True)

SUBSET = int(os.environ.get("SUBSET", "0"))  # 0 = use all


def maybe_subset(X, y):
    if SUBSET and X.shape[0] > SUBSET:
        return X[:SUBSET], y[:SUBSET]
    return X, y


def build_cnn():
    return Sequential(
        Conv2D(1, 8, 3, stride=1, rng=np.random.default_rng(42)),
        ReLU(),
        MaxPool2D(2),
        Conv2D(8, 16, 3, stride=1, rng=np.random.default_rng(43)),
        ReLU(),
        MaxPool2D(2),
        Flatten(),
        Linear(16 * 5 * 5, 64, rng=np.random.default_rng(44)),
        ReLU(),
        Dropout(0.25, seed=0),
        Linear(64, 10, rng=np.random.default_rng(45)),
    )


def build_mlp():
    return Sequential(
        Linear(784, 256, rng=np.random.default_rng(42)),
        ReLU(),
        Linear(256, 128, rng=np.random.default_rng(43)),
        ReLU(),
        Linear(128, 10, rng=np.random.default_rng(44)),
    )


def run_dataset(name, loader):
    print(f"\n{'#' * 60}\n# {name}\n{'#' * 60}")
    X_train_flat, y_train = loader("train")
    X_test_flat, y_test = loader("test")
    X_train_flat, y_train = maybe_subset(X_train_flat, y_train)
    print(f"Train: {X_train_flat.shape}, Test: {X_test_flat.shape}")

    X_train_img = to_image_shape(X_train_flat)
    X_test_img = to_image_shape(X_test_flat)

    print("\n--- MLP ---")
    mlp = build_mlp()
    train(mlp, X_train_flat, y_train, epochs=10, batch_size=64,
          optimiser=Adam(learning_rate=0.001),
          X_val=X_test_flat, y_val=y_test, verbose=False)
    acc_mlp = float((mlp.predict(X_test_flat) == y_test).mean())
    print(f"MLP test_acc: {acc_mlp:.4f}")

    print("\n--- CNN ---")
    cnn = build_cnn()
    train(cnn, X_train_img, y_train, epochs=10, batch_size=64,
          optimiser=Adam(learning_rate=0.001),
          X_val=X_test_img, y_val=y_test, verbose=False)
    acc_cnn = float((cnn.predict(X_test_img) == y_test).mean())
    print(f"CNN test_acc: {acc_cnn:.4f}")

    n_mlp = sum(p.data.size for p in mlp.parameters())
    n_cnn = sum(p.data.size for p in cnn.parameters())
    print(f"Parameters: MLP {n_mlp}, CNN {n_cnn} ({n_mlp / n_cnn:.1f}x fewer)")

    with open("results/mlp_vs_cnn_results.csv", "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([name, "MLP", f"{acc_mlp:.4f}", n_mlp])
        w.writerow([name, "CNN", f"{acc_cnn:.4f}", n_cnn])
    return acc_mlp, acc_cnn


if __name__ == "__main__":
    with open("results/mlp_vs_cnn_results.csv", "w", newline="") as f:
        csv.writer(f).writerow(["dataset", "architecture", "test_acc", "params"])
    run_dataset("MNIST", load_mnist)
    run_dataset("Fashion-MNIST", load_fashion_mnist)
    print("\nSaved results/mlp_vs_cnn_results.csv")
