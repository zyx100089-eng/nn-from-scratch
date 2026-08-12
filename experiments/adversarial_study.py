"""Experiment 5: the adversarial study — attacks, targeted attacks,
transfer, the defence, and the measured linearity argument.

This is the research half of the project: the framework from
experiments 1-4 is used to investigate WHY adversarial examples exist
and whether the standard defence works.  All numbers are saved to CSV.

Run:  python3 experiments/adversarial_study.py
"""

import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from src.attacks import (attack_success, fgsm, linearity_check, pgd,
                         pgd_targeted, target_success, transfer_attack)
from src.dataset import load_mnist
from src.nn import Linear, ReLU, Sequential
from src.optimisers import SGD
from src.train import train, train_adversarial

os.makedirs("results", exist_ok=True)


def build_model(seed=1):
    return Sequential(Linear(784, 128, rng=np.random.default_rng(seed)),
                      ReLU(),
                      Linear(128, 64, rng=np.random.default_rng(seed + 1)),
                      ReLU(),
                      Linear(64, 10, rng=np.random.default_rng(seed + 2)))


def run():
    print("Loading MNIST...")
    X, y = load_mnist("train", max_n=12000)
    Xt, yt = load_mnist("test", max_n=2000)
    rng = np.random.default_rng(0)
    sub = rng.choice(len(Xt), 300, replace=False)
    xs, ys = Xt[sub], yt[sub]

    print("\n=== Training the model ===")
    model = build_model()
    train(model, X, y, epochs=4, batch_size=256,
          optimiser=SGD(learning_rate=0.4, weight_decay=1e-4),
          seed=2, verbose=False)
    acc = float((model.predict(Xt) == yt).mean())
    print(f"  test accuracy: {acc:.3f}")

    print("\n=== Attack flip rates ===")
    rows = []
    for eps in (0.05, 0.1, 0.3):
        f = attack_success(model, xs, ys, eps, attack=fgsm)
        p = attack_success(model, xs, ys, eps, attack=pgd, steps=10)
        print(f"  eps={eps}: FGSM flip {f:.3f}   PGD flip {p:.3f}")
        rows.append({"eps": eps, "fgsm": round(f, 4), "pgd": round(p, 4)})
    with open("results/attack_flip_rates.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["eps", "fgsm", "pgd"])
        w.writeheader()
        w.writerows(rows)

    print("\n=== Targeted attack ===")
    target = 3
    sr = target_success(model, xs, ys, target, 0.1,
                        attack=pgd_targeted, steps=10)
    print(f"  PGD-targeted -> {target}: flip rate {sr:.3f}")

    print("\n=== The linearity argument, measured ===")
    for eps in (1e-3, 0.1):
        lc = linearity_check(model, xs[:50], ys[:50], eps=eps)
        print(f"  eps={eps}: actual/predicted logit change ratio {lc['ratio']:.2f}")

    print("\n=== The defence: adversarial training ===")
    plain = build_model()
    train(plain, X, y, epochs=4, batch_size=256,
          optimiser=SGD(learning_rate=0.4, weight_decay=1e-4),
          seed=2, verbose=False)
    robust = build_model()
    train_adversarial(robust, X, y, epochs=4, batch_size=256,
                      optimiser=SGD(learning_rate=0.2, weight_decay=1e-4),
                      eps=0.1, pgd_steps=5, seed=2, verbose=False)
    rows = []
    for eps in (0.05, 0.1, 0.3):
        s_plain = 1.0 - attack_success(plain, xs, ys, eps, attack=pgd, steps=10)
        s_robust = 1.0 - attack_success(robust, xs, ys, eps, attack=pgd, steps=10)
        print(f"  eps={eps}: plain {s_plain:.3f}   adversarially trained {s_robust:.3f}")
        rows.append({"eps": eps, "plain": round(s_plain, 4),
                     "adversarially_trained": round(s_robust, 4)})
    with open("results/adversarial_training.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["eps", "plain", "adversarially_trained"])
        w.writeheader()
        w.writerows(rows)

    print("\n=== Transfer attacks ===")
    B = Sequential(Linear(784, 256, rng=np.random.default_rng(42)),
                   ReLU(),
                   Linear(256, 10, rng=np.random.default_rng(43)))
    train(B, X, y, epochs=4, batch_size=256,
          optimiser=SGD(learning_rate=0.4, weight_decay=1e-4),
          seed=3, verbose=False)
    for eps in (0.1, 0.3):
        wb = attack_success(model, xs, ys, eps, attack=pgd, steps=10)
        tr = transfer_attack(model, B, xs, ys, eps, attack=pgd, steps=10)
        print(f"  eps={eps}: white-box {wb:.3f}   transfer {tr:.3f}")

    print("\nDone.")


if __name__ == "__main__":
    run()
