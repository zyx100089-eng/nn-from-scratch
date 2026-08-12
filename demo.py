"""The story of the project, in one runnable walkthrough.

1. Autodiff from scratch — verify gradients are correct (chain rule).
2. Train an honest MLP on MNIST — it reaches >90% test accuracy.
3. FGSM: one tiny gradient step flips the model's mind.  Show the
   original, the perturbation (magnified), and the adversarial image.
4. PGD: iterating + projecting makes the attack stronger at the same
   budget.
5. Why it works (Goodfellow's linearity argument): the decision
   function is locally linear, so the input changes the most along the
   gradient — and a 0.1-pixel-magnitude change in 784 dimensions
   easily crosses a boundary.
6. The defence: adversarial training.
7. Transfer: examples crafted on one model fool another.

Run:  python3 demo.py
"""

from __future__ import annotations

import os

import numpy as np
from PIL import Image

from src.attacks import (attack_success, fgsm, linearity_check, pgd,
                         pgd_targeted, target_success, transfer_attack)
from src.dataset import load_mnist
from src.nn import Linear, ReLU, Sequential
from src.optimisers import SGD
from src.train import train, train_adversarial

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def build_model(seed=1):
    return Sequential(Linear(784, 128, rng=np.random.default_rng(seed)),
                      ReLU(),
                      Linear(128, 64, rng=np.random.default_rng(seed + 1)),
                      ReLU(),
                      Linear(64, 10, rng=np.random.default_rng(seed + 2)))


def _save_grid(images: list[tuple[str, np.ndarray]], path: str, title: str) -> None:
    """Render a row of 28x28 images side by side, saved to disk."""
    os.makedirs(OUT_DIR, exist_ok=True)
    cells = []
    for label, img in images:
        arr = (np.clip(img, 0, 1).reshape(28, 28) * 255).astype(np.uint8)
        cells.append(Image.fromarray(arr, "L"))
    widths = [c.width for c in cells]
    height = max(c.height for c in cells)
    canvas = Image.new("L", (sum(widths) + 12 * len(cells), height + 12), 255)
    x = 0
    for c in cells:
        canvas.paste(c, (x + 6, 6))
        x += c.width + 12
    canvas.save(path)
    print(f"  [{title}] -> {path}")


def demo_autodiff() -> None:
    print("=" * 70)
    print("1. AUTODIFF FROM SCRATCH")
    print("   Reverse-mode autodiff = the chain rule on a computation")
    print("   graph.  Verified against numeric differentiation:")
    print("   every gradient agrees to ~1e-9.")
    print("=" * 70)
    from verify import check_autodiff
    check_autodiff()


def demo_train() -> tuple[Sequential, np.ndarray, np.ndarray]:
    print("=" * 70)
    print("2. TRAINING AN HONEST MLP ON MNIST")
    print("   (784 -> 128 -> 64 -> 10, ReLU, SGD + weight decay,")
    print("    softmax cross-entropy, all from scratch)")
    print("=" * 70)
    X, y = load_mnist("train", max_n=12000)
    Xt, yt = load_mnist("test", max_n=2000)
    model = build_model()
    train(model, X, y, epochs=4, batch_size=256,
          optimiser=SGD(learning_rate=0.4, weight_decay=1e-4),
          seed=2, verbose=True)
    acc = float((model.predict(Xt) == yt).mean())
    print(f"  -> test accuracy {acc:.3f}")
    return model, Xt, yt


def demo_fgsm(model, Xt, yt) -> None:
    print("=" * 70)
    print("3. FGSM: ONE GRADIENT STEP FLIPS THE MODEL")
    print("   eps = 0.1 means each pixel moves by at most 0.1 of its")
    print("   range - invisible to the eye, fatal to the model.")
    print("=" * 70)
    rng = np.random.default_rng(0)
    idx = int(rng.integers(0, len(Xt)))
    x = Xt[idx:idx + 1]
    true = yt[idx]
    pred = model.predict(x)[0]
    while pred != true:
        idx = int(rng.integers(0, len(Xt)))
        x = Xt[idx:idx + 1]
        true = yt[idx]
        pred = model.predict(x)[0]

    eps = 0.1
    adv = fgsm(model, x, np.array([true]), eps)
    adv_pred = model.predict(adv)[0]
    prob_before = model.softmax_probs(x)[0, true]
    prob_after = model.softmax_probs(adv)[0, adv_pred]
    delta = (adv - x).reshape(28, 28)

    print(f"  sample {idx}: true={true}, predicted={pred}")
    print(f"  FGSM eps={eps}: now predicted={adv_pred}")
    print(f"  P(true) before={prob_before:.3f}  after={prob_after:.3f}")
    _save_grid([
        ("original", x[0]),
        ("adversarial", adv[0]),
        ("|perturbation| x10", np.abs(delta) * 10),
    ], os.path.join(OUT_DIR, "fgsm.png"), "FGSM demo")
    print(f"  max |pixel change| = {np.abs(adv - x).max():.3f}")


def demo_pgd(model, Xt, yt) -> None:
    print("=" * 70)
    print("4. PGD: ITERATING + PROJECTING IS STRONGER")
    print("   Same eps budget, but 10 projected steps instead of 1.")
    print("=" * 70)
    rng = np.random.default_rng(0)
    sub = rng.choice(len(Xt), 300, replace=False)
    xs, ys = Xt[sub], yt[sub]
    for eps in (0.05, 0.1, 0.3):
        f = attack_success(model, xs, ys, eps, attack=fgsm)
        p = attack_success(model, xs, ys, eps, attack=pgd, steps=10)
        print(f"  eps={eps}: FGSM flip {f:.3f}   PGD flip {p:.3f}")


def demo_targeted(model, Xt, yt) -> None:
    print("=" * 70)
    print("5. TARGETED ATTACKS: MAKE A 4 INTO A 9 ON PURPOSE")
    print("   Untargeted attacks just want a wrong label.  Targeted")
    print("   attacks minimise the TARGET class's loss, so the image")
    print("   is steered into a chosen class.")
    print("=" * 70)
    rng = np.random.default_rng(0)
    idx = int(rng.integers(0, len(Xt)))
    x = Xt[idx:idx + 1]
    true = yt[idx]
    target = (true + 5) % 10  # a class far from the truth
    pred = model.predict(x)[0]
    while pred != true:
        idx = int(rng.integers(0, len(Xt)))
        x = Xt[idx:idx + 1]
        true = yt[idx]
        target = (true + 5) % 10
        pred = model.predict(x)[0]

    adv = pgd_targeted(model, x, np.array([target]), 0.3, steps=10)
    adv_pred = model.predict(adv)[0]
    print(f"  sample {idx}: true={true}, predicted={pred}, attacking -> {target}")
    print(f"  after PGD-targeted: predicted={adv_pred}")
    _save_grid([
        ("original", x[0]),
        (f"targeted at {target}", adv[0]),
        ("|perturbation| x10", np.abs(adv - x).reshape(28, 28) * 10),
    ], os.path.join(OUT_DIR, "targeted.png"), "targeted attack demo")
    sr = target_success(model, Xt[:500], yt[:500], target, 0.1,
                        attack=pgd_targeted, steps=10)
    print(f"  flip-into-class rate at eps=0.1: {sr:.3f}")


def demo_linearity(model, Xt, yt) -> None:
    print("=" * 70)
    print("6. THE LINEARITY ARGUMENT, MEASURED")
    print("   Goodfellow: a ReLU net is piecewise-linear in its input,")
    print("   so a perturbation of norm d changes the logits by ~grad.d,")
    print("   i.e. by O(d*sqrt(784)) in aggregate.  Measured:")
    print("=" * 70)
    rng = np.random.default_rng(0)
    sub = rng.choice(len(Xt), 50, replace=False)
    for eps in (1e-3, 0.1):
        lc = linearity_check(model, Xt[sub], yt[sub], eps=eps)
        print(f"  eps={eps}: actual/predicted logit change ratio "
              f"{lc['ratio']:.2f}")
    print("  ratio ~1 at small eps = the model really is locally")
    print("  linear, so the gradient points across a boundary nearby.")


def demo_defence(model, X, y, Xt, yt) -> None:
    print("=" * 70)
    print("7. THE DEFENCE: ADVERSARIAL TRAINING")
    print("   Train on clean + PGD-attacked minibatches.  This flattens")
    print("   the loss landscape near inputs, so small perturbations")
    print("   no longer cross a boundary.")
    print("=" * 70)
    rng = np.random.default_rng(0)
    sub = rng.choice(len(Xt), 300, replace=False)
    xs, ys = Xt[sub], yt[sub]
    robust = build_model()
    train_adversarial(robust, X, y, epochs=4, batch_size=256,
                      optimiser=SGD(learning_rate=0.2, weight_decay=1e-4),
                      eps=0.1, pgd_steps=5, seed=2, verbose=False)
    print("  survival under PGD-10 (accuracy after attack):")
    for eps in (0.05, 0.1, 0.3):
        s = 1.0 - attack_success(robust, xs, ys, eps, attack=pgd, steps=10)
        s_plain = 1.0 - attack_success(model, xs, ys, eps, attack=pgd, steps=10)
        print(f"  eps={eps}: plain {s_plain:.3f}   adversarially trained {s:.3f}")


def demo_transfer(model, X, y, Xt, yt) -> None:
    print("=" * 70)
    print("8. TRANSFER: EXAMPLES CRAFTED ON ONE MODEL FOOL ANOTHER")
    print("   A famous result: adversarial examples transfer between")
    print("   models (even different architectures), because they")
    print("   both approximate the same decision boundary.")
    print("=" * 70)
    rng = np.random.default_rng(0)
    sub = rng.choice(len(Xt), 300, replace=False)
    xs, ys = Xt[sub], yt[sub]
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


def main() -> None:
    demo_autodiff()
    model, Xt, yt = demo_train()
    demo_fgsm(model, Xt, yt)
    demo_pgd(model, Xt, yt)
    demo_targeted(model, Xt, yt)
    demo_linearity(model, Xt, yt)
    X, y = load_mnist("train", max_n=12000)
    demo_defence(model, X, y, Xt, yt)
    demo_transfer(model, X, y, Xt, yt)
    print("\nDone.")


if __name__ == "__main__":
    main()
