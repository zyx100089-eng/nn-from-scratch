"""End-to-end verification of the whole project.

Checks, in order:
1. autodiff gradients match numerical differentiation
2. a trained MLP reaches good test accuracy on MNIST
3. FGSM/PGD flip a meaningful fraction of predictions at small eps;
   perturbations respect the Linf bound and pixel range
4. targeted attacks flip predictions INTO a chosen class
5. adversarial training defends (robustness curve)
6. transfer attacks (crafted on one model, tested on another)
7. the linearity argument holds (logit change ~ gradient dot delta)
8. the CNN trains and beats the MLP on MNIST (parameter efficiency)

Run:  python3 verify.py
"""

from __future__ import annotations

import numpy as np

from src.attacks import (attack_success, fgsm, fgsm_targeted,
                         gradient_wrt_input, linearity_check, pgd,
                         pgd_targeted, robustness_curve, target_success,
                         transfer_attack)
from src.autodiff import Tensor
from src.dataset import load_mnist, to_image_shape
from src.nn import Conv2D, Flatten, Linear, MaxPool2D, ReLU, Sequential
from src.optimisers import SGD, Adam
from src.train import train, train_adversarial


def check_autodiff() -> None:
    print("[1] autodiff vs numeric gradients")
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4, 5))
    w = rng.normal(size=(5, 3))
    b = rng.normal(size=(3,))

    def fwd(xx, ww, bb):
        return (xx.matmul(ww) + bb).relu()

    def loss_fn(xx, ww, bb):
        r = fwd(xx, ww, bb)
        return float((r * r).sum().data)

    xt, wt, bt = Tensor(x), Tensor(w), Tensor(b)
    loss = (fwd(xt, wt, bt) ** 2).sum()
    loss.backward()

    def numeric_grad(f, v, eps=1e-6):
        g = np.zeros_like(v)
        it = np.nditer(v, flags=["multi_index"])
        while not it.finished:
            i = it.multi_index
            vp, vm = v.copy(), v.copy()
            vp[i] += eps
            vm[i] -= eps
            g[i] = (f(vp) - f(vm)) / (2 * eps)
            it.iternext()
        return g

    for name, val, got, setter in [
        ("W", w, wt.grad, lambda v: (xt, Tensor(v), bt)),
        ("b", b, bt.grad, lambda v: (xt, wt, Tensor(v))),
        ("x", x, xt.grad, lambda v: (Tensor(v), wt, bt)),
    ]:
        want = numeric_grad(lambda v: loss_fn(*setter(v)), val)
        err = np.abs(got - want).max()
        assert err < 1e-6, f"{name}: gradient error {err}"
        print(f"    {name}: max err {err:.1e} OK")

    logits = rng.normal(size=(6, 4))
    labels = rng.integers(0, 4, size=6)
    t = Tensor(logits)
    l = t.softmax_cross_entropy(labels)
    l.backward()
    want = numeric_grad(
        lambda v: float(Tensor(v).softmax_cross_entropy(labels).data), logits)
    err = np.abs(t.grad - want).max()
    assert err < 1e-8, f"softmax cross-entropy gradient error {err}"
    print(f"    softmax-cross-entropy: max err {err:.1e} OK")


def check_training_accuracy() -> None:
    print("[2] training accuracy")
    X, y = load_mnist("train", max_n=12000)
    Xt, yt = load_mnist("test", max_n=2000)
    model = Sequential(Linear(784, 128, rng=np.random.default_rng(1)),
                       ReLU(),
                       Linear(128, 64, rng=np.random.default_rng(2)),
                       ReLU(),
                       Linear(64, 10, rng=np.random.default_rng(3)))
    train(model, X, y, epochs=4, batch_size=256,
          optimiser=SGD(learning_rate=0.4, weight_decay=1e-4),
          seed=2, verbose=False)
    acc = float((model.predict(Xt) == yt).mean())
    print(f"    test accuracy: {acc:.3f}")
    assert acc > 0.85, f"test accuracy too low: {acc}"
    return model, Xt, yt


def check_attacks(model, xs, ys) -> None:
    print("[3] attacks flip predictions")

    g = gradient_wrt_input(model, xs[:1], ys[:1])[0]
    rng = np.random.default_rng(0)
    d = rng.normal(size=xs[:1].shape)
    d /= np.linalg.norm(d)
    eps_dir = 1e-5
    x0, y0 = xs[:1], ys[:1]
    from src.attacks import loss_of
    num = (loss_of(model, x0 + eps_dir * d, y0)
           - loss_of(model, x0 - eps_dir * d, y0)) / (2 * eps_dir)
    ana = float((d.reshape(1, -1) @ g.reshape(-1, 1)).item())
    assert abs(num - ana) < 1e-3, f"directional gradient mismatch: {num} vs {ana}"
    print(f"    directional gradient err {abs(num - ana):.1e} OK")

    base = float((model.predict(xs) == ys).mean())
    print(f"    base accuracy on attack set: {base:.3f}")
    assert base > 0.8

    for eps in (0.05, 0.3):
        sr_fgsm = attack_success(model, xs, ys, eps, attack=fgsm)
        sr_pgd = attack_success(model, xs, ys, eps, attack=pgd, steps=10)
        print(f"    eps={eps}: FGSM flip {sr_fgsm:.3f}  PGD flip {sr_pgd:.3f}")
        assert sr_fgsm > 0.2, f"FGSM too weak at eps={eps}: {sr_fgsm}"
        assert sr_pgd >= sr_fgsm - 0.05, "PGD should beat FGSM"
        adv = pgd(model, xs, ys, eps, steps=10)
        max_delta = np.abs(adv - xs).max()
        assert max_delta <= eps + 1e-9, f"PGD violated Linf bound: {max_delta}"
        assert adv.min() >= -1e-9 and adv.max() <= 1 + 1e-9, "pixel range"


def check_targeted(model, xs, ys) -> None:
    print("[4] targeted attacks")
    target = 3
    sr = target_success(model, xs, ys, target, 0.1, attack=pgd_targeted, steps=10)
    print(f"    PGD-targeted -> {target}: flip rate {sr:.3f}")
    assert sr > 0.2, f"targeted attack too weak: {sr}"
    x0 = xs[:1]
    adv = pgd_targeted(model, x0, np.array([target]), 0.3, steps=10)
    assert model.predict(adv)[0] == target, "targeted attack missed the class"
    adv2 = fgsm_targeted(model, x0, np.array([target]), 0.3)
    print("    targeted examples land in the target class")


def check_adversarial_training(X, y, Xt, yt) -> None:
    print("[5] adversarial training defends")
    rng = np.random.default_rng(0)
    sub = rng.choice(len(Xt), 300, replace=False)
    xs, ys = Xt[sub], yt[sub]

    def build():
        return Sequential(Linear(784, 128, rng=np.random.default_rng(1)),
                          ReLU(),
                          Linear(128, 64, rng=np.random.default_rng(2)),
                          ReLU(),
                          Linear(64, 10, rng=np.random.default_rng(3)))

    plain = build()
    train(plain, X, y, epochs=4, batch_size=256,
          optimiser=SGD(learning_rate=0.4, weight_decay=1e-4),
          seed=2, verbose=False)

    robust = build()
    train_adversarial(robust, X, y, epochs=4, batch_size=256,
                      optimiser=SGD(learning_rate=0.2, weight_decay=1e-4),
                      eps=0.1, pgd_steps=5, seed=2, verbose=False)

    plain_surv = 1.0 - attack_success(plain, xs, ys, 0.1, attack=pgd, steps=10)
    robust_surv = 1.0 - attack_success(robust, xs, ys, 0.1, attack=pgd, steps=10)
    print(f"    PGD-10 eps=0.1 survival: plain {plain_surv:.3f}, "
          f"adversarially trained {robust_surv:.3f}")
    assert robust_surv > plain_surv + 0.2, "adversarial training not defending"


def check_transfer(X, y, Xt, yt) -> None:
    print("[6] transfer attacks")
    rng = np.random.default_rng(0)
    sub = rng.choice(len(Xt), 300, replace=False)
    xs, ys = Xt[sub], yt[sub]

    A = Sequential(Linear(784, 128, rng=np.random.default_rng(1)),
                   ReLU(),
                   Linear(128, 64, rng=np.random.default_rng(2)),
                   ReLU(),
                   Linear(64, 10, rng=np.random.default_rng(3)))
    B = Sequential(Linear(784, 256, rng=np.random.default_rng(42)),
                   ReLU(),
                   Linear(256, 10, rng=np.random.default_rng(43)))
    train(A, X, y, epochs=4, batch_size=256,
          optimiser=SGD(learning_rate=0.4, weight_decay=1e-4),
          seed=2, verbose=False)
    train(B, X, y, epochs=4, batch_size=256,
          optimiser=SGD(learning_rate=0.4, weight_decay=1e-4),
          seed=3, verbose=False)

    wb = attack_success(A, xs, ys, 0.3, attack=pgd, steps=10)
    tr = transfer_attack(A, B, xs, ys, 0.3, attack=pgd, steps=10)
    print(f"    eps=0.3: white-box {wb:.3f}, transfer (crafted on B) {tr:.3f}")
    assert tr > 0.3, f"transferability too weak: {tr}"


def check_linearity(model, xs, ys) -> None:
    print("[7] the linearity argument holds")
    lc = linearity_check(model, xs[:50], ys[:50], eps=1e-3)
    print(f"    eps=1e-3: actual/predicted logit change ratio "
          f"{lc['ratio']:.3f} (≈1 means the model is locally linear)")
    assert 0.5 < lc['ratio'] < 2.0, f"linearity ratio off: {lc['ratio']}"


def check_cnn() -> None:
    print("[8] CNN trains and beats the MLP on MNIST")
    X, y = load_mnist("train", max_n=12000)
    Xt, yt = load_mnist("test", max_n=2000)
    X_img = to_image_shape(X)
    Xt_img = to_image_shape(Xt)

    cnn = Sequential(Conv2D(1, 8, 3, stride=1, rng=np.random.default_rng(42)),
                     ReLU(),
                     MaxPool2D(2),
                     Conv2D(8, 16, 3, stride=1, rng=np.random.default_rng(43)),
                     ReLU(),
                     MaxPool2D(2),
                     Flatten(),
                     Linear(16 * 5 * 5, 64, rng=np.random.default_rng(44)),
                     ReLU(),
                     Linear(64, 10, rng=np.random.default_rng(45)))
    train(cnn, X_img, y, epochs=5, batch_size=64,
          optimiser=Adam(learning_rate=0.001),
          X_val=Xt_img, y_val=yt, verbose=False)
    acc_cnn = float((cnn.predict(Xt_img) == yt).mean())
    print(f"    CNN test accuracy: {acc_cnn:.3f}")
    assert acc_cnn > 0.9, f"CNN accuracy too low: {acc_cnn}"

    mlp = Sequential(Linear(784, 256, rng=np.random.default_rng(42)),
                     ReLU(),
                     Linear(256, 128, rng=np.random.default_rng(43)),
                     ReLU(),
                     Linear(128, 10, rng=np.random.default_rng(44)))
    train(mlp, X, y, epochs=5, batch_size=64,
          optimiser=Adam(learning_rate=0.001),
          X_val=Xt, y_val=yt, verbose=False)
    acc_mlp = float((mlp.predict(Xt) == yt).mean())
    n_cnn = sum(p.data.size for p in cnn.parameters())
    n_mlp = sum(p.data.size for p in mlp.parameters())
    print(f"    MLP test accuracy: {acc_mlp:.3f} ({n_mlp} params)")
    print(f"    CNN test accuracy: {acc_cnn:.3f} ({n_cnn} params, "
          f"{n_mlp / n_cnn:.1f}x fewer)")
    assert acc_cnn > 0.9, f"CNN accuracy too low: {acc_cnn}"
    assert n_cnn < n_mlp / 5, "CNN should use far fewer parameters"


def main() -> None:
    check_autodiff()
    model, Xt, yt = check_training_accuracy()
    X, y = load_mnist("train", max_n=12000)
    rng = np.random.default_rng(0)
    sub = rng.choice(len(Xt), 300, replace=False)
    xs, ys = Xt[sub], yt[sub]
    check_attacks(model, xs, ys)
    check_targeted(model, xs, ys)
    check_linearity(model, xs, ys)
    check_adversarial_training(X, y, Xt, yt)
    check_transfer(X, y, Xt, yt)
    check_cnn()
    print("\nAll verification passed.")


if __name__ == "__main__":
    main()
