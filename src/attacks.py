"""Adversarial attacks: FGSM and PGD.

The core idea (Goodfellow et al. 2014): a ReLU network is
piecewise-linear in its input, so a single gradient step along the
direction that *raises* the loss moves the input across a decision
boundary.  FGSM does this in one shot; PGD iterates the same idea with
a projection back into the allowed perturbation ball.

The gradient of the loss with respect to the INPUT is computed with the
same autodiff engine used for training — the model's parameters are
constants in the graph, and the input's gradient is read after one
backward pass.
"""

from __future__ import annotations

import numpy as np

from .autodiff import Tensor
from .nn import Sequential


def loss_of(model: Sequential, x: np.ndarray, y: np.ndarray) -> float:
    """Cross-entropy loss for a batch (no graph retained)."""
    logits = model.forward(Tensor(x), training=False)
    return float(logits.softmax_cross_entropy(y).data)


def gradient_wrt_input(model: Sequential, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Gradient of the mean cross-entropy loss wrt the input batch.

    The graph is built on a fresh copy of x.  The model's weight
    tensors are also leaves of this graph (they are created in __init__
    and reused by forward), so backward() accumulates gradients into
    their .grad buffers too.  We zero those buffers before returning so
    attack code never leaves stale gradients behind.
    """
    xt = Tensor(x)
    logits = model.forward(xt, training=False)
    loss = logits.softmax_cross_entropy(y)
    loss.backward()
    for p in model.parameters():
        p.grad.fill(0.0)
    return xt.grad


def fgsm(model: Sequential, x: np.ndarray, y: np.ndarray, eps: float,
         targeted: bool = False) -> np.ndarray:
    """Fast Gradient Sign Method: one step of eps in the sign of the
    loss gradient (or the negative sign for a targeted attack).

    x: single image (784,) or batch (N, 784).  Returns perturbed x.
    """
    x2d = x.reshape(1, -1) if x.ndim == 1 else x
    y1d = np.atleast_1d(y)
    g = gradient_wrt_input(model, x2d, y1d)
    sign = np.sign(g)
    if targeted:
        sign = -sign
    adv = np.clip(x2d + eps * sign, 0.0, 1.0)
    return adv.reshape(x.shape)


def fgsm_targeted(model: Sequential, x: np.ndarray, target: np.ndarray,
                  eps: float) -> np.ndarray:
    """FGSM toward a SPECIFIC class: minimise the target class's loss
    so its logit rises above the others.

    Unlike the sign-flip in fgsm(..., targeted=True) (which just
    pushes away from the true class), this actually steers the image
    into the chosen class.
    """
    x2d = x.reshape(1, -1) if x.ndim == 1 else x
    t = np.atleast_1d(target)
    g = gradient_wrt_input(model, x2d, t)
    adv = np.clip(x2d - eps * np.sign(g), 0.0, 1.0)
    return adv.reshape(x.shape)


def pgd(model: Sequential, x: np.ndarray, y: np.ndarray, eps: float,
        steps: int, step_size: float | None = None, targeted: bool = False,
        clip: tuple[float, float] = (0.0, 1.0)) -> np.ndarray:
    """Projected Gradient Descent: `steps` FGSM-style updates, each
    followed by a projection back into the Linf ball of radius eps
    around x (and the pixel range)."""
    step_size = step_size or eps / max(steps, 1)
    x2d = x.reshape(1, -1) if x.ndim == 1 else x
    y1d = np.atleast_1d(y)
    lo, hi = clip
    adv = np.clip(x2d.copy(), lo, hi)
    for _ in range(steps):
        g = gradient_wrt_input(model, adv, y1d)
        sign = np.sign(g)
        if targeted:
            sign = -sign
        adv = adv + step_size * sign
        adv = np.clip(adv, x2d - eps, x2d + eps)  # stay in the ball
        adv = np.clip(adv, lo, hi)                # stay in pixel range
    return adv.reshape(x.shape)


def pgd_targeted(model: Sequential, x: np.ndarray, target: np.ndarray,
                 eps: float, steps: int, step_size: float | None = None,
                 clip: tuple[float, float] = (0.0, 1.0)) -> np.ndarray:
    """Projected Gradient Descent toward a specific class."""
    step_size = step_size or eps / max(steps, 1)
    x2d = x.reshape(1, -1) if x.ndim == 1 else x
    t = np.atleast_1d(target)
    lo, hi = clip
    adv = np.clip(x2d.copy(), lo, hi)
    for _ in range(steps):
        g = gradient_wrt_input(model, adv, t)
        adv = adv - step_size * np.sign(g)
        adv = np.clip(adv, x2d - eps, x2d + eps)
        adv = np.clip(adv, lo, hi)
    return adv.reshape(x.shape)


def attack_success(model: Sequential, x: np.ndarray, y: np.ndarray,
                   eps: float, attack=pgd, **kw) -> float:
    """Fraction of correctly-classified samples that the attack flips."""
    pred = model.predict(x)
    base = pred == y
    if base.sum() == 0:
        return 0.0
    adv = attack(model, x, y, eps, **kw)
    adv_pred = model.predict(adv)
    return float((base & (adv_pred != y)).sum() / base.sum())


def target_success(model: Sequential, x: np.ndarray, y: np.ndarray,
                   target: int, eps: float, attack=pgd_targeted, **kw) -> float:
    """Fraction of correctly-classified samples that the attack flips
    INTO the target class."""
    pred = model.predict(x)
    base = pred == y
    if base.sum() == 0:
        return 0.0
    adv = attack(model, x, y * 0 + target, eps, **kw)
    adv_pred = model.predict(adv)
    return float((base & (adv_pred == target)).sum() / base.sum())


def transfer_attack(model: Sequential, source: Sequential, x: np.ndarray,
                    y: np.ndarray, eps: float, attack=pgd, **kw) -> float:
    """Attack `model` with perturbations crafted against `source`.

    The famous transferability result: adversarial examples crafted
    against one model often fool a different one, because both models
    approximate the same decision boundary.  Returns the flip rate of
    `model` on examples crafted against `source`.
    """
    pred = model.predict(x)
    base = pred == y
    if base.sum() == 0:
        return 0.0
    adv = attack(source, x, y, eps, **kw)
    adv_pred = model.predict(adv)
    return float((base & (adv_pred != y)).sum() / base.sum())


def robustness_curve(model: Sequential, X: np.ndarray, y: np.ndarray,
                     epsilons=(0.0, 0.03, 0.05, 0.1, 0.3),
                     attack=pgd, **kw) -> list[float]:
    """Accuracy at each eps budget after an attack: the robustness
    curve.  eps=0 is the clean accuracy."""
    out = []
    for eps in epsilons:
        if eps == 0:
            out.append(float((model.predict(X) == y).mean()))
            continue
        adv = attack(model, X, y, eps, **kw)
        out.append(float((model.predict(adv) == y).mean()))
    return out


def linearity_check(model: Sequential, x: np.ndarray, y: np.ndarray,
                    eps: float = 0.1) -> dict:
    """Quantify Goodfellow's linearity argument on a batch.

    The claim: a ReLU network is piecewise-linear in its input, so a
    perturbation delta changes the predicted-class logit by roughly
    g·delta, where g is the logit's gradient wrt the input.  We
    measure:

      - actual logit change from an FGSM-sized step
      - predicted logit change via the logit gradient
      - the ratio (actual/predicted): ~1 means the model is locally
        linear in the input in the attacked direction (the argument
        holds); >> 1 means it is not.

    We use the PREDICTED class's logit (not the loss) because the
    loss is softmax-nonlinear; the network itself is what's claimed
    to be piecewise-linear.
    """
    pred = model.predict(x)
    xt = Tensor(x)
    logits = model.forward(xt, training=False)
    picked = (logits * _onehot(pred, logits.data.shape[1])).sum()
    picked.backward()
    g = xt.grad

    sign = np.sign(g)
    adv = np.clip(x + eps * sign, 0.0, 1.0)
    actual = _logits(model, adv) - _logits(model, x)
    delta = adv - x
    predicted = np.abs((g * delta).sum(axis=1))
    actual_pred = np.abs(actual[np.arange(len(x)), pred])
    ratio = float(np.mean(actual_pred / np.maximum(predicted, 1e-12)))
    return {
        "ratio": ratio,
        "mean_actual_logit_change": float(actual_pred.mean()),
        "mean_predicted_logit_change": float(predicted.mean()),
    }


def _onehot(idx: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.zeros((len(idx), n_classes), dtype=np.float64)
    out[np.arange(len(idx)), idx] = 1.0
    return out


def _logits(model: Sequential, x: np.ndarray) -> np.ndarray:
    return model.forward(Tensor(x), training=False).data
