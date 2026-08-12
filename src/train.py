"""Training loop: minibatch SGD over the autodiff graph.

One step: build the graph for a batch, compute the loss, `backward()`
to accumulate gradients into every parameter, then `optimiser.step()`
to apply and zero them.  `train_adversarial` is the same loop with
every minibatch augmented by a PGD-attacked copy of itself (the
classic defence, Goodfellow et al. 2014).
"""

from __future__ import annotations

import numpy as np

from .autodiff import Tensor
from .nn import Sequential


def _validate(model, X, y):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    if X.ndim not in (2, 4):
        raise ValueError("X must be 2D (n_samples, n_features) "
                         "or 4D (n_samples, channels, H, W)")
    if y.shape != (X.shape[0],):
        raise ValueError(f"y shape {y.shape} does not match X rows {X.shape[0]}")
    n_out = model.layers[-1].W.data.shape[1]
    if y.min() < 0 or y.max() >= n_out:
        raise ValueError(f"labels must be in [0, {n_out})")
    return X, y


def train(model: Sequential, X, y, *, epochs, batch_size, optimiser,
          X_val=None, y_val=None, seed=0, verbose=True, eval_every=1):
    """Minibatch training.  Returns a history dict with train loss /
    accuracy and (if given) validation accuracy per epoch."""
    X, y = _validate(model, X, y)
    if X_val is not None:
        X_val = np.asarray(X_val, dtype=np.float64)
        y_val = np.asarray(y_val)
        if y_val.shape != (X_val.shape[0],):
            raise ValueError("val labels do not match val rows")
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    params = model.parameters()
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        perm = rng.permutation(n)
        epoch_loss = 0.0
        n_batches = 0
        correct = 0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            xb, yb = X[idx], y[idx]
            logits = model.forward(Tensor(xb), training=True)
            loss = logits.softmax_cross_entropy(yb)
            loss.backward()
            optimiser.step(params)
            epoch_loss += float(loss.data)
            n_batches += 1
            correct += int((model.predict(xb) == yb).sum())

        history["train_loss"].append(epoch_loss / n_batches)
        history["train_acc"].append(correct / n)
        if X_val is not None:
            val_logits = model.forward(Tensor(X_val), training=False)
            history["val_loss"].append(float(val_logits.softmax_cross_entropy(y_val).data))
            history["val_acc"].append(float((model.predict(X_val) == y_val).mean()))
        else:
            history["val_loss"].append(None)
            history["val_acc"].append(None)

        if verbose and (epoch % eval_every == 0 or epoch == epochs):
            line = (f"  epoch {epoch:3d}  loss {history['train_loss'][-1]:.4f}"
                    f"  train acc {history['train_acc'][-1]:.3f}")
            if history["val_acc"][-1] is not None:
                line += f"  val acc {history['val_acc'][-1]:.3f}"
            print(line)
    return history


def train_adversarial(model: Sequential, X, y, *, epochs, batch_size,
                      optimiser, eps=0.1, pgd_steps=5,
                      X_val=None, y_val=None, seed=0, verbose=True):
    """Adversarial training: every minibatch is augmented with a
    PGD-perturbed copy of itself, and the model trains to classify both
    the clean and the adversarial examples.  This flattens the loss
    landscape near the inputs, so small perturbations stop crossing
    decision boundaries."""
    from .attacks import pgd

    X, y = _validate(model, X, y)
    if X_val is not None:
        X_val = np.asarray(X_val, dtype=np.float64)
        y_val = np.asarray(y_val)
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    params = model.parameters()
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        perm = rng.permutation(n)
        epoch_loss = 0.0
        n_batches = 0
        correct = 0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            xb, yb = X[idx], y[idx]
            adv = pgd(model, xb, yb, eps, steps=pgd_steps)
            Xcat = np.concatenate([xb, adv])
            ycat = np.concatenate([yb, yb])
            logits = model.forward(Tensor(Xcat), training=True)
            loss = logits.softmax_cross_entropy(ycat)
            loss.backward()
            optimiser.step(params)
            epoch_loss += float(loss.data)
            n_batches += 1
            correct += int((model.predict(xb) == yb).sum())

        history["train_loss"].append(epoch_loss / n_batches)
        history["train_acc"].append(correct / n)
        if X_val is not None:
            val_logits = model.forward(Tensor(X_val), training=False)
            history["val_loss"].append(float(val_logits.softmax_cross_entropy(y_val).data))
            history["val_acc"].append(float((model.predict(X_val) == y_val).mean()))
        else:
            history["val_loss"].append(None)
            history["val_acc"].append(None)

        if verbose:
            line = (f"  epoch {epoch:3d}  loss {history['train_loss'][-1]:.4f}"
                    f"  train acc {history['train_acc'][-1]:.3f}")
            if history["val_acc"][-1] is not None:
                line += f"  val acc {history['val_acc'][-1]:.3f}"
            print(line)
    return history
