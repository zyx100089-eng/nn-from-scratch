"""Training-loop tests: the model learns, inputs are validated, and
seeds are reproducible.  All on tiny synthetic data (no downloads)."""

import numpy as np
import pytest

from src.autodiff import Tensor
from src.nn import Linear, ReLU, Sequential
from src.optimisers import SGD, Adam
from src.train import train


def make_data(n=80, d=6, k=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    centers = rng.normal(size=(k, d))
    y = np.argmin(np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2),
                  axis=1)
    return X, y


def make_model(d=6, k=3, seed=1):
    return Sequential(Linear(d, 12, rng=np.random.default_rng(seed)),
                      ReLU(),
                      Linear(12, k, rng=np.random.default_rng(seed + 1)))


class TestTraining:
    def test_train_learns(self):
        X, y = make_data()
        model = make_model()
        train(model, X, y, epochs=30, batch_size=16,
              optimiser=SGD(learning_rate=0.3), verbose=False)
        acc = float((model.predict(X) == y).mean())
        assert acc > 0.9, f"training failed to learn: acc {acc}"

    def test_train_validates_inputs(self):
        X, y = make_data()
        model = make_model()
        with pytest.raises(ValueError):
            train(model, X, y[:40], epochs=1, batch_size=8,
                  optimiser=SGD(0.1))
        with pytest.raises(ValueError):
            train(model, X, np.array([0, 5] * 40), epochs=1,
                  batch_size=8, optimiser=SGD(0.1))

    def test_predict_matches_forward(self):
        X, y = make_data()
        model = make_model()
        train(model, X, y, epochs=5, batch_size=16,
              optimiser=SGD(0.3), verbose=False)
        logits = model.forward(Tensor(X)).data
        assert np.array_equal(logits.argmax(axis=1), model.predict(X))

    def test_seed_reproducible(self):
        X, y = make_data()
        a = make_model(seed=7)
        b = make_model(seed=7)
        assert np.allclose(a.layers[0].W.data, b.layers[0].W.data)

    def test_adam_trains(self):
        X, y = make_data()
        model = make_model()
        train(model, X, y, epochs=20, batch_size=16,
              optimiser=Adam(learning_rate=0.01), verbose=False)
        acc = float((model.predict(X) == y).mean())
        assert acc > 0.9

    def test_single_step_reduces_loss(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(20, 4))
        y = rng.integers(0, 3, 20)
        model = Sequential(Linear(4, 8, rng=rng), ReLU(),
                           Linear(8, 3, rng=rng))
        opt = SGD(learning_rate=0.1)
        logits = model.forward(Tensor(X))
        loss_before = float(logits.softmax_cross_entropy(y).data)
        loss = logits.softmax_cross_entropy(y)
        loss.backward()
        opt.step(model.parameters())
        logits2 = model.forward(Tensor(X))
        loss_after = float(logits2.softmax_cross_entropy(y).data)
        assert loss_after < loss_before
