"""Attack tests: perturbation budgets are respected, targeted attacks
land in the target class, attacks actually flip predictions, and the
linearity ratio holds at small eps.  All on tiny synthetic data."""

import numpy as np
import pytest

from src.attacks import (fgsm, fgsm_targeted, linearity_check, pgd,
                         pgd_targeted, target_success, transfer_attack)
from src.autodiff import Tensor
from src.nn import Linear, ReLU, Sequential
from src.optimisers import SGD
from src.train import train


def make_data(n=200, d=6, k=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    centers = rng.normal(size=(k, d))
    y = np.argmin(np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2),
                  axis=1)
    return X, y


def make_model(d=6, k=3, seed=1):
    return Sequential(Linear(d, 16, rng=np.random.default_rng(seed)),
                      ReLU(),
                      Linear(16, k, rng=np.random.default_rng(seed + 1)))


@pytest.fixture(autouse=True)
def _model():
    X, y = make_data()
    model = make_model()
    train(model, X, y, epochs=30, batch_size=32,
          optimiser=SGD(learning_rate=0.3), verbose=False)
    return model


class TestAttacks:
    def test_fgsm_linf_budget(self, _model):
        x = np.random.default_rng(0).random((8, 6))
        y = np.zeros(8, dtype=int)
        adv = fgsm(_model, x, y, eps=0.2)
        assert np.abs(adv - x).max() <= 0.2 + 1e-9
        assert adv.min() >= -1e-9 and adv.max() <= 1 + 1e-9

    def test_pgd_linf_budget(self, _model):
        x = np.random.default_rng(1).random((8, 6))
        y = np.zeros(8, dtype=int)
        adv = pgd(_model, x, y, eps=0.15, steps=10)
        assert np.abs(adv - x).max() <= 0.15 + 1e-9

    def test_pgd_zero_steps_identity(self, _model):
        x = np.random.default_rng(2).random((4, 6))
        y = np.zeros(4, dtype=int)
        assert np.array_equal(pgd(_model, x, y, eps=0.3, steps=0), x)

    def test_fgsm_eps_zero_identity(self, _model):
        x = np.random.default_rng(3).random((4, 6))
        y = np.zeros(4, dtype=int)
        assert np.array_equal(fgsm(_model, x, y, eps=0.0), x)

    def test_targeted_lands_in_target(self, _model):
        x = np.random.default_rng(4).random((1, 6))
        adv = pgd_targeted(_model, x, np.array([2]), 0.5, steps=10)
        assert _model.predict(adv)[0] == 2

    def test_attack_actually_flips(self, _model):
        rng = np.random.default_rng(5)
        x = rng.random((50, 6))
        y = _model.predict(x)
        adv = pgd(_model, x, y, eps=0.4, steps=10)
        flipped = (_model.predict(x) == y) & (_model.predict(adv) != y)
        assert flipped.sum() > 0, "PGD flipped nothing on this synthetic set"

    def test_linearity_holds_small_eps(self, _model):
        x = np.random.default_rng(6).random((20, 6))
        y = _model.predict(x)
        lc = linearity_check(_model, x, y, eps=1e-4)
        assert 0.5 < lc["ratio"] < 2.0, f"ratio {lc['ratio']}"

    def test_transfer_is_weaker_than_whitebox(self, _model):
        rng = np.random.default_rng(7)
        x = rng.random((30, 6))
        y = _model.predict(x)
        other = make_model(seed=9)
        X, yy = make_data(n=200, seed=3)
        train(other, X, yy, epochs=30, batch_size=32,
              optimiser=SGD(learning_rate=0.3), verbose=False)
        both = (_model.predict(x) == y) & (other.predict(x) == y)
        xb, yb = x[both], y[both]
        if len(xb) == 0:
            pytest.skip("no samples both models classify correctly")
        wb = transfer_attack(_model, _model, xb, yb, 0.3,
                             attack=pgd, steps=10)
        tr = transfer_attack(_model, other, xb, yb, 0.3,
                             attack=pgd, steps=10)
        assert tr < wb + 0.05

    def test_targeted_success_rate(self, _model):
        x = np.random.default_rng(8).random((40, 6))
        y = _model.predict(x)
        sr = target_success(_model, x, y, 2, 0.4,
                            attack=pgd_targeted, steps=10)
        assert sr >= 0.5, f"targeted success rate was only {sr:.2f}"
