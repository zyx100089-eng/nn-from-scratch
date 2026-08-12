"""Gradient checks: every backward rule vs numerical differentiation.

The discipline of this project: no gradient is trusted until it has
been checked against central finite differences.  These tests cover the
engine ops, the fused softmax cross-entropy, and the hand-derived
convolution / max-pool / batch-norm backward passes.
"""

import numpy as np
import pytest

from src.autodiff import Tensor, batch_norm, conv2d, max_pool2d
from src.nn import BatchNorm, Conv2D, Dropout, Linear, Sequential


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


class TestEngineOps:
    def test_linear_layer_grads(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(4, 5))
        w = rng.normal(size=(5, 3))
        b = rng.normal(size=(3,))
        xt, wt, bt = Tensor(x), Tensor(w), Tensor(b)
        loss = ((xt.matmul(wt) + bt).relu() ** 2).sum()
        loss.backward()

        def loss_fn(xx, ww, bb):
            return float(((xx.matmul(ww) + bb).relu() ** 2).sum().data)

        for name, val, got, setter in [
            ("W", w, wt.grad, lambda v: (xt, Tensor(v), bt)),
            ("b", b, bt.grad, lambda v: (xt, wt, Tensor(v))),
            ("x", x, xt.grad, lambda v: (Tensor(v), wt, bt)),
        ]:
            want = numeric_grad(lambda v: loss_fn(*setter(v)), val)
            assert np.abs(got - want).max() < 1e-6, name

    def test_broadcast_bias_grad(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=(4, 3))
        b = rng.normal(size=(3,))
        xt, bt = Tensor(x), Tensor(b)
        loss = (xt + bt).sum()
        loss.backward()
        assert bt.grad.shape == (3,)
        assert np.allclose(bt.grad, 4.0)

    def test_softmax_ce_grad(self):
        rng = np.random.default_rng(2)
        logits = rng.normal(size=(6, 4))
        labels = rng.integers(0, 4, size=6)
        t = Tensor(logits)
        loss = t.softmax_cross_entropy(labels)
        loss.backward()
        want = numeric_grad(
            lambda v: float(Tensor(v).softmax_cross_entropy(labels).data),
            logits)
        assert np.abs(t.grad - want).max() < 1e-8

    def test_softmax_grad(self):
        rng = np.random.default_rng(3)
        x = rng.normal(size=(5, 4))
        t = Tensor(x)
        loss = (t.softmax() ** 2).sum()
        loss.backward()
        want = numeric_grad(
            lambda v: float((Tensor(v).softmax() ** 2).sum().data), x)
        assert np.abs(t.grad - want).max() < 1e-6

    def test_sigmoid_tanh_grads(self):
        rng = np.random.default_rng(4)
        x = rng.normal(size=(3, 4))
        for op in ("sigmoid", "tanh"):
            t = Tensor(x)
            loss = (getattr(t, op)() ** 2).sum()
            loss.backward()
            want = numeric_grad(
                lambda v: float((getattr(Tensor(v), op)() ** 2).sum().data), x)
            assert np.abs(t.grad - want).max() < 1e-6, op

    def test_requires_grad_false(self):
        c = Tensor(np.array([5.0]), requires_grad=False)
        t = Tensor(np.array([2.0]))
        loss = (t * c).sum()
        loss.backward()
        assert c.grad.tolist() == [0.0]
        assert t.grad.tolist() == [5.0]

    def test_shared_tensor_accumulates(self):
        # a tensor used twice must receive the sum of both gradients
        t = Tensor(np.array([2.0]))
        loss = (t * t).sum()
        loss.backward()
        assert np.allclose(t.grad, [4.0])


class TestConv2D:
    def test_forward_shape(self):
        rng = np.random.default_rng(0)
        conv = Conv2D(3, 5, kernel_size=3, stride=1)
        X = rng.normal(size=(4, 3, 8, 8))
        out = conv.forward(Tensor(X))
        assert out.data.shape == (4, 5, 6, 6)

    def test_forward_shape_stride2(self):
        rng = np.random.default_rng(0)
        conv = Conv2D(2, 4, kernel_size=3, stride=2)
        X = rng.normal(size=(3, 2, 9, 9))
        out = conv.forward(Tensor(X))
        assert out.data.shape == (3, 4, 4, 4)

    def test_bias_addition(self):
        conv = Conv2D(1, 1, kernel_size=1)
        conv.W.data[:] = 0
        conv.b.data[:] = 3.0
        X = np.ones((1, 1, 4, 4))
        out = conv.forward(Tensor(X))
        np.testing.assert_allclose(out.data, np.full((1, 1, 4, 4), 3.0))

    def test_gradient_check_weights(self):
        rng = np.random.default_rng(1)
        conv = Conv2D(2, 3, kernel_size=3)
        X = rng.normal(size=(2, 2, 5, 5))
        out = conv.forward(Tensor(X))
        loss = (out ** 2).sum()
        loss.backward()

        def loss_fn(w):
            c = Conv2D(2, 3, kernel_size=3)
            c.W.data[:] = w.reshape(c.W.data.shape)
            c.b.data[:] = conv.b.data
            return float((c.forward(Tensor(X)) ** 2).sum().data)

        want = numeric_grad(loss_fn, conv.W.data.reshape(-1))
        assert np.abs(conv.W.grad.reshape(-1) - want).max() < 1e-5

    def test_gradient_check_input(self):
        rng = np.random.default_rng(2)
        conv = Conv2D(2, 2, kernel_size=3)
        X = rng.normal(size=(1, 2, 5, 5))
        xt = Tensor(X)
        out = conv.forward(xt)
        loss = (out ** 2).sum()
        loss.backward()

        def loss_fn(x):
            return float((conv.forward(Tensor(x)) ** 2).sum().data)

        want = numeric_grad(loss_fn, X)
        assert np.abs(xt.grad - want).max() < 1e-5

    def test_gradient_check_bias(self):
        rng = np.random.default_rng(3)
        conv = Conv2D(2, 2, kernel_size=3)
        X = rng.normal(size=(1, 2, 5, 5))
        out = conv.forward(Tensor(X))
        loss = (out ** 2).sum()
        loss.backward()

        def loss_fn(b):
            c = Conv2D(2, 2, kernel_size=3)
            c.W.data[:] = conv.W.data
            c.b.data[:] = b
            return float((c.forward(Tensor(X)) ** 2).sum().data)

        want = numeric_grad(loss_fn, conv.b.data)
        assert np.abs(conv.b.grad - want).max() < 1e-5


class TestMaxPool2D:
    def test_forward_max(self):
        X = np.arange(16, dtype=np.float64).reshape(1, 1, 4, 4)
        out = max_pool2d(Tensor(X), 2)
        np.testing.assert_allclose(out.data[0, 0], [[5, 7], [13, 15]])

    def test_backward_routes_to_max(self):
        X = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float64).reshape(1, 1, 2, 4)
        xt = Tensor(X)
        out = max_pool2d(xt, 2)
        (out * Tensor(np.array([[[[1.0, 2.0]]]]))).sum().backward()
        expected = np.array([[0, 0, 0, 0], [0, 1, 0, 2]], dtype=np.float64).reshape(1, 1, 2, 4)
        np.testing.assert_allclose(xt.grad, expected)

    def test_gradient_check(self):
        rng = np.random.default_rng(4)
        X = rng.normal(size=(2, 3, 6, 6))
        xt = Tensor(X)
        out = max_pool2d(xt, 2)
        loss = (out ** 2).sum()
        loss.backward()
        want = numeric_grad(
            lambda v: float((max_pool2d(Tensor(v), 2) ** 2).sum().data), X)
        assert np.abs(xt.grad - want).max() < 1e-6


class TestBatchNorm:
    def test_normalises_output(self):
        bn = BatchNorm(3)
        X = np.random.randn(100, 3) * 5 + 10
        out = bn.forward(Tensor(X), training=True)
        np.testing.assert_allclose(np.mean(out.data, axis=0), 0, atol=1e-6)
        np.testing.assert_allclose(np.std(out.data, axis=0), 1, atol=1e-1)

    def test_inference_uses_running_stats(self):
        bn = BatchNorm(3)
        X = np.random.randn(50, 3) * 2 + 3
        bn.forward(Tensor(X), training=True)
        X_test = np.random.randn(10, 3) * 2 + 3
        out = bn.forward(Tensor(X_test), training=False)
        assert out.data.shape == (10, 3)

    def test_gradient_check(self):
        """The full backward through dvar and dmean, checked numerically."""
        rng = np.random.default_rng(0)
        bn = BatchNorm(3)
        X = rng.standard_normal((16, 3))

        def loss_fn(x, gamma, beta):
            b = BatchNorm(3)
            b.gamma.data[:] = gamma
            b.beta.data[:] = beta
            return float((b.forward(Tensor(x), training=True) ** 2).sum().data)

        xt = Tensor(X)
        out = bn.forward(xt, training=True)
        loss = (out ** 2).sum()
        loss.backward()

        assert np.abs(xt.grad - numeric_grad(
            lambda v: loss_fn(v, bn.gamma.data, bn.beta.data), X)).max() < 1e-5
        assert np.abs(bn.gamma.grad - numeric_grad(
            lambda v: loss_fn(X, v, bn.beta.data), bn.gamma.data)).max() < 1e-5
        assert np.abs(bn.beta.grad - numeric_grad(
            lambda v: loss_fn(X, bn.gamma.data, v), bn.beta.data)).max() < 1e-5


class TestDropout:
    def test_forward_training(self):
        layer = Dropout(drop_prob=0.5, seed=42)
        X = np.ones((100, 10))
        out = layer.forward(Tensor(X), training=True)
        assert np.any(out.data == 0)
        active = out.data[out.data != 0]
        np.testing.assert_allclose(active, 2.0, atol=1e-10)

    def test_forward_inference(self):
        layer = Dropout(drop_prob=0.5)
        X = np.ones((10, 5))
        out = layer.forward(Tensor(X), training=False)
        np.testing.assert_array_equal(out.data, X)
