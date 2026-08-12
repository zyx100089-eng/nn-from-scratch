"""Reverse-mode autodiff engine.

A minimal tensor library: the forward pass builds a computation graph,
a backward pass propagates gradients from the loss to every leaf.
This is the chain rule applied to a directed acyclic graph — the same
machinery as PyTorch's autograd, written from scratch.

Design:
- A `Tensor` holds a value (numpy array), a gradient buffer, and a
  backward closure per operation.  Each op records how it was produced.
- `backward()` walks the graph in reverse topological order (iterative
  post-order: deterministic, no recursion limit) and *accumulates*
  gradients into each tensor's buffer.  Accumulation is what makes the
  chain rule compose when a tensor is used more than once.
- Broadcasting is handled by `_reduce_grad`: gradients are summed back
  over the axes that broadcasting added (e.g. a bias (3,) added to a
  batch (N, 3) must receive the sum of its gradient over N).

Everything in this project is a graph op on this engine: Linear layers,
convolution, max pooling, dropout, batch norm, the attacks, and the
losses.  Every backward pass is verified against numerical
differentiation in the test suite.
"""

from __future__ import annotations

import numpy as np


def _reduce_grad(grad: np.ndarray, shape: tuple) -> np.ndarray:
    """Sum a gradient over axes that broadcasting added, so it can be
    added to a parent of the given shape."""
    if grad.shape == shape:
        return grad
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis, (gs, ss) in enumerate(zip(grad.shape, shape)):
        if ss == 1 and gs != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad


class Tensor:
    def __init__(self, data, requires_grad=True):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data, dtype=np.float64)
        self.requires_grad = requires_grad
        self._backward = None  # fn() accumulates into parents' .grad
        self._parents: list["Tensor"] = []

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _record(self, parent_list, backward_fn) -> "Tensor":
        self._parents = list(parent_list)
        if any(p.requires_grad for p in parent_list):
            self._backward = backward_fn
        return self

    def backward(self) -> None:
        """Propagate gradients from this tensor (assumed scalar loss)."""
        assert self.data.ndim == 0, "backward() needs a scalar loss"
        self.grad = np.ones((), dtype=np.float64)
        order: list[Tensor] = []
        seen = set()

        def visit(t: Tensor) -> None:
            if id(t) in seen:
                return
            seen.add(id(t))
            for p in t._parents:
                visit(p)
            order.append(t)

        visit(self)
        for t in reversed(order):
            if t._backward is not None:
                t._backward()

    # ------------------------------------------------------------------
    # Elementwise ops (broadcast-safe backward)
    # ------------------------------------------------------------------

    def __add__(self, other):
        other = _coerce(other)
        out = Tensor(self.data + other.data)

        def bw():
            self.grad += _reduce_grad(out.grad, self.data.shape)
            if other.requires_grad:
                other.grad += _reduce_grad(out.grad, other.data.shape)

        return out._record([self, other], bw)

    def __sub__(self, other):
        other = _coerce(other)
        out = Tensor(self.data - other.data)

        def bw():
            self.grad += _reduce_grad(out.grad, self.data.shape)
            if other.requires_grad:
                other.grad -= _reduce_grad(out.grad, other.data.shape)

        return out._record([self, other], bw)

    def __mul__(self, other):
        other = _coerce(other)
        out = Tensor(self.data * other.data)

        def bw():
            self.grad += _reduce_grad(out.grad * other.data, self.data.shape)
            if other.requires_grad:
                other.grad += _reduce_grad(out.grad * self.data, other.data.shape)

        return out._record([self, other], bw)

    def __neg__(self):
        return self * -1.0

    def __pow__(self, power: float):
        out = Tensor(self.data ** power)

        def bw():
            self.grad += out.grad * power * self.data ** (power - 1)

        return out._record([self], bw)

    def __truediv__(self, other):
        other = _coerce(other)
        out = Tensor(self.data / other.data)

        def bw():
            self.grad += _reduce_grad(out.grad / other.data, self.data.shape)
            if other.requires_grad:
                other.grad -= _reduce_grad(out.grad * self.data / (other.data ** 2),
                                           other.data.shape)

        return out._record([self, other], bw)

    # ------------------------------------------------------------------
    # Linear algebra / reductions
    # ------------------------------------------------------------------

    def matmul(self, other):
        out = Tensor(self.data @ other.data)

        def bw():
            self.grad += out.grad @ other.data.T
            if other.requires_grad:
                other.grad += self.data.T @ out.grad

        return out._record([self, other], bw)

    def sum(self):
        out = Tensor(self.data.sum())

        def bw():
            self.grad += out.grad

        return out._record([self], bw)

    def mean(self):
        out = Tensor(self.data.mean())
        n = self.data.size

        def bw():
            self.grad += out.grad / n

        return out._record([self], bw)

    # ------------------------------------------------------------------
    # Activations
    # ------------------------------------------------------------------

    def relu(self):
        out = Tensor(np.maximum(self.data, 0.0))

        def bw():
            self.grad += out.grad * (self.data > 0)

        return out._record([self], bw)

    def sigmoid(self):
        out = Tensor(1.0 / (1.0 + np.exp(-np.clip(self.data, -500, 500))))

        def bw():
            self.grad += out.grad * out.data * (1.0 - out.data)

        return out._record([self], bw)

    def tanh(self):
        out = Tensor(np.tanh(self.data))

        def bw():
            self.grad += out.grad * (1.0 - out.data ** 2)

        return out._record([self], bw)

    def softmax(self, axis=1):
        m = self.data.max(axis=axis, keepdims=True)
        e = np.exp(self.data - m)
        out = Tensor(e / e.sum(axis=axis, keepdims=True))

        def bw():
            s = out.data
            dot = (out.grad * s).sum(axis=axis, keepdims=True)
            self.grad += s * (out.grad - dot)

        return out._record([self], bw)

    # ------------------------------------------------------------------
    # Losses (labels are integer class indices)
    # ------------------------------------------------------------------

    def softmax_cross_entropy(self, labels):
        """Fused softmax + cross-entropy, numerically stable via the
        max-shift trick.  Gradient is (softmax - onehot)/N, exact even
        where a naive softmax would underflow."""
        m = self.data.max(axis=1, keepdims=True)
        e = np.exp(self.data - m)
        p = e / e.sum(axis=1, keepdims=True)
        N = self.data.shape[0]
        row_idx = np.arange(N)
        p_true = np.maximum(p[row_idx, labels], 1e-12)
        loss = float(-np.log(p_true).mean())
        out = Tensor(np.array(loss))
        g = p.copy() / N
        g[row_idx, labels] = (p[row_idx, labels] - 1) / N

        def bw():
            self.grad += out.grad * g

        return out._record([self], bw)


# ----------------------------------------------------------------------
# Conv / pool / dropout / batchnorm as graph ops
# ----------------------------------------------------------------------

def _get_patches(X, kh, kw, stride):
    """im2col-style sliding-window extraction.  X: (N, C, H, W) ->
    patches (N, C, kh, kw, oh, ow) as a strided view."""
    n, c, h, w = X.shape
    out_h = (h - kh) // stride + 1
    out_w = (w - kw) // stride + 1
    s0, s1, s2, s3 = X.strides
    shape = (n, c, kh, kw, out_h, out_w)
    strides = (s0, s1, s2, s3, s2 * stride, s3 * stride)
    return np.lib.stride_tricks.as_strided(X, shape=shape, strides=strides)


def _col2im(dcols, X_shape, kh, kw, stride):
    """Inverse of im2col: scatter (N*oh*ow, C*kh*kw) gradients back
    into an (N, C, H, W) array, accumulating overlaps."""
    n, c, h, w = X_shape
    out_h = (h - kh) // stride + 1
    out_w = (w - kw) // stride + 1
    dpatches = dcols.reshape(n, out_h, out_w, c, kh, kw).transpose(0, 3, 4, 5, 1, 2)
    dX = np.zeros(X_shape, dtype=np.float64)
    for i in range(kh):
        for j in range(kw):
            dX[:, :, i:i + stride * out_h:stride, j:j + stride * out_w:stride] \
                += dpatches[:, :, i, j, :, :]
    return dX


def conv2d(x, w, b, stride=1):
    """2D convolution (cross-correlation) as a graph op.

    x: (N, C_in, H, W)   w: (C_out, C_in, k, k)   b: (C_out,)
    out: (N, C_out, H', W') with H' = (H - k) // stride + 1
    """
    x_data, w_data = x.data, w.data
    b_data = b.data
    n, c_in, h, w_in = x_data.shape
    c_out, _, k, _ = w_data.shape
    out_h = (h - k) // stride + 1
    out_w = (w_in - k) // stride + 1

    patches = _get_patches(x_data, k, k, stride).transpose(0, 4, 5, 1, 2, 3)
    cols = patches.reshape(n * out_h * out_w, -1)
    W_col = w_data.reshape(c_out, -1)
    out_data = (cols @ W_col.T + b_data.reshape(1, -1))
    out_data = out_data.reshape(n, out_h, out_w, c_out).transpose(0, 3, 1, 2)
    out = Tensor(out_data)

    def bw():
        dout = out.grad
        dout_r = dout.transpose(0, 2, 3, 1).reshape(n * out_h * out_w, c_out)
        w.grad += (dout_r.T @ cols).reshape(w_data.shape)
        b.grad += dout.sum(axis=(0, 2, 3))
        dcols = dout_r @ W_col
        x.grad += _col2im(dcols, x_data.shape, k, k, stride)

    return out._record([x, w, b], bw)


def max_pool2d(x, pool_size=2, stride=None):
    """Max pooling as a graph op; backward routes each gradient to the
    argmax position of its window."""
    stride = stride or pool_size
    x_data = x.data
    n, c, h, w = x_data.shape
    k, s = pool_size, stride
    out_h = (h - k) // s + 1
    out_w = (w - k) // s + 1
    patches = _get_patches(x_data, k, k, s)
    flat = patches.reshape(n, c, k * k, out_h, out_w)
    argmax = flat.argmax(axis=2)
    out = Tensor(flat.max(axis=2))

    def bw():
        dX = np.zeros_like(x_data)
        dout = out.grad
        for i in range(k):
            for j in range(k):
                mask = (argmax == i * k + j)
                dX[:, :, i:i + s * out_h:s, j:j + s * out_w:s] += dout * mask
        x.grad += dX

    return out._record([x], bw)


def dropout(x, drop_prob, training=True, rng=None):
    """Inverted dropout as a graph op.  At inference it is the identity."""
    if not training or drop_prob == 0.0:
        return x
    rng = rng or np.random.default_rng()
    mask = (rng.random(x.data.shape) > drop_prob).astype(np.float64)
    out = Tensor(x.data * mask / (1.0 - drop_prob))

    def bw():
        x.grad += out.grad * mask / (1.0 - drop_prob)

    return out._record([x], bw)


def batch_norm(x, gamma, beta, running_mean, running_var, momentum=0.9,
               epsilon=1e-5, training=True):
    """Batch norm as a graph op with a hand-derived backward.

    Training: normalises with the batch statistics and updates the
    running statistics as a side effect.  Inference: normalises with
    the running statistics (no graph needed).

    The backward is the full derivation through dvar and dmean — the
    trickiest hand-derived gradient in the project, verified against
    finite differences in the test suite.
    """
    if not training:
        X_norm = (x.data - running_mean) / np.sqrt(running_var + epsilon)
        return Tensor(gamma.data * X_norm + beta.data, requires_grad=False)

    mean = x.data.mean(axis=0, keepdims=True)
    var = x.data.var(axis=0, keepdims=True)
    running_mean[:] = momentum * running_mean + (1 - momentum) * mean
    running_var[:] = momentum * running_var + (1 - momentum) * var
    X_norm = (x.data - mean) / np.sqrt(var + epsilon)
    out = Tensor(gamma.data * X_norm + beta.data)

    def bw():
        dout = out.grad
        n = x.data.shape[0]
        dX_norm = dout * gamma.data
        inv_std = 1.0 / np.sqrt(var + epsilon)
        gamma.grad += (dout * X_norm).sum(axis=0, keepdims=True)
        beta.grad += dout.sum(axis=0, keepdims=True)
        dvar = (dX_norm * (x.data - mean) * -0.5 * inv_std ** 3).sum(axis=0, keepdims=True)
        dmean = (dX_norm * -inv_std).sum(axis=0, keepdims=True) \
            + dvar * np.mean(-2.0 * (x.data - mean), axis=0, keepdims=True)
        x.grad += dX_norm * inv_std + dvar * 2.0 * (x.data - mean) / n + dmean / n

    return out._record([x, gamma, beta], bw)


def reshape(x, shape):
    """View op; backward is the inverse reshape."""
    out = Tensor(x.data.reshape(shape))

    def bw():
        x.grad += out.grad.reshape(x.data.shape)

    return out._record([x], bw)


def transpose(x, axes):
    """View op; backward is the inverse transpose."""
    out = Tensor(x.data.transpose(axes))

    def bw():
        x.grad += out.grad.transpose(np.argsort(axes))

    return out._record([x], bw)


def _coerce(x):
    if isinstance(x, Tensor):
        return x
    return Tensor(np.asarray(x, dtype=np.float64), requires_grad=False)
