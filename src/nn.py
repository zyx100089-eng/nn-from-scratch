"""Network layers as graph ops on the autodiff engine.

Each layer is a class that owns its parameter tensors once (created in
__init__ and reused by every forward call) and returns the graph-op
result of `forward`.  All gradients flow through the engine — there is
no hand-coded backprop here, only hand-derived backward rules in
`autodiff.py`.

Why persistent parameters matter (this was a real bug during
development): if `forward()` created fresh tensors for the weights each
call, gradients would accumulate into throwaway objects and the
optimizer would step on zeros — training would sit at chance accuracy
forever.
"""

from __future__ import annotations

import numpy as np

from .autodiff import (Tensor, batch_norm, conv2d, dropout, max_pool2d,
                       reshape)


class Linear:
    def __init__(self, in_features, out_features, rng=None):
        rng = rng or np.random.default_rng()
        scale = np.sqrt(2.0 / in_features)
        self.W = Tensor(rng.normal(0.0, scale, size=(in_features, out_features)))
        self.b = Tensor(np.zeros(out_features, dtype=np.float64))

    def forward(self, x, training=True):
        return x.matmul(self.W) + self.b

    def parameters(self):
        return [self.W, self.b]


class Conv2D:
    """(N, C_in, H, W) -> (N, C_out, H', W'), H' = (H - k)//stride + 1."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, rng=None):
        rng = rng or np.random.default_rng()
        scale = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.W = Tensor(rng.normal(0.0, scale,
                                   size=(out_channels, in_channels,
                                         kernel_size, kernel_size)))
        self.b = Tensor(np.zeros(out_channels, dtype=np.float64))
        self.stride = stride

    def forward(self, x, training=True):
        return conv2d(x, self.W, self.b, stride=self.stride)

    def parameters(self):
        return [self.W, self.b]


class MaxPool2D:
    def __init__(self, pool_size=2, stride=None):
        self.pool_size = pool_size
        self.stride = stride or pool_size

    def forward(self, x, training=True):
        return max_pool2d(x, self.pool_size, self.stride)

    def parameters(self):
        return []


class Flatten:
    def forward(self, x, training=True):
        return reshape(x, (x.data.shape[0], -1))

    def parameters(self):
        return []


class Dropout:
    def __init__(self, drop_prob=0.5, seed=0):
        if not 0.0 <= drop_prob < 1.0:
            raise ValueError("drop_prob must be in [0, 1)")
        self.drop_prob = drop_prob
        self._rng = np.random.default_rng(seed)

    def forward(self, x, training=True):
        return dropout(x, self.drop_prob, training=training, rng=self._rng)

    def parameters(self):
        return []


class BatchNorm:
    def __init__(self, dim, momentum=0.9, epsilon=1e-5):
        self.gamma = Tensor(np.ones((1, dim), dtype=np.float64))
        self.beta = Tensor(np.zeros((1, dim), dtype=np.float64))
        self.momentum = momentum
        self.epsilon = epsilon
        self.running_mean = np.zeros((1, dim), dtype=np.float64)
        self.running_var = np.ones((1, dim), dtype=np.float64)

    def forward(self, x, training=True):
        return batch_norm(x, self.gamma, self.beta,
                          self.running_mean, self.running_var,
                          momentum=self.momentum, epsilon=self.epsilon,
                          training=training)

    def parameters(self):
        return [self.gamma, self.beta]


class ReLU:
    def forward(self, x, training=True):
        return x.relu()

    def parameters(self):
        return []


class Sigmoid:
    def forward(self, x, training=True):
        return x.sigmoid()

    def parameters(self):
        return []


class Tanh:
    def forward(self, x, training=True):
        return x.tanh()

    def parameters(self):
        return []


class Sequential:
    """A model: a list of layers.  Forward builds the graph through all
    of them; parameters() collects every parameter tensor exactly once.
    `training` is threaded through so Dropout and BatchNorm know whether
    this is a training or inference pass."""

    def __init__(self, *layers):
        self.layers = list(layers)

    def forward(self, x, training=True):
        for layer in self.layers:
            x = layer.forward(x, training=training)
        return x

    def predict(self, x):
        """Class indices for a batch, no graph needed (fast path)."""
        x = np.asarray(x, dtype=np.float64)
        for layer in self.layers:
            if isinstance(layer, (Dropout, BatchNorm)):
                x = layer.forward(Tensor(x), training=False).data
            elif isinstance(layer, (ReLU, Sigmoid, Tanh)):
                x = layer.forward(Tensor(x), training=False).data
            elif isinstance(layer, Linear):
                x = x @ layer.W.data + layer.b.data
            elif isinstance(layer, Conv2D):
                x = _conv_np(x, layer.W.data, layer.b.data, layer.stride)
            elif isinstance(layer, MaxPool2D):
                x = _pool_np(x, layer.pool_size, layer.stride)
            elif isinstance(layer, Flatten):
                x = x.reshape(x.shape[0], -1)
            else:
                x = layer.forward(Tensor(x), training=False).data
        return x.argmax(axis=1)

    def softmax_probs(self, x):
        """Row-wise softmax probabilities for a batch (fast path)."""
        x = np.asarray(x, dtype=np.float64)
        for layer in self.layers:
            if isinstance(layer, (Dropout, BatchNorm)):
                x = layer.forward(Tensor(x), training=False).data
            elif isinstance(layer, (ReLU, Sigmoid, Tanh)):
                x = layer.forward(Tensor(x), training=False).data
            elif isinstance(layer, Linear):
                x = x @ layer.W.data + layer.b.data
            elif isinstance(layer, Conv2D):
                x = _conv_np(x, layer.W.data, layer.b.data, layer.stride)
            elif isinstance(layer, MaxPool2D):
                x = _pool_np(x, layer.pool_size, layer.stride)
            elif isinstance(layer, Flatten):
                x = x.reshape(x.shape[0], -1)
            else:
                x = layer.forward(Tensor(x), training=False).data
        m = x.max(axis=1, keepdims=True)
        e = np.exp(x - m)
        return e / e.sum(axis=1, keepdims=True)

    def parameters(self):
        params = []
        seen = set()
        for layer in self.layers:
            for p in layer.parameters():
                if id(p) not in seen:
                    seen.add(id(p))
                    params.append(p)
        return params

    def __len__(self):
        return len(self.layers)


def _conv_np(x, w, b, stride):
    """NumPy-only conv forward for the predict fast path."""
    from .autodiff import _get_patches
    n, c_in, h, w_in = x.shape
    c_out, _, k, _ = w.shape
    out_h = (h - k) // stride + 1
    out_w = (w_in - k) // stride + 1
    patches = _get_patches(x, k, k, stride).transpose(0, 4, 5, 1, 2, 3)
    cols = patches.reshape(n * out_h * out_w, -1)
    out = (cols @ w.reshape(c_out, -1).T + b.reshape(1, -1))
    return out.reshape(n, out_h, out_w, c_out).transpose(0, 3, 1, 2)


def _pool_np(x, pool_size, stride):
    from .autodiff import _get_patches
    n, c, h, w = x.shape
    k, s = pool_size, stride
    out_h = (h - k) // s + 1
    out_w = (w - k) // s + 1
    patches = _get_patches(x, k, k, s)
    return patches.reshape(n, c, k * k, out_h, out_w).max(axis=2)
