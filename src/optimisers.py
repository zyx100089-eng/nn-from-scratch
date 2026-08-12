"""Optimisers: SGD, SGD with momentum, Adam.

Each optimiser steps a list of parameter tensors using their
accumulated `.grad` buffers, then zeroes those buffers.  Adam keeps
first/second moment estimates with bias correction (Kingma & Ba, 2015).
"""

from __future__ import annotations

import numpy as np

from .autodiff import Tensor


class SGD:
    def __init__(self, learning_rate=0.01, weight_decay=0.0):
        self.lr = learning_rate
        self.weight_decay = weight_decay

    def step(self, params):
        for p in params:
            grad = p.grad
            if self.weight_decay:
                grad = grad + self.weight_decay * p.data
            p.data -= self.lr * grad
            p.grad.fill(0.0)


class SGDMomentum:
    def __init__(self, learning_rate=0.01, momentum=0.9, weight_decay=0.0):
        self.lr = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self._velocities = {}

    def step(self, params):
        for p in params:
            if id(p) not in self._velocities:
                self._velocities[id(p)] = np.zeros_like(p.data)
            v = self._velocities[id(p)]
            grad = p.grad
            if self.weight_decay:
                grad = grad + self.weight_decay * p.data
            v[:] = self.momentum * v - self.lr * grad
            p.data += v
            p.grad.fill(0.0)


class Adam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999,
                 epsilon=1e-8, weight_decay=0.0):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self._m = {}
        self._v = {}
        self._t = 0

    def step(self, params):
        self._t += 1
        for p in params:
            if id(p) not in self._m:
                self._m[id(p)] = np.zeros_like(p.data)
                self._v[id(p)] = np.zeros_like(p.data)
            grad = p.grad
            if self.weight_decay:
                grad = grad + self.weight_decay * p.data
            m = self._m[id(p)]
            v = self._v[id(p)]
            m[:] = self.beta1 * m + (1 - self.beta1) * grad
            v[:] = self.beta2 * v + (1 - self.beta2) * grad ** 2
            m_hat = m / (1 - self.beta1 ** self._t)
            v_hat = v / (1 - self.beta2 ** self._t)
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
            p.grad.fill(0.0)
