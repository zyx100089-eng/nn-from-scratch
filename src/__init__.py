"""nn-from-scratch: a neural network framework and an adversarial study.

One autodiff engine, everything rebuilt on top of it: Linear, Conv2D,
MaxPool2D, Dropout, BatchNorm, activations, optimisers, the training
loop, and the FGSM/PGD attacks.
"""

from .autodiff import Tensor
from .attacks import (attack_success, fgsm, fgsm_targeted,
                      gradient_wrt_input, linearity_check, loss_of, pgd,
                      pgd_targeted, robustness_curve, target_success,
                      transfer_attack)
from .dataset import load_fashion_mnist, load_mnist, to_image_shape
from .nn import (BatchNorm, Conv2D, Dropout, Flatten, Linear, MaxPool2D,
                 ReLU, Sequential, Sigmoid, Tanh)
from .optimisers import Adam, SGD, SGDMomentum
from .train import train, train_adversarial

__all__ = [
    "Tensor",
    "Sequential", "Linear", "Conv2D", "MaxPool2D", "Flatten",
    "Dropout", "BatchNorm", "ReLU", "Sigmoid", "Tanh",
    "SGD", "SGDMomentum", "Adam",
    "train", "train_adversarial",
    "load_mnist", "load_fashion_mnist", "to_image_shape",
    "fgsm", "fgsm_targeted", "pgd", "pgd_targeted",
    "attack_success", "target_success", "transfer_attack",
    "robustness_curve", "linearity_check", "gradient_wrt_input", "loss_of",
]
