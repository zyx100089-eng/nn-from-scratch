"""Dataset loading: MNIST and Fashion-MNIST.

Both are read from the raw IDX binary format (magic number + dims +
big-endian byte arrays) straight out of the gzipped files, so no
external library is needed.  Files are cached to `data/` after the
first download.

If the download fails, MNIST falls back to a tiny synthetic dataset so
the rest of the project always runs.
"""

from __future__ import annotations

import gzip
import os
import struct
import urllib.request

import numpy as np

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

MNIST_MIRRORS = [
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
]
FASHION_BASE = "https://fashion-mnist.s3.amazonaws.com/"

MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}
FASHION_FILES = MNIST_FILES


def _read_idx(path: str) -> np.ndarray:
    """Parse an IDX file (gzipped or not) into a numpy array."""
    with gzip.open(path, "rb") if path.endswith(".gz") else open(path, "rb") as f:
        magic = struct.unpack(">I", f.read(4))[0]
        ndim = magic & 0xFF
        dims = struct.unpack(f">{ndim}I", f.read(4 * ndim))
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(dims)
    return data


def _download(name: str, files: dict, mirrors: list[str], subdir: str) -> str:
    path = os.path.join(_DATA_DIR, subdir, files[name])
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for base in mirrors:
        url = base + files[name]
        try:
            print(f"  downloading {url}")
            urllib.request.urlretrieve(url, path)
            return path
        except Exception as ex:
            print(f"  failed {url}: {ex}")
    raise RuntimeError(f"could not download {files[name]}")


def _load(split: str, max_n: int | None, files: dict, mirrors: list[str],
          subdir: str):
    if split not in ("train", "test"):
        raise ValueError("split must be 'train' or 'test'")
    if max_n is not None and max_n < 0:
        raise ValueError("max_n must be non-negative")
    prefix = "train" if split == "train" else "test"
    x_path = _download(f"{prefix}_images", files, mirrors, subdir)
    y_path = _download(f"{prefix}_labels", files, mirrors, subdir)
    X = _read_idx(x_path).astype(np.float64) / 255.0
    y = _read_idx(y_path).astype(np.int64)
    if max_n is not None:
        X, y = X[:max_n], y[:max_n]
    return X.reshape(X.shape[0], -1), y


def load_mnist(split: str = "train", max_n: int | None = None):
    """(X, y): X in [0,1] float, shape (N, 784); y int class indices."""
    try:
        return _load(split, max_n, MNIST_FILES, MNIST_MIRRORS, "mnist")
    except RuntimeError as ex:
        print(f"[dataset] MNIST unavailable ({ex}); using synthetic fallback")
        return _synthetic(12000 if split == "train" else 2000)


def load_fashion_mnist(split: str = "train", max_n: int | None = None):
    """Same interface as load_mnist, for Fashion-MNIST."""
    return _load(split, max_n, FASHION_FILES, [FASHION_BASE], "fashion_mnist")


def to_image_shape(X):
    """Reshape flattened (N, 784) images to (N, 1, 28, 28) for conv layers."""
    return X.reshape(-1, 1, 28, 28)


def _synthetic(n: int):
    """Gaussian blobs a linear model can separate (last resort)."""
    rng = np.random.default_rng(0)
    X = np.zeros((n, 784), dtype=np.float64)
    y = np.zeros(n, dtype=np.int64)
    for i in range(n):
        c = rng.integers(0, 10)
        y[i] = c
        r, cc = rng.integers(2, 26, size=2)
        rng2 = np.random.default_rng(i)
        for _ in range(30):
            X[i, (r + rng2.integers(-3, 4)) * 28 + (cc + rng2.integers(-3, 4))] += 1.0
    return np.clip(X, 0, 1), y
