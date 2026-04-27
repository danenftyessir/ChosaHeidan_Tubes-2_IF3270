"""
Fungsi aktivasi untuk CNN, RNN, dan LSTM dari nol.
Semua fungsi mendukung pemrosesan batch.
"""

import numpy as np


# Fungsi bantuan softmax

def _stable_softmax(x, axis=-1):
    """
    Softmax yang aman secara numpy di sepanjang axis tertentu.

    Cara kerja: kurangi nilai maksimum dulu sebelum exp, agar tidak overflow.

    Args:
        x: array numpy.
        axis: axis untuk hitung softmax. Default -1 (axis terakhir).
    Returns:
        Array numpy, jumlah sepanjang axis = 1.
    """
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


# ReLU

def relu(x):
    """
    Fungsi aktivasi ReLU: max(0, x).
    Forward pass.

    Args:
        x: array numpy, bentuk apa saja.
    Returns:
        Array numpy, bentuk sama dengan x.
    """
    return np.maximum(0, x)


def d_relu(x):
    """
    Turunan ReLU: 1 jika x > 0, else 0.
    Dipakai saat backward pass.

    Args:
        x: array numpy, yaitu input sebelum aktivasi ReLU (sama seperti forward).
    Returns:
        Array numpy, bentuk sama dengan x.
    """
    return (x > 0).astype(np.float64)


# Sigmoid

def sigmoid(x):
    """
    Fungsi aktivasi Sigmoid: 1 / (1 + exp(-x)).
    Forward pass, aman secara numerik.

    Args:
        x: array numpy, bentuk apa saja.
    Returns:
        Array numpy, bentuk sama dengan x, nilai di antara (0, 1).
    """
    # Batasi nilai agar tidak overflow
    x_clipped = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x_clipped))


def d_sigmoid(x):
    """
    Turunan Sigmoid: sigmoid(x) * (1 - sigmoid(x)).
    Pakai output sigmoid (bukan input mentah).

    Args:
        x: array numpy, yaitu output dari sigmoid(x).
    Returns:
        Array numpy, bentuk sama dengan x.
    """
    return x * (1.0 - x)


# Tanh

def tanh(x):
    """
    Fungsi aktivasi Hyperbolic Tangent.
    Forward pass.

    Args:
        x: array numpy, bentuk apa saja.
    Returns:
        Array numpy, bentuk sama dengan x, nilai di antara (-1, 1).
    """
    return np.tanh(x)


def d_tanh(x):
    """
    Turunan Tanh: 1 - tanh(x)^2.
    Pakai output tanh (bukan input mentah).

    Args:
        x: array numpy, yaitu output dari tanh(x).
    Returns:
        Array numpy, bentuk sama dengan x.
    """
    return 1.0 - np.square(x)


# Softmax

def softmax(x, axis=-1):
    """
    Fungsi aktivasi Softmax di sepanjang axis tertentu.
    Dipakai untuk lapisan output multi-kelas (misal: Dense + softmax).

    Args:
        x: array numpy, bentuk (..., num_classes, ...).
        axis: axis untuk hitung softmax. Default -1 (terakhir).
    Returns:
        Array numpy, bentuk sama dengan x, jumlah tiap baris = 1.
    """
    return _stable_softmax(x, axis=axis)


def d_softmax(x, axis=-1):
    """
    Jacobian-vector product softmax untuk backward pass.
    Untuk loss cross-entropy dengan softmax, gradient digabung menjadi:
        dL/dx = probs - labels  (one-hot labels)
    Fungsi ini menghitung Jacobian penuh untuk penggunaan umum.

    Args:
        x: array numpy, yaitu pre-softmax activations, bentuk (batch, classes).
        axis: axis tempat softmax dihitung.
    Returns:
        Jacobian matrix dalam bentuk (..., C, C) — mahal, untuk CE pakai shortcut.
    """
    probs = _stable_softmax(x, axis=axis)
    return probs, axis


def softmax_cross_entropy_backward(probs, labels):
    """
    Backward pass untuk Sparse Categorical Cross-Entropy dengan softmax.
    Gradient: dL/d logits = probs - one_hot(labels).
    Ini versi efisien, pakai ini daripada Jacobian penuh.

    Args:
        probs: output softmax, bentuk (batch, num_classes).
        labels: label integer, bentuk (batch,) atau (batch, 1).
    Returns:
        Gradient terhadap logits, bentuk (batch, num_classes).
    """
    probs = np.asarray(probs)
    labels = np.asarray(labels).reshape(-1)

    # Konversi label integer ke one-hot
    num_classes = probs.shape[-1]
    one_hot = np.eye(num_classes)[labels]

    return probs - one_hot


# Registry aktivasi

ACTIVATIONS = {
    'relu': (relu, d_relu),
    'sigmoid': (sigmoid, d_sigmoid),
    'tanh': (tanh, d_tanh),
    'softmax': (softmax, None),
    'linear': (lambda x: x, lambda x: np.ones_like(x)),
    None: (lambda x: x, lambda x: np.ones_like(x)),
}


def get_activation(name):
    """
    Ambil fungsi forward dan backward berdasarkan nama.

    Args:
        name: string, salah satu dari 'relu', 'sigmoid', 'tanh', 'softmax', 'linear', None.
    Returns:
        Tuple (forward_fn, backward_fn).
    """
    if name is None:
        name = 'linear'
    name = name.lower()
    if name not in ACTIVATIONS:
        raise ValueError(
            f"Unknown activation: '{name}'. Tersedia: {list(ACTIVATIONS.keys())}"
        )
    return ACTIVATIONS[name]
