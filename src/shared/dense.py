"""
Lapisan Dense (Fully Connected) dari nol.
Bisa dipakai ulang oleh CNN, RNN, dan LSTM.
Implementasi forward dan backward propagation pakai NumPy saja.
Mendukung pemrosesan batch.
"""

import numpy as np

from .activations import get_activation


class Dense:
    """
    Lapisan Dense / Fully Connected.

    Forward pass:
        z = X @ W + b
        a = activation(z)

    Backward pass:
        - dL/dz = dL/da * d_activation/dz
        - dL/dW = X.T @ dL/dz
        - dL/dX = dL/dz @ W.T
        - dL/db = sum(dL/dz, axis=0)

    Inisialisasi bobot: He (Kaiming) untuk keluarga ReLU.

    Args:
        input_dim (int): jumlah fitur input.
        units (int): jumlah unit output.
        activation (str atau None): nama fungsi aktivasi.
            Pilihan: 'relu', 'sigmoid', 'tanh', 'softmax', 'linear', None.
        kernel_regularizer (float): koefisien regularisasi L2.
        bias_regularizer (float): sama untuk bias.
    """

    def __init__(self, input_dim, units, activation='relu',
                 kernel_regularizer=0.0, bias_regularizer=0.0):
        self.input_dim = input_dim
        self.units = units
        self.activation_name = activation

        # Bobot dan bias
        self.weights = None
        self.bias = None
        self._init_weights()

        # Fungsi aktivasi
        self.forward_fn, self.backward_fn = get_activation(activation)

        # Cache untuk backward pass
        self.input_cache = None
        self.z_cache = None

    def _init_weights(self):
        """Inisialisasi bobot pakai He initialization."""
        std = np.sqrt(2.0 / self.input_dim)
        self.weights = np.random.randn(self.input_dim, self.units).astype(np.float64) * std
        self.bias = np.zeros(self.units, dtype=np.float64)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: array numpy, bentuk (batch_size, input_dim) atau (input_dim,).
               Untuk single sample, otomatis di-expand jadi batch of 1.
        Returns:
            output: array numpy, bentuk (batch_size, units) atau (units,).
        """
        # Pastikan ada batch dimension
        if x.ndim == 1:
            x = x.reshape(1, -1)
        self.input_cache = x

        # Transformasi linear: z = X @ W + b
        z = x @ self.weights + self.bias
        self.z_cache = z

        # Aplikasikan aktivasi
        output = self.forward_fn(z)
        return output

    def backward(self, dout):
        """
        Backward pass: hitung gradient terhadap input, bobot, dan bias.

        Args:
            dout: upstream gradient, bentuk (batch_size, units).
                  Untuk CE+softmax: sudah dalam bentuk (probs - labels).
        Returns:
            dx: gradient terhadap input X, bentuk (batch_size, input_dim).
        """
        if dout.ndim == 1:
            dout = dout.reshape(1, -1)

        z = self.z_cache
        x = self.input_cache

        # dL/dz = dL/da * d_activation/dz
        if self.backward_fn is not None:
            if self.activation_name == 'softmax':
                dz = dout
            else:
                da = dout
                act_out = self.backward_fn(z + 1e-8)
                if act_out.ndim == da.ndim:
                    dz = da * act_out
                else:
                    dz = da * act_out.reshape(da.shape)
        else:
            dz = dout

        # Gradient terhadap bobot: dL/dW = X.T @ dL/dz
        dW = x.T @ dz

        # Gradient terhadap bias: dL/db = sum over batch
        db = np.sum(dz, axis=0)

        # Gradient terhadap input: dL/dX = dL/dz @ W.T
        dx = dz @ self.weights.T

        self._grad_weights = dW
        self._grad_bias = db

        return dx

    def get_weights(self):
        """Kembalikan bobot dan bias."""
        return self.weights, self.bias

    def set_weights(self, weights, bias):
        """Setel bobot dari sumber eksternal (misal: Keras)."""
        self.weights = weights.astype(np.float64)
        self.bias = bias.astype(np.float64)
        self.input_dim = weights.shape[0]
        self.units = weights.shape[1]

    def get_grad_weights(self):
        """Kembalikan gradient bobot (harus dipanggil setelah backward)."""
        return self._grad_weights, self._grad_bias

    def summary(self):
        """Kembalikan info lapisan."""
        return (f"Dense(input_dim={self.input_dim}, units={self.units}, "
                f"activation='{self.activation_name}')")
