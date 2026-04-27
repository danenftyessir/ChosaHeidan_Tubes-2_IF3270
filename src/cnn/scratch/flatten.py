"""
Lapisan Flatten dari nol (NumPy only).
Mengubah tensor 4D (N, H, W, C) menjadi vektor 2D (N, H*W*C).
Mendukung forward pass dan backward pass.
"""

import numpy as np


class Flatten:
    """
    Lapisan Flatten: reshape tensor menjadi vektor 1D.

    Forward pass:
        input shape:  (N, H, W, C)
        output shape: (N, H*W*C)     [row-major / C-order, konsisten dengan Keras]

    Backward pass:
        Reshape gradient kembali ke bentuk spatial asli.
    """

    def __init__(self, order='C'):
        """
        Args:
            order (str): 'C' untuk row-major (C-order, default Keras),
                        'F' untuk column-major (Fortran-order).
        """
        self.order = order.upper()
        self.input_shape = None

    def forward(self, x):
        """
        Forward pass Flatten.

        Args:
            x: array numpy, bentuk (N, H, W, C) atau (H, W, C).
        Returns:
            output: array numpy, bentuk (N, H*W*C).
        """
        self.input_shape = x.shape

        if self.order == 'C':
            output = x.reshape(x.shape[0], -1)
        else:
            output = x.reshape(x.shape[0], -1, order='F')

        return output

    def backward(self, dout):
        """
        Backward pass Flatten.

        Args:
            dout: gradient dari layer atas, bentuk (N, H*W*C).
        Returns:
            dx: gradient terhadap input, bentuk (N, H, W, C).
        """
        if self.order == 'C':
            dx = dout.reshape(self.input_shape)
        else:
            dx = dout.reshape(self.input_shape[0], *self.input_shape[1:], order='F')

        return dx

    def summary(self):
        """Kembalikan info lapisan."""
        return f"Flatten(order='{self.order}')"