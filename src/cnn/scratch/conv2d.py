"""
Lapisan Conv2D dari nol (NumPy only).
Mendukung forward pass, backward pass, dan batch inference.
Parameter di-share antar seluruh spatial location.
"""

import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))
from activations import get_activation


class Conv2D:
    """
    Lapisan konvolusi 2D dengan parameter sharing.

    Forward pass:
        Output[b, i, j, k] = sum over (di,dj,c) of:
            input[b, i*stride+di, j*stride+dj, c] * kernel[di, dj, c, k] + bias[k]

    Args:
        filters (int): jumlah filter output (C_out).
        kernel_size (int atau tuple): ukuran kernel (kH, kW).
            Jika int, kH = kW = kernel_size.
        strides (int atau tuple): langkah geser filter. Default (1, 1).
        padding (str): 'same' (output sama ukuran) atau 'valid' (tanpa padding).
            Default: 'same'.
        activation (str atau None): fungsi aktivasi setelah konvolusi.
            Default: 'relu'.
        kernel_regularizer (float): bobot regularisasi L2 (hanya untuk kompatibilitas).
    """

    def __init__(self, filters, kernel_size, strides=(1, 1), padding='same',
                 activation='relu', kernel_regularizer=0.0):
        self.filters = filters

        # Normalisasi kernel_size ke tuple (kH, kW)
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = tuple(kernel_size)

        # Normalisasi strides ke tuple (sh, sw)
        if isinstance(strides, int):
            self.strides = (strides, strides)
        else:
            self.strides = tuple(strides)

        self.padding = padding.lower()
        self.activation_name = activation

        # Bobot dan bias
        self.kernel = None
        self.bias = None
        self.C_in = None

        # Fungsi aktivasi
        self.activation_fn, _ = get_activation(activation)

        # Cache untuk backward pass
        self.input_cache = None
        self.output_cache = None
        self.output_after_act_cache = None

    def _init_weights(self, C_in):
        """
        Inisialisasi bobot kernel dan bias pakai He initialization.

        Args:
            C_in (int): jumlah channel input.
        """
        kH, kW = self.kernel_size
        std = np.sqrt(2.0 / (kH * kW * C_in))
        self.kernel = np.random.randn(kH, kW, C_in, self.filters).astype(np.float64) * std
        self.bias = np.zeros(self.filters, dtype=np.float64)
        self.C_in = C_in

    def _pad_input(self, x):
        """
        Tambahkan padding ke input jika padding='same'.

        Args:
            x: input tensor, bentuk (N, H, W, C).
        Returns:
            x_padded: tensor dengan padding.
        """
        if self.padding == 'valid':
            return x

        N, H, W, C = x.shape
        kH, kW = self.kernel_size
        sh, sw = self.strides

        # Hitung output spatial dimensions
        H_out = (H + sh - 1) // sh
        W_out = (W + sw - 1) // sw

        # Hitung padding yang dibutuhkan
        pad_h_total = (H_out - 1) * sh + kH - H
        pad_w_total = (W_out - 1) * sw + kW - W

        pad_h_top = pad_h_total // 2
        pad_h_bottom = pad_h_total - pad_h_top
        pad_w_left = pad_w_total // 2
        pad_w_right = pad_w_total - pad_w_left

        return np.pad(x,
                      ((0, 0), (pad_h_top, pad_h_bottom),
                       (pad_w_left, pad_w_right), (0, 0)),
                      mode='constant', constant_values=0)

    def forward(self, x):
        """
        Forward pass untuk lapisan Conv2D.

        Args:
            x: array numpy, bentuk (N, H, W, C_in). N = batch_size.
        Returns:
            output: array numpy, bentuk (N, H_out, W_out, filters).
        """
        N, H, W, C = x.shape

        # Inisialisasi bobot jika belum ada
        if self.kernel is None:
            self._init_weights(C)

        kH, kW = self.kernel_size
        sh, sw = self.strides

        # Hitung output spatial dimensions
        if self.padding == 'same':
            H_out = int(np.ceil(H / sh))
            W_out = int(np.ceil(W / sw))
        else:
            H_out = (H - kH) // sh + 1
            W_out = (W - kW) // sw + 1

        # Padding input
        x_padded = self._pad_input(x)

        # Bangun matriks kolom: extract semua patch ke kolom
        col_out = np.zeros((N * H_out * W_out, kH * kW * C), dtype=np.float64)

        idx = 0
        for b in range(N):
            for i in range(H_out):
                for j in range(W_out):
                    r0 = i * sh
                    c0 = j * sw
                    patch = x_padded[b, r0:r0 + kH, c0:c0 + kW, :]
                    col_out[idx] = patch.flatten()
                    idx += 1

        # Kernel: (kH, kW, C_in, filters) -> (kH*kW*C_in, filters)
        kernel_flat = self.kernel.reshape(-1, self.filters)

        # Matrix multiply: (N*H_out*W_out, kH*kW*C_in) @ (kH*kW*C_in, filters)
        out_cols = col_out @ kernel_flat

        # Tambah bias (broadcast)
        out_cols += self.bias

        # Reshape ke (N, H_out, W_out, filters)
        output = out_cols.reshape(N, H_out, W_out, self.filters)

        # Cache untuk backward
        self.input_cache = x
        self.output_cache = output.copy()

        # Aplikasikan aktivasi
        if self.activation_fn is not None:
            output = self.activation_fn(output)
            self.output_after_act_cache = output

        return output

    def backward(self, dout):
        """
        Backward pass untuk lapisan Conv2D.

        Args:
            dout: gradient dari layer atas, bentuk (N, H_out, W_out, filters).
        Returns:
            dx: gradient terhadap input, bentuk (N, H, W, C_in).
        """
        N, H, W, C = self.input_cache.shape
        kH, kW = self.kernel_size
        sh, sw = self.strides

        # Gradient sebelum aktivasi (dz)
        if self.activation_fn is not None and self.output_cache is not None:
            z = self.output_cache
            mask = (z > 0).astype(np.float64)
            dz = dout * mask
        else:
            dz = dout

        # Pad input untuk backward
        x_padded = self._pad_input(self.input_cache)
        N2, H2, W2, C2 = x_padded.shape

        # Hitung H_out, W_out
        if self.padding == 'same':
            H_out = int(np.ceil(H / sh))
            W_out = int(np.ceil(W / sw))
        else:
            H_out = (H - kH) // sh + 1
            W_out = (W - kW) // sw + 1

        # Gradient terhadap bias: dL/db = sum(dz)
        db = np.sum(dz, axis=(0, 1, 2))

        # Gradient terhadap kernel
        dz_flat = dz.reshape(N * H_out * W_out, self.filters)

        col_out = np.zeros((N * H_out * W_out, kH * kW * C), dtype=np.float64)
        idx = 0
        for b in range(N):
            for i in range(H_out):
                for j in range(W_out):
                    r0 = i * sh
                    c0 = j * sw
                    patch = x_padded[b, r0:r0+kH, c0:c0+kW, :].flatten()
                    col_out[idx] = patch
                    idx += 1

        dkernel_flat = col_out.T @ dz_flat
        dkernel = dkernel_flat.reshape(kH, kW, C, self.filters)

        # Gradient terhadap input
        kernel_flat = self.kernel.reshape(-1, self.filters)
        dcol = dz_flat @ kernel_flat.T

        dx_padded = np.zeros((N2, H2, W2, C2), dtype=np.float64)

        idx = 0
        for b in range(N):
            for i in range(H_out):
                for j in range(W_out):
                    r0 = i * sh
                    c0 = j * sw
                    grad_patch = dcol[idx].reshape(kH, kW, C2)
                    dx_padded[b, r0:r0+kH, c0:c0+kW, :] += grad_patch
                    idx += 1

        # Crop padding jika ada
        if self.padding == 'same':
            N_in, H_in, W_in, C_in = self.input_cache.shape
            _, H_pad, W_pad, _ = dx_padded.shape
            pad_h_top = (H_pad - H_in) // 2
            pad_w_left = (W_pad - W_in) // 2
            dx = dx_padded[:, pad_h_top:pad_h_top+H_in,
                           pad_w_left:pad_w_left+W_in, :]
        else:
            dx = dx_padded

        self._dkernel = dkernel
        self._dbias = db

        return dx

    def get_weights(self):
        """Kembalikan bobot kernel dan bias."""
        return self.kernel, self.bias

    def set_weights(self, kernel, bias):
        """
        Setel bobot dari sumber eksternal (misal: Keras).

        Args:
            kernel: array, bentuk (kH, kW, C_in, C_out).
            bias: array, bentuk (C_out,).
        """
        self.kernel = kernel.astype(np.float64)
        self.bias = bias.astype(np.float64)
        self.C_in = kernel.shape[2]
        self.filters = kernel.shape[3]

    def get_grad_weights(self):
        """Kembalikan gradient bobot (harus dipanggil setelah backward)."""
        return self._dkernel, self._dbias

    def summary(self):
        """Kembalikan info lapisan."""
        return (f"Conv2D(filters={self.filters}, kernel_size={self.kernel_size}, "
                f"strides={self.strides}, padding='{self.padding}', "
                f"activation='{self.activation_name}')")