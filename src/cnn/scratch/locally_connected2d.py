"""
Lapisan LocallyConnected2D dari nol (NumPy only).
Berbeda dari Conv2D: setiap posisi output punya filter sendiri (TIDAK di-share).
Mendukung forward pass, backward pass, dan batch inference.
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))
from activations import get_activation


class LocallyConnected2D:
    """
    Lapisan konvolusi lokal TANPA parameter sharing.

    Berbeda dengan Conv2D:
    - Conv2D: filter yang SAMA digeser ke seluruh posisi (parameter sharing)
    - LocallyConnected2D: setiap posisi output punya filter UNIK

    Bobot disimpan sebagai: (output_rows * output_cols, kH * kW * C_in, C_out)
    artinya setiap posisi output memiliki bobot yang berbeda.

    Forward pass:
        Output[b, i, j, k] = sum over (di,dj,c) of:
            input[b, i*stride+di, j*stride+dj, c] * weights[i,j, di*kW*C_in+c, k]
            + bias[i, j, k]

    Args:
        filters (int): jumlah filter output per posisi (C_out).
        kernel_size (int atau tuple): ukuran kernel (kH, kW).
        strides (int atau tuple): langkah. Default (1, 1).
        padding (str): 'same' atau 'valid'. Default 'same'.
        activation (str atau None): fungsi aktivasi. Default 'relu'.
    """

    def __init__(self, filters, kernel_size, strides=(1, 1), padding='same',
                 activation='relu'):
        self.filters = filters

        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = tuple(kernel_size)

        if isinstance(strides, int):
            self.strides = (strides, strides)
        else:
            self.strides = tuple(strides)

        self.padding = padding.lower()
        self.activation_name = activation

        # Bobot dan bias
        self.weights = None
        self.bias = None
        self.out_rows = None
        self.out_cols = None
        self.C_in = None

        self.activation_fn, _ = get_activation(activation)

        # Cache
        self.input_cache = None
        self.output_cache = None

    def _init_weights(self, C_in, out_rows, out_cols):
        """
        Inisialisasi bobot untuk setiap posisi output.

        Args:
            C_in (int): jumlah channel input.
            out_rows (int): jumlah baris output spatial.
            out_cols (int): jumlah kolom output spatial.
        """
        kH, kW = self.kernel_size
        patch_size = kH * kW * C_in
        num_positions = out_rows * out_cols

        std = np.sqrt(2.0 / (kH * kW * C_in))
        self.weights = np.random.randn(num_positions, patch_size,
                                       self.filters).astype(np.float64) * std
        self.bias = np.zeros((num_positions, self.filters), dtype=np.float64)
        self.C_in = C_in
        self.out_rows = out_rows
        self.out_cols = out_cols

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

        H_out = int(np.ceil(H / sh))
        W_out = int(np.ceil(W / sw))

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
        Forward pass LocallyConnected2D.

        Args:
            x: array numpy, bentuk (N, H, W, C_in).
        Returns:
            output: array numpy, bentuk (N, H_out, W_out, filters).
        """
        N, H, W, C = x.shape

        kH, kW = self.kernel_size
        sh, sw = self.strides

        if self.padding == 'same':
            H_out = int(np.ceil(H / sh))
            W_out = int(np.ceil(W / sw))
        else:
            H_out = (H - kH) // sh + 1
            W_out = (W - kW) // sw + 1

        # Inisialisasi bobot jika belum ada
        if self.weights is None:
            self._init_weights(C, H_out, W_out)

        # Padding
        x_padded = self._pad_input(x)

        # Output array
        output = np.zeros((N, H_out, W_out, self.filters), dtype=np.float64)

        patch_size = kH * kW * C
        pos_idx = 0
        for i in range(H_out):
            for j in range(W_out):
                r0 = i * sh
                c0 = j * sw
                patch = x_padded[:, r0:r0+kH, c0:c0+kW, :]
                patch_flat = patch.reshape(N, patch_size)

                # Dot product: (N, patch_size) @ (patch_size, filters) -> (N, filters)
                output[:, i, j, :] = patch_flat @ self.weights[pos_idx] + self.bias[pos_idx]
                pos_idx += 1

        # Cache
        self.input_cache = x
        self.output_cache = output.copy()

        # Aktivasi
        if self.activation_fn is not None:
            output = self.activation_fn(output)

        return output

    def backward(self, dout):
        """
        Backward pass LocallyConnected2D.

        Args:
            dout: gradient dari layer atas, bentuk (N, H_out, W_out, filters).
        Returns:
            dx: gradient terhadap input, bentuk (N, H, W, C_in).
        """
        N, H, W, C = self.input_cache.shape
        kH, kW = self.kernel_size
        sh, sw = self.strides

        if self.padding == 'same':
            H_out = int(np.ceil(H / sh))
            W_out = int(np.ceil(W / sw))
        else:
            H_out = (H - kH) // sh + 1
            W_out = (W - kW) // sw + 1

        # Gradient sebelum aktivasi
        if self.activation_fn is not None and self.output_cache is not None:
            z = self.output_cache
            mask = (z > 0).astype(np.float64)
            dz = dout * mask
        else:
            dz = dout

        # Padding input
        x_padded = self._pad_input(self.input_cache)
        N_pad, H_pad, W_pad, C_pad = x_padded.shape

        # Gradient arrays
        num_positions = H_out * W_out
        patch_size = kH * kW * C_pad

        dweights = np.zeros_like(self.weights)
        dbias = np.zeros_like(self.bias)
        dx_padded = np.zeros((N_pad, H_pad, W_pad, C_pad), dtype=np.float64)

        pos_idx = 0
        for i in range(H_out):
            for j in range(W_out):
                r0 = i * sh
                c0 = j * sw

                patch = x_padded[:, r0:r0+kH, c0:c0+kW, :]
                patch_flat = patch.reshape(N, patch_size)

                dz_pos = dz[:, i, j, :]

                # dL/dw_pos = patch_flat.T @ dz_pos
                dweights[pos_idx] = patch_flat.T @ dz_pos

                # dL/db_pos = sum over batch
                dbias[pos_idx] = np.sum(dz_pos, axis=0)

                # dL/dpatch_flat = dz_pos @ w_pos.T
                dpatch_flat = dz_pos @ self.weights[pos_idx].T
                dpatch = dpatch_flat.reshape(N, kH, kW, C_pad)

                # Accumulate gradient ke dx_padded
                dx_padded[:, r0:r0+kH, c0:c0+kW, :] += dpatch

                pos_idx += 1

        # Crop padding
        if self.padding == 'same':
            pad_h_top = (H_pad - H) // 2
            pad_w_left = (W_pad - W) // 2
            dx = dx_padded[:, pad_h_top:pad_h_top+H,
                           pad_w_left:pad_w_left+W, :]
        else:
            dx = dx_padded

        self._dweights = dweights
        self._dbias = dbias

        return dx

    def get_weights(self):
        """Kembalikan bobot dan bias."""
        return self.weights, self.bias

    def set_weights(self, weights, bias):
        """
        Setel bobot dari sumber eksternal.

        Args:
            weights: array, bentuk (out_rows*out_cols, kH*kW*C_in, C_out).
            bias: array, bentuk (out_rows*out_cols, C_out).
        """
        self.weights = weights.astype(np.float64)
        self.bias = bias.astype(np.float64)
        self.filters = bias.shape[1]
        self.out_rows = None  # Caller harus set manual jika perlu
        self.out_cols = None
        self.C_in = None

    def get_grad_weights(self):
        """Kembalikan gradient bobot."""
        return self._dweights, self._dbias

    def summary(self):
        """Kembalikan info lapisan."""
        return (f"LocallyConnected2D(filters={self.filters}, "
                f"kernel_size={self.kernel_size}, strides={self.strides}, "
                f"padding='{self.padding}', activation='{self.activation_name}')")