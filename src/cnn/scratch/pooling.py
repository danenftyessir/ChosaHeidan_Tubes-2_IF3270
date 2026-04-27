"""
Lapisan Pooling dari nol (NumPy only).
Termasuk MaxPooling2D, AveragePooling2D, GlobalMaxPooling2D, GlobalAveragePooling2D.
Mendukung forward pass, backward pass, dan batch inference.
"""

import numpy as np


# Fungsi bantuan: hitung dimensi output spatial untuk pooling

def _compute_output_shape(H, W, pool_size, strides):
    """
    Hitung dimensi output spatial untuk pooling.

    Args:
        H, W: dimensi input spatial.
        pool_size: tuple (ph, pw).
        strides: tuple (sh, sw).
    Returns:
        H_out, W_out: dimensi output spatial.
    """
    if strides is None:
        strides = pool_size
    sh, sw = strides
    ph, pw = pool_size

    H_out = (H - ph) // sh + 1
    W_out = (W - pw) // sw + 1
    return H_out, W_out


def _extract_windows(x, pool_size, strides):
    """
    Extract semua window pooling dari input.

    Args:
        x: array, bentuk (N, H, W, C).
        pool_size: (ph, pw).
        strides: (sh, sw).
    Returns:
        windows: array, bentuk (N, H_out, W_out, C, ph, pw).
    """
    N, H, W, C = x.shape
    ph, pw = pool_size
    sh, sw = strides
    H_out, W_out = _compute_output_shape(H, W, pool_size, strides)

    windows = np.zeros((N, H_out, W_out, C, ph, pw), dtype=np.float64)

    for i in range(H_out):
        for j in range(W_out):
            r0 = i * sh
            c0 = j * sw
            windows[:, i, j, :, :, :] = x[:, r0:r0+ph, c0:c0+pw, :]

    return windows


# MaxPooling2D

class MaxPooling2D:
    """
    Lapisan Max Pooling 2D.

    Forward pass:
        output[b, i, j, c] = max over window of input[b, i*sh:i*sh+ph, j*sw:j*sw+pw, c]

    Backward pass:
        Gradient mengalir HANYA ke posisi yang berisi nilai maksimum (mask-based).
        Posisi lain mendapat gradient = 0.
    """

    def __init__(self, pool_size=(2, 2), strides=None):
        if isinstance(pool_size, int):
            self.pool_size = (pool_size, pool_size)
        else:
            self.pool_size = tuple(pool_size)

        if strides is None:
            self.strides = self.pool_size
        elif isinstance(strides, int):
            self.strides = (strides, strides)
        else:
            self.strides = tuple(strides)

        # Cache untuk backward
        self.input_cache = None
        self.mask_cache = None

    def forward(self, x):
        """
        Forward pass MaxPooling2D.

        Args:
            x: array numpy, bentuk (N, H, W, C).
        Returns:
            output: array numpy, bentuk (N, H_out, W_out, C).
        """
        N, H, W, C = x.shape
        ph, pw = self.pool_size
        sh, sw = self.strides

        H_out, W_out = _compute_output_shape(H, W, self.pool_size, self.strides)

        output = np.zeros((N, H_out, W_out, C), dtype=np.float64)
        mask = np.zeros_like(x)

        for i in range(H_out):
            for j in range(W_out):
                r0 = i * sh
                c0 = j * sw
                window = x[:, r0:r0+ph, c0:c0+pw, :]

                max_val = np.max(window, axis=(1, 2))
                output[:, i, j, :] = max_val

                # Buat mask untuk backward
                for b in range(N):
                    for c_idx in range(C):
                        max_r, max_c = np.unravel_index(
                            window[b, :, :, c_idx].argmax(),
                            (ph, pw)
                        )
                        mask[b, r0 + max_r, c0 + max_c, c_idx] = 1.0

        self.input_cache = x
        self.mask_cache = mask

        return output

    def backward(self, dout):
        """
        Backward pass MaxPooling2D.

        Args:
            dout: gradient dari layer atas, bentuk (N, H_out, W_out, C).
        Returns:
            dx: gradient terhadap input, bentuk (N, H, W, C).
        """
        N, H, W, C = self.input_cache.shape
        ph, pw = self.pool_size
        sh, sw = self.strides
        H_out, W_out = _compute_output_shape(H, W, self.pool_size, self.strides)

        dx = np.zeros_like(self.input_cache)

        for i in range(H_out):
            for j in range(W_out):
                r0 = i * sh
                c0 = j * sw
                dx[:, r0:r0+ph, c0:c0+pw, :] += \
                    self.mask_cache[:, r0:r0+ph, c0:c0+pw, :] * dout[:, i:i+1, j:j+1, :]

        return dx

    def summary(self):
        """Kembalikan info lapisan."""
        return f"MaxPooling2D(pool_size={self.pool_size}, strides={self.strides})"


# AveragePooling2D

class AveragePooling2D:
    """
    Lapisan Average Pooling 2D.

    Forward pass:
        output[b, i, j, c] = mean over window of input[b, ...]

    Backward pass:
        Gradient didistribusikan merata ke seluruh posisi dalam window:
        dL/dinput[pos] = dL/doutput / (ph * pw)
    """

    def __init__(self, pool_size=(2, 2), strides=None):
        if isinstance(pool_size, int):
            self.pool_size = (pool_size, pool_size)
        else:
            self.pool_size = tuple(pool_size)

        if strides is None:
            self.strides = self.pool_size
        elif isinstance(strides, int):
            self.strides = (strides, strides)
        else:
            self.strides = tuple(strides)

        self.input_cache = None

    def forward(self, x):
        """
        Forward pass AveragePooling2D.

        Args:
            x: array numpy, bentuk (N, H, W, C).
        Returns:
            output: array numpy, bentuk (N, H_out, W_out, C).
        """
        N, H, W, C = x.shape
        ph, pw = self.pool_size
        sh, sw = self.strides

        H_out, W_out = _compute_output_shape(H, W, self.pool_size, self.strides)

        output = np.zeros((N, H_out, W_out, C), dtype=np.float64)

        for i in range(H_out):
            for j in range(W_out):
                r0 = i * sh
                c0 = j * sw
                window = x[:, r0:r0+ph, c0:c0+pw, :]
                avg_val = np.mean(window, axis=(1, 2))
                output[:, i, j, :] = avg_val

        self.input_cache = x

        return output

    def backward(self, dout):
        """
        Backward pass AveragePooling2D.

        Gradient didistribusikan merata: gradient_per_pos = dout / (ph * pw)

        Args:
            dout: gradient dari layer atas, bentuk (N, H_out, W_out, C).
        Returns:
            dx: gradient terhadap input, bentuk (N, H, W, C).
        """
        N, H, W, C = self.input_cache.shape
        ph, pw = self.pool_size
        sh, sw = self.strides
        H_out, W_out = _compute_output_shape(H, W, self.pool_size, self.strides)

        dx = np.zeros_like(self.input_cache)
        pool_area = ph * pw

        for i in range(H_out):
            for j in range(W_out):
                r0 = i * sh
                c0 = j * sw
                dx[:, r0:r0+ph, c0:c0+pw, :] += \
                    dout[:, i:i+1, j:j+1, :] / pool_area

        return dx

    def summary(self):
        """Kembalikan info lapisan."""
        return f"AveragePooling2D(pool_size={self.pool_size}, strides={self.strides})"


# GlobalAveragePooling2D

class GlobalAveragePooling2D:
    """
    Global Average Pooling: rata-rata SELURUH spatial dimension (H x W) per channel.

    Forward pass:
        output[b, c] = mean over all (h, w) of input[b, h, w, c]
        Output shape: (N, C) — spatial dimension hilang

    Backward pass:
        Gradient broadcast ke seluruh spatial, bagi rata:
            dx[b, h, w, c] = dout[b, c] / (H * W)
    """

    def __init__(self):
        self.input_cache = None
        self.spatial_shape = None

    def forward(self, x):
        """
        Forward pass GlobalAveragePooling2D.

        Args:
            x: array numpy, bentuk (N, H, W, C).
        Returns:
            output: array numpy, bentuk (N, C).
        """
        N, H, W, C = x.shape
        output = np.mean(x, axis=(1, 2))

        self.input_cache = x
        self.spatial_shape = (H, W)

        return output

    def backward(self, dout):
        """
        Backward pass GlobalAveragePooling2D.

        Args:
            dout: gradient dari layer atas, bentuk (N, C).
        Returns:
            dx: gradient terhadap input, bentuk (N, H, W, C).
        """
        N, H, W, C = self.input_cache.shape
        H_avg, W_avg = self.spatial_shape

        dx = dout[:, np.newaxis, np.newaxis, :] / (H_avg * W_avg)
        dx = np.broadcast_to(dx, (N, H, W, C)).astype(np.float64)

        return dx

    def summary(self):
        """Kembalikan info lapisan."""
        return "GlobalAveragePooling2D()"


# GlobalMaxPooling2D

class GlobalMaxPooling2D:
    """
    Global Max Pooling: nilai maksimum SELURUH spatial dimension (H x W) per channel.

    Forward pass:
        output[b, c] = max over all (h, w) of input[b, h, w, c]
        Output shape: (N, C)

    Backward pass:
        Gradient mengalir ke posisi max (one-hot per spatial).
            dx[b, h, w, c] = dout[b, c] jika (h,w) = argmax of input[b,:,:,c]
    """

    def __init__(self):
        self.input_cache = None
        self.max_mask = None

    def forward(self, x):
        """
        Forward pass GlobalMaxPooling2D.

        Args:
            x: array numpy, bentuk (N, H, W, C).
        Returns:
            output: array numpy, bentuk (N, C).
        """
        N, H, W, C = x.shape
        output = np.max(x, axis=(1, 2))

        max_mask = np.zeros_like(x, dtype=np.float64)
        for b in range(N):
            for c_idx in range(C):
                flat_idx = x[b, :, :, c_idx].argmax()
                r, c = np.unravel_index(flat_idx, (H, W))
                max_mask[b, r, c, c_idx] = 1.0

        self.input_cache = x
        self.max_mask = max_mask

        return output

    def backward(self, dout):
        """
        Backward pass GlobalMaxPooling2D.

        Args:
            dout: gradient dari layer atas, bentuk (N, C).
        Returns:
            dx: gradient terhadap input, bentuk (N, H, W, C).
        """
        N, H, W, C = self.input_cache.shape
        dx = self.max_mask * dout[:, np.newaxis, np.newaxis, :]

        return dx

    def summary(self):
        """Kembalikan info lapisan."""
        return "GlobalMaxPooling2D()"


# Pooling factory

POOLING_TYPES = {
    'max': MaxPooling2D,
    'average': AveragePooling2D,
    'avg': AveragePooling2D,
    'global_max': GlobalMaxPooling2D,
    'global_average': GlobalAveragePooling2D,
    'global_avg': GlobalAveragePooling2D,
}


def get_pooling_layer(pool_type, **kwargs):
    """
    Factory function untuk mendapatkan lapisan pooling.

    Args:
        pool_type (str): 'max', 'average', 'global_max', 'global_average'.
        **kwargs: argumen untuk konstruktor.
    Returns:
        Instance pooling layer.
    """
    key = pool_type.lower().strip()
    if key not in POOLING_TYPES:
        raise ValueError(
            f"Unknown pooling type: '{pool_type}'. "
            f"Tersedia: {list(POOLING_TYPES.keys())}"
        )
    return POOLING_TYPES[key](**kwargs)