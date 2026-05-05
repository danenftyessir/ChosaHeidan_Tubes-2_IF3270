"""
Model CNN dari nol — menyusun semua lapisan (Conv2D, LocallyConnected2D, Pooling, Flatten, Dense).
Mendukung forward pass dari input gambar hingga prediksi kelas.
Mendukung batch inference dengan configurable batch_size.
Load bobot dari file .h5 hasil pelatihan Keras.
"""

import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))
from dense import Dense


class CNNScratch:
    """
    Model CNN lengkap dari nol.

    Menerima input shape (N, H, W, C) dan menghasilkan prediksi kelas.

    Args:
        layers (list): daftar objek layer (Conv2D, LocallyConnected2D,
            pooling, Flatten, Dense, dst).
        batch_size (int): ukuran batch untuk inference. Default 32.
        num_classes (int): jumlah kelas output. Default 6 (Intel Image).
        input_shape (tuple): (H, W, C) dari input. Default (150, 150, 3).
    """

    def __init__(self, layers=None, batch_size=32, num_classes=6,
                 input_shape=(150, 150, 3)):
        if layers is None:
            layers = []
        self.layers = layers
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.input_shape = input_shape
        self._is_built = False

    def build(self, input_shape=None):
        """
        Bangun model: inisialisasi bobot untuk setiap lapisan berdasarkan
        bentuk input aktual.

        Args:
            input_shape: tuple (H, W, C) untuk input pertama.
        """
        if input_shape is not None:
            self.input_shape = input_shape

        current_shape = self.input_shape
        for layer in self.layers:
            layer_name = layer.__class__.__name__
            if layer_name in ('Conv2D', 'LocallyConnected2D'):
                if hasattr(layer, 'kernel') and layer.kernel is None:
                    C_in = current_shape[-1]
                    if hasattr(layer, '_init_weights'):
                        layer._init_weights(C_in)

            elif layer_name == 'Dense':
                if hasattr(layer, 'weights') and layer.weights is None:
                    flat_dim = np.prod(current_shape).item()
                    layer.input_dim = flat_dim
                    layer._init_weights()

            current_shape = self._get_output_shape(layer, current_shape)

        self._is_built = True

    def _get_output_shape(self, layer, input_shape):
        """Dapatkan bentuk output dari suatu layer."""
        N, H, W, C = input_shape
        name = layer.__class__.__name__

        if name == 'Conv2D':
            kH, kW = layer.kernel_size
            sh, sw = layer.strides
            if layer.padding == 'same':
                H_out = int(np.ceil(H / sh))
                W_out = int(np.ceil(W / sw))
            else:
                H_out = (H - kH) // sh + 1
                W_out = (W - kW) // sw + 1
            return (H_out, W_out, layer.filters)

        elif name == 'LocallyConnected2D':
            kH, kW = layer.kernel_size
            sh, sw = layer.strides
            if layer.padding == 'same':
                H_out = int(np.ceil(H / sh))
                W_out = int(np.ceil(W / sw))
            else:
                H_out = (H - kH) // sh + 1
                W_out = (W - kW) // sw + 1
            return (H_out, W_out, layer.filters)

        elif name in ('MaxPooling2D', 'AveragePooling2D'):
            ph, pw = layer.pool_size
            sh, sw = layer.strides
            H_out = (H - ph) // sh + 1
            W_out = (W - pw) // sw + 1
            return (H_out, W_out, C)

        elif name in ('GlobalAveragePooling2D', 'GlobalMaxPooling2D'):
            return (1, 1, current_shape[2])  # (H=1, W=1, C=channels)

        elif name == 'Flatten':
            return (H * W * C,)

        elif name == 'Dense':
            return (layer.units,)

        return input_shape

    def forward(self, x, verbose=False):
        """
        Forward pass tunggal — memproses SATU batch.

        Args:
            x: array numpy, bentuk (N, H, W, C) atau (H, W, C).
            verbose (bool): cetak info setiap layer.
        Returns:
            output: array numpy, bentuk (N, num_classes), distribusi probabilitas.
        """
        if x.ndim == 3:
            x = x[np.newaxis, ...]

        current = x

        for i, layer in enumerate(self.layers):
            name = layer.__class__.__name__
            current = layer.forward(current)
            if verbose:
                print(f"  Layer {i} {name}: output shape = {current.shape}")

        return current

    def predict(self, X, batch_size=None, verbose=False):
        """
        Prediksi untuk seluruh dataset dengan batch inference.

        Membagi input menjadi chunk sesuai batch_size, forward pass per chunk,
        lalu concatenate hasilnya.

        Args:
            X: array numpy, bentuk (N, H, W, C) atau (H, W, C).
            batch_size: ukuran batch. Jika None, gunakan self.batch_size.
            verbose: cetak progress.
        Returns:
            probs: array numpy, bentuk (N, num_classes), distribusi probabilitas.
        """
        if X.ndim == 3:
            X = X[np.newaxis, ...]

        N = X.shape[0]
        if batch_size is None:
            batch_size = self.batch_size

        all_probs = []

        num_batches = (N + batch_size - 1) // batch_size
        for i in range(0, N, batch_size):
            end = min(i + batch_size, N)
            batch = X[i:end]
            probs = self.forward(batch, verbose=False)
            all_probs.append(probs)

            if verbose:
                print(f"Batch {i//batch_size + 1}/{num_batches}: "
                      f"samples {i}-{end-1}, shape={batch.shape}")

        return np.concatenate(all_probs, axis=0)

    def load_weights_from_h5(self, h5_path):
        """
        Load bobot dari file .h5 hasil pelatihan Keras.

        Keras menyimpan bobot dalam format layer.indices:
            layer_{idx}/kernel:.000, layer_{idx}/bias:...

        Args:
            h5_path: path ke file .h5.
        """
        import h5py

        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"File tidak ditemukan: {h5_path}")

        with h5py.File(h5_path, 'r') as f:
            layer_names = []
            f.visititems(
                lambda name, obj: layer_names.append(name)
                if isinstance(obj, h5py.Dataset) and 'kernel' in name else None
            )

            weight_groups = {}
            for name in layer_names:
                parts = name.split('/')
                layer_key = parts[0]
                if layer_key not in weight_groups:
                    weight_groups[layer_key] = {}
                weight_groups[layer_key][parts[-1]] = np.array(f[name])

            sorted_keys = sorted(
                weight_groups.keys(),
                key=lambda k: int(k.split('_')[-1])
            )

            scratch_layer_idx = 0
            for key in sorted_keys:
                wg = weight_groups[key]
                kernel = wg['kernel'] if 'kernel' in wg else None
                bias = wg['bias'] if 'bias' in wg else None

                if kernel is None:
                    continue

                while scratch_layer_idx < len(self.layers):
                    layer = self.layers[scratch_layer_idx]
                    name = layer.__class__.__name__
                    if name in ('Conv2D', 'LocallyConnected2D'):
                        # Deteksi LocallyConnected2D: kernel Keras 6D, Conv2D 4D
                        if kernel.ndim == 6:
                            # Keras LocallyConnected2D kernel: (H_out, W_out, kH, kW, C_in, C_out)
                            # Scratch weights: (H_out*W_out, kH*kW*C_in, C_out)
                            H_out, W_out, kH, kW, C_in, C_out = kernel.shape
                            # Transpose: (H,W,kH,kW,C,C) → (H,W,C,kH,kW,C) → (H*W, C*kH*kW, C)
                            kernel_transposed = kernel.transpose(0, 1, 4, 2, 3, 5)
                            kernel_reshaped = kernel_transposed.reshape(H_out * W_out, kH * kW * C_in, C_out)
                            # Bias: (H_out, W_out, C_out) → (H_out*W_out, C_out)
                            bias_reshaped = bias.reshape(H_out * W_out, C_out)
                            layer.set_weights(kernel_reshaped, bias_reshaped)
                            layer.padding = 'valid'  # Keras default for LocallyConnected2D
                        else:
                            # Conv2D kernel: (kH, kW, C_in, C_out) — compatible
                            layer.set_weights(kernel, bias)
                        scratch_layer_idx += 1
                        break
                    elif name == 'Dense':
                        layer.set_weights(kernel, bias)
                        scratch_layer_idx += 1
                        break
                    else:
                        scratch_layer_idx += 1

        print(f"Bobot berhasil dimuat dari: {h5_path}")

    def load_weights(self, path):
        """Alias untuk load_weights_from_h5."""
        self.load_weights_from_h5(path)

    def add_layer(self, layer):
        """Tambahkan lapisan ke model."""
        self.layers.append(layer)

    def summary(self):
        """Cetak arsitektur model."""
        print("CNNScratch Architecture:")
        print("=" * 60)
        for i, layer in enumerate(self.layers):
            print(f"  [{i}] {layer.summary()}")
        print("=" * 60)

    def get_layer(self, idx):
        """Dapatkan lapisan berdasarkan indeks."""
        return self.layers[idx]

    def set_batch_size(self, batch_size):
        """Ubah batch_size untuk inference."""
        self.batch_size = batch_size


def build_cnn_from_config(config, num_classes=6, input_shape=(150, 150, 3)):
    """
    Helper function: bangun model CNN dari konfigurasi dict.

    Args:
        config (dict): konfigurasi arsitektur.
            Contoh:
            {
                'layers': [
                    {'type': 'conv2d', 'filters': 32, 'kernel_size': 3, 'activation': 'relu'},
                    {'type': 'maxpool', 'pool_size': 2},
                    {'type': 'conv2d', 'filters': 64, 'kernel_size': 3, 'activation': 'relu'},
                    {'type': 'globalavgpool'},
                    {'type': 'dense', 'units': num_classes, 'activation': 'softmax'},
                ]
            }
        num_classes (int): jumlah kelas output.
        input_shape (tuple): bentuk input.
    Returns:
        CNNScratch instance.
    """
    from .conv2d import Conv2D
    from .locally_connected2d import LocallyConnected2D
    from .pooling import (
        MaxPooling2D, AveragePooling2D,
        GlobalAveragePooling2D, GlobalMaxPooling2D
    )
    from .flatten import Flatten
    from shared.dense import Dense

    layers = []
    for layer_config in config.get('layers', []):
        ltype = layer_config['type'].lower()

        if ltype == 'conv2d':
            layers.append(Conv2D(
                filters=layer_config['filters'],
                kernel_size=layer_config.get('kernel_size', 3),
                strides=layer_config.get('strides', (1, 1)),
                padding=layer_config.get('padding', 'same'),
                activation=layer_config.get('activation', 'relu'),
            ))
        elif ltype == 'locally_connected2d':
            layers.append(LocallyConnected2D(
                filters=layer_config['filters'],
                kernel_size=layer_config.get('kernel_size', 3),
                strides=layer_config.get('strides', (1, 1)),
                padding=layer_config.get('padding', 'same'),
                activation=layer_config.get('activation', 'relu'),
            ))
        elif ltype == 'maxpool':
            layers.append(MaxPooling2D(
                pool_size=layer_config.get('pool_size', (2, 2)),
                strides=layer_config.get('strides'),
            ))
        elif ltype == 'avgpool':
            layers.append(AveragePooling2D(
                pool_size=layer_config.get('pool_size', (2, 2)),
                strides=layer_config.get('strides'),
            ))
        elif ltype == 'globalavgpool':
            layers.append(GlobalAveragePooling2D())
        elif ltype == 'globalmaxpool':
            layers.append(GlobalMaxPooling2D())
        elif ltype == 'flatten':
            layers.append(Flatten())
        elif ltype == 'dense':
            layers.append(Dense(
                input_dim=0,
                units=layer_config['units'],
                activation=layer_config.get('activation', 'relu'),
            ))

    model = CNNScratch(layers=layers, num_classes=num_classes, input_shape=input_shape)
    model.build()
    return model