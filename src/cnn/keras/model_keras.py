"""
Arsitektur CNN dengan Keras (TensorFlow).
Mendukung dua varian lapisan konvolusi:
  - Conv2D: parameter di-share di seluruh spatial location
  - LocallyConnected2D: parameter TIDAK di-share (tiap lokasi punya bobot sendiri)

Variasi hyperparameter:
  - 3 variasi jumlah layer konvolusi: [2, 3, 4]
  - 3 variasi jumlah filter: [32, 64, 128]
  - 3 variasi ukuran filter: [(3,3), (5,5), (7,7)]
  - 2 variasi pooling: 'max' vs 'average'

Dataset: Intel Image Classification (6 kelas: buildings, forest, glacier,
  mountain, sea, street)
Input shape: (150, 150, 3)
"""

import os
import numpy as np


# ============================================================================
# Konfigurasi Global
# ============================================================================

INTEL_CLASSES = [
    'buildings',  # 0
    'forest',     # 1
    'glacier',    # 2
    'mountain',   # 3
    'sea',        # 4
    'street',     # 5
]

NUM_CLASSES = len(INTEL_CLASSES)
INPUT_SHAPE = (150, 150, 3)
DEFAULT_WEIGHTS_DIR = 'weights/cnn'


# ============================================================================
# Arsitektur Base
# ============================================================================

def _get_optimizer(optimizer='adam', lr=0.001):
    """
    Dapatkan optimizer Keras.

    Args:
        optimizer (str): 'adam', 'sgd', 'rmsprop'
        lr (float): learning rate
    Returns:
        optimizer instance
    """
    if optimizer == 'adam':
        from tensorflow.keras.optimizers import Adam
        return Adam(learning_rate=lr)
    elif optimizer == 'sgd':
        from tensorflow.keras.optimizers import SGD
        return SGD(learning_rate=lr, momentum=0.9)
    elif optimizer == 'rmsprop':
        from tensorflow.keras.optimizers import RMSprop
        return RMSprop(learning_rate=lr)
    else:
        raise ValueError(f"Optimizer tidak dikenal: {optimizer}")


def _build_conv_block(x, filters, kernel_size, pooling_type='max', idx=0):
    """
    Bangun satu block konvolusi: Conv2D → BatchNorm → Activation → Pooling.

    Args:
        x: tensor input
        filters (int): jumlah filter
        kernel_size (int atau tuple): ukuran kernel
        pooling_type (str): 'max' atau 'average'
        idx (int): indeks block (untuk nama layer)
    Returns:
        tensor output
    """
    from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation
    from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D

    prefix = f'block{idx}'

    x = Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=(1, 1),
        padding='same',
        use_bias=False,
        name=f'{prefix}_conv'
    )(x)

    x = BatchNormalization(name=f'{prefix}_bn')(x)
    x = Activation('relu', name=f'{prefix}_act')(x)

    if pooling_type == 'max':
        x = MaxPooling2D(pool_size=(2, 2), name=f'{prefix}_pool')(x)
    else:
        x = AveragePooling2D(pool_size=(2, 2), name=f'{prefix}_pool')(x)

    return x


# ============================================================================
# Conv2D Baseline — Parameter Sharing
# ============================================================================

def build_cnn_conv2d(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES,
                      num_conv_layers=3, num_filters=64, kernel_size=(3, 3),
                      pooling_type='max', optimizer='adam', lr=0.001):
    """
    Bangun CNN dengan Conv2D (parameter sharing) untuk Intel Image Classification.

    Arsitektur:
        Input (150, 150, 3)
        → [Conv2D → BN → ReLU → Pool] × num_conv_layers
        → Flatten
        → Dense(256) → ReLU → Dropout(0.5)
        → Dense(num_classes) → Softmax

    Parameter TIDAK di-share di seluruh spatial location.

    Args:
        input_shape (tuple): bentuk input gambar (H, W, C). Default (150, 150, 3).
        num_classes (int): jumlah kelas output. Default 6.
        num_conv_layers (int): jumlah block konvolusi. Default 3.
        num_filters (int): jumlah filter per layer konvolusi. Default 64.
        kernel_size (tuple): ukuran kernel konvolusi. Default (3, 3).
        pooling_type (str): 'max' atau 'average'. Default 'max'.
        optimizer (str): jenis optimizer. Default 'adam'.
        lr (float): learning rate. Default 0.001.
    Returns:
        tuple: (model, config_dict)
    """
    from tensorflow.keras import Model
    from tensorflow.keras.layers import Input, Flatten, Dense, Dropout

    # Input
    inputs = Input(shape=input_shape, name='input')
    x = inputs

    # Block konvolusi
    for i in range(num_conv_layers):
        # Tambah filter tiap layer (doubling)
        block_filters = num_filters * (2 ** i)
        x = _build_conv_block(x, block_filters, kernel_size, pooling_type, idx=i + 1)

    # Classification head
    x = Flatten(name='flatten')(x)
    x = Dense(256, activation='relu', name='fc1')(x)
    x = Dropout(0.5, name='dropout')(x)
    outputs = Dense(num_classes, activation='softmax', name='output')(x)

    model = Model(inputs=inputs, outputs=outputs, name='CNN_Conv2D')

    model.compile(
        optimizer=_get_optimizer(optimizer, lr),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    config = {
        'type': 'conv2d',
        'input_shape': input_shape,
        'num_classes': num_classes,
        'num_conv_layers': num_conv_layers,
        'num_filters': num_filters,
        'kernel_size': kernel_size,
        'pooling_type': pooling_type,
        'optimizer': optimizer,
        'lr': lr,
    }

    return model, config


# ============================================================================
# LocallyConnected2D — Parameter NON-Sharing
# ============================================================================

def build_cnn_locallyconnected(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES,
                                num_conv_layers=3, num_filters=64,
                                kernel_size=(3, 3), pooling_type='max',
                                optimizer='adam', lr=0.001):
    """
    Bangun CNN dengan LocallyConnected2D (parameter NON-sharing) untuk Intel.

    Perbedaan dengan Conv2D:
        Conv2D: bobot filter di-share → satu set bobot untuk seluruh spatial lokasi
        LocallyConnected2D: setiap lokasi spasial punya bobot sendiri
                           → lebih banyak parameter, bisa menangkap pola lokal lebih spesifik

    Arsitektur sama dengan Conv2D, tapi menggunakan LocallyConnected2D
    di posisi yang sama dengan Conv2D.

    Args:
        input_shape (tuple): bentuk input gambar (H, W, C). Default (150, 150, 3).
        num_classes (int): jumlah kelas output. Default 6.
        num_conv_layers (int): jumlah block konvolusi. Default 3.
        num_filters (int): jumlah filter per layer. Default 64.
        kernel_size (tuple): ukuran kernel. Default (3, 3).
        pooling_type (str): 'max' atau 'average'. Default 'max'.
        optimizer (str): jenis optimizer. Default 'adam'.
        lr (float): learning rate. Default 0.001.
    Returns:
        tuple: (model, config_dict)
    """
    from tensorflow.keras import Model
    from tensorflow.keras.layers import (Input, LocallyConnected2D,
                                          BatchNormalization, Activation)
    from tensorflow.keras.layers import (Flatten, Dense, Dropout,
                                          MaxPooling2D, AveragePooling2D)

    inputs = Input(shape=input_shape, name='input')
    x = inputs

    # Block dengan LocallyConnected2D
    for i in range(num_conv_layers):
        block_filters = num_filters * (2 ** i)
        prefix = f'block{i + 1}'

        # LocallyConnected2D: parameter TIDAK di-share
        x = LocallyConnected2D(
            filters=block_filters,
            kernel_size=kernel_size,
            strides=(1, 1),
            padding='valid',
            use_bias=False,
            name=f'{prefix}_locconv'
        )(x)

        x = BatchNormalization(name=f'{prefix}_bn')(x)
        x = Activation('relu', name=f'{prefix}_act')(x)

        if pooling_type == 'max':
            x = MaxPooling2D(pool_size=(2, 2), name=f'{prefix}_pool')(x)
        else:
            x = AveragePooling2D(pool_size=(2, 2), name=f'{prefix}_pool')(x)

    # Classification head
    x = Flatten(name='flatten')(x)
    x = Dense(256, activation='relu', name='fc1')(x)
    x = Dropout(0.5, name='dropout')(x)
    outputs = Dense(num_classes, activation='softmax', name='output')(x)

    model = Model(inputs=inputs, outputs=outputs, name='CNN_LocallyConnected')

    model.compile(
        optimizer=_get_optimizer(optimizer, lr),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    config = {
        'type': 'locallyconnected',
        'input_shape': input_shape,
        'num_classes': num_classes,
        'num_conv_layers': num_conv_layers,
        'num_filters': num_filters,
        'kernel_size': kernel_size,
        'pooling_type': pooling_type,
        'optimizer': optimizer,
        'lr': lr,
    }

    return model, config


# ============================================================================
# VGG-style Deep CNN Variants
# ============================================================================

def build_cnn_vgg_style(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES,
                        variant='small', pooling_type='max',
                        layer_type='conv2d', optimizer='adam', lr=0.001):
    """
    Bangun CNN dengan arsitektur VGG-style untuk Intel Classification.

    Varian:
        - 'tiny': 2 conv layers (32, 64) + FC(128) → 6 classes
        - 'small': 3 conv layers (32, 64, 128) + FC(256) → 6 classes
        - 'medium': 4 conv layers (32, 64, 128, 256) + FC(256) → 6 classes
        - 'large': 4 conv layers (64, 128, 256, 512) + FC(512) → 6 classes

    Args:
        input_shape (tuple): bentuk input
        num_classes (int): jumlah kelas
        variant (str): 'tiny', 'small', 'medium', 'large'
        pooling_type (str): 'max' atau 'average'
        layer_type (str): 'conv2d' atau 'locallyconnected'
        optimizer (str): jenis optimizer
        lr (float): learning rate
    Returns:
        tuple: (model, config_dict)
    """
    from tensorflow.keras import Model
    from tensorflow.keras.layers import (Input, Conv2D, LocallyConnected2D,
                                          BatchNormalization, Activation,
                                          Flatten, Dense, Dropout,
                                          MaxPooling2D, AveragePooling2D)

    configs = {
        'tiny':    {'layers': [32, 64],           'fc': 128},
        'small':   {'layers': [32, 64, 128],      'fc': 256},
        'medium':  {'layers': [32, 64, 128, 256], 'fc': 256},
        'large':   {'layers': [64, 128, 256, 512], 'fc': 512},
    }

    if variant not in configs:
        raise ValueError(f"Varian tidak dikenal: {variant}. Pilih: {list(configs.keys())}")

    layer_config = configs[variant]
    conv_layers = layer_config['layers']
    fc_units = layer_config['fc']

    inputs = Input(shape=input_shape, name='input')
    x = inputs

    for i, filters in enumerate(conv_layers):
        conv_cls = Conv2D if layer_type == 'conv2d' else LocallyConnected2D
        conv_name = f'block{i + 1}_{layer_type}'

        padding = 'same' if layer_type == 'conv2d' else 'valid'

        x = conv_cls(
            filters=filters,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=padding,
            use_bias=False,
            name=conv_name
        )(x)

        x = BatchNormalization(name=f'bn{i + 1}')(x)
        x = Activation('relu', name=f'act{i + 1}')(x)

        if pooling_type == 'max':
            x = MaxPooling2D(pool_size=(2, 2), name=f'pool{i + 1}')(x)
        else:
            x = AveragePooling2D(pool_size=(2, 2), name=f'pool{i + 1}')(x)

    x = Flatten(name='flatten')(x)
    x = Dense(fc_units, activation='relu', name='fc1')(x)
    x = Dropout(0.5, name='dropout')(x)
    outputs = Dense(num_classes, activation='softmax', name='output')(x)

    model_name = f'VGG_{variant.upper()}_{layer_type}'
    model = Model(inputs=inputs, outputs=outputs, name=model_name)

    model.compile(
        optimizer=_get_optimizer(optimizer, lr),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    config = {
        'type': f'vgg_{variant}_{layer_type}',
        'variant': variant,
        'layer_type': layer_type,
        'conv_layers': conv_layers,
        'fc_units': fc_units,
        'pooling_type': pooling_type,
        'optimizer': optimizer,
        'lr': lr,
    }

    return model, config


# ============================================================================
# Factory Functions — Semua Variasi Hyperparameter
# ============================================================================

def build_cnn_factory(arch_type='conv2d', num_conv_layers=3, num_filters=64,
                      kernel_size=(3, 3), pooling_type='max',
                      input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES,
                      optimizer='adam', lr=0.001):
    """
    Factory function: bangun CNN dengan variasi hyperparameter tertentu.

    Args:
        arch_type (str): 'conv2d' atau 'locallyconnected'
        num_conv_layers (int): 1-5
        num_filters (int): jumlah filter base
        kernel_size (tuple): ukuran kernel
        pooling_type (str): 'max' atau 'average'
        input_shape (tuple): bentuk input
        num_classes (int): jumlah kelas
        optimizer (str): optimizer
        lr (float): learning rate
    Returns:
        tuple: (model, config)
    """
    if arch_type == 'conv2d':
        return build_cnn_conv2d(
            input_shape=input_shape, num_classes=num_classes,
            num_conv_layers=num_conv_layers, num_filters=num_filters,
            kernel_size=kernel_size, pooling_type=pooling_type,
            optimizer=optimizer, lr=lr
        )
    elif arch_type == 'locallyconnected':
        return build_cnn_locallyconnected(
            input_shape=input_shape, num_classes=num_classes,
            num_conv_layers=num_conv_layers, num_filters=num_filters,
            kernel_size=kernel_size, pooling_type=pooling_type,
            optimizer=optimizer, lr=lr
        )
    else:
        raise ValueError(f"Tipe arsitektur tidak dikenal: {arch_type}")


def build_all_variations(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES,
                          pooling_types=('max', 'average'),
                          layer_types=('conv2d', 'locallyconnected'),
                          optimizer='adam', lr=0.001):
    """
    Bangun SEMUA kombinasi variasi hyperparameter.

    Variasi:
        - layer_types: conv2d, locallyconnected (2)
        - num_conv_layers: [2, 3, 4] (3)
        - num_filters: [32, 64, 128] (3)
        - kernel_sizes: [(3,3), (5,5), (7,7)] (3)
        - pooling_types: ['max', 'average'] (2)

    Total: 2 × 3 × 3 × 3 × 2 = 108 variasi!

    Untuk eksperimen terpisah, gunakan subset variasi.

    Args:
        input_shape (tuple): bentuk input
        num_classes (int): jumlah kelas
        pooling_types (tuple): jenis pooling
        layer_types (tuple): jenis layer konvolusi
        optimizer (str): optimizer
        lr (float): learning rate
    Returns:
        dict: {config_name: (model, config)}
    """
    results = {}

    for layer_type in layer_types:
        for num_layers in [2, 3, 4]:
            for base_filters in [32, 64, 128]:
                for kernel_size in [(3, 3), (5, 5), (7, 7)]:
                    for pooling in pooling_types:
                        model, config = build_cnn_factory(
                            arch_type=layer_type,
                            num_conv_layers=num_layers,
                            num_filters=base_filters,
                            kernel_size=kernel_size,
                            pooling_type=pooling,
                            input_shape=input_shape,
                            num_classes=num_classes,
                            optimizer=optimizer,
                            lr=lr
                        )

                        name = (
                            f"{layer_type}_L{num_layers}_F{base_filters}_"
                            f"K{kernel_size[0]}_{pooling}"
                        )
                        results[name] = (model, config)

    return results


# ============================================================================
# Model Summary Helpers
# ============================================================================

def count_parameters(model):
    """
    Hitung jumlah parameter model.

    Args:
        model: Keras Model
    Returns:
        dict: {total, trainable, non_trainable}
    """
    trainable = int(np.sum([np.prod(v.shape) for v in model.trainable_weights]))
    non_trainable = int(np.sum([np.prod(v.shape) for v in model.non_trainable_weights]))
    total = trainable + non_trainable
    return {'total': total, 'trainable': trainable, 'non_trainable': non_trainable}


def compare_model_sizes(arch_type='conv2d', num_conv_layers=3,
                        num_filters=64, kernel_size=(3, 3),
                        input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES):
    """
    Bandingkan jumlah parameter Conv2D vs LocallyConnected2D.

    Args:
        arch_type (str): 'conv2d' atau 'locallyconnected'
        num_conv_layers (int): jumlah layer konvolusi
        num_filters (int): filter base
        kernel_size (tuple): ukuran kernel
        input_shape (tuple): bentuk input
        num_classes (int): jumlah kelas
    Returns:
        dict: {conv2d_params, locallyconnected_params, ratio}
    """
    conv2d_model, _ = build_cnn_conv2d(
        input_shape=input_shape, num_classes=num_classes,
        num_conv_layers=num_conv_layers, num_filters=num_filters,
        kernel_size=kernel_size
    )
    local_model, _ = build_cnn_locallyconnected(
        input_shape=input_shape, num_classes=num_classes,
        num_conv_layers=num_conv_layers, num_filters=num_filters,
        kernel_size=kernel_size
    )

    conv_params = count_parameters(conv2d_model)
    local_params = count_parameters(local_model)

    return {
        'conv2d': conv_params,
        'locallyconnected': local_params,
        'ratio': local_params['total'] / conv_params['total'] if conv_params['total'] > 0 else 0,
    }


def print_model_summary(model, config=None):
    """Cetak ringkasan model."""
    print("=" * 70)
    print(f"Model: {model.name}")
    if config:
        print(f"  Type:          {config.get('type', 'N/A')}")
        print(f"  Conv Layers:   {config.get('num_conv_layers', config.get('conv_layers', 'N/A'))}")
        print(f"  Filters:       {config.get('num_filters', config.get('conv_layers', 'N/A'))}")
        print(f"  Kernel Size:   {config.get('kernel_size', 'N/A')}")
        print(f"  Pooling:       {config.get('pooling_type', 'N/A')}")
    params = count_parameters(model)
    print(f"  Parameters:    {params['total']:,} (trainable: {params['trainable']:,})")
    print("=" * 70)


if __name__ == '__main__':
    print("[Test] CNN Keras Architecture Models...")

    # Test Conv2D
    model_conv, cfg_conv = build_cnn_conv2d(
        num_conv_layers=3, num_filters=64, kernel_size=(3, 3), pooling_type='max'
    )
    print_model_summary(model_conv, cfg_conv)

    # Test LocallyConnected2D
    model_local, cfg_local = build_cnn_locallyconnected(
        num_conv_layers=3, num_filters=64, kernel_size=(3, 3), pooling_type='max'
    )
    print_model_summary(model_local, cfg_local)

    # Bandingkan ukuran parameter
    print("\n[Parameter Comparison]")
    comp = compare_model_sizes(
        num_conv_layers=3, num_filters=64, kernel_size=(3, 3)
    )
    print(f"  Conv2D params:            {comp['conv2d']['total']:,}")
    print(f"  LocallyConnected2D params: {comp['locallyconnected']['total']:,}")
    print(f"  Ratio (local/conv2d):     {comp['ratio']:.1f}x")

    # Test VGG-style
    print("\n[VGG-style Variants]")
    for variant in ['tiny', 'small', 'medium']:
        model, cfg = build_cnn_vgg_style(variant=variant, layer_type='conv2d')
        params = count_parameters(model)
        print(f"  {variant:8s}: {params['total']:,} params")

    print("\n[Test] Selesai.")
