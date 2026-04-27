"""
Visualisasi Intermediate Feature Maps dari lapisan konvolusi CNN.
Menampilkan aktivasi setiap filter pada setiap layer konvolusi.
"""

import numpy as np
import os


def visualize_feature_maps(activations, n_filters=None, save_path=None,
                           cmap='viridis', figsize=None):
    """
    Visualisasi feature maps dari suatu lapisan konvolusi.

    Menampilkan grid NxN dari feature maps (filter activations).
    Setiap subplot menampilkan satu feature map (channel) dalam 2D spatial.

    Args:
        activations: array numpy, bentuk (H, W, C_out) atau (1, H, W, C_out).
                     Hasil output dari suatu Conv2D layer.
        n_filters: jumlah filter yang ingin ditampilkan. Jika None, tampilkan semua.
        save_path: jika diberikan, simpan figure ke path ini.
        cmap: colormap matplotlib. Default 'viridis'.
        figsize: tuple (width, height) untuk figure. Jika None, dihitung otomatis.
    Returns:
        fig, axes: matplotlib figure dan axes.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib diperlukan. Install: pip install matplotlib")

    # Hilangkan batch dimension jika ada
    if activations.ndim == 4:
        activations = activations[0]

    H, W, C_out = activations.shape

    if n_filters is None:
        n_filters = C_out
    else:
        n_filters = min(n_filters, C_out)

    # Layout grid:尽量正方形
    n_cols = int(np.ceil(np.sqrt(n_filters)))
    n_rows = int(np.ceil(n_filters / n_cols))

    if figsize is None:
        figsize = (n_cols * 2, n_rows * 2)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    fig.suptitle(f'Feature Maps — {n_filters} filters dari {C_out} total\n'
                 f'Shape: ({H}, {W}, {C_out})', fontsize=12)

    for idx in range(n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]

        if idx < n_filters:
            feature_map = activations[:, :, idx]
            ax.imshow(feature_map, cmap=cmap)
            ax.set_title(f'Filter {idx}', fontsize=8)
            ax.axis('off')
        else:
            ax.axis('off')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Feature maps disimpan ke: {save_path}")

    return fig, axes


def visualize_feature_maps_comparison(layer_outputs_dict, n_filters=None,
                                      save_path=None, cmap='viridis'):
    """
    Bandingkan feature maps dari beberapa layer secara bersamaan.

    Args:
        layer_outputs_dict: dict {layer_name: activations}.
                           activations bentuk: (H, W, C_out) atau (1, H, W, C_out).
        n_filters: jumlah filter yang ditampilkan per layer.
        save_path: path untuk menyimpan figure.
        cmap: colormap.
    Returns:
        fig, axes.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib diperlukan. Install: pip install matplotlib")

    n_layers = len(layer_outputs_dict)
    layer_names = list(layer_outputs_dict.keys())

    min_filters = n_filters
    if min_filters is None:
        min_filters = min(
            act.shape[-1] if act.ndim == 4 else act.shape[-1]
            for act in layer_outputs_dict.values()
        )

    n_cols = min_filters
    n_rows = n_layers

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.5, n_rows * 2),
                             squeeze=False)

    for row, layer_name in enumerate(layer_names):
        activations = layer_outputs_dict[layer_name]
        if activations.ndim == 4:
            activations = activations[0]

        C_out = activations.shape[-1]
        filters_to_show = min(n_filters, C_out) if n_filters else C_out

        for col in range(n_cols):
            ax = axes[row, col]
            if col < filters_to_show:
                feature_map = activations[:, :, col]
                ax.imshow(feature_map, cmap=cmap)
                ax.axis('off')
            else:
                ax.axis('off')

        axes[row, 0].set_ylabel(layer_name, fontsize=9, rotation=0,
                                ha='right', va='center')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Feature maps comparison disimpan ke: {save_path}")

    return fig, axes


def plot_feature_map_statistics(activations, save_path=None):
    """
    Plot statistik feature maps: mean, std, min, max per filter.

    Berguna untuk menganalisis distribusi aktivasi setiap filter.

    Args:
        activations: array numpy, bentuk (H, W, C_out) atau (1, H, W, C_out).
        save_path: path untuk menyimpan plot.
    Returns:
        fig, axes.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib diperlukan. Install: pip install matplotlib")

    if activations.ndim == 4:
        activations = activations[0]

    H, W, C_out = activations.shape

    means = np.mean(activations, axis=(0, 1))
    stds = np.std(activations, axis=(0, 1))
    mins = np.min(activations, axis=(0, 1))
    maxs = np.max(activations, axis=(0, 1))

    x = np.arange(C_out)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    axes[0, 0].bar(x, means)
    axes[0, 0].set_title('Mean Activation per Filter')
    axes[0, 0].set_xlabel('Filter Index')
    axes[0, 0].set_ylabel('Mean')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].bar(x, stds)
    axes[0, 1].set_title('Std Dev Activation per Filter')
    axes[0, 1].set_xlabel('Filter Index')
    axes[0, 1].set_ylabel('Std')
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].bar(x, mins)
    axes[1, 0].set_title('Min Activation per Filter')
    axes[1, 0].set_xlabel('Filter Index')
    axes[1, 0].set_ylabel('Min')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].bar(x, maxs)
    axes[1, 1].set_title('Max Activation per Filter')
    axes[1, 1].set_xlabel('Filter Index')
    axes[1, 1].set_ylabel('Max')
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle(f'Feature Map Statistics — {C_out} Filters, Shape ({H}, {W})',
                 fontsize=12)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Stats disimpan ke: {save_path}")

    return fig, axes


def extract_intermediate_features(model, X, layer_indices=None, layer_names=None):
    """
    Ekstrak intermediate activations dari beberapa layer sekaligus.

    Args:
        model: instance CNNScratch.
        X: input, bentuk (1, H, W, C).
        layer_indices: list indeks layer yang akan di-extract.
        layer_names: list nama layer yang akan di-extract.
    Returns:
        dict {layer_name: activations}.
    """
    if X.ndim == 3:
        X = X[np.newaxis, ...]

    intermediate = {}

    current = X
    for i, layer in enumerate(model.layers):
        current = layer.forward(current)

        if layer_indices is not None and i in layer_indices:
            intermediate[f'layer_{i}_{layer.__class__.__name__}'] = current.copy()
        elif layer_names is not None:
            layer_name = layer.__class__.__name__
            if layer_name in layer_names or f'layer_{i}_{layer_name}' in layer_names:
                intermediate[f'layer_{i}_{layer_name}'] = current.copy()

    return intermediate


def visualize_filter_weights(conv_layer, filter_idx=None, save_path=None):
    """
    Visualisasi bobot kernel suatu lapisan Conv2D.

    Args:
        conv_layer: instance Conv2D layer.
        filter_idx: indeks filter yang ingin ditampilkan. Jika None, tampilkan semua.
        save_path: path untuk menyimpan plot.
    Returns:
        fig, axes.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib diperlukan. Install: pip install matplotlib")

    kernel = conv_layer.kernel
    kH, kW, C_in, C_out = kernel.shape

    if filter_idx is None:
        filters_to_show = C_out
    else:
        filters_to_show = 1
        filter_idx = min(filter_idx, C_out - 1)

    # Tampilkan filter sebagai RGB jika C_in == 3
    if C_in == 3:
        n_cols = filters_to_show
        n_rows = 1

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, 2), squeeze=False)

        for i in range(filters_to_show):
            if filters_to_show == 1:
                ax = axes[0, 0]
                filter_weights = kernel[:, :, :, filter_idx]
            else:
                ax = axes[0, i]
                filter_weights = kernel[:, :, :, i]

            fw = filter_weights - filter_weights.min()
            if fw.max() > 0:
                fw = fw / fw.max()

            ax.imshow(fw.astype(np.float64))
            ax.set_title(f'Filter {i if filter_idx is None else filter_idx}', fontsize=8)
            ax.axis('off')
    else:
        n_cols = filters_to_show
        n_rows = C_in

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2),
                                 squeeze=False)

        for i in range(filters_to_show):
            for c in range(C_in):
                ax = axes[c, i]
                filter_weights = kernel[:, :, c, i]

                fw = filter_weights - filter_weights.min()
                if fw.max() > 0:
                    fw = fw / fw.max()

                ax.imshow(fw.astype(np.float64), cmap='gray')
                ax.axis('off')

                if c == 0:
                    ax.set_title(f'Filter {i}', fontsize=8)

    fig.suptitle(f'Kernel Weights — {kH}x{kW}, C_in={C_in}, C_out={C_out}',
                 fontsize=12)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Filter weights disimpan ke: {save_path}")

    return fig, axes