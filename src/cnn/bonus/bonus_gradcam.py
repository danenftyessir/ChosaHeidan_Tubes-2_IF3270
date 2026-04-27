"""
Grad-CAM (Gradient-weighted Class Activation Mapping) untuk CNN dari nol.
Memvisualisasikan region gambar yang paling berpengaruh terhadap prediksi model.

Referensi: Selvaraju et al. (2017) - Grad-CAM: Visual Explanations from Deep
Networks via Gradient-based Localization.

Algoritma:
1. Forward pass → catat activations lapisan konvolusi target
2. Backward pass → hitung gradient loss terhadap setiap filter di layer target
3. Global Average Pooling pada gradient → bobot importance per filter
4. Weighted sum activations dengan bobot → class activation map
5. ReLU → hanya activation positif yang relevan
6. Resize ke ukuran gambar asli → overlay dengan gambar
"""

import numpy as np
import os


def gradcam(model, image, layer_name=None, class_idx=None, return_heatmap=True):
    """
    Grad-CAM: visualisasi region paling berpengaruh untuk prediksi.

    Args:
        model: instance CNNScratch yang sudah di-load bobotnya.
        image: array numpy, bentuk (H, W, C) atau (1, H, W, C).
               Gambar input yang sudah dinormalisasi ke [0, 1].
        layer_name: nama layer konvolusi yang akan digunakan.
                    Jika None, gunakan layer konvolusi terakhir.
        class_idx: indeks kelas yang akan di-explain.
                   Jika None, gunakan kelas dengan probabilitas tertinggi.
        return_heatmap: jika True, return juga heatmap.
    Returns:
        Jika return_heatmap=True:
            heatmap: array numpy, bentuk (H_out, W_out) — class activation map.
            overlay: array numpy, bentuk (H, W, C) — gambar + heatmap overlay.
            class_idx: kelas yang di-explain.
            probs: distribusi probabilitas.
        Jika return_heatmap=False:
            overlay, class_idx, probs.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
    except ImportError:
        raise ImportError("matplotlib diperlukan. Install: pip install matplotlib")

    # Siapkan input
    if image.ndim == 3:
        image = image[np.newaxis, ...]

    # Cari target layer
    target_layer = _find_target_conv_layer(model, layer_name)
    if target_layer is None:
        raise ValueError("Tidak ada lapisan Conv2D ditemukan di model.")

    target_layer_idx = model.layers.index(target_layer)
    original_H, original_W = image.shape[1:3]

    # Forward pass: catat activations
    activations_list = []
    intermediate_cache = {}

    def forward_hook(layer, input_data, output_data):
        """Hook untuk menangkap activations."""
        activations_list.append(output_data.copy())

    hook_handle = _register_hook(target_layer, forward_hook)

    output = image.copy()
    for i, layer in enumerate(model.layers):
        output = layer.forward(output)
        if i == target_layer_idx:
            intermediate_cache['target_input'] = intermediate_cache.get('prev_output',
                                                                        image.copy())
        if i > 0:
            intermediate_cache['prev_output'] = output

    probs = output
    intermediate_cache['final_output'] = output

    hook_handle.remove()

    # Ambil activations
    activations = activations_list[0]
    if activations.ndim == 4:
        activations = activations[0]

    H_out, W_out, num_filters = activations.shape

    # Tentukan kelas target
    if class_idx is None:
        class_idx = int(np.argmax(probs, axis=-1)[0])

    one_hot = np.zeros_like(probs)
    one_hot[0, class_idx] = 1.0

    # Backward pass: gradient terhadap target layer
    grad_output = one_hot

    current_grad = grad_output
    for i in range(len(model.layers) - 1, target_layer_idx, -1):
        layer = model.layers[i]
        current_grad = layer.backward(current_grad)

    if current_grad.ndim == 4:
        grad_weights = current_grad[0]

    # Global Average Pooling pada gradient → bobot per filter
    alpha = np.mean(grad_weights, axis=(0, 1))

    # Weighted sum: Grad-CAM = ReLU(sum_k alpha_k * A^k)
    cam = np.sum(alpha[np.newaxis, np.newaxis, :] *
                 activations, axis=-1)

    cam = np.maximum(cam, 0)

    # Normalize ke [0, 1]
    if cam.max() > 0:
        cam = cam / cam.max()

    # Resize ke ukuran gambar asli
    overlay = _resize_heatmap_to_image(cam, original_H, original_W, image[0])

    if return_heatmap:
        return cam, overlay, class_idx, probs

    return overlay, class_idx, probs


def _find_target_conv_layer(model, layer_name=None):
    """
    Cari lapisan Conv2D untuk Grad-CAM.

    Jika layer_name diberikan, cari berdasarkan nama.
    Jika tidak, gunakan lapisan konvolusi TERAKHIR.
    """
    conv_layers = [(i, layer) for i, layer in enumerate(model.layers)
                   if layer.__class__.__name__ == 'Conv2D']

    if not conv_layers:
        return None

    if layer_name is not None:
        for idx, layer in conv_layers:
            if layer_name.lower() in layer.__class__.__name__.lower():
                return layer
        return conv_layers[-1][1]
    else:
        return conv_layers[-1][1]


def _register_hook(layer, hook_fn):
    """Pasang forward hook ke lapisan."""
    original_forward = layer.forward

    def hooked_forward(x):
        output = original_forward(x)
        hook_fn(layer, x, output)
        return output

    layer.forward = hooked_forward
    return HookHandle(layer, original_forward)


class HookHandle:
    """Handle untuk melepas hook."""
    def __init__(self, layer, original_forward):
        self.layer = layer
        self.original_forward = original_forward

    def remove(self):
        """Lepas hook, kembalikan metode forward asli."""
        self.layer.forward = self.original_forward


def _resize_heatmap_to_image(cam, target_H, target_W, original_image):
    """
    Resize heatmap ke ukuran gambar asli menggunakan bilinear interpolation.

    Args:
        cam: array numpy 2D (H_out, W_out).
        target_H, target_W: dimensi target.
        original_image: array numpy (H, W, C).
    Returns:
        overlay: array numpy (H, W, C).
    """
    try:
        from scipy.ndimage import zoom

        H_out, W_out = cam.shape
        zoom_h = target_H / H_out
        zoom_w = target_W / W_out

        cam_resized = zoom(cam, (zoom_h, zoom_w), order=1)

    except Exception:
        import cv2
        cam_resized = cv2.resize(cam, (target_W, target_H),
                                  interpolation=cv2.INTER_LINEAR)

    # Normalize
    cam_resized = cam_resized - cam_resized.min()
    if cam_resized.max() > 0:
        cam_resized = cam_resized / cam_resized.max()

    # Colormap
    cmap = _get_colormap()
    heatmap = cmap(cam_resized)

    # Gabungkan dengan gambar asli
    original_image_clamped = np.clip(original_image, 0, 1)

    if original_image_clamped.ndim == 2:
        original_image_clamped = np.stack(
            [original_image_clamped] * 3, axis=-1
        )

    heatmap_rgb = heatmap[:, :, :3]
    alpha = heatmap[:, :, 3:4] * 0.6 + 0.4

    overlay = (original_image_clamped * (1 - alpha) +
               heatmap_rgb * alpha)

    return overlay


def _get_colormap():
    """Dapatkan colormap viridis."""
    try:
        import matplotlib.cm as cm
        return cm.viridis
    except ImportError:
        return lambda x: np.stack([x, x, x * 0.5, np.ones_like(x)], axis=-1)


def visualize_gradcam(model, image, layer_name=None, class_idx=None,
                      class_names=None, save_path=None, figsize=(12, 5)):
    """
    Visualisasi Grad-CAM lengkap: gambar asli + heatmap overlay + prediksi.

    Args:
        model: instance CNNScratch.
        image: input, bentuk (H, W, C) atau (1, H, W, C).
        layer_name: nama layer konvolusi target.
        class_idx: indeks kelas yang di-explain.
        class_names: list nama kelas (misal: ['buildings', 'forest', ...]).
        save_path: path untuk menyimpan figure.
        figsize: ukuran figure.
    Returns:
        fig, axes.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib diperlukan. Install: pip install matplotlib")

    if image.ndim == 3:
        image_for_plot = image
    else:
        image_for_plot = image[0]

    heatmap, overlay, pred_class, probs = gradcam(
        model, image, layer_name=layer_name, class_idx=class_idx,
        return_heatmap=True
    )

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    axes[0].imshow(image_for_plot)
    axes[0].set_title('Gambar Asli', fontsize=11)
    axes[0].axis('off')

    axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap', fontsize=11)
    axes[1].axis('off')

    axes[2].imshow(overlay)
    axes[2].set_title('Overlay', fontsize=11)
    axes[2].axis('off')

    class_label = class_names[pred_class] if class_names else f'Kelas {pred_class}'
    prob_label = f'{probs[0, pred_class]:.3f}'
    title = f'Grad-CAM — {class_label} ({prob_label})'
    fig.suptitle(title, fontsize=13, fontweight='bold')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Grad-CAM disimpan ke: {save_path}")

    plt.show()

    return fig, axes


def visualize_multiple_gradcam(model, images, layer_name=None,
                               class_names=None, n_cols=3, save_path=None):
    """
    Visualisasi Grad-CAM untuk banyak gambar sekaligus.

    Args:
        model: instance CNNScratch.
        images: list array numpy, masing-masing (H, W, C).
        layer_name: nama layer konvolusi target.
        class_names: list nama kelas.
        n_cols: jumlah kolom per baris.
        save_path: path untuk menyimpan.
    Returns:
        fig.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib diperlukan. Install: pip install matplotlib")

    n_images = len(images)
    n_rows = int(np.ceil(n_images / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols * 2, figsize=(n_cols * 6, n_rows * 3),
                             squeeze=False)

    for idx, img in enumerate(images):
        row = idx // n_cols
        col_base = (idx % n_cols) * 2

        ax_orig = axes[row, col_base]
        ax_orig.imshow(img if img.ndim == 3 else img[0])
        ax_orig.axis('off')

        _, overlay, pred_class, probs = gradcam(
            model, img, layer_name=layer_name, return_heatmap=False
        )

        ax_grad = axes[row, col_base + 1]
        ax_grad.imshow(overlay)
        ax_grad.axis('off')

        class_label = class_names[pred_class] if class_names else f'Kelas {pred_class}'
        prob_label = f'{probs[0, pred_class]:.2f}'
        ax_orig.set_title(f'{class_label} ({prob_label})', fontsize=9)

    for idx in range(n_images, n_rows * n_cols):
        row = idx // n_cols
        col_base = (idx % n_cols) * 2
        axes[row, col_base].axis('off')
        if col_base + 1 < axes.shape[1]:
            axes[row, col_base + 1].axis('off')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Multi Grad-CAM disimpan ke: {save_path}")

    plt.show()

    return fig


def get_gradcam_config(model):
    """
    Dapatkan konfigurasi Grad-CAM untuk model.

    Mencetak daftar lapisan konvolusi yang bisa digunakan sebagai target,
    beserta dimensi spatial output masing-masing.

    Args:
        model: instance CNNScratch.
    Returns:
        list of dicts dengan info setiap Conv2D layer.
    """
    conv_layers = []
    for i, layer in enumerate(model.layers):
        if layer.__class__.__name__ == 'Conv2D':
            info = {
                'index': i,
                'name': layer.summary(),
                'filters': layer.filters,
                'kernel_size': layer.kernel_size,
                'strides': layer.strides,
            }
            conv_layers.append(info)

    print(f"\nDaftar Conv2D layers untuk Grad-CAM ({len(conv_layers)} layer):")
    for info in conv_layers:
        print(f"  [{info['index']}] {info['name']}")

    return conv_layers