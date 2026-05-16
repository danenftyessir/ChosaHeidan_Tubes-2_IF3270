"""
Grad-CAM (Gradient-weighted Class Activation Mapping) untuk model Keras.
Memvisualisasikan region gambar yang paling berpengaruh terhadap prediksi model.

Referensi: Selvaraju et al. (2017) - Grad-CAM: Visual Explanations from Deep
Networks via Gradient-based Localization.

Algoritma:
1. Buat sub-model yang menghasilkan aktivasi layer konvolusi target + output akhir
2. Forward pass → catat activations lapisan konvolusi target
3. Hitung gradient output kelas terhadap activations menggunakan tf.GradientTape
4. Global Average Pooling pada gradient → bobot importance per filter
5. Weighted sum activations dengan bobot → class activation map
6. ReLU → hanya activation positif yang relevan
7. Resize ke ukuran gambar asli → overlay dengan gambar
"""

import numpy as np
import os
import tensorflow as tf


def _find_best_gradcam_layer(model):
    """
    Cari target layer terbaik untuk Grad-CAM.

    Preferensi: layer Activation terakhir dengan output 4D (post-ReLU, sebelum MaxPool
    selanjutnya atau GAP). Ini memberikan:
    - Resolusi spasial lebih tinggi (75x75 vs 37x37 setelah MaxPool)
    - Aktivasi non-negatif (setelah ReLU)
    - Gradient lebih bermakna secara spasial

    Fallback: layer 4D terakhir sebelum GAP → fallback ke MaxPool terakhir.
    Final fallback: Conv2D terakhir.
    """
    last_activation = None
    last_spatial = None

    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.GlobalAveragePooling2D):
            break
        try:
            if len(layer.output_shape) == 4:
                last_spatial = layer
                if isinstance(layer, tf.keras.layers.Activation):
                    last_activation = layer
        except Exception:
            pass

    if last_activation is not None:
        return last_activation
    if last_spatial is not None:
        return last_spatial
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer
    return None


def gradcam_keras(model, image, layer_name=None, class_idx=None):
    """
    Grad-CAM untuk model Keras menggunakan tf.GradientTape.

    Args:
        model: Keras Model.
        image: array numpy (H, W, C) atau (1, H, W, C), dinormalisasi [0, 1].
        layer_name: nama layer Conv2D target. Jika None, gunakan Conv2D terakhir.
        class_idx: indeks kelas yang di-explain. Jika None, gunakan prediksi tertinggi.
    Returns:
        heatmap: array numpy 2D (H_out, W_out) — class activation map ternormalisasi.
        class_idx: int — kelas yang di-explain.
        probs: array numpy (1, num_classes) — distribusi probabilitas.
    """
    if layer_name is None:
        target_layer = _find_best_gradcam_layer(model)
        if target_layer is None:
            raise ValueError("Tidak ada layer spatial ditemukan di model.")
    else:
        target_layer = model.get_layer(layer_name)

    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[target_layer.output, model.output]
    )

    if image.ndim == 3:
        image = image[np.newaxis, ...]
    image_tensor = tf.cast(image, tf.float32)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_tensor, training=False)
        if class_idx is None:
            class_idx = int(np.argmax(predictions[0]))
        class_score = predictions[:, class_idx]

    grads = tape.gradient(class_score, conv_outputs)  # (1, H_out, W_out, C)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # (C,)

    conv_outputs = conv_outputs[0]  # (H_out, W_out, C)
    cam = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1).numpy()  # (H_out, W_out)

    cam = np.maximum(cam, 0)
    if cam.max() > 0:
        cam = cam / cam.max()

    return cam, class_idx, predictions.numpy()


def _to_display(image):
    """
    Normalisasi gambar ke [0, 1] untuk display/overlay.
    Bekerja untuk range input apapun ([0,1], [0,1/255], [0,255], dll).
    """
    img = np.clip(image, 0, None).astype(np.float32)
    max_val = img.max()
    if max_val > 0:
        img = img / max_val
    return img


def _resize_and_overlay(cam, original_image):
    """
    Resize heatmap ke ukuran gambar asli dan buat overlay.

    Args:
        cam: array numpy 2D (H_out, W_out).
        original_image: array numpy (H, W, C), range bebas (auto-dinormalisasi untuk display).
    Returns:
        heatmap_colored: array numpy (H, W, C) — heatmap dengan colormap jet.
        overlay: array numpy (H, W, C) — gambar + heatmap overlay.
    """
    target_H, target_W = original_image.shape[:2]

    try:
        from scipy.ndimage import zoom
        H_out, W_out = cam.shape
        cam_resized = zoom(cam, (target_H / H_out, target_W / W_out), order=1)
    except ImportError:
        import cv2
        cam_resized = cv2.resize(cam, (target_W, target_H), interpolation=cv2.INTER_LINEAR)

    cam_resized = cam_resized - cam_resized.min()
    if cam_resized.max() > 0:
        cam_resized = cam_resized / cam_resized.max()

    try:
        import matplotlib.cm as cm
        heatmap_rgba = cm.jet(cam_resized)
    except ImportError:
        heatmap_rgba = np.stack(
            [cam_resized, cam_resized * 0.5, np.zeros_like(cam_resized), np.ones_like(cam_resized)],
            axis=-1
        )

    heatmap_colored = heatmap_rgba[:, :, :3]

    # Auto-normalisasi gambar asli untuk overlay (bebas dari range input model)
    original = _to_display(original_image)
    if original.ndim == 2:
        original = np.stack([original] * 3, axis=-1)

    overlay = original * 0.6 + heatmap_colored * 0.4

    return heatmap_colored, overlay


def visualize_gradcam(model, image, layer_name=None, class_idx=None,
                      class_names=None, save_path=None, figsize=(12, 5)):
    """
    Visualisasi Grad-CAM: gambar asli + heatmap + overlay.

    Args:
        model: Keras Model.
        image: array numpy (H, W, C) atau (1, H, W, C), dinormalisasi [0, 1].
        layer_name: nama layer Conv2D target (None = Conv2D terakhir).
        class_idx: indeks kelas yang di-explain (None = prediksi tertinggi).
        class_names: list nama kelas.
        save_path: path untuk menyimpan figure.
        figsize: ukuran figure.
    Returns:
        fig, axes.
    """
    import matplotlib.pyplot as plt

    image_for_plot = image[0] if image.ndim == 4 else image

    cam, pred_class, probs = gradcam_keras(
        model, image_for_plot, layer_name=layer_name, class_idx=class_idx
    )
    heatmap_colored, overlay = _resize_and_overlay(cam, image_for_plot)

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    axes[0].imshow(_to_display(image_for_plot))
    axes[0].set_title('Gambar Asli', fontsize=11)
    axes[0].axis('off')

    axes[1].imshow(cam, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap', fontsize=11)
    axes[1].axis('off')

    axes[2].imshow(np.clip(overlay, 0, 1))
    axes[2].set_title('Overlay', fontsize=11)
    axes[2].axis('off')

    class_label = class_names[pred_class] if class_names else f'Kelas {pred_class}'
    prob_label = f'{probs[0, pred_class]:.3f}'
    fig.suptitle(f'Grad-CAM — {class_label} ({prob_label})', fontsize=13, fontweight='bold')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Grad-CAM disimpan ke: {save_path}")

    plt.show()
    return fig, axes


def visualize_multiple_gradcam(model, images, layer_name=None,
                               class_names=None, n_cols=3, save_path=None):
    """
    Visualisasi Grad-CAM untuk banyak gambar: gambar asli | overlay per gambar.

    Args:
        model: Keras Model.
        images: list array numpy (H, W, C).
        layer_name: nama layer Conv2D target.
        class_names: list nama kelas.
        n_cols: jumlah kolom per baris.
        save_path: path untuk menyimpan.
    Returns:
        fig.
    """
    import matplotlib.pyplot as plt

    n_images = len(images)
    n_rows = int(np.ceil(n_images / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols * 2,
                             figsize=(n_cols * 6, n_rows * 3),
                             squeeze=False)

    for idx, img in enumerate(images):
        row = idx // n_cols
        col_base = (idx % n_cols) * 2

        cam, pred_class, probs = gradcam_keras(model, img, layer_name=layer_name)
        _, overlay = _resize_and_overlay(cam, img)

        axes[row, col_base].imshow(_to_display(img))
        axes[row, col_base].axis('off')

        axes[row, col_base + 1].imshow(np.clip(overlay, 0, 1))
        axes[row, col_base + 1].axis('off')

        class_label = class_names[pred_class] if class_names else f'Kelas {pred_class}'
        prob_label = f'{probs[0, pred_class]:.2f}'
        axes[row, col_base].set_title(f'{class_label} ({prob_label})', fontsize=9)

    for idx in range(n_images, n_rows * n_cols):
        row = idx // n_cols
        col_base = (idx % n_cols) * 2
        axes[row, col_base].axis('off')
        if col_base + 1 < axes.shape[1]:
            axes[row, col_base + 1].axis('off')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Multi Grad-CAM disimpan ke: {save_path}")

    plt.show()
    return fig


def get_gradcam_config(model):
    """
    Dapatkan daftar semua layer spatial (4D output) yang bisa digunakan
    sebagai target Grad-CAM, beserta layer default yang dipilih otomatis.

    Args:
        model: Keras Model.
    Returns:
        list of dicts dengan info setiap layer spatial.
    """
    spatial_layers = []
    for i, layer in enumerate(model.layers):
        try:
            if len(layer.output_shape) == 4:
                spatial_layers.append({
                    'index': i,
                    'name': layer.name,
                    'type': layer.__class__.__name__,
                    'output_shape': layer.output_shape,
                })
        except Exception:
            pass

    default_layer = _find_best_gradcam_layer(model)
    default_name = default_layer.name if default_layer else 'N/A'

    print(f"\nLayer spatial untuk Grad-CAM ({len(spatial_layers)} layer):")
    for info in spatial_layers:
        marker = '  <-- DEFAULT' if info['name'] == default_name else ''
        print(f"  [{info['index']}] {info['name']:<20} "
              f"({info['type']:<20}) output={info['output_shape']}{marker}")

    return spatial_layers
