"""
Utility functions untuk pemrosesan data image menggunakan PIL/Pillow dan NumPy.
Tidak menggunakan Keras preprocessing — murni PIL + NumPy.

Fungsi utama:
  - load_image:       load satu gambar dari file path
  - load_batch:       load sekumpulan gambar menjadi batch numpy array
  - extract_features: ekstraksi feature vector menggunakan frozen Keras CNN encoder
"""

import os
import numpy as np
from PIL import Image


def load_image(path, target_size=(150, 150)):
    """
    Load gambar dari file path menggunakan PIL, resize, dan normalisasi ke [0, 1].

    Args:
        path (str): path ke file gambar
        target_size (tuple): ukuran target (H, W). Default (150, 150).
    Returns:
        np.ndarray: array shape (H, W, 3), dtype float32, nilai [0.0, 1.0]
    """
    img = Image.open(path).convert('RGB')
    img = img.resize((target_size[1], target_size[0]), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr


def load_batch(paths, target_size=(150, 150)):
    """
    Load dan proses sekumpulan gambar dari list file path menjadi numpy array.

    Args:
        paths (list[str]): list path ke file gambar
        target_size (tuple): ukuran target (H, W). Default (150, 150).
    Returns:
        np.ndarray: shape (N, H, W, C), dtype float32, nilai [0.0, 1.0]
                    Gambar yang gagal di-load akan dilewati (tidak dimasukkan).
    """
    images = []
    for path in paths:
        try:
            img = load_image(path, target_size=target_size)
            images.append(img)
        except Exception as e:
            print(f"[Warning] Gagal load {path}: {e}")
    if not images:
        H, W = target_size
        return np.zeros((0, H, W, 3), dtype=np.float32)
    return np.stack(images, axis=0)


def extract_features(image_paths, encoder_model, target_size=(150, 150),
                     batch_size=32, save_path=None):
    """
    Ekstraksi feature vectors menggunakan frozen Keras CNN encoder.

    Encoder harus sudah di-freeze sebelum dipanggil. Hasil disimpan ke disk
    dalam format .npy agar tidak perlu diekstraksi ulang.

    Args:
        image_paths (list[str]): list path ke gambar
        encoder_model: Keras model (frozen) yang menerima input (N, H, W, C)
                       dan menghasilkan feature vectors (N, D)
        target_size (tuple): ukuran resize gambar. Default (150, 150).
        batch_size (int): jumlah gambar per batch saat inferensi. Default 32.
        save_path (str|None): path file .npy untuk menyimpan hasil.
                              Jika None, hasil tidak disimpan ke disk.
    Returns:
        np.ndarray: shape (N, D) — feature vectors untuk semua gambar
    """
    all_features = []
    n = len(image_paths)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_paths = image_paths[start:end]
        batch = load_batch(batch_paths, target_size=target_size)

        if batch.shape[0] == 0:
            continue

        feats = encoder_model.predict(batch, verbose=0)
        all_features.append(feats)

        if (start // batch_size + 1) % 10 == 0:
            print(f"  [extract_features] {end}/{n} gambar diproses")

    if not all_features:
        return np.zeros((0,), dtype=np.float32)

    features = np.concatenate(all_features, axis=0)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
        np.save(save_path, features)
        print(f"[extract_features] Features disimpan ke: {save_path} (shape: {features.shape})")

    return features
