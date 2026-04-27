"""
Fungsi utilitas untuk CNN: memuat gambar, pemrosesan batch, dan ekstraksi fitur.
Semua implementasi pakai PIL/Pillow dan NumPy saja (tanpa Keras preprocessing).
"""

import os
import numpy as np
from PIL import Image


# Muat gambar

def load_image(path, target_size=(150, 150)):
    """
    Muat satu gambar dari path file.

    Args:
        path (str): path ke file gambar, bisa absolut atau relatif.
        target_size (tuple): (tinggi, lebar) untuk resize gambar.
                             Default: (150, 150).
    Returns:
        img_array: array numpy, bentuk (H, W, C), dtype float64,
                   nilai pixel dinormalisasi ke [0, 1].
    """
    img = Image.open(path)

    # Kalau grayscale, konversi ke RGB
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # Resize
    img = img.resize((target_size[1], target_size[0]), Image.BILINEAR)

    # Konversi ke array numpy
    img_array = np.array(img, dtype=np.float64)

    # Normalisasi ke [0, 1]
    img_array = img_array / 255.0

    return img_array


def save_image(img_array, save_path):
    """
    Simpan array numpy jadi file gambar.

    Args:
        img_array: array numpy, nilai di [0, 1], bentuk (H, W, C) atau (H, W).
        save_path: path tujuan file.
    """
    img = (np.clip(img_array, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(img).save(save_path)


# Muat batch gambar

def batch_loader(file_paths, batch_size, target_size=(150, 150),
                 shuffle=False, seed=None):
    """
    Generator yang menghasilkan batch gambar sebagai array numpy.

    Args:
        file_paths (list of str): daftar path file gambar.
        batch_size (int): jumlah gambar per batch.
        target_size (tuple): (H, W) untuk resize setiap gambar.
        shuffle (bool): apakah akan acak urutan file_paths.
        seed (int atau None): seed random untuk shuffle.
    Yields:
        batch: array numpy, bentuk (batch_size, H, W, C), dtype float64,
               dinormalisasi ke [0, 1].
    """
    paths = list(file_paths)
    n = len(paths)

    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(paths)

    start = 0
    while start < n:
        end = min(start + batch_size, n)
        batch_paths = paths[start:end]
        batch_size_actual = len(batch_paths)

        # Muat gambar pertama untuk tahu dimensi
        first = load_image(batch_paths[0], target_size)
        H, W, C = first.shape
        dtype = first.dtype

        # Pre-allocate batch array
        batch = np.zeros((batch_size_actual, H, W, C), dtype=dtype)
        for i, fp in enumerate(batch_paths):
            batch[i] = load_image(fp, target_size)

        yield batch
        start = end


def load_image_dataset(folder_path, batch_size, target_size=(150, 150),
                       shuffle=False, seed=None):
    """
    Fungsi pembantu: muat semua gambar dari struktur folder.

    Asumsi struktur folder: folder_path/nama_kelas/*.jpg (atau .png)
    Menghasilkan (gambar, label) untuk setiap batch.

    Args:
        folder_path (str): folder root yang berisi subfolder kelas.
        batch_size (int): ukuran batch.
        target_size (tuple): (H, W) untuk resize.
        shuffle (bool): apakah diacak.
        seed (int): seed random.
    Yields:
        (images_batch, labels_batch): keduanya array numpy.
    """
    class_names = sorted(os.listdir(folder_path))
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    file_paths = []
    labels = []
    for class_name in class_names:
        class_dir = os.path.join(folder_path, class_name)
        if not os.path.isdir(class_dir):
            continue
        for fname in os.listdir(class_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                file_paths.append(os.path.join(class_dir, fname))
                labels.append(class_to_idx[class_name])

    for batch, lab in batch_loader(file_paths, batch_size, target_size,
                                    shuffle=shuffle, seed=seed):
        yield batch, np.array(lab[:batch.shape[0]])


# Ekstraksi fitur CNN (untuk bonus visualisasi)

def extract_cnn_features(image_paths, encoder='inceptionv3',
                         target_size=(150, 150), save_path=None):
    """
    Ekstrak vektor fitur CNN dari gambar pakai encoder Keras yang di-freeze.

    Args:
        image_paths (list of str): daftar path file gambar.
        encoder (str): 'inceptionv3' atau 'vgg16'.
        target_size (tuple): ukuran input yang diharapkan encoder.
            - InceptionV3: (299, 299)
            - VGG16: (224, 224)
        save_path (str atau None): jika ada, simpan fitur ke file .npy.
    Returns:
        features: array numpy, bentuk (N, feature_dim).
            - InceptionV3: (N, 2048)
            - VGG16: (N, 4096)
    """
    try:
        import tensorflow as tf
        from tensorflow.keras.applications import InceptionV3, VGG16
        from tensorflow.keras.applications.inception_v3 import preprocess_input as inc_prep
        from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_prep
    except ImportError:
        raise ImportError(
            "TensorFlow diperlukan untuk ekstraksi fitur. "
            "Install: pip install tensorflow"
        )

    if encoder == 'inceptionv3':
        base_model = InceptionV3(include_top=True, weights='imagenet')
        preprocess_fn = inc_prep
        feature_dim = 2048
        resize_to = (299, 299)
    elif encoder == 'vgg16':
        base_model = VGG16(include_top=True, weights='imagenet')
        preprocess_fn = vgg_prep
        feature_dim = 4096
        resize_to = (224, 224)
    else:
        raise ValueError(f"Encoder tidak dikenal: '{encoder}'. Gunakan 'inceptionv3' atau 'vgg16'.")

    # Freeze semua layer
    for layer in base_model.layers:
        layer.trainable = False

    n = len(image_paths)
    features = np.zeros((n, feature_dim), dtype=np.float32)

    ekstrak_batch = 32
    for start in range(0, n, ekstrak_batch):
        end = min(start + ekstrak_batch, n)
        batch_paths = image_paths[start:end]

        images = []
        for fp in batch_paths:
            img = Image.open(fp)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img = img.resize(resize_to, Image.BILINEAR)
            arr = np.array(img, dtype=np.float32)
            images.append(arr)

        batch_arr = np.stack(images)
        batch_arr = preprocess_fn(batch_arr)

        preds = base_model.predict(batch_arr, verbose=0)
        features[start:end] = preds

    if save_path is not None:
        np.save(save_path, features)
        print(f"Fitur disimpan ke: {save_path}")

    return features


# Utilitas label encoding

def get_class_names(train_folder):
    """Ambil nama kelas dari struktur folder (diurutkan alfabet)."""
    return sorted([d for d in os.listdir(train_folder)
                   if os.path.isdir(os.path.join(train_folder, d))])


def one_hot_encode(labels, num_classes=None):
    """Konversi label integer ke one-hot encoding."""
    labels = np.asarray(labels).reshape(-1)
    if num_classes is None:
        num_classes = int(labels.max()) + 1
    one_hot = np.zeros((len(labels), num_classes), dtype=np.float64)
    one_hot[np.arange(len(labels)), labels] = 1.0
    return one_hot
