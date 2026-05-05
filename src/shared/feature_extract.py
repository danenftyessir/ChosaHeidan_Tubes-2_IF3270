"""
Ekstraksi CNN feature vectors dari gambar Flickr8k.
Menggunakan pretrained CNN (InceptionV3/VGG16) dari Keras.
Hasil disimpan ke disk (.npy) agar tidak perlu ekstraksi ulang.
Mendukung batch processing.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'cnn', 'utils'))
from utils import load_image, batch_loader


# ============================================================================
# CNN Feature Extraction
# ============================================================================

def extract_features_inceptionv3(image_paths, image_dir=None, batch_size=32,
                                 verbose=True):
    """
    Ekstrak feature vectors dari gambar menggunakan InceptionV3.
    Feature vector shape: (N, 2048).

    Cara kerja:
        1. Load gambar dan preprocess (resize ke 299x299, normalize)
        2. Forward pass melalui InceptionV3 (tanpa top layer)
        3. Global Average Pooling → feature vector (2048,)
        4. Simpan ke array numpy dan return

    Args:
        image_paths (list): list path gambar (relative terhadap image_dir
                           atau absolute).
        image_dir (str): direktori dasar untuk path relatif.
                       Jika None, path dianggap absolute.
        batch_size (int): ukuran batch untuk forward pass.
        verbose (bool): cetak progress.
    Returns:
        features: np.ndarray shape (N, 2048).
    """
    try:
        import tensorflow as tf
        from tensorflow.keras.applications import InceptionV3
        from tensorflow.keras.models import Model
        from tensorflow.keras.applications.inception_v3 import preprocess_input
    except ImportError:
        print("[ERROR] TensorFlow tidak tersedia. "
              "Gunakan pretrained weights yang sudah ada.")
        raise ImportError("TensorFlow diperlukan untuk feature extraction")

    if verbose:
        print(f"[Feature Extraction] InceptionV3 — {len(image_paths)} gambar, "
              f"batch_size={batch_size}")

    # Load InceptionV3 tanpa top layer
    base_model = InceptionV3(weights='imagenet', include_top=False,
                             pooling='avg', input_shape=(299, 299, 3))
    # Freeze semua layer
    for layer in base_model.layers:
        layer.trainable = False

    model = Model(inputs=base_model.input, outputs=base_model.output)
    feature_dim = 2048

    # Feature storage
    all_features = []
    N = len(image_paths)
    num_batches = (N + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, N)
        batch_paths = image_paths[start:end]

        # Load dan preprocess batch
        images = []
        for path in batch_paths:
            if image_dir:
                full_path = os.path.join(image_dir, path)
            else:
                full_path = path

            img = load_image(full_path, target_size=(299, 299))
            # InceptionV3 preprocessing: scale ke [-1, 1]
            img = img * 2.0 - 1.0
            images.append(img)

        batch_array = np.stack(images, axis=0)

        # Forward pass
        features = model.predict(batch_array, verbose=0)
        all_features.append(features)

        if verbose and (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx + 1}/{num_batches} selesai "
                  f"({end}/{N} gambar)")

    result = np.concatenate(all_features, axis=0)
    if verbose:
        print(f"[Feature Extraction] Selesai. Shape: {result.shape}")

    return result


def extract_features_vgg16(image_paths, image_dir=None, batch_size=32,
                           verbose=True):
    """
    Ekstrak feature vectors dari gambar menggunakan VGG16.
    Feature vector shape: (N, 512*7*7) setelah flatten, atau (N, 4096) dengan FC.

    Menggunakan VGG16 tanpa top layer + GlobalAveragePooling.

    Args:
        image_paths (list): list path gambar.
        image_dir (str): direktori dasar.
        batch_size (int): ukuran batch.
        verbose (bool): cetak progress.
    Returns:
        features: np.ndarray shape (N, 512).
    """
    try:
        import tensorflow as tf
        from tensorflow.keras.applications import VGG16
        from tensorflow.keras.models import Model
    except ImportError:
        print("[ERROR] TensorFlow tidak tersedia.")
        raise ImportError("TensorFlow diperlukan untuk feature extraction")

    if verbose:
        print(f"[Feature Extraction] VGG16 — {len(image_paths)} gambar, "
              f"batch_size={batch_size}")

    # Load VGG16 tanpa top layer, dengan GlobalAveragePooling
    base_model = VGG16(weights='imagenet', include_top=False,
                       pooling='avg', input_shape=(224, 224, 3))
    for layer in base_model.layers:
        layer.trainable = False

    model = Model(inputs=base_model.input, outputs=base_model.output)
    feature_dim = 512  # VGG16 global avg pooling output

    all_features = []
    N = len(image_paths)
    num_batches = (N + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, N)
        batch_paths = image_paths[start:end]

        images = []
        for path in batch_paths:
            if image_dir:
                full_path = os.path.join(image_dir, path)
            else:
                full_path = path

            img = load_image(full_path, target_size=(224, 224))
            img = img.astype(np.float32)
            images.append(img)

        batch_array = np.stack(images, axis=0)
        # VGG16 preprocess: ImageNet normalization
        from tensorflow.keras.applications.vgg16 import preprocess_input
        batch_array = preprocess_input(batch_array)

        features = model.predict(batch_array, verbose=0)
        all_features.append(features)

        if verbose and (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx + 1}/{num_batches} selesai")

    result = np.concatenate(all_features, axis=0)
    if verbose:
        print(f"[Feature Extraction] Selesai. Shape: {result.shape}")

    return result


def extract_features_custom(image_paths, image_dir, cnn_model_path,
                            input_shape=(299, 299, 3), batch_size=32,
                            preprocess_fn=None, verbose=True):
    """
    Ekstrak feature menggunakan custom trained CNN dari proyek ini.

    Args:
        image_paths (list): list path gambar.
        image_dir (str): direktori gambar.
        cnn_model_path (str): path ke model CNN Keras (.h5).
        input_shape (tuple): bentuk input CNN.
        batch_size (int): ukuran batch.
        preprocess_fn (callable): fungsi preprocessing opsional.
        verbose (bool): cetak progress.
    Returns:
        features: np.ndarray shape (N, feature_dim).
    """
    try:
        from tensorflow.keras.models import load_model
        from tensorflow.keras import Model as KerasModel
    except ImportError:
        raise ImportError("TensorFlow diperlukan")

    if verbose:
        print(f"[Feature Extraction] Custom CNN — {len(image_paths)} gambar")

    # Load model
    cnn = load_model(cnn_model_path, compile=False)
    cnn.trainable = False

    # Remove classification head: get last conv/pooling layer output
    # Asumsi: model punya GlobalAveragePooling sebelum Dense
    model = KerasModel(inputs=cnn.input, outputs=cnn.output)
    H, W, C = input_shape

    all_features = []
    N = len(image_paths)
    num_batches = (N + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, N)
        batch_paths = image_paths[start:end]

        images = []
        for path in batch_paths:
            full_path = os.path.join(image_dir, path)
            img = load_image(full_path, target_size=(H, W))
            if preprocess_fn:
                img = preprocess_fn(img)
            images.append(img)

        batch_array = np.stack(images, axis=0).astype(np.float32)
        features = model.predict(batch_array, verbose=0)
        all_features.append(features)

        if verbose and (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx + 1}/{num_batches} selesai")

    result = np.concatenate(all_features, axis=0)
    if verbose:
        print(f"[Feature Extraction] Selesai. Shape: {result.shape}")

    return result


# ============================================================================
# Save / Load Features
# ============================================================================

def save_cnn_features(features, path):
    """
    Simpan feature vectors ke file .npy.

    Args:
        features (np.ndarray): array fitur, bentuk (N, feature_dim).
        path (str): path file output.
    """
    np.save(path, features)
    print(f"[Feature Extraction] Features disimpan ke: {path} "
          f"(shape: {features.shape})")


def load_cnn_features(path):
    """
    Load feature vectors dari file .npy.

    Args:
        path (str): path ke file .npy.
    Returns:
        np.ndarray: array fitur.
    """
    features = np.load(path)
    print(f"[Feature Extraction] Features dimuat dari: {path} "
          f"(shape: {features.shape})")
    return features


def load_features_with_ids(feature_path, image_ids):
    """
    Load features dan filter berdasarkan image IDs.
    Mengembalikan dictionary {image_id: feature_vector}.

    Args:
        feature_path (str): path ke file .npy.
        image_ids (list): list image IDs yang akan diambil.
    Returns:
        dict: {image_id: feature_vector}.
    """
    features = load_cnn_features(feature_path)
    return {img_id: features[i] for i, img_id in enumerate(image_ids)
            if i < len(features)}


# ============================================================================
# Feature Extraction Pipeline untuk Flickr8k
# ============================================================================

def extract_flickr8k_features(image_dir, captions_dir, output_path,
                              encoder='inceptionv3', batch_size=32,
                              verbose=True):
    """
    Ekstrak dan simpan CNN features untuk seluruh dataset Flickr8k.

    Args:
        image_dir (str): direktori gambar Flickr8k.
        captions_dir (str): direktori caption (untuk dapat list image IDs).
        output_path (str): path output .npy.
        encoder (str): 'inceptionv3' atau 'vgg16'.
        batch_size (int): ukuran batch.
        verbose (bool): cetak progress.
    """
    # Load semua image IDs dari captions
    train_ids = []
    val_ids = []
    test_ids = []

    for fname in ['train_ids.txt', 'val_ids.txt', 'test_ids.txt']:
        path = os.path.join(captions_dir, fname)
        if os.path.exists(path):
            with open(path, 'r') as f:
                ids = [line.strip() for line in f if line.strip()]
                if 'train' in fname:
                    train_ids = ids
                elif 'val' in fname:
                    val_ids = ids
                elif 'test' in fname:
                    test_ids = ids

    all_ids = train_ids + val_ids + test_ids

    if verbose:
        print(f"[Flickr8k Feature Extraction] Total gambar: {len(all_ids)}")

    # Extract features
    if encoder.lower() == 'inceptionv3':
        features = extract_features_inceptionv3(all_ids, image_dir,
                                               batch_size=batch_size,
                                               verbose=verbose)
    elif encoder.lower() == 'vgg16':
        features = extract_features_vgg16(all_ids, image_dir,
                                          batch_size=batch_size,
                                          verbose=verbose)
    else:
        raise ValueError(f"Unknown encoder: {encoder}")

    # Simpan
    save_cnn_features(features, output_path)

    return features


# ============================================================================
# Feature Statistics
# ============================================================================

def compute_feature_statistics(features):
    """
    Hitung statistik feature vectors.

    Args:
        features (np.ndarray): shape (N, feature_dim).
    Returns:
        dict: statistik (mean, std, min, max, median).
    """
    stats = {
        'mean': float(np.mean(features)),
        'std': float(np.std(features)),
        'min': float(np.min(features)),
        'max': float(np.max(features)),
        'median': float(np.median(features)),
        'norm_mean': float(np.mean(np.linalg.norm(features, axis=1))),
    }
    return stats


def find_nearest_neighbors(features_query, features_database, k=5):
    """
    Temukan k gambar terdekat dari database berdasarkan cosine similarity.

    Args:
        features_query (np.ndarray): query features, shape (N_query, dim).
        features_database (np.ndarray): database features, shape (N_db, dim).
        k (int): jumlah neighbors.
    Returns:
        list: list of indices (N_query, k).
    """
    # Normalize
    query_norm = features_query / (np.linalg.norm(features_query, axis=1, keepdims=True) + 1e-10)
    db_norm = features_database / (np.linalg.norm(features_database, axis=1, keepdims=True) + 1e-10)

    # Cosine similarity
    similarities = np.dot(query_norm, db_norm.T)

    # Top-k
    top_k_idx = np.argsort(similarities, axis=1)[:, -k:][:, ::-1]
    return top_k_idx


class FeatureExtractor:
    """
    Wrapper class untuk feature extraction.
    Cache model CNN agar tidak perlu reload setiap kali.
    """

    def __init__(self, encoder='inceptionv3'):
        self.encoder = encoder.lower()
        self.model = None
        self.input_size = None

    def _load_model(self):
        """Load model CNN."""
        if self.model is not None:
            return

        if self.encoder == 'inceptionv3':
            try:
                from tensorflow.keras.applications import InceptionV3
                from tensorflow.keras.models import Model
                base = InceptionV3(weights='imagenet', include_top=False,
                                   pooling='avg', input_shape=(299, 299, 3))
                for layer in base.layers:
                    layer.trainable = False
                self.model = Model(inputs=base.input, outputs=base.output)
                self.input_size = (299, 299)
                self.preprocess = lambda x: x * 2.0 - 1.0
            except ImportError:
                raise ImportError("TensorFlow diperlukan untuk InceptionV3")

        elif self.encoder == 'vgg16':
            try:
                from tensorflow.keras.applications import VGG16
                from tensorflow.keras.models import Model
                base = VGG16(weights='imagenet', include_top=False,
                            pooling='avg', input_shape=(224, 224, 3))
                for layer in base.layers:
                    layer.trainable = False
                self.model = Model(inputs=base.input, outputs=base.output)
                self.input_size = (224, 224)
                from tensorflow.keras.applications.vgg16 import preprocess_input
                self.preprocess = preprocess_input
            except ImportError:
                raise ImportError("TensorFlow diperlukan untuk VGG16")
        else:
            raise ValueError(f"Unknown encoder: {self.encoder}")

        print(f"[FeatureExtractor] Model {self.encoder} berhasil dimuat")

    def extract(self, image_paths, image_dir=None, batch_size=32, verbose=True):
        """
        Ekstrak features dari list gambar.

        Args:
            image_paths (list): list path gambar.
            image_dir (str): direktori dasar.
            batch_size (int): ukuran batch.
            verbose (bool): cetak progress.
        Returns:
            np.ndarray: features shape (N, feature_dim).
        """
        self._load_model()

        H, W = self.input_size
        all_features = []
        N = len(image_paths)
        num_batches = (N + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, N)
            batch_paths = image_paths[start:end]

            images = []
            for path in batch_paths:
                if image_dir:
                    full_path = os.path.join(image_dir, path)
                else:
                    full_path = path

                img = load_image(full_path, target_size=(H, W))
                img = img.astype(np.float32)
                img = self.preprocess(img)
                images.append(img)

            batch_array = np.stack(images, axis=0)
            features = self.model.predict(batch_array, verbose=0)
            all_features.append(features)

            if verbose and (batch_idx + 1) % 20 == 0:
                print(f"  Batch {batch_idx + 1}/{num_batches} selesai")

        result = np.concatenate(all_features, axis=0)
        if verbose:
            print(f"[FeatureExtractor] Selesai. Shape: {result.shape}")

        return result
