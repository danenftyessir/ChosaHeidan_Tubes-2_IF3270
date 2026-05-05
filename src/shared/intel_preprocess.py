"""
Preprocessing untuk Intel Image Classification dataset.
Dataset ini terdiri dari ~25.000 gambar dalam 6 kategori:
  - buildings, forest, glacier, mountain, sea, street

Struktur folder:
  seg_train/   -> training images (~14k)
  seg_test/    -> test images (~3k)
  seg_val/     -> validation images (~3k)

Mendukung:
  - Auto-label extraction dari nama folder
  - Batch loading dan preprocessing
  - Train/val/test split ready-to-use
  - Label encoding dan one-hot conversion
"""

import os
import numpy as np
from collections import defaultdict


# ============================================================================
# Konstanta Intel Image Classification
# ============================================================================

# 6 kategori Intel Image Classification
INTEL_CLASSES = [
    'buildings',  # 0
    'forest',     # 1
    'glacier',    # 2
    'mountain',   # 3
    'sea',        # 4
    'street',     # 5
]

INTEL_CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(INTEL_CLASSES)}
INTEL_IDX_TO_CLASS = {idx: cls for idx, cls in enumerate(INTEL_CLASSES)}

NUM_INTEL_CLASSES = len(INTEL_CLASSES)


# ============================================================================
# Fungsi Preprocessing Intel Image Classification
# ============================================================================

def load_intel_dataset(data_dir, split='train'):
    """
    Load Intel Image Classification dataset untuk split tertentu.

    Args:
        data_dir (str): path ke root dataset intel-ic.
                       Struktur: intel-ic/seg_train/, intel-ic/seg_test/, intel-ic/seg_val/
        split (str): 'train', 'test', atau 'val'
    Returns:
        tuple: (image_paths, labels)
            - image_paths: list of path ke gambar
            - labels: list of integer labels (0-5)
    """
    if split not in ['train', 'test', 'val']:
        raise ValueError(f"Split harus 'train', 'test', atau 'val'. Dapat: {split}")

    folder_name = f'seg_{split}'
    split_dir = os.path.join(data_dir, folder_name)

    if not os.path.exists(split_dir):
        raise FileNotFoundError(f"Direktori tidak ditemukan: {split_dir}")

    image_paths = []
    labels = []

    for class_name in INTEL_CLASSES:
        class_dir = os.path.join(split_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"  [WARNING] Kelas {class_name} tidak ditemukan di {split_dir}")
            continue

        class_idx = INTEL_CLASS_TO_IDX[class_name]

        # List semua file gambar
        for filename in os.listdir(class_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(class_dir, filename)
                image_paths.append(img_path)
                labels.append(class_idx)

    print(f"[Intel {split.title()}] Dimuat: {len(image_paths)} gambar")
    return image_paths, labels


def load_intel_train_val_test(data_dir):
    """
    Load seluruh Intel dataset (train, val, test) sekaligus.

    Args:
        data_dir (str): path ke root dataset intel-ic
    Returns:
        tuple: (train_paths, train_labels, val_paths, val_labels, test_paths, test_labels)
    """
    train_paths, train_labels = load_intel_dataset(data_dir, split='train')
    val_paths, val_labels = load_intel_dataset(data_dir, split='val')
    test_paths, test_labels = load_intel_dataset(data_dir, split='test')

    return train_paths, train_labels, val_paths, val_labels, test_paths, test_labels


def get_intel_class_distribution(labels):
    """
    Hitung distribusi kelas dalam dataset.

    Args:
        labels (list): list of integer labels (0-5)
    Returns:
        dict: {class_name: count}
    """
    from collections import Counter
    counter = Counter(labels)
    distribution = {}
    for class_name, idx in INTEL_CLASS_TO_IDX.items():
        distribution[class_name] = counter.get(idx, 0)
    return distribution


def print_intel_dataset_stats(data_dir):
    """
    Cetak statistik lengkap Intel Image Classification dataset.

    Args:
        data_dir (str): path ke root dataset intel-ic
    """
    print("=" * 70)
    print("INTEL IMAGE CLASSIFICATION — Dataset Statistics")
    print("=" * 70)

    for split in ['train', 'val', 'test']:
        try:
            paths, labels = load_intel_dataset(data_dir, split=split)
            dist = get_intel_class_distribution(labels)

            print(f"\n{split.upper()} SET:")
            print(f"  Total gambar: {len(paths)}")
            print(f"  Distribusi kelas:")
            for class_name, count in dist.items():
                pct = count / len(labels) * 100 if len(labels) > 0 else 0
                print(f"    {class_name:10s}: {count:5d} ({pct:5.1f}%)")
        except FileNotFoundError:
            print(f"\n{split.upper()} SET: Tidak ditemukan")

    print("\n" + "=" * 70)


def labels_to_onehot(labels, num_classes=NUM_INTEL_CLASSES):
    """
    Konversi label integers ke one-hot encoding.

    Args:
        labels (array-like): list atau array of integer labels (0-5)
        num_classes (int): jumlah kelas (default 6)
    Returns:
        np.ndarray: shape (N, num_classes), one-hot encoded
    """
    labels = np.array(labels)
    N = len(labels)
    onehot = np.zeros((N, num_classes), dtype=np.float32)
    onehot[np.arange(N), labels] = 1.0
    return onehot


def onehot_to_labels(onehot):
    """
    Konversi one-hot encoding kembali ke label integers.

    Args:
        onehot (np.ndarray): shape (N, num_classes)
    Returns:
        list: list of integer labels
    """
    return np.argmax(onehot, axis=1).tolist()


# ============================================================================
# Batch Loading untuk Intel Dataset
# ============================================================================

def create_intel_batches(image_paths, labels, batch_size=32, shuffle=True,
                         preprocess_fn=None, verbose=True):
    """
    Generator untuk batch loading Intel dataset.

    Args:
        image_paths (list): list path ke gambar
        labels (list): list label integers (0-5)
        batch_size (int): ukuran batch
        shuffle (bool): shuffle dataset sebelum batching
        preprocess_fn (callable): fungsi preprocessing gambar (opsional)
        verbose (bool): cetak progress
    Yields:
        tuple: (batch_images, batch_labels)
            - batch_images: np.ndarray shape (batch_size, H, W, C)
            - batch_labels: np.ndarray shape (batch_size,) integer labels
    """
    from ..cnn.utils.utils import load_image

    N = len(image_paths)
    indices = np.arange(N)

    if shuffle:
        np.random.shuffle(indices)

    num_batches = (N + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, N)
        batch_indices = indices[start:end]

        batch_images = []
        batch_labels = []

        for idx in batch_indices:
            img_path = image_paths[idx]
            label = labels[idx]

            try:
                img = load_image(img_path, target_size=(150, 150))

                if preprocess_fn:
                    img = preprocess_fn(img)
                else:
                    # Default preprocessing: normalize ke [0, 1]
                    img = img.astype(np.float32) / 255.0

                batch_images.append(img)
                batch_labels.append(label)
            except Exception as e:
                if verbose:
                    print(f"  [WARNING] Gagal load {img_path}: {e}")
                continue

        if batch_images:
            batch_images = np.stack(batch_images, axis=0)
            batch_labels = np.array(batch_labels, dtype=np.int32)

            if verbose and (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx + 1}/{num_batches} selesai")

            yield batch_images, batch_labels


# ============================================================================
# Intel Data Preprocessor Class
# ============================================================================

class IntelImagePreprocessor:
    """
    Wrapper class untuk preprocessing Intel Image Classification dataset.

    Menyimpan paths, labels, dan konfigurasi preprocessing untuk reuse.
    """

    def __init__(self, data_dir, target_size=(150, 150), normalize=True):
        """
        Inisialisasi IntelImagePreprocessor.

        Args:
            data_dir (str): path ke root dataset intel-ic
            target_size (tuple): ukuran target gambar (H, W). Default (150, 150)
            normalize (bool): normalize pixel values ke [0, 1]. Default True
        """
        self.data_dir = data_dir
        self.target_size = target_size
        self.normalize = normalize

        self.train_paths = None
        self.train_labels = None
        self.val_paths = None
        self.val_labels = None
        self.test_paths = None
        self.test_labels = None

        self._is_loaded = False

    def load_data(self):
        """Load seluruh dataset (train, val, test)."""
        print("[IntelImagePreprocessor] Loading Intel Image Classification dataset...")
        self.train_paths, self.train_labels = load_intel_dataset(self.data_dir, 'train')
        self.val_paths, self.val_labels = load_intel_dataset(self.data_dir, 'val')
        self.test_paths, self.test_labels = load_intel_dataset(self.data_dir, 'test')
        self._is_loaded = True
        print(f"[IntelImagePreprocessor] Selesai. Train: {len(self.train_paths)}, "
              f"Val: {len(self.val_paths)}, Test: {len(self.test_paths)}")

    def get_train_batches(self, batch_size=32, shuffle=True):
        """
        Generator untuk training batches.

        Args:
            batch_size (int): ukuran batch
            shuffle (bool): shuffle dataset
        Yields:
            tuple: (batch_images, batch_labels)
        """
        if not self._is_loaded:
            self.load_data()

        return create_intel_batches(
            self.train_paths, self.train_labels,
            batch_size=batch_size, shuffle=shuffle,
            preprocess_fn=self._preprocess_image,
            verbose=True
        )

    def get_val_batches(self, batch_size=32, shuffle=False):
        """Generator untuk validation batches."""
        if not self._is_loaded:
            self.load_data()

        return create_intel_batches(
            self.val_paths, self.val_labels,
            batch_size=batch_size, shuffle=shuffle,
            preprocess_fn=self._preprocess_image,
            verbose=False
        )

    def get_test_batches(self, batch_size=32, shuffle=False):
        """Generator untuk test batches."""
        if not self._is_loaded:
            self.load_data()

        return create_intel_batches(
            self.test_paths, self.test_labels,
            batch_size=batch_size, shuffle=shuffle,
            preprocess_fn=self._preprocess_image,
            verbose=False
        )

    def _preprocess_image(self, img):
        """
        Preprocess single image: resize + normalize.

        Args:
            img: loaded image array
        Returns:
            np.ndarray: preprocessed image
        """
        from ..cnn.utils.utils import load_image

        # Resize (if needed)
        if img.shape[:2] != self.target_size:
            img = load_image(img, target_size=self.target_size)

        # Normalize
        if self.normalize:
            img = img.astype(np.float32) / 255.0

        return img

    def get_train_data(self):
        """Return training paths and labels."""
        if not self._is_loaded:
            self.load_data()
        return self.train_paths, self.train_labels

    def get_val_data(self):
        """Return validation paths and labels."""
        if not self._is_loaded:
            self.load_data()
        return self.val_paths, self.val_labels

    def get_test_data(self):
        """Return test paths and labels."""
        if not self._is_loaded:
            self.load_data()
        return self.test_paths, self.test_labels

    def summary(self):
        """Cetak summary dataset."""
        if not self._is_loaded:
            self.load_data()

        print("\n" + "=" * 70)
        print("INTEL IMAGE CLASSIFICATION — Summary")
        print("=" * 70)
        print(f"Data Directory: {self.data_dir}")
        print(f"Target Size: {self.target_size}")
        print(f"Normalize: {self.normalize}")
        print(f"\nNum Classes: {NUM_INTEL_CLASSES}")
        print(f"Classes: {INTEL_CLASSES}")

        for split_name, paths, labels in [
            ('Train', self.train_paths, self.train_labels),
            ('Val', self.val_paths, self.val_labels),
            ('Test', self.test_paths, self.test_labels),
        ]:
            print(f"\n{split_name} Set:")
            print(f"  Total: {len(paths)} gambar")
            if paths:
                dist = get_intel_class_distribution(labels)
                for class_name, count in dist.items():
                    pct = count / len(labels) * 100
                    print(f"    {class_name:10s}: {count:5d} ({pct:5.1f}%)")

        print("=" * 70)


# ============================================================================
# Utility Functions
# ============================================================================

def encode_intel_labels(class_names):
    """
    Encode class names ke integers.

    Args:
        class_names (list): list nama kelas ('buildings', 'forest', ...)
    Returns:
        list: list integer labels (0-5)
    """
    return [INTEL_CLASS_TO_IDX.get(name, -1) for name in class_names]


def decode_intel_labels(label_indices):
    """
    Decode integer labels ke class names.

    Args:
        label_indices (list): list integer labels (0-5)
    Returns:
        list: list nama kelas
    """
    return [INTEL_IDX_TO_CLASS.get(idx, '<unknown>') for idx in label_indices]


def verify_intel_dataset(data_dir):
    """
    Verifikasi struktur dataset Intel Image Classification.

    Args:
        data_dir (str): path ke root dataset intel-ic
    Returns:
        bool: True jika struktur valid
    """
    print("[Verify] Memeriksa struktur Intel Image Classification dataset...")

    required_dirs = [
        os.path.join(data_dir, 'seg_train'),
        os.path.join(data_dir, 'seg_val'),
        os.path.join(data_dir, 'seg_test'),
    ]

    required_classes = INTEL_CLASSES

    all_valid = True

    for split_dir in required_dirs:
        if not os.path.exists(split_dir):
            print(f"  [ERROR] Direktori tidak ditemukan: {split_dir}")
            all_valid = False
            continue

        print(f"  [OK] {os.path.basename(split_dir)}/ ditemukan")

        # Cek kelas
        for class_name in required_classes:
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"    [WARNING] Kelas {class_name} tidak ditemukan di {split_dir}")
            else:
                num_images = len([f for f in os.listdir(class_dir)
                                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                print(f"    [OK] {class_name:10s}: {num_images} gambar")

    if all_valid:
        print("[Verify] Struktur dataset VALID ✓")
    else:
        print("[Verify] Struktur dataset INVALID ✗")

    return all_valid


if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'cnn', 'utils'))
    from utils import load_image

    # Contoh usage
    print("[Test] Intel Image Classification Preprocessing")

    # Ganti dengan path dataset yang sesuai
    data_dir = 'data/intel-ic'

    # Verify dataset
    # verify_intel_dataset(data_dir)

    # Print stats
    # print_intel_dataset_stats(data_dir)

    # Atau gunakan class wrapper
    preprocessor = IntelImagePreprocessor(data_dir, target_size=(150, 150), normalize=True)
    # preprocessor.summary()

    print("\n[Test] Selesai.")
