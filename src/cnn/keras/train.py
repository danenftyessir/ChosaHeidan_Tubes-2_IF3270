"""
Training loop CNN dengan Keras.
Melatih model CNN (Conv2D dan LocallyConnected2D) pada Intel Image Classification.
Menggunakan macro F1-score sebagai metrik utama.
Menyimpan bobot terbaik ke file .h5.
"""

import os
import sys
import time
import json
import numpy as np


# ============================================================================
# Macro F1-Score Metric (NumPy implementation, compatible dengan semua backend)
# ============================================================================

def compute_confusion_matrix(y_true, y_pred, num_classes):
    """
    Hitung confusion matrix.

    Args:
        y_true: array of true labels (N,)
        y_pred: array of predicted labels (N,)
        num_classes: jumlah kelas
    Returns:
        np.ndarray: confusion matrix shape (num_classes, num_classes)
    """
    cm = np.zeros((num_classes, num_classes), dtype=np.int32)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def compute_f1_from_cm(cm):
    """
    Hitung macro F1-score dari confusion matrix.

    Args:
        cm: confusion matrix (num_classes, num_classes)
    Returns:
        dict: {precision, recall, f1_score, per_class}
    """
    num_classes = cm.shape[0]
    precisions = []
    recalls = []
    f1_scores = []

    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    macro_precision = np.mean(precisions)
    macro_recall = np.mean(recalls)
    macro_f1 = np.mean(f1_scores)

    return {
        'precision': macro_precision,
        'recall': macro_recall,
        'f1_score': macro_f1,
        'per_class': {
            'precision': precisions,
            'recall': recalls,
            'f1_score': f1_scores,
        }
    }


def macro_f1_score(y_true, y_pred, num_classes=6):
    """
    Hitung macro F1-score untuk klasifikasi multi-kelas.

    Args:
        y_true: array of true labels (N,)
        y_pred: array of predicted labels (N,)
        num_classes: jumlah kelas
    Returns:
        float: macro F1-score
    """
    cm = compute_confusion_matrix(y_true, y_pred, num_classes)
    result = compute_f1_from_cm(cm)
    return result['f1_score']


# ============================================================================
# Data Generators
# ============================================================================

def create_numpy_generator(image_paths, labels, batch_size=32, shuffle=True,
                           target_size=(150, 150)):
    """
    Buat generator dari numpy arrays (numpy-only, tidak butuh Keras ImageDataGenerator).

    Args:
        image_paths (list): list path ke gambar
        labels (list): list label integers
        batch_size (int): ukuran batch
        shuffle (bool): shuffle data
        target_size (tuple): ukuran gambar (H, W)
    Yields:
        tuple: (X_batch, y_batch)
    """
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))
    try:
        from utils import load_image
    except ImportError:
        from cnn.utils.utils import load_image

    N = len(image_paths)
    indices = np.arange(N)

    while True:
        if shuffle:
            np.random.shuffle(indices)

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch_idx = indices[start:end]

            batch_images = []
            batch_labels = []

            for idx in batch_idx:
                img_path = image_paths[idx]
                label = labels[idx]

                try:
                    img = load_image(img_path, target_size=target_size)
                    img = img.astype(np.float32) / 255.0
                    batch_images.append(img)
                    batch_labels.append(label)
                except Exception as e:
                    print(f"[Warning] Gagal load {img_path}: {e}")
                    continue

            if batch_images:
                X_batch = np.stack(batch_images, axis=0)
                y_batch = np.array(batch_labels, dtype=np.int32)
                yield X_batch, y_batch


# ============================================================================
# Training Loop Utama
# ============================================================================

def train_cnn(model, train_gen, val_gen=None, epochs=30, batch_size=32,
              callbacks=None, steps_per_epoch=None, val_steps=None,
              num_classes=6, verbose=1):
    """
    Training loop utama untuk CNN Keras.

    Args:
        model: Keras Model yang sudah di-compile
        train_gen: training data generator
        val_gen: validation data generator (optional)
        epochs (int): jumlah epoch
        batch_size (int): ukuran batch
        callbacks (list): list Keras callbacks
        steps_per_epoch (int): steps per epoch (default: len(train_gen))
        val_steps (int): validation steps (default: len(val_gen))
        num_classes (int): jumlah kelas
        verbose (int): 0=silent, 1=progress bar, 2=one line per epoch
    Returns:
        History object
    """
    if hasattr(train_gen, '__len__'):
        if steps_per_epoch is None:
            steps_per_epoch = len(train_gen)
    else:
        steps_per_epoch = steps_per_epoch or 100

    if val_gen is not None:
        if hasattr(val_gen, '__len__'):
            val_steps = val_steps if val_steps is not None else len(val_gen)
        else:
            val_steps = val_steps or 50

    history = model.fit(
        train_gen,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=val_gen,
        steps_per_epoch=steps_per_epoch,
        validation_steps=val_steps,
        callbacks=callbacks,
        verbose=verbose
    )

    return history


# ============================================================================
# Callbacks
# ============================================================================

def get_callbacks(model_name, weights_dir='weights/cnn',
                  monitor='val_f1', patience=7):
    """
    Dapatkan list Keras callbacks.

    Args:
        model_name (str): nama model (untuk nama file bobot)
        weights_dir (str): direktori penyimpanan bobot
        monitor (str): metrik yang dimonitor
        patience (int): early stopping patience (epochs)
    Returns:
        list: Keras callbacks
    """
    from tensorflow.keras.callbacks import (
        EarlyStopping, ModelCheckpoint,
        ReduceLROnPlateau
    )

    os.makedirs(weights_dir, exist_ok=True)
    filepath = os.path.join(weights_dir, f'{model_name}.h5')

    return [
        # Simpan bobot terbaik
        ModelCheckpoint(
            filepath=filepath,
            monitor=monitor,
            save_best_only=True,
            save_weights_only=False,
            mode='max',
            verbose=1
        ),
        # Early stopping
        EarlyStopping(
            monitor=monitor,
            patience=patience,
            mode='max',
            restore_best_weights=True,
            verbose=1
        ),
        # Reduce learning rate
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
    ]


# ============================================================================
# Macro F1 Callback
# ============================================================================

class MacroF1Callback:
    """
    Keras Callback untuk menghitung macro F1-score per epoch pada validation set.

    Menambahkan metrik 'val_f1' ke history setelah setiap epoch.
    Compatible dengan generator-based training.
    """

    def __init__(self, val_data=None, batch_size=32, num_classes=6,
                 monitor='val_f1', steps=None):
        """
        Args:
            val_data: tuple (image_paths, labels) atau None
            batch_size (int): ukuran batch untuk evaluasi
            num_classes (int): jumlah kelas
            monitor (str): nama metrik di history
            steps (int): jumlah steps per evaluasi
        """
        self.val_data = val_data
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.monitor = monitor
        self.steps = steps

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        if self.val_data is not None:
            val_paths, val_labels = self.val_data

            try:
                val_gen = create_numpy_generator(
                    val_paths, val_labels,
                    batch_size=self.batch_size,
                    shuffle=False
                )

                all_preds = []
                all_true = []

                max_steps = self.steps or (len(val_paths) // self.batch_size)

                for step, (X_batch, y_batch) in enumerate(val_gen):
                    if step >= max_steps:
                        break
                    preds = self._model.predict(X_batch, verbose=0)
                    all_preds.extend(np.argmax(preds, axis=1))
                    all_true.extend(y_batch)

                if all_preds:
                    f1 = macro_f1_score(
                        np.array(all_true),
                        np.array(all_preds),
                        num_classes=self.num_classes
                    )
                    logs[self.monitor] = float(f1)
                    print(f" - {self.monitor}: {f1:.4f}")
            except Exception:
                pass

    def set_model(self, model):
        """Set model reference (dipanggil oleh Keras)."""
        self._model = model


# ============================================================================
# Training dengan Variasi Hyperparameter
# ============================================================================

def train_with_variations(data_dir, arch_type='conv2d',
                          layer_variations=[2, 3, 4],
                          filter_variations=[32, 64, 128],
                          kernel_variations=[(3, 3), (5, 5), (7, 7)],
                          pooling_variations=['max', 'average'],
                          epochs=30, batch_size=32,
                          weights_dir='weights/cnn',
                          results_path='results/cnn_variations.json',
                          verbose=True):
    """
    Latih CNN dengan berbagai kombinasi hyperparameter.

    Args:
        data_dir (str): path ke dataset Intel
        arch_type (str): 'conv2d' atau 'locallyconnected'
        layer_variations (list): jumlah conv layers [2, 3, 4]
        filter_variations (list): filter base [32, 64, 128]
        kernel_variations (list): kernel sizes [(3,3), (5,5), (7,7)]
        pooling_variations (list): ['max', 'average']
        epochs (int): epochs per konfigurasi
        batch_size (int): ukuran batch
        weights_dir (str): direktori bobot
        results_path (str): path untuk menyimpan hasil
        verbose (bool): cetak progress
    Returns:
        dict: {config_name: results_dict}
    """
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))
    from .model_keras import build_cnn_factory, count_parameters

    os.makedirs(weights_dir, exist_ok=True)
    results_dir = os.path.dirname(results_path) or '.'
    os.makedirs(results_dir, exist_ok=True)

    # Load data
    print("[Train] Memuat dataset Intel...")
    preprocessor = IntelImagePreprocessor(data_dir, target_size=(150, 150))
    preprocessor.load_data()

    train_paths = preprocessor.train_paths
    train_labels = preprocessor.train_labels
    val_paths = preprocessor.val_paths
    val_labels = preprocessor.val_labels

    print(f"  Train: {len(train_paths)} gambar")
    print(f"  Val:   {len(val_paths)} gambar")

    results = {}
    total_configs = (
        len(layer_variations) * len(filter_variations) *
        len(kernel_variations) * len(pooling_variations)
    )
    config_idx = 0

    for num_layers in layer_variations:
        for base_filters in filter_variations:
            for kernel_size in kernel_variations:
                for pooling in pooling_variations:
                    config_idx += 1

                    config_name = (
                        f"{arch_type}_L{num_layers}_F{base_filters}_"
                        f"K{kernel_size[0]}_{pooling}"
                    )

                    if verbose:
                        print(f"\n{'=' * 60}")
                        print(f"[{config_idx}/{total_configs}] {config_name}")
                        print(f"{'=' * 60}")

                    start_time = time.time()

                    # Build model
                    model, config = build_cnn_factory(
                        arch_type=arch_type,
                        num_conv_layers=num_layers,
                        num_filters=base_filters,
                        kernel_size=kernel_size,
                        pooling_type=pooling,
                        num_classes=6
                    )

                    params = count_parameters(model)
                    if verbose:
                        print(f"  Parameters: {params['total']:,}")

                    # Callbacks
                    callbacks = get_callbacks(
                        config_name, weights_dir=weights_dir,
                        monitor='val_f1', patience=7
                    )

                    # F1 Callback
                    f1_cb = MacroF1Callback(
                        val_data=(val_paths, val_labels),
                        batch_size=batch_size,
                        num_classes=6,
                        steps=len(val_paths) // batch_size
                    )
                    f1_cb.set_model(model)
                    callbacks.append(f1_cb)

                    # Data generators
                    train_gen = create_numpy_generator(
                        train_paths, train_labels,
                        batch_size=batch_size, shuffle=True
                    )
                    val_gen = create_numpy_generator(
                        val_paths, val_labels,
                        batch_size=batch_size, shuffle=False
                    )

                    steps_per_epoch = max(1, len(train_paths) // batch_size)
                    val_steps = max(1, len(val_paths) // batch_size)

                    try:
                        history = train_cnn(
                            model, train_gen, val_gen,
                            epochs=epochs,
                            batch_size=batch_size,
                            callbacks=callbacks,
                            steps_per_epoch=steps_per_epoch,
                            val_steps=val_steps,
                            verbose=2
                        )

                        elapsed = time.time() - start_time
                        best_val_f1 = max(history.history.get('val_f1', [0.0]))
                        best_val_acc = max(history.history.get('val_accuracy', [0.0]))

                        results[config_name] = {
                            'train_loss': [float(x) for x in history.history.get('loss', [])],
                            'val_loss': [float(x) for x in history.history.get('val_loss', [])],
                            'val_accuracy': [float(x) for x in history.history.get('val_accuracy', [])],
                            'val_f1': [float(x) for x in history.history.get('val_f1', [])],
                            'best_val_f1': float(best_val_f1),
                            'best_val_accuracy': float(best_val_acc),
                            'elapsed_seconds': float(elapsed),
                            'config': config,
                            'parameters': params,
                        }

                        # Generate training history plot
                        try:
                            import matplotlib
                            matplotlib.use('Agg')  # Non-interactive backend
                            import matplotlib.pyplot as plt

                            plot_dir = os.path.join(os.path.dirname(weights_dir), 'plots', arch_type)
                            os.makedirs(plot_dir, exist_ok=True)

                            history_dict = {
                                'train_loss': [float(x) for x in history.history.get('loss', [])],
                                'val_loss': [float(x) for x in history.history.get('val_loss', [])],
                            }
                            save_path = os.path.join(plot_dir, f'{config_name}_loss.png')
                            plt.figure(figsize=(10, 6))
                            epochs_range = range(1, len(history_dict['train_loss']) + 1)
                            if history_dict['train_loss']:
                                plt.plot(epochs_range, history_dict['train_loss'], 'b-o',
                                         label='Training Loss', linewidth=2, markersize=4)
                            if history_dict['val_loss']:
                                plt.plot(epochs_range, history_dict['val_loss'], 'r-s',
                                         label='Validation Loss', linewidth=2, markersize=4)
                            plt.xlabel('Epoch', fontsize=12)
                            plt.ylabel('Loss', fontsize=12)
                            plt.title(f'{config_name} - Training & Validation Loss', fontsize=13)
                            plt.legend(fontsize=11)
                            plt.grid(True, alpha=0.3)
                            plt.tight_layout()
                            plt.savefig(save_path, dpi=150, bbox_inches='tight')
                            plt.close()
                            if verbose:
                                print(f"  [Plot] Training history disimpan ke: {save_path}")
                        except Exception:
                            pass  # Plot generation non-critical, skip if fails

                        if verbose:
                            print(f"\n  Best Val F1: {best_val_f1:.4f}")
                            print(f"  Best Val Acc: {best_val_acc:.4f}")
                            print(f"  Training Time: {elapsed:.1f}s")

                    except Exception as e:
                        print(f"  [ERROR] Training gagal: {e}")
                        results[config_name] = {
                            'error': str(e),
                            'config': config,
                        }

    # Save results
    serializable_results = {}
    for k, v in results.items():
        try:
            serializable_results[k] = v
        except Exception:
            serializable_results[k] = str(v)

    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, default=str)

    print(f"\n[Train] Semua variasi selesai. Hasil disimpan ke: {results_path}")

    return results


# ============================================================================
# Training Standalone untuk Satu Konfigurasi
# ============================================================================

def train_single_cnn(data_dir, arch_type='conv2d',
                     num_conv_layers=3, num_filters=64,
                     kernel_size=(3, 3), pooling_type='max',
                     epochs=30, batch_size=32,
                     model_name=None,
                     weights_dir='weights/cnn',
                     verbose=1):
    """
    Latih SATU konfigurasi CNN.

    Args:
        data_dir (str): path ke dataset Intel
        arch_type (str): 'conv2d' atau 'locallyconnected'
        num_conv_layers (int): jumlah conv layers
        num_filters (int): filter base
        kernel_size (tuple): ukuran kernel
        pooling_type (str): 'max' atau 'average'
        epochs (int): jumlah epoch
        batch_size (int): ukuran batch
        model_name (str): nama file bobot (auto-generate jika None)
        weights_dir (str): direktori penyimpanan
        verbose (int): verbose level
    Returns:
        tuple: (model, history, config)
    """
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))
    from .model_keras import build_cnn_factory, count_parameters

    if model_name is None:
        model_name = (
            f"{arch_type}_L{num_conv_layers}_F{num_filters}_"
            f"K{kernel_size[0]}_{pooling_type}"
        )

    os.makedirs(weights_dir, exist_ok=True)

    # Build model
    print(f"[Train] Membangun model: {model_name}")
    model, config = build_cnn_factory(
        arch_type=arch_type,
        num_conv_layers=num_conv_layers,
        num_filters=num_filters,
        kernel_size=kernel_size,
        pooling_type=pooling_type,
        num_classes=6
    )

    params = count_parameters(model)
    print(f"  Parameters: {params['total']:,}")

    # Load data
    print(f"[Train] Memuat dataset dari: {data_dir}")
    preprocessor = IntelImagePreprocessor(data_dir, target_size=(150, 150))
    preprocessor.load_data()

    # Create generators
    train_gen = create_numpy_generator(
        preprocessor.train_paths, preprocessor.train_labels,
        batch_size=batch_size, shuffle=True
    )
    val_gen = create_numpy_generator(
        preprocessor.val_paths, preprocessor.val_labels,
        batch_size=batch_size, shuffle=False
    )

    steps_per_epoch = max(1, len(preprocessor.train_paths) // batch_size)
    val_steps = max(1, len(preprocessor.val_paths) // batch_size)

    # Callbacks
    callbacks = get_callbacks(
        model_name, weights_dir=weights_dir,
        monitor='val_f1', patience=7
    )

    f1_cb = MacroF1Callback(
        val_data=(preprocessor.val_paths, preprocessor.val_labels),
        batch_size=batch_size,
        num_classes=6,
        steps=val_steps
    )
    f1_cb.set_model(model)
    callbacks.append(f1_cb)

    # Train
    print(f"[Train] Mulai training: epochs={epochs}, batch_size={batch_size}")
    start_time = time.time()

    history = train_cnn(
        model, train_gen, val_gen,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        steps_per_epoch=steps_per_epoch,
        val_steps=val_steps,
        verbose=verbose
    )

    elapsed = time.time() - start_time
    print(f"\n[Train] Selesai dalam {elapsed:.1f}s")

    return model, history, config


if __name__ == '__main__':
    print("[Test] CNN Training Loop...")

    print("\n[Test] Selesai.")
    print("\nUsage:")
    print("  from keras.train import train_single_cnn, train_with_variations")
    print("  model, history, config = train_single_cnn(data_dir, arch_type='conv2d')")
    print("  results = train_with_variations(data_dir, epochs=20)")
