"""
Evaluasi CNN pada Intel Image Classification.
Menggunakan macro F1-score, confusion matrix, dan classification report.
Mendukung perbandingan Conv2D (shared) vs LocallyConnected2D (non-shared).
"""

import os
import sys
import json
import numpy as np
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))
from intel_preprocess import IntelImagePreprocessor, INTEL_CLASSES


# ============================================================================
# Macro F1 & Confusion Matrix (sama dengan train.py)
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


def compute_f1_from_cm(cm, class_names=None):
    """
    Hitung precision, recall, F1-score dari confusion matrix.

    Args:
        cm: confusion matrix (num_classes, num_classes)
        class_names (list): nama kelas untuk report
    Returns:
        dict: metrics lengkap
    """
    num_classes = cm.shape[0]
    results = {
        'per_class': {},
        'macro': {},
    }

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp

        total_tp += tp
        total_fp += fp
        total_fn += fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        name = class_names[i] if class_names else f'class_{i}'

        results['per_class'][name] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'support': int(cm[i, :].sum()),
        }

    all_precision = [v['precision'] for v in results['per_class'].values()]
    all_recall = [v['recall'] for v in results['per_class'].values()]
    all_f1 = [v['f1_score'] for v in results['per_class'].values()]

    results['macro'] = {
        'precision': float(np.mean(all_precision)),
        'recall': float(np.mean(all_recall)),
        'f1_score': float(np.mean(all_f1)),
        'accuracy': total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0,
        'total_support': int(total_tp + total_fn),
    }

    return results


def print_classification_report(results, class_names=None):
    """Cetak classification report yang bagus."""
    print("=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)

    # Header
    print(f"{'Class':<15} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>8}")
    print("-" * 70)

    for name, metrics in results['per_class'].items():
        print(f"{name:<15} "
              f"{metrics['precision']:>10.4f} "
              f"{metrics['recall']:>10.4f} "
              f"{metrics['f1_score']:>10.4f} "
              f"{metrics['support']:>8}")

    print("-" * 70)
    macro = results['macro']
    print(f"{'Macro Avg':<15} "
          f"{macro['precision']:>10.4f} "
          f"{macro['recall']:>10.4f} "
          f"{macro['f1_score']:>10.4f} "
          f"{macro['total_support']:>8}")
    print(f"{'Accuracy':<15} {'':10}{'':10}{macro['accuracy']:>10.4f}")
    print("=" * 70)


# ============================================================================
# Data Loading untuk Evaluasi
# ============================================================================

def create_numpy_generator(image_paths, labels, batch_size=32, shuffle=False,
                           target_size=(150, 150)):
    """
    Generator numpy untuk evaluasi (sama dengan train.py).

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


def load_model_weights(model, weights_path):
    """
    Load bobot ke model Keras.

    Args:
        model: Keras Model
        weights_path (str): path ke file .h5
    Returns:
        model: model dengan bobot yang dimuat
    """
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"File bobot tidak ditemukan: {weights_path}")

    model.load_weights(weights_path)
    print(f"Bobot berhasil dimuat dari: {weights_path}")
    return model


# ============================================================================
# Evaluasi Tunggal
# ============================================================================

def evaluate_cnn(model, image_paths, labels, batch_size=32,
                 class_names=None, verbose=True):
    """
    Evaluasi CNN pada dataset.

    Args:
        model: Keras Model yang sudah di-load bobotnya
        image_paths (list): list path ke gambar
        labels (list): list label integers (N,)
        batch_size (int): ukuran batch
        class_names (list): nama kelas untuk report
        verbose (bool): cetak hasil
    Returns:
        dict: hasil evaluasi lengkap
    """
    if class_names is None:
        class_names = INTEL_CLASSES

    num_classes = len(class_names)

    # Collect predictions
    all_preds = []
    all_true = []

    if verbose:
        print(f"[Evaluate] Memproses {len(image_paths)} gambar...")

    gen = create_numpy_generator(image_paths, labels, batch_size=batch_size)

    for batch_idx, (X_batch, y_batch) in enumerate(gen):
        preds = model.predict(X_batch, verbose=0)
        pred_labels = np.argmax(preds, axis=1)

        all_preds.extend(pred_labels)
        all_true.extend(y_batch)

        if verbose and (batch_idx + 1) % 20 == 0:
            print(f"  Batch {batch_idx + 1} selesai")

    all_preds = np.array(all_preds)
    all_true = np.array(all_true)

    # Compute metrics
    cm = compute_confusion_matrix(all_true, all_preds, num_classes)
    results = compute_f1_from_cm(cm, class_names=class_names)

    # Overall accuracy
    accuracy = np.mean(all_preds == all_true)

    results['accuracy'] = accuracy
    results['num_samples'] = len(all_true)
    results['confusion_matrix'] = cm.tolist()

    if verbose:
        print(f"\n[Evaluate] Accuracy: {accuracy:.4f}")
        print_classification_report(results, class_names=class_names)

    return results


def evaluate_cnn_keras(model, data_dir, split='test', batch_size=32,
                      class_names=None, verbose=True):
    """
    Evaluasi CNN langsung dari direktori dataset (tanpa manual path loading).

    Args:
        model: Keras Model
        data_dir (str): path ke root dataset intel-ic
        split (str): 'train', 'val', atau 'test'
        batch_size (int): ukuran batch
        class_names (list): nama kelas
        verbose (bool): cetak progress
    Returns:
        dict: hasil evaluasi
    """
    preprocessor = IntelImagePreprocessor(data_dir, target_size=(150, 150))
    preprocessor.load_data()

    if split == 'train':
        paths, labels = preprocessor.train_paths, preprocessor.train_labels
    elif split == 'val':
        paths, labels = preprocessor.val_paths, preprocessor.val_labels
    elif split == 'test':
        paths, labels = preprocessor.test_paths, preprocessor.test_labels
    else:
        raise ValueError(f"Split tidak dikenal: {split}")

    return evaluate_cnn(
        model, paths, labels,
        batch_size=batch_size,
        class_names=class_names or INTEL_CLASSES,
        verbose=verbose
    )


# ============================================================================
# Evaluasi dengan Bobot File
# ============================================================================

def evaluate_from_weights(model, weights_path, data_dir, split='test',
                          batch_size=32, class_names=None, verbose=True):
    """
    Load bobot lalu evaluasi model.

    Args:
        model: Keras Model architecture
        weights_path (str): path ke file .h5
        data_dir (str): path ke dataset Intel
        split (str): 'train', 'val', atau 'test'
        batch_size (int): ukuran batch
        class_names (list): nama kelas
        verbose (bool): cetak hasil
    Returns:
        dict: hasil evaluasi
    """
    model = load_model_weights(model, weights_path)

    results = evaluate_cnn_keras(
        model, data_dir, split=split,
        batch_size=batch_size,
        class_names=class_names,
        verbose=verbose
    )

    results['weights_path'] = weights_path
    return results


# ============================================================================
# Perbandingan Conv2D vs LocallyConnected2D
# ============================================================================

def compare_shared_vs_non_shared(conv2d_weights, local_weights,
                                  data_dir, split='test',
                                  batch_size=32, class_names=None,
                                  verbose=True):
    """
    Bandingkan performa Conv2D (shared) vs LocallyConnected2D (non-shared).

    Args:
        conv2d_weights (str): path ke bobot Conv2D (.h5)
        local_weights (str): path ke bobot LocallyConnected2D (.h5)
        data_dir (str): path ke dataset Intel
        split (str): split yang digunakan untuk evaluasi
        batch_size (int): ukuran batch
        class_names (list): nama kelas
        verbose (bool): cetak hasil
    Returns:
        dict: {conv2d_results, locallyconnected_results, comparison}
    """
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))
    from .model_keras import build_cnn_conv2d, build_cnn_locallyconnected

    if class_names is None:
        class_names = INTEL_CLASSES

    results = {}

    # Conv2D
    print("\n" + "=" * 60)
    print("[Compare] Evaluating Conv2D (Shared Parameters)...")
    print("=" * 60)

    model_conv, _ = build_cnn_conv2d(num_classes=len(class_names))
    results['conv2d'] = evaluate_from_weights(
        model_conv, conv2d_weights, data_dir, split=split,
        batch_size=batch_size, class_names=class_names, verbose=verbose
    )

    # LocallyConnected2D
    print("\n" + "=" * 60)
    print("[Compare] Evaluating LocallyConnected2D (Non-Shared Parameters)...")
    print("=" * 60)

    model_local, _ = build_cnn_locallyconnected(num_classes=len(class_names))
    results['locallyconnected'] = evaluate_from_weights(
        model_local, local_weights, data_dir, split=split,
        batch_size=batch_size, class_names=class_names, verbose=verbose
    )

    # Comparison summary
    conv2d_f1 = results['conv2d']['macro']['f1_score']
    local_f1 = results['locallyconnected']['macro']['f1_score']

    conv2d_acc = results['conv2d']['accuracy']
    local_acc = results['locallyconnected']['accuracy']

    # Hitung jumlah parameter untuk kedua arsitektur
    model_conv_count, _ = build_cnn_conv2d(num_classes=len(class_names))
    model_local_count, _ = build_cnn_locallyconnected(num_classes=len(class_names))

    from .model_keras import count_parameters
    conv2d_params = count_parameters(model_conv_count)['total']
    local_params = count_parameters(model_local_count)['total']

    comparison = {
        'conv2d': {
            'macro_f1': conv2d_f1,
            'accuracy': conv2d_acc,
            'total_params': conv2d_params,
        },
        'locallyconnected': {
            'macro_f1': local_f1,
            'accuracy': local_acc,
            'total_params': local_params,
        },
        'winner_f1': 'conv2d' if conv2d_f1 > local_f1 else 'locallyconnected',
        'winner_acc': 'conv2d' if conv2d_acc > local_acc else 'locallyconnected',
        'f1_diff': abs(conv2d_f1 - local_f1),
        'acc_diff': abs(conv2d_acc - local_acc),
        'param_ratio': local_params / conv2d_params if conv2d_params > 0 else 0,
    }

    results['comparison'] = comparison

    if verbose:
        print("\n" + "=" * 70)
        print("COMPARISON SUMMARY: Conv2D vs LocallyConnected2D")
        print("=" * 70)
        print(f"{'Metric':<20} {'Conv2D (Shared)':>20} {'LocalConv (Non-Shared)':>22} {'Difference':>12}")
        print("-" * 70)
        print(f"{'Macro F1-Score':<20} {conv2d_f1:>20.4f} {local_f1:>22.4f} {comparison['f1_diff']:>12.4f}")
        print(f"{'Accuracy':<20} {conv2d_acc:>20.4f} {local_acc:>22.4f} {comparison['acc_diff']:>12.4f}")
        print(f"{'Total Parameters':<20} {conv2d_params:>20,} {local_params:>22,} {comparison['param_ratio']:>11.1f}x")
        print("=" * 70)
        print(f"Best F1:     {comparison['winner_f1']}")
        print(f"Best Acc:    {comparison['winner_acc']}")
        print(f"Param Ratio: LocalConv has {comparison['param_ratio']:.1f}x more parameters than Conv2D")

    return results


def compare_multiple_configs(results_path, data_dir, split='test',
                              class_names=None, verbose=True):
    """
    Bandingkan hasil dari file JSON training variations.

    Args:
        results_path (str): path ke file hasil training (cnn_variations.json)
        data_dir (str): path ke dataset Intel
        split (str): split yang dipakai untuk evaluasi
        class_names (list): nama kelas
        verbose (bool): cetak hasil
    Returns:
        dict: ranking semua konfigurasi berdasarkan macro F1
    """
    with open(results_path, 'r', encoding='utf-8') as f:
        training_results = json.load(f)

    # Load best weights untuk setiap konfigurasi
    best_configs = {}

    for config_name, data in training_results.items():
        if 'error' in data:
            continue

        best_f1 = data.get('best_val_f1', 0.0)

        # Estimasi parameter count
        params = data.get('parameters', {})
        param_count = params.get('total', 0)

        best_configs[config_name] = {
            'best_val_f1': best_f1,
            'best_val_acc': data.get('best_val_accuracy', 0.0),
            'total_params': param_count,
            'elapsed_seconds': data.get('elapsed_seconds', 0),
            'config': data.get('config', {}),
        }

    # Sort by val F1
    sorted_configs = sorted(
        best_configs.items(),
        key=lambda x: x[1]['best_val_f1'],
        reverse=True
    )

    ranking = {name: {'rank': i + 1, **data}
               for i, (name, data) in enumerate(sorted_configs)}

    if verbose:
        print("\n" + "=" * 70)
        print("RANKING: Semua Konfigurasi berdasarkan Macro F1 (Validation)")
        print("=" * 70)
        print(f"{'Rank':<4} {'Config Name':<45} {'F1':>8} {'Acc':>8} {'Params':>12}")
        print("-" * 70)
        for i, (name, data) in enumerate(sorted_configs[:10]):
            print(f"{i + 1:<4} {name:<45} {data['best_val_f1']:>8.4f} "
                  f"{data['best_val_acc']:>8.4f} {data['total_params']:>12,}")
        print("=" * 70)

    return ranking


# ============================================================================
# Inference Speed Benchmark
# ============================================================================

def benchmark_inference_speed(model, image_paths, batch_sizes=[1, 8, 16, 32],
                               warmup=3, num_runs=10, verbose=True):
    """
    Ukur inference speed model untuk berbagai batch_size.

    Args:
        model: Keras Model
        image_paths (list): list path ke gambar
        batch_sizes (list): list batch_size yang akan diuji
        warmup (int): jumlah warmup runs
        num_runs (int): jumlah runs untuk averaging
        verbose (bool): cetak hasil
    Returns:
        dict: timing results per batch_size
    """
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))
    try:
        from utils import load_image
    except ImportError:
        from cnn.utils.utils import load_image

    # Load sample images
    sample_images = []
    for path in image_paths[:max(batch_sizes)]:
        try:
            img = load_image(path, target_size=(150, 150))
            img = img.astype(np.float32) / 255.0
            sample_images.append(img)
        except Exception:
            continue

    if not sample_images:
        print("[Warning] Tidak ada gambar yang berhasil dimuat.")
        return {}

    sample_batch = np.stack(sample_images, axis=0)

    results = {}

    for bs in batch_sizes:
        if verbose:
            print(f"\n[Timing] batch_size={bs}...")

        # Warmup
        for w in range(warmup):
            _ = model.predict(sample_batch[:min(bs, len(sample_images))], verbose=0)

        # Timing runs
        timings = []
        for r in range(num_runs):
            batch_input = sample_batch[:min(bs, len(sample_images))]
            start = time.time()
            _ = model.predict(batch_input, verbose=0)
            elapsed = time.time() - start
            timings.append(elapsed)

        import statistics
        mean_time = statistics.mean(timings)
        std_time = statistics.stdev(timings) if len(timings) > 1 else 0

        throughput = bs / mean_time if mean_time > 0 else 0
        latency = mean_time / bs if bs > 0 else 0

        results[bs] = {
            'mean_ms': mean_time * 1000,
            'std_ms': std_time * 1000,
            'images_per_sec': throughput,
            'sec_per_image': latency,
        }

        if verbose:
            print(f"  Mean: {mean_time * 1000:.2f} ms")
            print(f"  Std:  {std_time * 1000:.2f} ms")
            print(f"  Throughput: {throughput:.1f} gambar/detik")
            print(f"  Latency: {latency * 1000:.1f} ms/gambar")

    return results


# ============================================================================
# Save Hasil Evaluasi
# ============================================================================

def save_evaluation_results(results, save_path):
    """
    Simpan hasil evaluasi ke file JSON.

    Args:
        results (dict): hasil dari evaluate_cnn atau compare_shared_vs_non_shared
        save_path (str): path untuk menyimpan hasil
    """
    def make_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [make_serializable(v) for v in obj]
        return obj

    serializable = make_serializable(results)

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)

    print(f"Hasil evaluasi disimpan ke: {save_path}")


def print_confusion_matrix(cm, class_names=None):
    """
    Cetak confusion matrix yang bagus.

    Args:
        cm: confusion matrix (num_classes, num_classes)
        class_names (list): nama kelas
    """
    if class_names is None:
        class_names = [f'C{i}' for i in range(cm.shape[0])]

    print("\nConfusion Matrix:")
    print("=" * 70)

    # Header
    header = f"{'':>15}"
    for name in class_names:
        header += f" {name[:10]:>10}"
    print(header)
    print("-" * 70)

    # Rows
    for i, name in enumerate(class_names):
        row = f"{name[:15]:>15}"
        for j in range(cm.shape[1]):
            row += f" {cm[i, j]:>10}"
        print(row)

    print("=" * 70)
    print("(Baris = Ground Truth, Kolom = Prediksi)")


# ============================================================================
# Part 4 Pipeline: Scratch Forward Pass vs Keras
# ============================================================================

def run_part4_evaluation(results_json_path, weights_dir, data_dir, split='test',
                          batch_size=32, class_names=None, save_dir='results/cnn',
                          verbose=True):
    """
    Pipeline Part 4: baca JSON → load best model weights →
    scratch forward pass → compute macro F1 vs Keras.

    Args:
        results_json_path (str): path ke file hasil training (cnn_variations.json)
        weights_dir (str): direktori tempat bobot .h5 disimpan
        data_dir (str): path ke dataset Intel
        split (str): 'train', 'val', atau 'test'
        batch_size (int): ukuran batch
        class_names (list): nama kelas
        save_dir (str): direktori untuk menyimpan hasil
        verbose (bool): cetak progress
    Returns:
        dict: hasil comparison scratch vs keras
    """
    from .model_keras import build_cnn_conv2d, build_cnn_locallyconnected

    if class_names is None:
        class_names = INTEL_CLASSES

    os.makedirs(save_dir, exist_ok=True)

    # 1. Baca JSON, temukan best config
    with open(results_json_path, 'r', encoding='utf-8') as f:
        training_results = json.load(f)

    best_name = None
    best_f1 = 0.0
    for config_name, data in training_results.items():
        if 'error' in data:
            continue
        f1 = data.get('best_val_f1', 0.0)
        if f1 > best_f1:
            best_f1 = f1
            best_name = config_name

    if best_name is None:
        raise ValueError(f"Tidak ada konfigurasi valid di {results_json_path}")

    if verbose:
        print(f"\n[Part4] Best config: {best_name} (val_f1={best_f1:.4f})")

    best_data = training_results[best_name]
    config = best_data.get('config', {})
    arch_type = config.get('type', 'conv2d')

    # 2. Load test data
    preprocessor = IntelImagePreprocessor(data_dir, target_size=(150, 150))
    preprocessor.load_data()

    if split == 'train':
        paths, labels = preprocessor.train_paths, preprocessor.train_labels
    elif split == 'val':
        paths, labels = preprocessor.val_paths, preprocessor.val_labels
    elif split == 'test':
        paths, labels = preprocessor.test_paths, preprocessor.test_labels
    else:
        raise ValueError(f"Split tidak dikenal: {split}")

    num_classes = len(class_names)

    # Helper: load image batch → numpy array
    def load_image_batch(img_paths, target_size=(150, 150)):
        from cnn.utils.utils import load_image
        images = []
        for p in img_paths:
            try:
                img = load_image(p, target_size=target_size)
                images.append(img)
            except Exception:
                continue
        if not images:
            return np.array([])
        return np.array(images)

    # 3. Load Keras model + bobot
    if verbose:
        print(f"\n[Part4] Loading Keras model ({arch_type})...")

    if arch_type in ('conv2d', 'vgg') or 'conv2d' in arch_type:
        keras_model, _ = build_cnn_conv2d(
            num_classes=num_classes,
            num_conv_layers=config.get('num_conv_layers', 3),
            num_filters=config.get('num_filters', 64),
            kernel_size=tuple(config.get('kernel_size', (3, 3))),
            pooling_type=config.get('pooling_type', 'max')
        )
    else:
        keras_model, _ = build_cnn_locallyconnected(
            num_classes=num_classes,
            num_conv_layers=config.get('num_conv_layers', 3),
            num_filters=config.get('num_filters', 64),
            kernel_size=tuple(config.get('kernel_size', (3, 3))),
            pooling_type=config.get('pooling_type', 'max')
        )

    weight_file = os.path.join(weights_dir, f'{best_name}.h5')
    keras_model = load_model_weights(keras_model, weight_file)

    # 4. Keras predictions
    if verbose:
        print(f"\n[Part4] Running Keras inference ({len(paths)} images)...")

    keras_preds = []
    keras_probs = []

    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i:i+batch_size]
        batch_images = load_image_batch(batch_paths)
        if len(batch_images) == 0:
            continue

        # Keras expects (batch, H, W, C) float32 [0,255] or preprocessed
        # Convert [0,1] to [0,255]
        batch_for_keras = (batch_images * 255).astype(np.float32)
        preds = keras_model.predict(batch_for_keras, verbose=0)
        keras_probs.append(preds)
        keras_preds.extend(np.argmax(preds, axis=1).tolist())

    keras_probs_all = np.concatenate(keras_probs, axis=0)
    keras_labels_arr = np.array(labels[:len(keras_preds)])

    # 5. Scratch model: build + load weights + forward
    if verbose:
        print(f"\n[Part4] Building CNNScratch model ({arch_type})...")

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scratch'))
    from conv2d import Conv2D
    from pooling import MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
    from flatten import Flatten
    from model_scratch import CNNScratch
    import sys as _sys
    _sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))
    from dense import Dense

    num_conv = config.get('num_conv_layers', 3)
    num_filt = config.get('num_filters', 64)
    k_size = tuple(config.get('kernel_size', (3, 3)))
    pool_type = config.get('pooling_type', 'max')
    is_local = (arch_type in ('locallyconnected',) or 'locallyconnected' in arch_type)

    from locally_connected2d import LocallyConnected2D

    scratch_layers = []
    base_filters = num_filt

    for i in range(num_conv):
        filters_i = base_filters * (2 ** i)
        if is_local:
            layer = LocallyConnected2D(
                filters=filters_i, kernel_size=k_size,
                strides=(1, 1), padding='valid', activation='relu'
            )
        else:
            layer = Conv2D(
                filters=filters_i, kernel_size=k_size,
                strides=(1, 1), padding='same', activation='relu'
            )
        scratch_layers.append(layer)

        # Pooling layer
        if pool_type == 'max' or pool_type == 'average':
            PoolCls = MaxPooling2D if pool_type == 'max' else AveragePooling2D
            scratch_layers.append(PoolCls(pool_size=(2, 2)))

    scratch_layers.append(GlobalAveragePooling2D())
    scratch_layers.append(Flatten())
    scratch_layers.append(Dense(input_dim=0, units=256, activation='relu'))
    scratch_layers.append(Dense(input_dim=0, units=num_classes, activation='softmax'))

    scratch_model = CNNScratch(
        layers=scratch_layers,
        batch_size=batch_size,
        num_classes=num_classes,
        input_shape=(150, 150, 3)
    )
    scratch_model.build()

    # Load weights
    if verbose:
        print(f"[Part4] Loading weights from {weight_file}...")
    scratch_model.load_weights_from_h5(weight_file)

    # 6. Scratch predictions
    if verbose:
        print(f"\n[Part4] Running scratch forward pass...")

    scratch_probs = scratch_model.predict(
        load_image_batch(paths[:len(keras_preds)]).astype(np.float64),
        batch_size=batch_size
    )
    scratch_preds = np.argmax(scratch_probs, axis=1)

    # 7. Compute macro F1 for both
    from metrics import macro_f1_score

    keras_f1 = macro_f1_score(keras_labels_arr, np.array(keras_preds), num_classes)
    scratch_f1 = macro_f1_score(keras_labels_arr, scratch_preds, num_classes)

    keras_acc = np.mean(np.array(keras_preds) == keras_labels_arr)
    scratch_acc = np.mean(scratch_preds == keras_labels_arr[:len(scratch_preds)])

    # 8. Print & save results
    if verbose:
        print("\n" + "=" * 70)
        print("PART 4: KERAS vs FROM-SCRATCH COMPARISON")
        print("=" * 70)
        print(f"{'Metric':<20} {'Keras':>20} {'From-Scratch':>18} {'Diff':>12}")
        print("-" * 70)
        print(f"{'Macro F1-Score':<20} {keras_f1:>20.4f} {scratch_f1:>18.4f} {abs(keras_f1 - scratch_f1):>12.4f}")
        print(f"{'Accuracy':<20} {keras_acc:>20.4f} {scratch_acc:>18.4f} {abs(keras_acc - scratch_acc):>12.4f}")
        print("=" * 70)
        print(f"Best Config: {best_name}")
        print(f"Architecture: {arch_type}")

    comparison_result = {
        'best_config': best_name,
        'arch_type': arch_type,
        'keras': {'macro_f1': float(keras_f1), 'accuracy': float(keras_acc)},
        'scratch': {'macro_f1': float(scratch_f1), 'accuracy': float(scratch_acc)},
        'f1_diff': abs(float(keras_f1) - float(scratch_f1)),
        'acc_diff': abs(float(keras_acc) - float(scratch_acc)),
        'winner_f1': 'keras' if keras_f1 > scratch_f1 else 'scratch',
    }

    save_path = os.path.join(save_dir, 'part4_comparison.json')
    save_evaluation_results(comparison_result, save_path)
    if verbose:
        print(f"\n[Part4] Results saved to: {save_path}")

    return comparison_result


if __name__ == '__main__':
    print("[Test] CNN Evaluation Module...")

    print("\n[Test] Contoh penggunaan:")
    print("  from keras.evaluate import evaluate_from_weights, compare_shared_vs_non_shared")
    print("")
    print("  # Evaluasi single model")
    print("  model, _ = build_cnn_conv2d(num_classes=6)")
    print("  results = evaluate_from_weights(model, 'weights/cnn/conv2d.h5', data_dir)")
    print("")
    print("  # Bandingkan Conv2D vs LocallyConnected2D")
    print("  results = compare_shared_vs_non_shared(")
    print("      'weights/cnn/conv2d.h5', 'weights/cnn/locallyconnected.h5', data_dir)")

    print("\n[Test] Selesai.")