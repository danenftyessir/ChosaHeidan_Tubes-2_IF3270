"""
Batch Inference untuk CNN dari nol.
Mendukung inference dengan berbagai batch_size, chunk processing,
dan mode evaluasi (train vs inference).
"""

import numpy as np


def batch_inference(model, X, batch_size=None, verbose=True):
    """
    Lakukan inference untuk seluruh dataset dengan batching.

    Membagi input menjadi chunk sesuai batch_size, forward pass per chunk,
    lalu concatenate hasilnya.

    Args:
        model: instance CNNScratch.
        X: array numpy, bentuk (N, H, W, C) atau (H, W, C).
        batch_size: ukuran batch. Jika None, gunakan model.batch_size.
        verbose: cetak progress.
    Returns:
        probs: array numpy, bentuk (N, num_classes), distribusi probabilitas.
    """
    if X.ndim == 3:
        X = X[np.newaxis, ...]

    N = X.shape[0]

    if batch_size is None:
        batch_size = getattr(model, 'batch_size', 32)

    if verbose:
        print(f"[Batch Inference] Total {N} sampel, batch_size={batch_size}, "
              f"{(N + batch_size - 1) // batch_size} batches")

    all_probs = []
    num_batches = (N + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, N)
        batch = X[start:end]

        probs = model.forward(batch)
        all_probs.append(probs)

        if verbose and (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx + 1}/{num_batches} selesai "
                  f"(samples {start}-{end-1})")

    return np.concatenate(all_probs, axis=0)


def predict_class(model, X, batch_size=None):
    """
    Prediksi kelas (tanpa probabilitas).

    Args:
        model: instance CNNScratch.
        X: array numpy, bentuk (N, H, W, C).
        batch_size: ukuran batch.
    Returns:
        predictions: array numpy, bentuk (N,), kelas integer.
    """
    probs = batch_inference(model, X, batch_size=batch_size, verbose=False)
    return np.argmax(probs, axis=1)


def predict_top_k(model, X, k=3, batch_size=None):
    """
    Prediksi top-k kelas teratas.

    Args:
        model: instance CNNScratch.
        X: array numpy, bentuk (N, H, W, C).
        k: jumlah top predictions.
        batch_size: ukuran batch.
    Returns:
        top_k: array numpy, bentuk (N, k), indeks kelas top-k.
        top_k_probs: array numpy, bentuk (N, k), probabilitas corresponding.
    """
    probs = batch_inference(model, X, batch_size=batch_size, verbose=False)

    top_k_idx = np.argsort(probs, axis=1)[:, -k:][:, ::-1]
    top_k_probs = np.take_along_axis(probs, top_k_idx, axis=1)

    return top_k_idx, top_k_probs


def evaluate_model(model, X_test, y_test, batch_size=None, verbose=True):
    """
    Evaluasi model pada data test: hitung accuracy dan macro F1-score.

    Args:
        model: instance CNNScratch.
        X_test: array numpy, bentuk (N, H, W, C).
        y_test: array numpy, bentuk (N,) — label integer.
        batch_size: ukuran batch.
        verbose: cetak hasil.
    Returns:
        dict dengan 'accuracy' dan 'macro_f1'.
    """
    predictions = predict_class(model, X_test, batch_size=batch_size,
                                 verbose=verbose)

    accuracy = np.mean(predictions == y_test)
    macro_f1 = macro_f1_score(y_test, predictions)

    if verbose:
        print(f"\n[Evaluasi] Accuracy: {accuracy:.4f}")
        print(f"[Evaluasi] Macro F1-Score: {macro_f1:.4f}")

    return {'accuracy': accuracy, 'macro_f1': macro_f1}


def macro_f1_score(y_true, y_pred, num_classes=None):
    """
    Hitung Macro F1-Score.

    F1 = 2 * (precision * recall) / (precision + recall)
    Macro = rata-rata F1 per kelas.

    Args:
        y_true: array label integer ground truth.
        y_pred: array label integer prediksi.
        num_classes: jumlah kelas. Jika None, di-infer dari max(y_true).
    Returns:
        macro_f1: float.
    """
    if num_classes is None:
        num_classes = max(int(y_true.max()), int(y_pred.max())) + 1

    f1_scores = []
    for c in range(num_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)

    return np.mean(f1_scores)


class BatchInferenceRunner:
    """
    Runner untuk batch inference dengan berbagai konfigurasi batch_size.
    Cocok untuk eksperimen pengaruh batch_size terhadap throughput.
    """

    def __init__(self, model):
        self.model = model
        self.results = {}

    def run(self, X, batch_sizes=None, n_runs=3, verbose=True):
        """
        Jalankan batch inference dengan berbagai batch_size.

        Args:
            X: data input, bentuk (N, H, W, C).
            batch_sizes: list dari batch_size yang akan diuji.
                         Jika None, gunakan [1, 4, 8, 16, 32, 64].
            n_runs: jumlah runs per batch_size (untuk rata-rata waktu).
            verbose: cetak progress.
        Returns:
            results: dict dengan hasil per batch_size.
        """
        import time

        if batch_sizes is None:
            batch_sizes = [1, 4, 8, 16, 32, 64]

        results = {}

        for bs in batch_sizes:
            if verbose:
                print(f"\nBatch size = {bs}...")

            times = []
            probs_result = None

            for run in range(n_runs):
                start = time.time()
                probs_result = batch_inference(
                    self.model, X, batch_size=bs, verbose=False
                )
                elapsed = time.time() - start
                times.append(elapsed)

            avg_time = np.mean(times)
            throughput = X.shape[0] / avg_time

            results[bs] = {
                'avg_time': avg_time,
                'throughput': throughput,
                'probs': probs_result,
                'times': times,
            }

            if verbose:
                print(f"  Avg time: {avg_time:.4f}s, "
                      f"Throughput: {throughput:.2f} samples/s")

        self.results = results
        return results

    def compare_results(self, other_runner, X, y_true, batch_sizes=None):
        """
        Bandingkan hasil antara dua runner (misal: scratch vs Keras).

        Args:
            other_runner: BatchInferenceRunner lain.
            X: data input.
            y_true: label ground truth.
            batch_sizes: list batch_size.
        Returns:
            comparison: dict dengan perbandingan accuracy, F1, dan waktu.
        """
        if batch_sizes is None:
            batch_sizes = list(self.results.keys())

        comparison = {}

        for bs in batch_sizes:
            scratch_probs = self.results.get(bs, {}).get('probs')
            keras_probs = other_runner.results.get(bs, {}).get('probs')

            if scratch_probs is None or keras_probs is None:
                continue

            scratch_preds = np.argmax(scratch_probs, axis=1)
            keras_preds = np.argmax(keras_probs, axis=1)

            scratch_acc = np.mean(scratch_preds == y_true)
            keras_acc = np.mean(keras_preds == y_true)

            scratch_f1 = macro_f1_score(y_true, scratch_preds)
            keras_f1 = macro_f1_score(y_true, keras_preds)

            comparison[bs] = {
                'scratch': {'accuracy': scratch_acc, 'f1': scratch_f1},
                'keras': {'accuracy': keras_acc, 'f1': keras_f1},
                'diff_acc': scratch_acc - keras_acc,
                'diff_f1': scratch_f1 - keras_f1,
            }

        return comparison


class InferenceMode:
    """
    Context manager untuk mode inference.
    Men-disable gradient tracking untuk efisiensi memory dan speed.
    """

    def __init__(self):
        self.previous_state = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


def pad_to_batch_size(X, batch_size):
    """
    Tambahkan padding ke X agar jumlah sampel habis dibagi batch_size.
    Menambahkan sampel dummy (nol) yang akan di-mask saat evaluasi.

    Args:
        X: array numpy, bentuk (N, ...).
        batch_size: ukuran batch target.
    Returns:
        X_padded: array numpy dengan jumlah sampel yang habis dibagi batch_size.
        n_pad: jumlah padding yang ditambahkan.
    """
    N = X.shape[0]
    remainder = N % batch_size
    if remainder == 0:
        return X, 0

    n_pad = batch_size - remainder
    pad_shape = (n_pad, *X.shape[1:])
    X_padded = np.concatenate([X, np.zeros(pad_shape, dtype=X.dtype)], axis=0)
    return X_padded, n_pad


def remove_padding(probs, n_pad):
    """Hapus padding dari hasil inference."""
    if n_pad == 0:
        return probs
    return probs[:-n_pad]