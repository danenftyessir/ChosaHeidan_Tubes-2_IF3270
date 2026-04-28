"""
Batch Inference untuk LSTM dari nol.
Mendukung inference batch dengan berbagai ukuran batch_size.
Termasuk parallel batch decoding dan BLEU evaluation per batch.
"""

import numpy as np
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))
from caption_preprocess import END_IDX, START_IDX
from metrics import corpus_bleu_score, evaluate_batch


# ============================================================================
# Batch Inference Utilities
# ============================================================================

def batch_inference_lstm(model, cnn_features, token_seqs, batch_size=32,
                         verbose=True):
    """
    Batch inference: forward pass untuk batch gambar sekaligus.

    Args:
        model: LSTMScratch atau LSTMInitInject instance
        cnn_features: (N, feature_dim)
        token_seqs: list of token sequences untuk teacher forcing
        batch_size (int): ukuran batch
        verbose (bool): cetak progress
    Returns:
        np.ndarray: output probabilities shape (N, vocab_size)
    """
    N = cnn_features.shape[0]
    vocab_size = model.vocab_size
    all_outputs = []

    num_batches = (N + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, N)

        batch_cnn = cnn_features[start:end]
        batch_tokens = token_seqs[start:end]

        output = model.forward(batch_cnn, batch_tokens, training=False)
        all_outputs.append(output)

        if verbose and (batch_idx + 1) % 20 == 0:
            print(f"  Batch {batch_idx + 1}/{num_batches} selesai "
                  f"({end}/{N} gambar)")

    return np.concatenate(all_outputs, axis=0)


def batch_predict_lstm(model, cnn_features_batch, idx2word, max_length=30,
                       batch_size=32, verbose=True):
    """
    Batch prediction: generate captions untuk batch gambar sekaligus.
    Setiap gambar mendapat greedy decoding secara paralel.

    Args:
        model: LSTMScratch atau LSTMInitInject instance
        cnn_features_batch: (N, feature_dim)
        idx2word: dict, vocabulary
        max_length (int): panjang maksimum caption
        batch_size (int): ukuran batch
        verbose (bool): cetak progress
    Returns:
        list: list caption strings
    """
    N = cnn_features_batch.shape[0]
    captions = []

    num_batches = (N + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, N)

        batch_cnn = cnn_features_batch[start:end]
        batch_captions = model.greedy_decode_batch(batch_cnn, idx2word, max_length)

        for cap in batch_captions:
            captions.append(cap)

        if verbose and (batch_idx + 1) % 20 == 0:
            print(f"  Batch {batch_idx + 1}/{num_batches} selesai "
                  f"({end}/{N} gambar)")

    return captions


def batch_predict_lstm_with_beam(model, cnn_features_batch, idx2word,
                                  max_length=30, k=5, batch_size=32,
                                  verbose=True):
    """
    Batch prediction dengan beam search decoding (k=5).
    Beam search untuk setiap gambar dalam batch dilakukan sekuensial.

    Args:
        model: LSTMScratch instance dengan beam_search capability
        cnn_features_batch: (N, feature_dim)
        idx2word: dict
        max_length (int): panjang maksimum
        k (int): beam width (default 5)
        batch_size (int): ukuran batch per inference
        verbose (bool): cetak progress
    Returns:
        list: list caption strings
    """
    try:
        from .bonus_beam_search import beam_search_lstm
    except ImportError:
        from bonus_beam_search import beam_search_lstm

    N = cnn_features_batch.shape[0]
    captions = []

    num_batches = (N + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, N)
        batch_cnn = cnn_features_batch[start:end]

        for i in range(end - start):
            feat = batch_cnn[i:i+1]
            caption = beam_search_lstm(model, feat, idx2word, k=k, max_length=max_length)
            captions.append(caption)

        if verbose and (batch_idx + 1) % 20 == 0:
            print(f"  Batch {batch_idx + 1}/{num_batches} selesai")

    return captions


# ============================================================================
# Batch BLEU Evaluation
# ============================================================================

def evaluate_batch_bleu_lstm(model, cnn_features_test, gt_captions_test,
                            idx2word, max_length=30, batch_size=32,
                            metrics=('bleu1', 'bleu2', 'bleu3', 'bleu4'),
                            verbose=True):
    """
    Evaluasi BLEU score untuk batch test set (LSTM).

    Args:
        model: LSTMScratch instance
        cnn_features_test: (N, feature_dim)
        gt_captions_test: list of ground truth caption strings (N,)
        idx2word: dict
        max_length (int): panjang maksimum caption
        batch_size (int): ukuran batch
        metrics (tuple): metric yang dihitung
        verbose (bool): cetak progress
    Returns:
        dict: metrics results
    """
    if verbose:
        print(f"[Batch BLEU] Evaluasi {len(cnn_features_test)} gambar, "
              f"batch_size={batch_size}")

    start_time = time.time()

    pred_captions = batch_predict_lstm(
        model, cnn_features_test, idx2word,
        max_length=max_length, batch_size=batch_size,
        verbose=verbose
    )

    elapsed = time.time() - start_time

    results = evaluate_batch(gt_captions_test, pred_captions, metrics=metrics)
    results['elapsed_time'] = elapsed
    results['ms_per_image'] = elapsed / len(cnn_features_test) * 1000

    if verbose:
        print(f"\n  BLEU-1: {results.get('bleu1', 0):.4f}")
        print(f"  BLEU-2: {results.get('bleu2', 0):.4f}")
        print(f"  BLEU-3: {results.get('bleu3', 0):.4f}")
        print(f"  BLEU-4: {results.get('bleu4', 0):.4f}")
        print(f"  Waktu: {elapsed:.1f}s ({results['ms_per_image']:.1f} ms/gambar)")

    return results


def evaluate_batch_bleu_lstm_with_beam(model, cnn_features_test, gt_captions_test,
                                        idx2word, max_length=30, k=5,
                                        batch_size=32, verbose=True):
    """
    Evaluasi BLEU dengan beam search decoding untuk LSTM.

    Args:
        model: LSTMScratch instance
        cnn_features_test: (N, feature_dim)
        gt_captions_test: list of ground truth caption strings
        idx2word: dict
        max_length (int): panjang maksimum
        k (int): beam width (default 5)
        batch_size (int): ukuran batch
        verbose (bool): cetak progress
    Returns:
        dict: metrics results
    """
    if verbose:
        print(f"[Batch BLEU + Beam] Evaluasi {len(cnn_features_test)} gambar, "
              f"k={k}, batch_size={batch_size}")

    start_time = time.time()

    pred_captions = batch_predict_lstm_with_beam(
        model, cnn_features_test, idx2word,
        max_length=max_length, k=k, batch_size=batch_size,
        verbose=verbose
    )

    elapsed = time.time() - start_time

    results = evaluate_batch(gt_captions_test, pred_captions,
                            metrics=('bleu1', 'bleu2', 'bleu3', 'bleu4'))
    results['elapsed_time'] = elapsed
    results['ms_per_image'] = elapsed / len(cnn_features_test) * 1000

    if verbose:
        print(f"\n  BLEU-1: {results.get('bleu1', 0):.4f}")
        print(f"  BLEU-2: {results.get('bleu2', 0):.4f}")
        print(f"  BLEU-3: {results.get('bleu3', 0):.4f}")
        print(f"  BLEU-4: {results.get('bleu4', 0):.4f}")
        print(f"  Waktu: {elapsed:.1f}s ({results['ms_per_image']:.1f} ms/gambar)")

    return results


# ============================================================================
# Batch Comparison Utilities
# ============================================================================

def compare_batch_sizes_lstm(model, cnn_features, gt_captions, idx2word,
                             batch_sizes=(1, 8, 16, 32, 64),
                             max_length=30, n_samples=None,
                             verbose=True):
    """
    Bandingkan throughput dan latency untuk berbagai batch_size (LSTM).

    Args:
        model: LSTMScratch instance
        cnn_features: (N, feature_dim)
        gt_captions: list of ground truth captions
        idx2word: dict
        batch_sizes (tuple): list batch_size yang akan diuji
        max_length (int): panjang maksimum caption
        n_samples (int): jumlah sample untuk test (None = seluruh set)
        verbose (bool): cetak progress
    Returns:
        dict: {batch_size: {'throughput': ..., 'latency': ..., 'bleu4': ...}}
    """
    if n_samples is not None:
        cnn_features = cnn_features[:n_samples]
        gt_captions = gt_captions[:n_samples]

    results = {}

    for bs in batch_sizes:
        if verbose:
            print(f"\n[Batch Size] Testing batch_size={bs}...")

        start_time = time.time()
        pred_captions = batch_predict_lstm(
            model, cnn_features, idx2word,
            max_length=max_length, batch_size=bs, verbose=False
        )
        elapsed = time.time() - start_time

        bleu = corpus_bleu_score(gt_captions, pred_captions, n=4)

        throughput = len(cnn_features) / elapsed
        latency = elapsed / len(cnn_features)

        results[bs] = {
            'throughput': throughput,
            'latency': latency,
            'bleu4': bleu,
            'elapsed': elapsed,
        }

        if verbose:
            print(f"  Throughput: {throughput:.2f} gambar/detik")
            print(f"  Latency: {latency*1000:.1f} ms/gambar")
            print(f"  BLEU-4: {bleu:.4f}")

    return results


def batch_inference_timing_lstm(model, cnn_features, token_seqs, batch_sizes,
                                warmup=3, verbose=True):
    """
    Ukur timing untuk berbagai batch sizes (LSTM).
    Termasuk warmup untuk stabilisasi cache CPU.

    Args:
        model: LSTMScratch instance
        cnn_features: (N, feature_dim)
        token_seqs: list of token sequences
        batch_sizes (list): list batch_size yang akan diuji
        warmup (int): jumlah warmup runs
        verbose (bool): cetak progress
    Returns:
        dict: timing results per batch_size
    """
    results = {}

    for bs in batch_sizes:
        if verbose:
            print(f"\n[Timing] batch_size={bs}...")

        # Warmup
        for w in range(warmup):
            _ = model.forward(
                cnn_features[:bs],
                token_seqs[:bs],
                training=False
            )

        # Timing runs
        timings = []
        num_runs = 10

        for r in range(num_runs):
            start = time.time()
            _ = model.forward(
                cnn_features[:bs],
                token_seqs[:bs],
                training=False
            )
            elapsed = time.time() - start
            timings.append(elapsed)

        import statistics
        mean_time = statistics.mean(timings)
        std_time = statistics.stdev(timings) if len(timings) > 1 else 0

        results[bs] = {
            'mean_ms': mean_time * 1000,
            'std_ms': std_time * 1000,
            'images_per_sec': bs / mean_time,
        }

        if verbose:
            print(f"  Mean: {mean_time*1000:.2f} ms")
            print(f"  Std:  {std_time*1000:.2f} ms")
            print(f"  Throughput: {bs/mean_time:.2f} gambar/detik")

    return results


# ============================================================================
# Batch Helper Classes
# ============================================================================

class BatchInferenceEngineLSTM:
    """
    Wrapper untuk batch inference LSTM dengan caching dan statistik.

    Cache hasil inference agar tidak perlu re-compute untuk data yang sama.
    """

    def __init__(self, model, batch_size=32):
        self.model = model
        self.batch_size = batch_size
        self.cache = {}
        self.stats = {'hits': 0, 'misses': 0}

    def predict(self, cnn_features, idx2word, max_length=30, use_cache=True):
        """
        Predict captions dengan caching.

        Args:
            cnn_features: (N, feature_dim)
            idx2word: dict
            max_length (int): panjang maksimum
            use_cache (bool): gunakan cache jika tersedia
        Returns:
            list: list caption strings
        """
        if use_cache:
            cache_key = hash(cnn_features.tobytes())
            if cache_key in self.cache:
                self.stats['hits'] += 1
                return self.cache[cache_key]
            self.stats['misses'] += 1

        result = batch_predict_lstm(
            self.model, cnn_features, idx2word,
            max_length=max_length, batch_size=self.batch_size,
            verbose=False
        )

        if use_cache:
            self.cache[cache_key] = result

        return result

    def get_stats(self):
        """Return cache statistics."""
        total = self.stats['hits'] + self.stats['misses']
        hit_rate = self.stats['hits'] / total if total > 0 else 0
        return {
            'hits': self.stats['hits'],
            'misses': self.stats['misses'],
            'hit_rate': hit_rate,
        }

    def clear_cache(self):
        """Clear cache."""
        self.cache = {}


if __name__ == '__main__':
    print("[Test] Batch inference utilities untuk LSTM...")

    # Contoh usage
    print("\n[Test] Gunakan batch_predict_lstm() untuk generate captions.")
    print("  from bonus_batch_inference import batch_predict_lstm, evaluate_batch_bleu_lstm")
    print("\n[Test] Contoh:")
    print("  captions = batch_predict_lstm(model, cnn_features_test, idx2word, max_length=30, batch_size=32)")
    print("  results = evaluate_batch_bleu_lstm(model, cnn_features_test, gt_captions, idx2word)")

    print("\n[Test] Selesai.")
