"""
Evaluasi untuk decoder RNN menggunakan Keras.
BLEU-4, METEOR, qualitative analysis, dan perbandingan Keras vs Scratch.
"""

import os
import sys
import json
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))
from metrics import (
    evaluate_batch, corpus_bleu_score, evaluate_single_pair,
    print_metrics, save_metrics_to_file
)
from caption_preprocess import END_IDX, START_IDX, sequence_to_caption


# ============================================================================
# Evaluation Functions
# ============================================================================

def evaluate_model(model, cnn_features, gt_captions, idx2word,
                   max_length=30, batch_size=64, verbose=True):
    """
    Evaluasi model RNN pada dataset dengan greedy decoding.

    Args:
        model: Keras Model instance (pre-inject atau init-inject).
        cnn_features (np.ndarray): CNN features, shape (N, feature_dim).
        gt_captions (list): list ground truth captions (string).
        idx2word (dict): index → word mapping.
        max_length (int): panjang maksimal caption generation.
        batch_size (int): ukuran batch untuk inference.
        verbose (bool): cetak progress.
    Returns:
        dict: metrics (bleu1-4, meteor, dll).
    """
    N = cnn_features.shape[0]
    pred_captions = []

    num_batches = (N + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, N)
        batch_cnn = cnn_features[start:end]

        # Greedy decode per sample
        for i in range(end - start):
            feat = batch_cnn[i:i+1]  # (1, feature_dim)
            pred_seq = greedy_decode_keras(model, feat, max_length)
            caption = sequence_to_caption(pred_seq, idx2word)
            pred_captions.append(caption)

        if verbose and (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx + 1}/{num_batches} selesai")

    # Hitung BLEU scores
    metrics = evaluate_batch(gt_captions, pred_captions, metrics=['bleu1', 'bleu2', 'bleu3', 'bleu4', 'meteor'])

    return metrics, pred_captions


def greedy_decode_keras(model, cnn_feature, max_length=30):
    """
    Greedy decoding untuk model Keras.

    Args:
        model: Keras Model.
        cnn_feature: (1, feature_dim).
        max_length (int): panjang maksimal.
    Returns:
        list: sequence of token indices.
    """
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    vocab_size = model.output_shape[-1]
    tokens = []

    # Prepare input sequence dengan <start> token
    seq = [START_IDX]
    seq_padded = pad_sequences([seq], maxlen=max_length, padding='post')[0]

    for step in range(max_length):
        # Predict
        probs = model.predict([cnn_feature, seq_padded[np.newaxis, :]], verbose=0)[0]

        # Greedy: ambil token dengan probability tertinggi
        next_token = int(np.argmax(probs))

        if next_token == END_IDX or next_token == 0:  # 0 = pad
            break

        tokens.append(next_token)

        # Update sequence untuk step berikutnya
        seq.append(next_token)
        seq_padded = pad_sequences([seq], maxlen=max_length, padding='post')[0]

    return tokens


def beam_search_decode_keras(model, cnn_feature, idx2word, k=5, max_length=30):
    """
    Beam search decoding untuk model Keras.

    Args:
        model: Keras Model.
        cnn_feature: (1, feature_dim).
        idx2word (dict): index → word mapping.
        k (int): beam width.
        max_length (int): panjang maksimal.
    Returns:
        str: generated caption.
    """
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    beams = [([START_IDX], 0.0)]  # (token_sequence, log_probability)

    for step in range(max_length):
        all_candidates = []

        for seq, score in beams:
            if seq[-1] == END_IDX or seq[-1] == 0:
                all_candidates.append((seq, score))
                continue

            # Prepare input
            seq_padded = pad_sequences([seq], maxlen=max_length, padding='post')[0]

            # Predict
            probs = model.predict([cnn_feature, seq_padded[np.newaxis, :]], verbose=0)[0]

            # Top-k next tokens
            top_k_idx = np.argsort(probs)[-k:]
            for token in top_k_idx:
                new_seq = seq + [int(token)]
                new_score = score + np.log(probs[token] + 1e-10)
                all_candidates.append((new_seq, new_score))

        # Select top-k
        all_candidates.sort(key=lambda x: x[1], reverse=True)
        beams = all_candidates[:k]

        if all(seq[-1] == END_IDX or seq[-1] == 0 for seq, _ in beams):
            break

    best_seq = beams[0][0]
    return sequence_to_caption(best_seq, idx2word)


def qualitative_analysis(images, gt_captions, pred_captions, save_path=None, n=10):
    """
    Tampilkan qualitative examples: gambar + GT caption + predicted caption.

    Args:
        images (list): list path ke gambar.
        gt_captions (list): ground truth captions.
        pred_captions (list): predicted captions.
        save_path (str): path untuk menyimpan.
        n (int): jumlah sample.
    """
    try:
        import matplotlib.pyplot as plt
        from PIL import Image
    except ImportError:
        print("[WARNING] matplotlib/PIL tidak tersedia untuk qualitative analysis.")
        return

    n = min(n, len(images))
    fig, axes = plt.subplots(n, 1, figsize=(12, n * 2))

    if n == 1:
        axes = [axes]

    for idx, ax in enumerate(axes):
        img_path = images[idx]
        gt = gt_captions[idx] if idx < len(gt_captions) else ''
        pred = pred_captions[idx] if idx < len(pred_captions) else ''

        if os.path.exists(img_path):
            img = Image.open(img_path)
            ax.imshow(img)
            ax.axis('off')
        else:
            ax.text(0.5, 0.5, f'[Image not found]', ha='center', va='center')
            ax.axis('off')

        caption_text = f"GT: {gt}\nPred: {pred}"
        ax.set_title(caption_text, fontsize=9, loc='left', fontfamily='monospace')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Qualitative] Disimpan ke: {save_path}")

    try:
        plt.show()
    except Exception:
        pass

    plt.close()


# ============================================================================
# Comparison Functions
# ============================================================================

def compare_rnn_lstm(rnn_metrics, lstm_metrics, save_path=None):
    """
    Bandingkan hasil evaluasi RNN vs LSTM.

    Args:
        rnn_metrics (dict): metrics untuk RNN.
        lstm_metrics (dict): metrics untuk LSTM.
        save_path (str): path untuk menyimpan hasil.
    Returns:
        dict: comparison results.
    """
    comparison = {
        'rnn': rnn_metrics,
        'lstm': lstm_metrics,
    }

    print(f"\n{'='*60}")
    print(f"{'Comparison':^60}")
    print(f"{'='*60}")
    print(f"{'Metric':<15} {'RNN':>15} {'LSTM':>15} {'Diff':>15}")
    print(f"{'-'*60}")

    for key in ['bleu1', 'bleu2', 'bleu3', 'bleu4', 'meteor']:
        rnn_val = rnn_metrics.get(key, 0)
        lstm_val = lstm_metrics.get(key, 0)
        diff = lstm_val - rnn_val
        print(f"{key.upper():<15} {rnn_val:>15.4f} {lstm_val:>15.4f} {diff:>+15.4f}")

    print(f"{'='*60}\n")

    if save_path:
        with open(save_path, 'w') as f:
            json.dump(comparison, f, indent=2)
        print(f"Comparison disimpan ke: {save_path}")

    return comparison


def compare_keras_vs_scratch(keras_metrics, scratch_metrics, save_path=None):
    """Bandingkan hasil Keras vs Scratch implementation."""
    comparison = {
        'keras': keras_metrics,
        'scratch': scratch_metrics,
    }

    print(f"\n{'='*60}")
    print(f"{'Keras vs Scratch Comparison':^60}")
    print(f"{'='*60}")
    print(f"{'Metric':<15} {'Keras':>15} {'Scratch':>15} {'Diff':>15}")
    print(f"{'-'*60}")

    for key in ['bleu1', 'bleu2', 'bleu3', 'bleu4', 'meteor']:
        keras_val = keras_metrics.get(key, 0)
        scratch_val = scratch_metrics.get(key, 0)
        diff = scratch_val - keras_val
        print(f"{key.upper():<15} {keras_val:>15.4f} {scratch_val:>15.4f} {diff:>+15.4f}")

    print(f"{'='*60}\n")

    if save_path:
        with open(save_path, 'w') as f:
            json.dump(comparison, f, indent=2)

    return comparison


def compare_preinject_vs_initinject(preinject_metrics, initinject_metrics, save_path=None):
    """Bandingkan hasil Pre-Inject vs Init-Inject."""
    comparison = {
        'preinject': preinject_metrics,
        'initinject': initinject_metrics,
    }

    print(f"\n{'='*60}")
    print(f"{'Pre-Inject vs Init-Inject Comparison':^60}")
    print(f"{'='*60}")
    print(f"{'Metric':<15} {'Pre-Inject':>15} {'Init-Inject':>15} {'Diff':>15}")
    print(f"{'-'*60}")

    for key in ['bleu1', 'bleu2', 'bleu3', 'bleu4', 'meteor']:
        pre_val = preinject_metrics.get(key, 0)
        init_val = initinject_metrics.get(key, 0)
        diff = init_val - pre_val
        print(f"{key.upper():<15} {pre_val:>15.4f} {init_val:>15.4f} {diff:>+15.4f}")

    print(f"{'='*60}\n")

    if save_path:
        with open(save_path, 'w') as f:
            json.dump(comparison, f, indent=2)

    return comparison


# ============================================================================
# Full Evaluation Pipeline
# ============================================================================

def full_evaluation_pipeline(cnn_features_test, test_captions_clean,
                            test_image_ids, idx2word,
                            models_dict, images_dir,
                            results_dir='results/rnn', max_length=30,
                            batch_size=64):
    """
    Full evaluation pipeline untuk RNN decoder.

    Args:
        cnn_features_test (np.ndarray): CNN features untuk test set.
        test_captions_clean (dict): {image_id: [captions]}.
        test_image_ids (list): list image ID untuk test.
        idx2word (dict): index → word mapping.
        models_dict (dict): {model_name: keras_model}.
        images_dir (str): direktori gambar Flickr8k.
        results_dir (str): direktori untuk menyimpan hasil.
        max_length (int): panjang maksimal caption.
        batch_size (int): ukuran batch.
    Returns:
        dict: semua hasil evaluasi per model.
    """
    os.makedirs(results_dir, exist_ok=True)

    # Siapkan GT captions
    gt_captions = []
    for img_id in test_image_ids:
        if img_id in test_captions_clean:
            gt_captions.append(test_captions_clean[img_id][0])  # Ambil caption pertama
        else:
            gt_captions.append('')

    # Filter valid samples
    valid_indices = [i for i, cap in enumerate(gt_captions) if cap]
    gt_captions = [gt_captions[i] for i in valid_indices]
    cnn_features_valid = cnn_features_test[valid_indices]

    all_results = {}

    for model_name, model in models_dict.items():
        print(f"\n{'='*60}")
        print(f"[Evaluation] {model_name}")
        print(f"{'='*60}")

        start_time = time.time()
        metrics, pred_captions = evaluate_model(
            model, cnn_features_valid, gt_captions, idx2word,
            max_length=max_length, batch_size=batch_size, verbose=True
        )
        elapsed = time.time() - start_time
        metrics['elapsed_time'] = elapsed

        print_metrics(metrics, title=f"{model_name}")
        print(f"  Waktu inference: {elapsed:.1f} detik ({elapsed/len(gt_captions)*1000:.1f} ms/gambar)")

        # Save results
        metrics_path = os.path.join(results_dir, f'{model_name}_metrics.json')
        captions_path = os.path.join(results_dir, f'{model_name}_captions.json')

        save_metrics_to_file(metrics, metrics_path, model_name)

        with open(captions_path, 'w') as f:
            json.dump({
                'gt': gt_captions,
                'pred': pred_captions
            }, f, indent=2)

        all_results[model_name] = {
            'metrics': metrics,
            'pred_captions': pred_captions,
            'elapsed_time': elapsed,
        }

    # Save summary
    summary_path = os.path.join(results_dir, 'evaluation_summary.json')
    with open(summary_path, 'w') as f:
        json.dump({k: {'metrics': v['metrics'], 'elapsed_time': v['elapsed_time']}
                  for k, v in all_results.items()}, f, indent=2)

    return all_results


# ============================================================================
# Per-Epoch Evaluation
# ============================================================================

class EvaluationCallback:
    """
    Keras Callback untuk evaluasi per epoch.

    Args:
        cnn_features (np.ndarray): CNN features untuk evaluasi.
        gt_captions (list): ground truth captions.
        idx2word (dict): vocabulary.
        eval_every (int): evaluasi setiap N epoch.
        max_length (int): panjang maksimal.
    """

    def __init__(self, cnn_features, gt_captions, idx2word,
                 eval_every=5, max_length=30):
        self.cnn_features = cnn_features
        self.gt_captions = gt_captions
        self.idx2word = idx2word
        self.eval_every = eval_every
        self.max_length = max_length
        self.best_bleu4 = 0.0
        self.best_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        """Dipanggil setiap akhir epoch."""
        if (epoch + 1) % self.eval_every == 0:
            print(f"\n[Epoch {epoch+1}] Evaluasi BLEU-4...")
            # Implementasi evaluasi di sini


if __name__ == '__main__':
    print("[Evaluate] Jalankan setelah training selesai:")
    print("  1. Load bobot trained models")
    print("  2. Jalankan full_evaluation_pipeline()")
