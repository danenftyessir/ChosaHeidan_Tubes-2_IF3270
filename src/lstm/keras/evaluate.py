"""
Evaluasi untuk decoder LSTM menggunakan Keras.
BLEU-4, METEOR, qualitative analysis, dan perbandingan Keras vs Scratch.
Sama strukturnya dengan RNN evaluate.py.
"""

import os
import sys
import json
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))
from metrics import (
    evaluate_batch, corpus_bleu_score, print_metrics, save_metrics_to_file
)
from caption_preprocess import END_IDX, START_IDX, sequence_to_caption


# ============================================================================
# Evaluation Functions
# ============================================================================

def evaluate_model(model, cnn_features, gt_captions, idx2word,
                   max_length=30, batch_size=64, verbose=True,
                   repetition_penalty=1.5):
    """
    Evaluasi model LSTM pada dataset dengan batched greedy decoding.

    Args:
        model: Keras Model instance.
        cnn_features (np.ndarray): CNN features, shape (N, feature_dim).
        gt_captions (list): list ground truth captions (string).
        idx2word (dict): index → word mapping.
        max_length (int): panjang maksimal caption generation.
        batch_size (int): ukuran batch untuk inference.
        verbose (bool): cetak progress.
        repetition_penalty (float): faktor penalti untuk token berulang.
            Default 1.5.
    Returns:
        dict: metrics (bleu1-4, meteor).
    """
    import tensorflow as tf

    N = cnn_features.shape[0]
    seqs = np.zeros((N, max_length), dtype=np.int32)
    seqs[:, 0] = START_IDX
    done = np.zeros(N, dtype=bool)
    tokens_list = [[] for _ in range(N)]

    for step in range(max_length - 1):
        active = np.where(~done)[0]
        if len(active) == 0:
            break

        for b in range(0, len(active), batch_size):
            idx = active[b: b + batch_size]
            batch_cnn = tf.constant(cnn_features[idx], dtype=tf.float32)
            batch_seq = tf.constant(seqs[idx], dtype=tf.int32)
            probs = model([batch_cnn, batch_seq], training=False).numpy()
            # probs shape: (batch, max_length, vocab_size)
            step_probs = probs[:, step, :].copy()

            if repetition_penalty > 1.0:
                for local_i, sample_i in enumerate(idx):
                    for prev_tok in set(tokens_list[sample_i]):
                        step_probs[local_i, prev_tok] /= repetition_penalty

            next_tokens = np.argmax(step_probs, axis=-1)

            for local_i, sample_i in enumerate(idx):
                tok = int(next_tokens[local_i])
                if tok == END_IDX or tok == 0:
                    done[sample_i] = True
                else:
                    tokens_list[sample_i].append(tok)
                    seqs[sample_i, step + 1] = tok

        if verbose and (step + 1) % 5 == 0:
            remaining = int((~done).sum())
            print(f"  Step {step + 1}/{max_length - 1}  ({N - remaining}/{N} selesai)")

        if done.all():
            break

    pred_captions = [sequence_to_caption(toks, idx2word) for toks in tokens_list]

    metrics = evaluate_batch(gt_captions, pred_captions,
                             metrics=['bleu1', 'bleu2', 'bleu3', 'bleu4', 'meteor'],
                             smooth=True)

    return metrics, pred_captions


def greedy_decode_keras(model, cnn_feature, max_length=30, repetition_penalty=1.5):
    """
    Greedy decoding untuk model Keras LSTM.

    Args:
        model: Keras Model.
        cnn_feature: (1, feature_dim).
        max_length (int): panjang maksimal.
        repetition_penalty (float): penalti untuk token berulang. Default 1.5.
    Returns:
        list: sequence of token indices.
    """
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    tokens = []
    seq = [START_IDX]
    seq_padded = pad_sequences([seq], maxlen=max_length, padding='post')[0]

    for step in range(max_length):
        raw = model.predict([cnn_feature, seq_padded[np.newaxis, :]], verbose=0)[0]
        # raw shape: (seq_max_length, vocab_size) — ambil prediksi di posisi step
        step_probs = raw[step].copy() if raw.ndim == 2 else raw.copy()

        # Repetition penalty: kurangi prob token yang sudah di-generate
        if repetition_penalty > 1.0 and tokens:
            for prev_tok in set(tokens):
                step_probs[prev_tok] /= repetition_penalty

        next_token = int(np.argmax(step_probs))

        if next_token == END_IDX or next_token == 0:
            break

        tokens.append(next_token)

        seq.append(next_token)
        seq_padded = pad_sequences([seq], maxlen=max_length, padding='post')[0]

    return tokens


def beam_search_decode_keras(model, cnn_feature, idx2word, k=5, max_length=30):
    """
    Beam search decoding untuk model Keras LSTM.

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

            seq_padded = pad_sequences([seq], maxlen=max_length, padding='post')[0]
            probs = model.predict([cnn_feature, seq_padded[np.newaxis, :]], verbose=0)[0]

            top_k_idx = np.argsort(probs)[-k:]
            for token in top_k_idx:
                new_seq = seq + [int(token)]
                new_score = score + np.log(probs[token] + 1e-10)
                all_candidates.append((new_seq, new_score))

        all_candidates.sort(key=lambda x: x[1], reverse=True)
        beams = all_candidates[:k]

        if all(seq[-1] == END_IDX or seq[-1] == 0 for seq, _ in beams):
            break

    best_seq = beams[0][0]
    return sequence_to_caption(best_seq, idx2word)


def qualitative_analysis(images, gt_captions, pred_captions, save_path=None, n=10):
    """
    Tampilkan qualitative examples.

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
        print("[WARNING] matplotlib/PIL tidak tersedia.")
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

def compare_lstm_rnn(lstm_metrics, rnn_metrics, save_path=None):
    """Bandingkan hasil evaluasi LSTM vs RNN."""
    comparison = {
        'lstm': lstm_metrics,
        'rnn': rnn_metrics,
    }

    print(f"\n{'='*60}")
    print(f"{'LSTM vs RNN Comparison':^60}")
    print(f"{'='*60}")
    print(f"{'Metric':<15} {'LSTM':>15} {'RNN':>15} {'Diff':>15}")
    print(f"{'-'*60}")

    for key in ['bleu1', 'bleu2', 'bleu3', 'bleu4', 'meteor']:
        lstm_val = lstm_metrics.get(key, 0)
        rnn_val = rnn_metrics.get(key, 0)
        diff = lstm_val - rnn_val
        print(f"{key.upper():<15} {lstm_val:>15.4f} {rnn_val:>15.4f} {diff:>+15.4f}")

    print(f"{'='*60}\n")

    if save_path:
        with open(save_path, 'w') as f:
            json.dump(comparison, f, indent=2)

    return comparison


def compare_keras_vs_scratch(keras_metrics, scratch_metrics, save_path=None):
    """Bandingkan hasil Keras vs Scratch."""
    comparison = {
        'keras': keras_metrics,
        'scratch': scratch_metrics,
    }

    print(f"\n{'='*60}")
    print(f"{'Keras vs Scratch Comparison (LSTM)':^60}")
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
    """Bandingkan hasil Pre-Inject vs Init-Inject untuk LSTM."""
    comparison = {
        'preinject': preinject_metrics,
        'initinject': initinject_metrics,
    }

    print(f"\n{'='*60}")
    print(f"{'Pre-Inject vs Init-Inject Comparison (LSTM)':^60}")
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
                            results_dir='results/lstm', max_length=30,
                            batch_size=64):
    """
    Full evaluation pipeline untuk LSTM decoder.

    Args:
        cnn_features_test (np.ndarray): CNN features untuk test set.
        test_captions_clean (dict): {image_id: [captions]}.
        test_image_ids (list): list image ID untuk test.
        idx2word (dict): index → word mapping.
        models_dict (dict): {model_name: keras_model}.
        images_dir (str): direktori gambar.
        results_dir (str): direktori hasil.
        max_length (int): panjang maksimal caption.
        batch_size (int): ukuran batch.
    Returns:
        dict: semua hasil evaluasi per model.
    """
    os.makedirs(results_dir, exist_ok=True)

    gt_captions = []
    for img_id in test_image_ids:
        if img_id in test_captions_clean:
            gt_captions.append(test_captions_clean[img_id][0])
        else:
            gt_captions.append('')

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

    summary_path = os.path.join(results_dir, 'evaluation_summary.json')
    with open(summary_path, 'w') as f:
        json.dump({k: {'metrics': v['metrics'], 'elapsed_time': v['elapsed_time']}
                  for k, v in all_results.items()}, f, indent=2)

    return all_results


# ============================================================================
# Max Length Variation
# ============================================================================

def evaluate_max_length_variation(model, cnn_features, gt_captions, idx2word,
                                  max_lengths=None, batch_size=64):
    """
    Evaluasi pengaruh panjang maksimal caption terhadap BLEU-4.

    Args:
        model: Keras Model.
        cnn_features: CNN features.
        gt_captions: ground truth captions.
        idx2word: vocabulary.
        max_lengths (list): list max_length yang akan diuji.
        batch_size (int): ukuran batch.
    Returns:
        dict: {max_length: metrics}.
    """
    if max_lengths is None:
        max_lengths = [20, 30, 40, 50]

    results = {}

    for max_len in max_lengths:
        print(f"\n[Max Length] Evaluasi dengan max_length={max_len}...")
        metrics, _ = evaluate_model(
            model, cnn_features, gt_captions, idx2word,
            max_length=max_len, batch_size=batch_size, verbose=False
        )
        results[max_len] = metrics
        print(f"  BLEU-4: {metrics.get('bleu4', 0):.4f}")

    print(f"\n{'='*60}")
    print(f"{'Pengaruh Max Length terhadap BLEU-4':^60}")
    print(f"{'='*60}")
    print(f"{'Max Length':>12} {'BLEU-1':>10} {'BLEU-2':>10} {'BLEU-3':>10} {'BLEU-4':>10}")
    print(f"{'-'*60}")
    for max_len, m in sorted(results.items()):
        print(f"{max_len:>12} {m.get('bleu1', 0):>10.4f} {m.get('bleu2', 0):>10.4f} "
              f"{m.get('bleu3', 0):>10.4f} {m.get('bleu4', 0):>10.4f}")
    print(f"{'='*60}\n")

    return results


if __name__ == '__main__':
    print("[Evaluate] Jalankan setelah training selesai:")
    print("  1. Load bobot trained models")
    print("  2. Jalankan full_evaluation_pipeline()")
