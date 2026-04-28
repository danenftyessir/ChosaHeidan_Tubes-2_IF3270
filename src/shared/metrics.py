"""
Metric evaluasi untuk Image Captioning.
BLEU-1, BLEU-2, BLEU-3, BLEU-4, METEOR.
Dapat dipakai oleh RNN dan LSTM.
"""

import numpy as np
from collections import Counter
import sys

try:
    from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("[WARNING] nltk tidak terinstall. "
          "Install dengan: pip install nltk")


# ============================================================================
# BLEU Score (from scratch)
# ============================================================================

def _get_ngrams(tokens, n):
    """
    Ekstrak n-grams dari list token.

    Args:
        tokens (list): list token.
        n (int): ukuran n-gram.
    Returns:
        Counter: dictionary n-gram → count.
    """
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngrams.append(tuple(tokens[i:i+n]))
    return Counter(ngrams)


def _clip_ngrams(reference_ngrams, hypothesis_ngrams):
    """
    Clip n-grams hypothesis berdasarkan reference.
    Menghitung modified n-gram precision.

    Args:
        reference_ngrams (Counter): n-grams dari reference.
        hypothesis_ngrams (Counter): n-grams dari hypothesis.
    Returns:
        int: clipped count.
    """
    clipped = 0
    for ngram, count in hypothesis_ngrams.items():
        clipped += min(count, reference_ngrams.get(ngram, 0))
    return clipped


def bleu_score(reference, hypothesis, n=4, smooth=False):
    """
    Hitung BLEU score untuk satu pasang reference-hypothesis.
    Implementasi manual tanpa library eksternal.

    Cara kerja:
        1. Hitung modified n-gram precision untuk n=1,2,3,4
        2. Hitung brevity penalty (jika hypothesis terlalu pendek)
        3. BLEU = BP * exp(1/4 * sum(log(p_n)))

    Args:
        reference (list atau str): reference caption (tokenized atau string).
        hypothesis (list atau str): hypothesis caption (tokenized atau string).
        n (int): maksimum n-gram. Default 4 untuk BLEU-4.
        smooth (bool): smoothing untuk n-gram count (untuk sentence pendek).
    Returns:
        float: BLEU score antara 0 dan 1.
    """
    # Tokenize jika string
    if isinstance(reference, str):
        reference = reference.split()
    if isinstance(hypothesis, str):
        hypothesis = hypothesis.split()

    if len(hypothesis) == 0:
        return 0.0

    # Log precision scores
    log_precisions = []

    for i in range(1, n + 1):
        ref_ngrams = _get_ngrams(reference, i)
        hyp_ngrams = _get_ngrams(hypothesis, i)

        clipped_count = _clip_ngrams(ref_ngrams, hyp_ngrams)
        hyp_count = len(hypothesis) - i + 1

        if hyp_count > 0:
            precision = clipped_count / hyp_count
        else:
            precision = 0.0

        # Smoothing untuk n-gram count (opsional)
        if smooth and clipped_count == 0:
            precision = 1.0 / (hyp_count * 2 ** (n - i + 1))

        if precision > 0:
            log_precisions.append(np.log(precision))
        else:
            log_precisions.append(float('-inf'))

    # Brevity penalty
    ref_len = len(reference)
    hyp_len = len(hypothesis)

    if hyp_len > ref_len:
        bp = 1.0
    elif hyp_len == 0:
        bp = 0.0
    else:
        bp = np.exp(1 - ref_len / hyp_len)

    # Geometric mean dari precisions
    if len(log_precisions) == n and all(p != float('-inf') for p in log_precisions):
        geo_mean = np.exp(np.mean(log_precisions))
    else:
        # Jika ada precision = 0, return 0
        geo_mean = 0.0

    bleu = bp * geo_mean
    return bleu


def bleu_n_score(reference, hypothesis, n):
    """
    Hitung BLEU-n score (hanya n-gram spesifik).

    Args:
        reference (list atau str): reference caption.
        hypothesis (list atau str): hypothesis caption.
        n (int): n-gram (1-4).
    Returns:
        float: BLEU-n score.
    """
    return bleu_score(reference, hypothesis, n=n, smooth=False)


def corpus_bleu_score(references, hypotheses, max_n=4, smooth=False):
    """
    Hitung corpus-level BLEU score.
    Menggunakan cumulative n-gram precisions.

    Args:
        references (list): list reference captions.
        hypotheses (list): list hypothesis captions.
        max_n (int): maksimum n-gram (default 4 untuk BLEU-4).
        smooth (bool): smoothing.
    Returns:
        float: corpus BLEU score.
    """
    if len(references) != len(hypotheses):
        raise ValueError("Jumlah reference dan hypothesis harus sama")

    # Aggregate n-gram counts
    ref_counts = [Counter() for _ in range(max_n)]
    hyp_counts = [Counter() for _ in range(max_n)]
    ref_lengths = 0
    hyp_lengths = 0

    for ref, hyp in zip(references, hypotheses):
        if isinstance(ref, str):
            ref = ref.split()
        if isinstance(hyp, str):
            hyp = hyp.split()

        ref_lengths += len(ref)
        hyp_lengths += len(hyp)

        for i in range(1, max_n + 1):
            ref_ngrams = _get_ngrams(ref, i)
            hyp_ngrams = _get_ngrams(hyp, i)

            # Clip dan akumulasi
            for ngram, count in hyp_ngrams.items():
                clipped = min(count, ref_ngrams.get(ngram, 0))
                hyp_counts[i-1][ngram] += clipped
                ref_counts[i-1][ngram] += ref_ngrams.get(ngram, 0)

    # Hitung precisions
    log_precisions = []
    for i in range(max_n):
        if sum(hyp_counts[i].values()) == 0:
            precision = 0.0
        elif sum(ref_counts[i].values()) == 0:
            precision = 0.0
        else:
            clipped = sum(hyp_counts[i].values())
            total = sum(ref_counts[i].values())
            precision = clipped / total

        if smooth and precision == 0:
            precision = 1.0 / (sum(hyp_counts[i].values()) * 2 ** (max_n - i))

        if precision > 0:
            log_precisions.append(np.log(precision))
        else:
            log_precisions.append(float('-inf'))

    # Brevity penalty
    if hyp_lengths > ref_lengths:
        bp = 1.0
    elif hyp_lengths == 0:
        bp = 0.0
    else:
        bp = np.exp(1 - ref_lengths / hyp_lengths)

    # Geometric mean
    if all(p != float('-inf') for p in log_precisions):
        geo_mean = np.exp(np.mean(log_precisions))
    else:
        geo_mean = 0.0

    corpus_bleu = bp * geo_mean
    return corpus_bleu


# ============================================================================
# BLEU Score dengan NLTK (wrapper)
# ============================================================================

def bleu_score_nltk(reference, hypothesis, max_n=4, smooth=False):
    """
    BLEU score menggunakan NLTK (lebih akurat, handling edge case better).

    Args:
        reference (list): list reference tokens.
        hypothesis (list): list hypothesis tokens.
        max_n (int): maksimum n-gram.
        smooth (bool): gunakan smoothing.
    Returns:
        float: BLEU score.
    """
    if not NLTK_AVAILABLE:
        print("[WARNING] nltk tidak tersedia. Gunakan bleu_score() manual.")
        return bleu_score(reference, hypothesis, n=max_n, smooth=smooth)

    # NLTK expect reference sebagai list of list (multi-reference support)
    if isinstance(reference, str):
        reference = [reference.split()]
    elif not isinstance(reference[0], list):
        reference = [reference]

    if isinstance(hypothesis, str):
        hypothesis = hypothesis.split()

    smoothing = SmoothingFunction().method1 if smooth else None

    try:
        weights = tuple([1.0/max_n] * max_n)
        score = sentence_bleu(reference, hypothesis, weights=weights,
                             smoothing_function=smoothing)
        return score
    except ZeroDivisionError:
        return 0.0


def corpus_bleu_score_nltk(references, hypotheses, max_n=4):
    """
    Corpus-level BLEU menggunakan NLTK.

    Args:
        references (list): list reference captions (string atau tokenized).
        hypotheses (list): list hypothesis captions.
        max_n (int): maksimum n-gram.
    Returns:
        float: corpus BLEU score.
    """
    if not NLTK_AVAILABLE:
        print("[WARNING] nltk tidak tersedia. Gunakan corpus_bleu_score() manual.")
        return corpus_bleu_score(references, hypotheses, max_n=max_n)

    # Format untuk NLTK: list of list of references (multi-reference)
    refs_list = []
    for ref in references:
        if isinstance(ref, str):
            refs_list.append([ref.split()])
        elif isinstance(ref[0], str):
            refs_list.append([r.split() if isinstance(r, str) else r for r in ref])
        else:
            refs_list.append(ref)

    hyps_list = []
    for hyp in hypotheses:
        if isinstance(hyp, str):
            hyps_list.append(hyp.split())
        else:
            hyps_list.append(hyp)

    weights = tuple([1.0/max_n] * max_n)
    return corpus_bleu(refs_list, hyps_list, weights=weights)


# ============================================================================
# METEOR Score
# ============================================================================

def meteor_score_reference(reference, hypothesis):
    """
    METEOR score untuk satu pair reference-hypothesis.
    METEOR menggabungkan precision, recall, dan stemming.

    Args:
        reference (str atau list): reference caption.
        hypothesis (str atau list): hypothesis caption.
    Returns:
        float: METEOR score antara 0 dan 1.
    """
    if not NLTK_AVAILABLE:
        print("[WARNING] nltk tidak tersedia. METEOR tidak dapat dihitung.")
        return 0.0

    if isinstance(reference, str):
        reference = reference.split()
    if isinstance(hypothesis, str):
        hypothesis = hypothesis.split()

    try:
        return meteor_score([reference], hypothesis)
    except Exception:
        return 0.0


def corpus_meteor_score(references, hypotheses):
    """
    Rata-rata METEOR score untuk seluruh corpus.

    Args:
        references (list): list reference captions.
        hypotheses (list): list hypothesis captions.
    Returns:
        float: rata-rata METEOR score.
    """
    if not NLTK_AVAILABLE:
        print("[WARNING] nltk tidak tersedia. METEOR tidak dapat dihitung.")
        return 0.0

    scores = []
    for ref, hyp in zip(references, hypotheses):
        scores.append(meteor_score_reference(ref, hyp))

    return np.mean(scores) if scores else 0.0


# ============================================================================
# CIDEr Score (simplified)
# ============================================================================

def cosine_similarity(vec1, vec2):
    """Cosine similarity antara dua vectors."""
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 < 1e-10 or norm2 < 1e-10:
        return 0.0
    return np.dot(vec1, vec2) / (norm1 * norm2)


def cider_score(reference, hypothesis, n=4):
    """
    CIDEr score (Consensus-based Image Description Evaluation).
    Implementasi sederhana menggunakan TF-IDF weighted cosine similarity.

    CATATAN: Ini adalah implementasi dasar. CIDEr asli memerlukan
    corpus IDF yang lebih kompleks dari seluruh dataset.

    Args:
        reference (list atau str): reference caption.
        hypothesis (list atau str): hypothesis caption.
        n (int): maksimum n-gram.
    Returns:
        float: CIDEr score.
    """
    if isinstance(reference, str):
        reference = reference.split()
    if isinstance(hypothesis, str):
        hypothesis = hypothesis.split()

    # Bangun n-gram vectors
    ref_ngrams = Counter()
    hyp_ngrams = Counter()
    for i in range(1, n + 1):
        for ng in _get_ngrams(reference, i):
            ref_ngrams[ng] += 1
        for ng in _get_ngrams(hypothesis, i):
            hyp_ngrams[ng] += 1

    all_ngrams = list(set(ref_ngrams.keys()) | set(hyp_ngrams.keys()))
    if not all_ngrams:
        return 0.0

    # Simple TF-IDF
    ref_vec = np.array([ref_ngrams.get(ng, 0) for ng in all_ngrams], dtype=np.float64)
    hyp_vec = np.array([hyp_ngrams.get(ng, 0) for ng in all_ngrams], dtype=np.float64)

    # Normalisasi
    ref_norm = np.linalg.norm(ref_vec)
    hyp_norm = np.linalg.norm(hyp_vec)
    if ref_norm > 1e-10:
        ref_vec /= ref_norm
    if hyp_norm > 1e-10:
        hyp_vec /= hyp_norm

    return 10.0 * cosine_similarity(ref_vec, hyp_vec)


# ============================================================================
# Batch Evaluation
# ============================================================================

def evaluate_batch(references, hypotheses, metrics=None):
    """
    Evaluasi batch caption dengan berbagai metric.

    Args:
        references (list): list reference captions (string atau tokenized).
        hypotheses (list): list hypothesis captions (string atau tokenized).
        metrics (list): metric yang akan dihitung.
            Pilihan: 'bleu1', 'bleu2', 'bleu3', 'bleu4', 'meteor', 'cider'.
    Returns:
        dict: {metric_name: score}.
    """
    if metrics is None:
        metrics = ['bleu1', 'bleu2', 'bleu3', 'bleu4']

    results = {}

    if 'bleu1' in metrics:
        results['bleu1'] = corpus_bleu_score(references, hypotheses, max_n=1)

    if 'bleu2' in metrics:
        results['bleu2'] = corpus_bleu_score(references, hypotheses, max_n=2)

    if 'bleu3' in metrics:
        results['bleu3'] = corpus_bleu_score(references, hypotheses, max_n=3)

    if 'bleu4' in metrics:
        results['bleu4'] = corpus_bleu_score(references, hypotheses, max_n=4)

    if 'meteor' in metrics:
        results['meteor'] = corpus_meteor_score(references, hypotheses)

    if 'cider' in metrics:
        cider_scores = [cider_score(r, h) for r, h in zip(references, hypotheses)]
        results['cider'] = np.mean(cider_scores) if cider_scores else 0.0

    return results


def evaluate_single_pair(reference, hypothesis, max_n=4):
    """
    Evaluasi single reference-hypothesis pair.

    Args:
        reference (str atau list): reference caption.
        hypothesis (str atau list): hypothesis caption.
        max_n (int): maksimum n-gram untuk BLEU.
    Returns:
        dict: {bleu1, bleu2, bleu3, bleu4, meteor} scores.
    """
    results = {}
    results['bleu1'] = bleu_n_score(reference, hypothesis, 1)
    results['bleu2'] = bleu_n_score(reference, hypothesis, 2)
    results['bleu3'] = bleu_n_score(reference, hypothesis, 3)
    results['bleu4'] = bleu_n_score(reference, hypothesis, 4)

    if NLTK_AVAILABLE:
        results['meteor'] = meteor_score_reference(reference, hypothesis)

    return results


# ============================================================================
# Utilities
# ============================================================================

def print_metrics(metrics_dict, title="Metrics"):
    """
    Cetak metric evaluation dalam format yang rapi.

    Args:
        metrics_dict (dict): {metric_name: score}.
        title (str): judul evaluasi.
    """
    print(f"\n{'='*50}")
    print(f"{title}")
    print(f"{'='*50}")
    for metric, score in sorted(metrics_dict.items()):
        if isinstance(score, (int, float)):
            print(f"  {metric.upper():10s}: {float(score):.4f}")
        else:
            print(f"  {metric.upper():10s}: {score}")
    print(f"{'='*50}\n")


def save_metrics_to_file(metrics_dict, path, model_name="model"):
    """
    Simpan metric evaluation ke file.

    Args:
        metrics_dict (dict): {metric_name: score}.
        path (str): path file output.
        model_name (str): nama model untuk log.
    """
    import json
    from datetime import datetime

    output = {
        'model': model_name,
        'timestamp': datetime.now().isoformat(),
        'metrics': {k: float(v) if isinstance(v, (int, float)) else v
                   for k, v in metrics_dict.items()}
    }

    if path.endswith('.json'):
        with open(path, 'w') as f:
            json.dump(output, f, indent=2)
    else:
        with open(path, 'w') as f:
            f.write(f"Model: {model_name}\n")
            f.write(f"Timestamp: {output['timestamp']}\n")
            f.write("-" * 50 + "\n")
            for metric, score in sorted(metrics_dict.items()):
                f.write(f"{metric.upper()}: {score}\n")

    print(f"Metrics disimpan ke: {path}")