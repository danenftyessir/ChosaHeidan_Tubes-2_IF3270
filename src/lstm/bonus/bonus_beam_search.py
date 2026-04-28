"""
Beam Search Decoder untuk LSTM dari nol.
Beam search sebagai alternatif greedy decoding (k=5).
Dapat dipakai untuk batch inference.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))
from caption_preprocess import END_IDX, START_IDX


# ============================================================================
# Beam Search Decoder
# ============================================================================

class BeamSearchDecoderLSTM:
    """
    Beam Search Decoder untuk LSTM.

    Algoritma:
        1. Inisialisasi: beam dengan <start> token, score=0
        2. Setiap step: expand setiap beam → top-k by cumulative log probability
        3. Terminasi: <end> token atau max_length
        4. Return: top beam sequence

    Args:
        model: instance LSTMScratch.
        k (int): beam width. Default 5 (sesuai requirement).
        max_length (int): panjang maksimum caption.
    """

    def __init__(self, model, k=5, max_length=30):
        self.model = model
        self.k = k
        self.max_length = max_length

    def decode(self, cnn_feature, idx2word):
        """
        Beam search decode untuk SATU gambar.

        Args:
            cnn_feature: array, bentuk (feature_dim,) atau (1, feature_dim).
            idx2word (dict): index → word mapping.
        Returns:
            str: caption string.
        """
        if cnn_feature.ndim == 1:
            cnn_feature = cnn_feature.reshape(1, -1)

        # beams: (token_sequence, log_prob, hidden_state, cell_state)
        beams = [(
            [START_IDX],
            0.0,
            np.zeros((1, self.model.hidden_dim), dtype=np.float64),
            np.zeros((1, self.model.hidden_dim), dtype=np.float64)
        )]
        finished_beams = []

        for step in range(self.max_length):
            all_candidates = []

            for seq, score, h_state, c_state in beams:
                if seq[-1] == END_IDX:
                    all_candidates.append((seq, score, h_state, c_state))
                    continue

                probs, h_new, c_new = self._step_decode(
                    cnn_feature, seq[-1], h_state, c_state
                )

                top_k_idx = np.argsort(probs)[-self.k:]

                for token in top_k_idx:
                    new_seq = seq + [int(token)]
                    new_score = score + np.log(probs[token] + 1e-10)
                    all_candidates.append((new_seq, new_score, h_new, c_new))

            # Select top-k
            all_candidates.sort(key=lambda x: x[1], reverse=True)
            beams = all_candidates[:self.k]

            if all(seq[-1] == END_IDX for seq, _, _, _ in beams):
                break

        best_seq = beams[0][0]
        return self._seq_to_caption(best_seq, idx2word)

    def decode_batch(self, cnn_features, idx2word):
        """
        Beam search decode untuk batch gambar sekaligus.

        Args:
            cnn_features: array, bentuk (batch_size, feature_dim).
            idx2word (dict): index → word mapping.
        Returns:
            list: list caption strings.
        """
        if cnn_features.ndim == 1:
            cnn_features = cnn_features.reshape(1, -1)

        batch_size = cnn_features.shape[0]
        captions = []

        for b in range(batch_size):
            feat = cnn_features[b:b+1]
            caption = self.decode(feat, idx2word)
            captions.append(caption)

        return captions

    def _step_decode(self, cnn_feature, prev_token, h_prev, c_prev):
        """
        Single decoding step menggunakan model LSTM.

        Args:
            cnn_feature: (1, feature_dim)
            prev_token: integer token index
            h_prev: (1, hidden_dim)
            c_prev: (1, hidden_dim)
        Returns:
            probs: (vocab_size,) distribusi probabilitas
            h_new: (1, hidden_dim)
            c_new: (1, hidden_dim)
        """
        probs, h_new, c_new = self.model.step_forward(
            cnn_feature, prev_token, h_prev, c_prev
        )
        return probs, h_new, c_new

    def _seq_to_caption(self, sequence, idx2word):
        """Konversi sequence ke caption string."""
        words = []
        for token in sequence:
            word = idx2word.get(token, '<unk>')
            if word in ('<pad>', '<start>', '<end>'):
                continue
            words.append(word)
        return ' '.join(words)


# ============================================================================
# Beam Search Utilities
# ============================================================================

def beam_search_lstm(model, cnn_feature, idx2word, k=5, max_length=30):
    """
    Beam search decode (function-based) untuk LSTM.

    Args:
        model: LSTMScratch instance.
        cnn_feature: (feature_dim,) atau (1, feature_dim).
        idx2word (dict): vocabulary.
        k (int): beam width.
        max_length (int): panjang maksimal.
    Returns:
        str: generated caption.
    """
    if cnn_feature.ndim == 1:
        cnn_feature = cnn_feature.reshape(1, -1)

    beams = [(
        [START_IDX],
        0.0,
        np.zeros((1, model.hidden_dim), dtype=np.float64),
        np.zeros((1, model.hidden_dim), dtype=np.float64)
    )]

    for step in range(max_length):
        all_candidates = []

        for seq, score, h_state, c_state in beams:
            if seq[-1] == END_IDX:
                all_candidates.append((seq, score, h_state, c_state))
                continue

            probs, h_new, c_new = model.step_forward(
                cnn_feature, seq[-1], h_state, c_state
            )
            top_k_idx = np.argsort(probs)[-k:]

            for token in top_k_idx:
                new_seq = seq + [int(token)]
                new_score = score + np.log(probs[token] + 1e-10)
                all_candidates.append((new_seq, new_score, h_new, c_new))

        all_candidates.sort(key=lambda x: x[1], reverse=True)
        beams = all_candidates[:k]

        if all(seq[-1] == END_IDX for seq, _, _, _ in beams):
            break

    best_seq = beams[0][0]
    words = []
    for token in best_seq:
        word = idx2word.get(token, '<unk>')
        if word in ('<pad>', '<start>', '<end>'):
            continue
        words.append(word)
    return ' '.join(words)


def compare_lstm_beam_vs_greedy(model, cnn_features, idx2word, k=5,
                                 max_length=30, n_samples=10):
    """
    Bandingkan hasil beam search vs greedy decoding untuk LSTM.

    Args:
        model: LSTMScratch instance.
        cnn_features: batch CNN features.
        idx2word: vocabulary.
        k (int): beam width.
        max_length (int): panjang maksimal.
        n_samples (int): jumlah sample untuk dibandingkan.
    Returns:
        dict: comparison results.
    """
    results = {
        'beam': [],
        'greedy': [],
    }

    decoder = BeamSearchDecoderLSTM(model, k=k, max_length=max_length)

    for i in range(min(n_samples, len(cnn_features))):
        feat = cnn_features[i:i+1]

        beam_caption = decoder.decode(feat, idx2word)
        greedy_caption = model.greedy_decode(feat, idx2word, max_length)

        results['beam'].append(beam_caption)
        results['greedy'].append(greedy_caption)

        if i < 5:
            print(f"\n[Gambar {i+1}]")
            print(f"  Greedy:    {greedy_caption}")
            print(f"  Beam(k={k}): {beam_caption}")

    return results


# ============================================================================
# Batch Beam Search
# ============================================================================

def beam_search_batch_lstm(model, cnn_features, idx2word, k=5, max_length=30):
    """
    Beam search untuk batch gambar (parallel beam search).

    Args:
        model: LSTMScratch instance.
        cnn_features: (batch_size, feature_dim).
        idx2word: vocabulary.
        k (int): beam width.
        max_length (int): panjang maksimal.
    Returns:
        list: list caption strings.
    """
    if cnn_features.ndim == 1:
        cnn_features = cnn_features.reshape(1, -1)

    batch_size = cnn_features.shape[0]
    captions = []

    for b in range(batch_size):
        feat = cnn_features[b:b+1]
        caption = beam_search_lstm(model, feat, idx2word, k=k, max_length=max_length)
        captions.append(caption)

    return captions


# ============================================================================
# Length Penalty (untuk Length Normalization)
# ============================================================================

def beam_search_lstm_with_length_penalty(model, cnn_feature, idx2word, k=5,
                                          max_length=30, alpha=0.6):
    """
    Beam search dengan length normalization untuk LSTM.
    Score = log_prob / len^alpha (panjang lebih pendek lebih diapresiasi).

    Args:
        model: LSTMScratch instance.
        cnn_feature: (1, feature_dim).
        idx2word: vocabulary.
        k (int): beam width.
        max_length (int): panjang maksimal.
        alpha (float): length penalty coefficient.
    Returns:
        str: generated caption.
    """
    if cnn_feature.ndim == 1:
        cnn_feature = cnn_feature.reshape(1, -1)

    beams = [(
        [START_IDX],
        0.0,
        np.zeros((1, model.hidden_dim), dtype=np.float64),
        np.zeros((1, model.hidden_dim), dtype=np.float64)
    )]

    for step in range(max_length):
        all_candidates = []

        for seq, score, h_state, c_state in beams:
            if seq[-1] == END_IDX:
                length_penalty = len(seq) ** alpha
                norm_score = score / length_penalty
                all_candidates.append((seq, score, h_state, c_state, norm_score))
                continue

            probs, h_new, c_new = model.step_forward(
                cnn_feature, seq[-1], h_state, c_state
            )
            top_k_idx = np.argsort(probs)[-k:]

            for token in top_k_idx:
                new_seq = seq + [int(token)]
                new_score = score + np.log(probs[token] + 1e-10)
                all_candidates.append((new_seq, new_score, h_new, c_new, 0.0))

        all_candidates.sort(key=lambda x: x[4] if x[4] != 0.0 else x[1], reverse=True)
        beams = [(seq, score, h, c) for seq, score, h, c, _ in all_candidates[:k]]

        if all(seq[-1] == END_IDX for seq, _, _, _ in beams):
            break

    best_seq = beams[0][0]
    words = []
    for token in best_seq:
        word = idx2word.get(token, '<unk>')
        if word in ('<pad>', '<start>', '<end>'):
            continue
        words.append(word)
    return ' '.join(words)


if __name__ == '__main__':
    print("[Test] Beam Search Decoder untuk LSTM...")
    print("\n[Test] Contoh usage:")
    print("  from bonus_beam_search import beam_search_lstm, BeamSearchDecoderLSTM")
    print("  caption = beam_search_lstm(model, cnn_feature, idx2word, k=5)")
    print("\n[Test] Selesai.")
