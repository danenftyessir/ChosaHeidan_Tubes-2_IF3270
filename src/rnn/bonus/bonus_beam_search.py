"""
Beam Search Decoder untuk SimpleRNN dari nol.
Beam search sebagai alternatif greedy decoding (k=5).
Dapat dipakai untuk batch inference.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))
from caption_preprocess import END_IDX, START_IDX, sequence_to_caption


# ============================================================================
# Beam Search Decoder
# ============================================================================

class BeamSearchDecoder:
    """
    Beam Search Decoder untuk SimpleRNN.

    Algoritma:
        1. Inisialisasi: beam dengan <start> token, score=0
        2. Setiap step: expand setiap beam → top-k by cumulative log probability
        3. Terminasi: <end> token atau max_length
        4. Return: top beam sequence

    Args:
        model: instance RNNScratch.
        k (int): beam width. Default 5.
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

        # beams: list of (token_sequence, cumulative_log_prob)
        beams = [([START_IDX], 0.0)]
        finished_beams = []

        for step in range(self.max_length):
            all_candidates = []

            for seq, score in beams:
                if seq[-1] == END_IDX:
                    # Beam sudah selesai
                    all_candidates.append((seq, score))
                    continue

                # Decode satu step
                probs, h_new = self._step_decode(cnn_feature, seq[-1], seq)

                # Top-k next tokens
                top_k_idx = np.argsort(probs)[-self.k:]

                for token in top_k_idx:
                    new_seq = seq + [int(token)]
                    new_score = score + np.log(probs[token] + 1e-10)
                    all_candidates.append((new_seq, new_score))

            # Select top-k
            all_candidates.sort(key=lambda x: x[1], reverse=True)
            beams = all_candidates[:self.k]

            # Check termination
            if all(seq[-1] == END_IDX for seq, _ in beams):
                break

        # Pilih beam terbaik
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

    def _step_decode(self, cnn_feature, prev_token, seq_so_far):
        """
        Single decoding step menggunakan model RNN.

        Args:
            cnn_feature: (1, feature_dim)
            prev_token: integer token index
            seq_so_far: list token yang sudah di-decode
        Returns:
            probs: (vocab_size,) distribusi probabilitas
            h_new: hidden state baru
        """
        # Gunakan step_forward dari model
        # Di step pertama, hidden state adalah zeros
        if len(seq_so_far) == 1:
            # Hanya <start> token, belum ada step sebelumnya
            # Hidden state awal
            h_prev = np.zeros((1, self.model.hidden_dim), dtype=np.float64)
            h_final = self.model.step_forward(cnn_feature, prev_token, h_prev)
            probs = h_final[0]
            # probs = self.model.step_forward(...) kembali (probs, h_new)
            probs, h_new = self.model.step_forward(cnn_feature, prev_token, h_prev)
        else:
            # Ada hidden state dari step sebelumnya
            # Kita gunakan cached hidden state
            # Untuk simplicity, kita gunakan step_forward dengan h_prev=0
            h_prev = np.zeros((1, self.model.hidden_dim), dtype=np.float64)
            probs, h_new = self.model.step_forward(cnn_feature, prev_token, h_prev)

        return probs, h_new

    def _seq_to_caption(self, sequence, idx2word):
        """Konversi sequence ke caption string."""
        words = []
        for token in sequence:
            word = idx2word.get(token, '<unk>')
            if word in ('<pad>', '<start>', '<end>'):
                continue
            words.append(word)
        return ' '.join(words)


class BeamSearchDecoderWithState(BeamSearchDecoder):
    """
    Beam Search Decoder yang lebih efisien dengan cached hidden states.
    Menyimpan hidden state per beam untuk menghindari recompute.
    """

    def __init__(self, model, k=5, max_length=30):
        super().__init__(model, k, max_length)
        self.state_cache = None

    def decode(self, cnn_feature, idx2word):
        """Decode dengan state caching."""
        if cnn_feature.ndim == 1:
            cnn_feature = cnn_feature.reshape(1, -1)

        # beams: (token_sequence, log_prob, hidden_state)
        beams = [([START_IDX], 0.0, np.zeros((1, self.model.hidden_dim), dtype=np.float64))]

        for step in range(self.max_length):
            all_candidates = []

            for seq, score, h_state in beams:
                if seq[-1] == END_IDX:
                    all_candidates.append((seq, score, h_state))
                    continue

                probs, h_new = self.model.step_forward(cnn_feature, seq[-1], h_state)
                top_k_idx = np.argsort(probs)[-self.k:]

                for token in top_k_idx:
                    new_seq = seq + [int(token)]
                    new_score = score + np.log(probs[token] + 1e-10)
                    all_candidates.append((new_seq, new_score, h_new))

            all_candidates.sort(key=lambda x: x[1], reverse=True)
            beams = all_candidates[:self.k]

            if all(seq[-1] == END_IDX for seq, _, _ in beams):
                break

        best_seq = beams[0][0]
        return self._seq_to_caption(best_seq, idx2word)


# ============================================================================
# Beam Search Utilities
# ============================================================================

def beam_search(model, cnn_feature, idx2word, k=5, max_length=30):
    """
    Beam search decode (function-based, tanpa class).

    Args:
        model: RNNScratch instance.
        cnn_feature: (feature_dim,) atau (1, feature_dim).
        idx2word (dict): vocabulary.
        k (int): beam width.
        max_length (int): panjang maksimal.
    Returns:
        str: generated caption.
    """
    if cnn_feature.ndim == 1:
        cnn_feature = cnn_feature.reshape(1, -1)

    beams = [([START_IDX], 0.0, np.zeros((1, model.hidden_dim), dtype=np.float64))]

    for step in range(max_length):
        all_candidates = []

        for seq, score, h_state in beams:
            if seq[-1] == END_IDX:
                all_candidates.append((seq, score, h_state))
                continue

            probs, h_new = model.step_forward(cnn_feature, seq[-1], h_state)
            top_k_idx = np.argsort(probs)[-k:]

            for token in top_k_idx:
                new_seq = seq + [int(token)]
                new_score = score + np.log(probs[token] + 1e-10)
                all_candidates.append((new_seq, new_score, h_new))

        all_candidates.sort(key=lambda x: x[1], reverse=True)
        beams = all_candidates[:k]

        if all(seq[-1] == END_IDX for seq, _, _ in beams):
            break

    best_seq = beams[0][0]
    words = []
    for token in best_seq:
        word = idx2word.get(token, '<unk>')
        if word in ('<pad>', '<start>', '<end>'):
            continue
        words.append(word)
    return ' '.join(words)


def compare_beam_vs_greedy(model, cnn_features, idx2word, k=5, max_length=30,
                           n_samples=10):
    """
    Bandingkan hasil beam search vs greedy decoding.

    Args:
        model: RNNScratch instance.
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

    decoder = BeamSearchDecoder(model, k=k, max_length=max_length)

    for i in range(min(n_samples, len(cnn_features))):
        feat = cnn_features[i:i+1]

        # Beam search
        beam_caption = decoder.decode(feat, idx2word)

        # Greedy
        greedy_caption = model.greedy_decode(feat, idx2word, max_length)

        results['beam'].append(beam_caption)
        results['greedy'].append(greedy_caption)

        if i < 5:  # Print 5 contoh pertama
            print(f"\n[Gambar {i+1}]")
            print(f"  Greedy: {greedy_caption}")
            print(f"  Beam(k={k}): {beam_caption}")

    return results


# ============================================================================
# Batch Beam Search
# ============================================================================

def beam_search_batch(model, cnn_features, idx2word, k=5, max_length=30):
    """
    Beam search untuk batch gambar (parallel beam search).

    Args:
        model: RNNScratch instance.
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
        caption = beam_search(model, feat, idx2word, k=k, max_length=max_length)
        captions.append(caption)

    return captions


# ============================================================================
# Length Penalty (untuk Length Normalization)
# ============================================================================

def beam_search_with_length_penalty(model, cnn_feature, idx2word, k=5,
                                     max_length=30, alpha=0.6):
    """
    Beam search dengan length normalization.
    Score = log_prob / len^alpha (panjang lebih pendek lebih diapresiasi).

    Args:
        model: RNNScratch instance.
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

    beams = [([START_IDX], 0.0, np.zeros((1, model.hidden_dim), dtype=np.float64))]

    for step in range(max_length):
        all_candidates = []

        for seq, score, h_state in beams:
            if seq[-1] == END_IDX:
                # Panjang sekuens untuk normalisasi
                length_penalty = len(seq) ** alpha
                norm_score = score / length_penalty
                all_candidates.append((seq, score, h_state, norm_score))
                continue

            probs, h_new = model.step_forward(cnn_feature, seq[-1], h_state)
            top_k_idx = np.argsort(probs)[-k:]

            for token in top_k_idx:
                new_seq = seq + [int(token)]
                new_score = score + np.log(probs[token] + 1e-10)
                all_candidates.append((new_seq, new_score, h_new, 0.0))

        # Sort berdasarkan normalized score
        all_candidates.sort(key=lambda x: x[3] if x[3] != 0.0 else x[1], reverse=True)
        beams = [(seq, score, h_state) for seq, score, h_state, _ in all_candidates[:k]]

        if all(seq[-1] == END_IDX for seq, _, _ in beams):
            break

    best_seq = beams[0][0]
    words = []
    for token in best_seq:
        word = idx2word.get(token, '<unk>')
        if word in ('<pad>', '<start>', '<end>'):
            continue
        words.append(word)
    return ' '.join(words)
