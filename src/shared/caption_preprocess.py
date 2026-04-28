"""
Preprocessing untuk Image Captioning: tokenisasi, membangun vocabulary,
padding sequence, dan utilitas preprocessing caption Flickr8k.
Mendukung batch processing.
"""

import json
import re
import os
import numpy as np


# ============================================================================
# Konstanta Special Tokens
# ============================================================================

PAD_TOKEN = '<pad>'
START_TOKEN = '<start>'
END_TOKEN = '<end>'
UNK_TOKEN = '<unk>'

PAD_IDX = 0
START_IDX = 1
END_IDX = 2
UNK_IDX = 3


# ============================================================================
# Fungsi Preprocessing Caption
# ============================================================================

def clean_caption(caption):
    """
    Bersihkan caption: lowercase dan hapus tanda baca.
    Konsisten dengan preprocessing Flickr8k standar.

    Cara kerja:
        1. Lowercase seluruh teks
        2. Hapus tanda baca (kecuali apostrophe dalam kata)
        3. Hapus karakter non-alfabet di luar space
        4. Trim extra whitespace

    Args:
        caption (str): caption asli.
    Returns:
        str: caption yang sudah dibersihkan.
    """
    caption = caption.lower()
    # Hapus tanda baca, simpan apostrophe untuk contraction
    caption = re.sub(r"[^a-z\s']", ' ', caption)
    # Hapus spasi berlebih
    caption = re.sub(r'\s+', ' ', caption).strip()
    return caption


def tokenize_caption(caption):
    """
    Tokenisasi caption menjadi list kata.

    Args:
        caption (str): caption yang sudah dibersihkan.
    Returns:
        list: list token (kata).
    """
    return caption.split()


def split_captions_file(captions_path):
    """
    Parsing file captions.txt Flickr8k.
    Format tiap baris: image_name#0\tcaption_text

    Args:
        captions_path (str): path ke file captions.txt.
    Returns:
        dict: {image_id: [caption_0, caption_1, ..., caption_4]}.
    """
    captions_dict = {}
    with open(captions_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) != 2:
                continue
            img_caption = parts[0]
            caption_text = parts[1]

            # Parse: filename#0 → (filename, caption_num)
            try:
                img_name, cap_num = img_caption.rsplit('#', 1)
                cap_num = int(cap_num)
            except ValueError:
                continue

            if img_name not in captions_dict:
                captions_dict[img_name] = [''] * 5

            captions_dict[img_name][cap_num] = caption_text

    return captions_dict


def build_vocabulary(captions_dict, min_freq=5):
    """
    Bangun vocabulary dari dictionary caption.
    Kata dengan frekuensi < min_freq diabaikan.

    Cara kerja:
        1. Hitung frekuensi setiap kata dari seluruh caption
        2. Kata dengan freq >= min_freq masuk vocabulary
        3. Special tokens: <pad>=0, <start>=1, <end>=2, <unk>=3
        4. Sisanya: index 4+

    Args:
        captions_dict (dict): {image_id: [captions]}.
        min_freq (int): frekuensi minimum kata agar masuk vocab.
    Returns:
        tuple: (word2idx dict, idx2word dict, total word count).
    """
    # Hitung frekuensi kata
    word_freq = {}
    for img_id, captions in captions_dict.items():
        for cap in captions:
            if not cap:
                continue
            cleaned = clean_caption(cap)
            tokens = tokenize_caption(cleaned)
            for word in tokens:
                word_freq[word] = word_freq.get(word, 0) + 1

    # Bangun vocabulary dengan min_freq
    word2idx = {
        PAD_TOKEN: PAD_IDX,
        START_TOKEN: START_IDX,
        END_TOKEN: END_IDX,
        UNK_TOKEN: UNK_IDX,
    }

    idx = 4  # mulai dari 4 karena 0-3 reserved
    for word, freq in sorted(word_freq.items(), key=lambda x: -x[1]):
        if freq >= min_freq:
            word2idx[word] = idx
            idx += 1

    # Bangun idx2word
    idx2word = {int(v): k for k, v in word2idx.items()}

    return word2idx, idx2word, len(word2idx)


def caption_to_sequence(caption, word2idx, max_length=None,
                        add_start=True, add_end=True):
    """
    Konversi caption string ke sequence of indices.

    Args:
        caption (str): caption asli.
        word2idx (dict): word → index mapping.
        max_length (int atau None): panjang maksimal sequence.
            Jika None, tidak ada truncate.
        add_start (bool): tambahkan START_IDX di awal.
        add_end (bool): tambahkan END_IDX di akhir.
    Returns:
        list: sequence of integer indices.
    """
    cleaned = clean_caption(caption)
    tokens = tokenize_caption(cleaned)

    seq = []
    if add_start:
        seq.append(START_IDX)

    for token in tokens:
        seq.append(word2idx.get(token, UNK_IDX))

    if add_end:
        seq.append(END_IDX)

    # Truncate jika perlu
    if max_length is not None:
        if len(seq) > max_length:
            seq = seq[:max_length]
            if seq[-1] != END_IDX:
                seq[-1] = END_IDX

    return seq


def pad_sequence(sequence, max_length, pad_idx=PAD_IDX):
    """
    Pad sequence ke max_length menggunakan pad_idx.

    Args:
        sequence (list): sequence of indices.
        max_length (int): panjang target.
        pad_idx (int): index untuk padding. Default 0 (<pad>).
    Returns:
        np.ndarray: numpy array shape (max_length,).
    """
    seq = list(sequence)
    if len(seq) < max_length:
        seq += [pad_idx] * (max_length - len(seq))
    else:
        seq = seq[:max_length]
    return np.array(seq, dtype=np.int32)


def captions_to_sequences(captions_list, word2idx, max_length,
                          add_start=True, add_end=True):
    """
    Konversi list caption ke numpy array (padded sequences).
    Batch-friendly version.

    Args:
        captions_list (list): list caption string.
        word2idx (dict): word → index mapping.
        max_length (int): panjang maksimal sequence.
        add_start (bool): tambahkan start token.
        add_end (bool): tambahkan end token.
    Returns:
        np.ndarray: shape (N, max_length), dtype int32.
    """
    sequences = []
    for cap in captions_list:
        if not cap:
            seq = []
        else:
            seq = caption_to_sequence(cap, word2idx, max_length,
                                      add_start=add_start, add_end=add_end)
        sequences.append(seq)

    return pad_sequences_numpy(sequences, max_length, pad_idx=PAD_IDX)


def pad_sequences_numpy(sequences, max_length, pad_idx=PAD_IDX):
    """
    Pad list of sequences ke numpy array.

    Args:
        sequences (list): list of sequences (list of ints).
        max_length (int): panjang target.
        pad_idx (int): index untuk padding.
    Returns:
        np.ndarray: shape (N, max_length).
    """
    N = len(sequences)
    result = np.full((N, max_length), pad_idx, dtype=np.int32)

    for i, seq in enumerate(sequences):
        length = min(len(seq), max_length)
        result[i, :length] = seq[:length]

    return result


def sequence_to_caption(sequence, idx2word, remove_special=True):
    """
    Konversi sequence of indices kembali ke caption string.

    Args:
        sequence (list atau np.ndarray): sequence of indices.
        idx2word (dict): index → word mapping.
        remove_special (bool): hapus special tokens dari output.
    Returns:
        str: caption string.
    """
    words = []
    for idx in sequence:
        idx = int(idx)
        if idx == END_IDX or idx == PAD_IDX:
            break
        word = idx2word.get(idx, UNK_TOKEN)
        if remove_special and word in (START_TOKEN, END_TOKEN, PAD_TOKEN):
            continue
        words.append(word)

    return ' '.join(words)


def remove_start_token(sequence):
    """Hapus start token dari sequence jika ada."""
    seq = list(sequence)
    while seq and seq[0] == START_IDX:
        seq.pop(0)
    return seq


def remove_end_and_pad(sequence):
    """Hapus end token dan padding dari sequence."""
    seq = list(sequence)
    while seq and seq[-1] == PAD_IDX:
        seq.pop()
    while seq and seq[-1] == END_IDX:
        seq.pop()
    return seq


# ============================================================================
# Save / Load
# ============================================================================

def save_vocabulary(word2idx, path):
    """
    Simpan vocabulary ke file JSON.

    Args:
        word2idx (dict): word → index mapping.
        path (str): path file output.
    """
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(word2idx, f, ensure_ascii=False, indent=2)
    print(f"Vocabulary disimpan ke: {path} ({len(word2idx)} kata)")


def load_vocabulary(path):
    """
    Load vocabulary dari file JSON.

    Args:
        path (str): path ke file vocabulary JSON.
    Returns:
        tuple: (word2idx, idx2word).
    """
    with open(path, 'r', encoding='utf-8') as f:
        word2idx = json.load(f)
    idx2word = {int(v): k for k, v in word2idx.items()}
    print(f"Vocabulary dimuat dari: {path} ({len(word2idx)} kata)")
    return word2idx, idx2word


def save_captions_clean(captions_dict, captions_clean_path):
    """
    Simpan dictionary captions yang sudah dibersihkan ke JSON.

    Args:
        captions_dict (dict): {image_id: [cleaned_captions]}.
        captions_clean_path (str): path output JSON.
    """
    cleaned = {}
    for img_id, captions in captions_dict.items():
        cleaned[img_id] = [clean_caption(c) if c else '' for c in captions]

    with open(captions_clean_path, 'w', encoding='utf-8') as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)
    print(f"Cleaned captions disimpan ke: {captions_clean_path}")


def load_captions_clean(captions_clean_path):
    """Load cleaned captions dari JSON."""
    with open(captions_clean_path, 'r', encoding='utf-8') as f:
        return json.load(f)


# ============================================================================
# Data Loading Helpers
# ============================================================================

def load_train_val_test_ids(captions_dir):
    """
    Load train/val/test image IDs dari file txt.

    Args:
        captions_dir (str): direktori caption (ada train_ids.txt, val_ids.txt, test_ids.txt).
    Returns:
        tuple: (train_ids, val_ids, test_ids) sebagai list.
    """
    def _load_ids(path):
        if not os.path.exists(path):
            return []
        with open(path, 'r') as f:
            return [line.strip() for line in f if line.strip()]

    train_ids = _load_ids(os.path.join(captions_dir, 'train_ids.txt'))
    val_ids = _load_ids(os.path.join(captions_dir, 'val_ids.txt'))
    test_ids = _load_ids(os.path.join(captions_dir, 'test_ids.txt'))

    return train_ids, val_ids, test_ids


def prepare_training_data(captions_dict, image_ids, word2idx,
                          max_length, add_start=True, add_end=True):
    """
    Siapkan data training dari list image IDs.

    Args:
        captions_dict (dict): {image_id: [captions]}.
        image_ids (list): list image ID yang akan diambil.
        word2idx (dict): vocabulary.
        max_length (int): panjang maksimal sequence.
        add_start (bool): tambahkan start token.
        add_end (bool): tambahkan end token.
    Returns:
        tuple: (image_ids_array, sequences_array).
    """
    all_seqs = []
    matched_ids = []

    for img_id in image_ids:
        if img_id not in captions_dict:
            continue
        caps = captions_dict[img_id]
        for cap in caps:
            if not cap:
                continue
            seq = caption_to_sequence(cap, word2idx, max_length,
                                      add_start=add_start, add_end=add_end)
            all_seqs.append(seq)
            matched_ids.append(img_id)

    seqs_array = pad_sequences_numpy(all_seqs, max_length, pad_idx=PAD_IDX)
    return matched_ids, seqs_array


def get_max_caption_length(captions_dict, min_captions=1):
    """
    Hitung panjang maksimum caption dalam dataset.

    Args:
        captions_dict (dict): {image_id: [captions]}.
        min_captions (int): minimal caption yang ada per gambar.
    Returns:
        int: panjang maksimum.
    """
    max_len = 0
    for img_id, captions in captions_dict.items():
        for cap in captions[:min_captions]:
            if not cap:
                continue
            cleaned = clean_caption(cap)
            tokens = tokenize_caption(cleaned)
            max_len = max(max_len, len(tokens))

    return max_len


# ============================================================================
# Batch Helpers
# ============================================================================

def create_batches(sequences, batch_size, shuffle=True, drop_last=False):
    """
    Buat batch dari sequences.

    Args:
        sequences (np.ndarray): shape (N, seq_len).
        batch_size (int): ukuran batch.
        shuffle (bool): shuffle sebelum batching.
        drop_last (bool): drop batch terakhir jika tidak lengkap.
    Yields:
        np.ndarray: batch array shape (batch_size, seq_len).
    """
    N = sequences.shape[0]
    indices = np.arange(N)
    if shuffle:
        np.random.shuffle(indices)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        if drop_last and end - start < batch_size:
            break
        yield sequences[indices[start:end]]


class CaptionPreprocessor:
    """
    Wrapper class untuk preprocessing caption Flickr8k.
    Menyimpan vocabulary dan max_length untuk reuse.
    """

    def __init__(self, vocab_size=None, min_freq=5):
        self.word2idx = None
        self.idx2word = None
        self.vocab_size = None
        self.max_length = None
        self.min_freq = min_freq
        self.captions_dict = None

    def fit(self, captions_path, max_length=None, min_freq=None):
        """
        Build vocabulary dan prepare data dari file captions.

        Args:
            captions_path (str): path ke captions.txt.
            max_length (int atau None): max length untuk padding.
                Jika None, auto-detect dari data.
            min_freq (int): frekuensi minimum kata.
        """
        if min_freq is not None:
            self.min_freq = min_freq

        print("[CaptionPreprocessor] Memuat captions...")
        self.captions_dict = split_captions_file(captions_path)

        print("[CaptionPreprocessor] Membangun vocabulary...")
        self.word2idx, self.idx2word, self.vocab_size = build_vocabulary(
            self.captions_dict, min_freq=self.min_freq
        )

        if max_length is None:
            print("[CaptionPreprocessor] Auto-detect max_length...")
            raw_max = get_max_caption_length(self.captions_dict)
            self.max_length = raw_max + 2
        else:
            self.max_length = max_length

        print(f"[CaptionPreprocessor] Selesai: vocab_size={self.vocab_size}, "
              f"max_length={self.max_length}")

    def fit_from_clean(self, captions_clean_path, max_length=None, min_freq=None):
        """
        Load dari cleaned captions JSON (lebih cepat).

        Args:
            captions_clean_path (str): path ke captions_clean.json.
            max_length (int atau None): max length.
            min_freq (int): frekuensi minimum kata.
        """
        if min_freq is not None:
            self.min_freq = min_freq

        print("[CaptionPreprocessor] Memuat cleaned captions...")
        with open(captions_clean_path, 'r', encoding='utf-8') as f:
            self.captions_dict = json.load(f)

        print("[CaptionPreprocessor] Membangun vocabulary...")
        self.word2idx, self.idx2word, self.vocab_size = build_vocabulary(
            self.captions_dict, min_freq=self.min_freq
        )

        if max_length is None:
            print("[CaptionPreprocessor] Auto-detect max_length...")
            raw_max = get_max_caption_length(self.captions_dict)
            self.max_length = raw_max + 2
        else:
            self.max_length = max_length

        print(f"[CaptionPreprocessor] Selesai: vocab_size={self.vocab_size}, "
              f"max_length={self.max_length}")

    def transform(self, captions_list):
        """
        Konversi list caption ke padded sequences.

        Args:
            captions_list (list): list caption string.
        Returns:
            np.ndarray: shape (N, max_length).
        """
        return captions_to_sequences(
            captions_list, self.word2idx, self.max_length,
            add_start=True, add_end=True
        )

    def inverse_transform(self, sequences):
        """
        Konversi sequences kembali ke caption strings.

        Args:
            sequences (np.ndarray atau list): shape (N, max_length)
                atau list of indices.
        Returns:
            list: list caption string.
        """
        if isinstance(sequences, np.ndarray) and sequences.ndim == 2:
            return [sequence_to_caption(seq, self.idx2word) for seq in sequences]
        else:
            return [sequence_to_caption(seq, self.idx2word) for seq in sequences]

    def save(self, vocab_path, captions_clean_path=None):
        """Simpan vocabulary dan cleaned captions."""
        save_vocabulary(self.word2idx, vocab_path)
        if captions_clean_path and self.captions_dict:
            save_captions_clean(self.captions_dict, captions_clean_path)

    def load(self, vocab_path, captions_clean_path=None, max_length=None):
        """Load vocabulary dan cleaned captions."""
        self.word2idx, self.idx2word = load_vocabulary(vocab_path)
        self.vocab_size = len(self.word2idx)
        if captions_clean_path:
            self.captions_dict = load_captions_clean(captions_clean_path)
        if max_length is not None:
            self.max_length = max_length