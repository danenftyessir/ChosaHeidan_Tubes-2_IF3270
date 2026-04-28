"""
Lapisan Embedding dari nol (NumPy only).
Token integer → vektor dense.
Mendukung forward pass, backward pass, dan batch inference.
Dapat digunakan oleh RNN dan LSTM.
"""

import numpy as np


class Embedding:
    """
    Lapisan Embedding: memetakan token integer ke vektor dense.

    Forward pass:
        output[b, t] = self.weights[input[b, t]]

    Backward pass:
        dL/dW[d] = sum over t where input[b, t] == d of dout[b, t, :]
        (gradient terhadap embedding weight diakumulasi per token)

    Args:
        vocab_size (int): ukuran vocabulary (jumlah token unik).
        embed_dim (int): dimensi vektor embedding per token.
    """

    def __init__(self, vocab_size, embed_dim):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        # Embedding matrix: shape (vocab_size, embed_dim)
        self.weights = None
        self._init_weights()

        # Cache untuk backward pass
        self.input_cache = None
        self.output_cache = None

    def _init_weights(self, std=0.01):
        """
        Inisialisasi embedding weights dengan random normal.
        Standar deviasi kecil (0.01) untuk stabil di awal training.

        Args:
            std (float): standar deviasi untuk inisialisasi random normal.
        """
        self.weights = np.random.randn(
            self.vocab_size, self.embed_dim
        ).astype(np.float64) * std

    def forward(self, x):
        """
        Forward pass untuk lapisan Embedding.

        Args:
            x: array numpy, bentuk (batch_size, seq_len) atau (seq_len,).
               Berisi integer indices (0 sampai vocab_size-1).
        Returns:
            output: array numpy, bentuk (batch_size, seq_len, embed_dim)
                    atau (seq_len, embed_dim).
        """
        x_orig = x
        if x.ndim == 1:
            x = x.reshape(1, -1)

        batch_size, seq_len = x.shape

        # Foward pass: indexing ke embedding matrix
        # output[b, t, :] = weights[x[b, t], :]
        output = self.weights[x.astype(int), :]

        self.input_cache = x_orig
        self.output_cache = output.copy()

        return output

    def backward(self, dout):
        """
        Backward pass untuk lapisan Embedding.

        Gradient terhadap weights:
            Untuk setiap token d di vocabulary:
                grad_weights[d] = sum_{b, t where x[b,t]==d} dout[b, t, :]

        Args:
            dout: gradient dari layer atas,
                  bentuk (batch_size, seq_len, embed_dim)
                  atau (seq_len, embed_dim).
        Returns:
            dx: gradient terhadap input (jika diperlukan),
                bentuk sama dengan input_cache.
                (biasanya tidak dipakai, tapi untuk konsistensi)
        """
        # Restore batch dimension jika perlu
        input_x = self.input_cache
        if input_x.ndim == 1:
            input_x = input_x.reshape(1, -1)
            dout = dout.reshape(1, -1, self.embed_dim)

        batch_size, seq_len = input_x.shape
        embed_dim = self.embed_dim

        # Inisialisasi gradient weights
        grad_weights = np.zeros_like(self.weights)

        # Flatten batch dan sequence dimension
        x_flat = input_x.reshape(-1)
        dout_flat = dout.reshape(-1, embed_dim)

        # Akumulasi gradient per token
        for i, token_idx in enumerate(x_flat):
            grad_weights[int(token_idx)] += dout_flat[i]

        self._grad_weights = grad_weights

        # Gradient terhadap input (sebenarnya tidak dipakai,
        # tapi untuk chaining)
        dx = np.zeros_like(input_x, dtype=np.float64)

        return dx

    def get_weights(self):
        """
        Kembalikan embedding weights.

        Returns:
            weights: array numpy, bentuk (vocab_size, embed_dim).
        """
        return self.weights

    def set_weights(self, weights):
        """
        Set embedding weights dari sumber eksternal (misal: Keras).

        Args:
            weights: array numpy, bentuk (vocab_size, embed_dim).
        """
        self.weights = weights.astype(np.float64)
        self.vocab_size = weights.shape[0]
        self.embed_dim = weights.shape[1]

    def get_grad_weights(self):
        """
        Kembalikan gradient weights (harus dipanggil setelah backward).

        Returns:
            grad_weights: gradient terhadap embedding weights.
        """
        return self._grad_weights

    def summary(self):
        """Kembalikan info lapisan."""
        return (f"Embedding(vocab_size={self.vocab_size}, "
                f"embed_dim={self.embed_dim})")


class PositionalEmbedding:
    """
    Positional Embedding untuk RNN/LSTM.
    Kombinasi token embedding dan positional encoding.

    Dalam konteks Image Captioning, positional encoding TIDAK terlalu penting
    karena model sudah menangani posisi melalui sequential processing.
    Kelas ini disediakan untuk fleksibilitas.

    Args:
        vocab_size (int): ukuran vocabulary.
        embed_dim (int): dimensi embedding.
        max_seq_len (int): panjang maksimum sequence untuk positional encoding.
    """

    def __init__(self, vocab_size, embed_dim, max_seq_len=100):
        self.token_embedding = Embedding(vocab_size, embed_dim)
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim

        # Positional encoding matrix
        self.pos_encoding = self._build_positional_encoding()

    def _build_positional_encoding(self):
        """
        Bangun positional encoding matrix.
        Menggunakan sinusoidal positional encoding dari Vaswani et al., 2017.

        Returns:
            pos_encoding: array (max_seq_len, embed_dim).
        """
        pos = np.arange(self.max_seq_len)[:, np.newaxis]
        dim = np.arange(self.embed_dim)[np.newaxis, :]

        # Sinusoidal encoding
        angle_rates = 1 / np.power(10000, (2 * (dim // 2)) / self.embed_dim)
        angle_rads = pos * angle_rates

        # Apply sin ke dimensi genap, cos ke dimensi ganjil
        pos_encoding = np.zeros((self.max_seq_len, self.embed_dim))
        pos_encoding[:, 0::2] = np.sin(angle_rads[:, 0::2])
        pos_encoding[:, 1::2] = np.cos(angle_rads[:, 1::2])

        return pos_encoding.astype(np.float64)

    def forward(self, x):
        """
        Forward pass dengan positional encoding.

        Args:
            x: array (batch_size, seq_len) token indices.
        Returns:
            output: array (batch_size, seq_len, embed_dim).
        """
        # Token embedding
        token_emb = self.token_embedding.forward(x)

        # Tambah positional encoding
        batch_size, seq_len = x.shape
        pos_emb = self.pos_encoding[:seq_len, :]
        output = token_emb + pos_emb[np.newaxis, :, :]

        return output

    def backward(self, dout):
        """
        Backward pass.

        Args:
            dout: gradient dari atas.
        Returns:
            dx: gradient terhadap input.
        """
        # Gradient hanya mengalir ke token embedding
        # (positional encoding adalah fixed, tidak trained)
        return self.token_embedding.backward(dout)

    def get_weights(self):
        """Return token embedding weights."""
        return self.token_embedding.get_weights()

    def set_weights(self, weights):
        """Set token embedding weights."""
        self.token_embedding.set_weights(weights)

    def get_grad_weights(self):
        """Return gradient token embedding."""
        return self.token_embedding.get_grad_weights()

    def summary(self):
        """Return info layer."""
        return (f"PositionalEmbedding(vocab_size={self.token_embedding.vocab_size}, "
                f"embed_dim={self.embed_dim}, max_seq_len={self.max_seq_len})")


# ============================================================================
# Pre-trained Embedding Loader
# ============================================================================

def load_glove_embeddings(vocab, glove_path, embed_dim=300):
    """
    Load pre-trained GloVe embeddings untuk vocabulary.

    Args:
        vocab (dict): word2idx dictionary.
        glove_path (str): path ke file GloVe (misal: glove.6B.300d.txt).
        embed_dim (int): dimensi GloVe embedding.
    Returns:
        embedding_matrix: numpy array (vocab_size, embed_dim).
    """
    vocab_size = len(vocab)
    embedding_matrix = np.random.randn(vocab_size, embed_dim).astype(np.float64) * 0.01

    # Special tokens: zero initialization
    embedding_matrix[0:4] = 0.0  # <pad>, <start>, <end>, <unk>

    found = 0
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != embed_dim + 1:
                continue
            word = parts[0]
            if word in vocab:
                idx = vocab[word]
                embedding_matrix[idx] = np.array(parts[1:], dtype=np.float64)
                found += 1

    print(f"[GloVe] Loaded {found}/{vocab_size} words from {glove_path}")
    return embedding_matrix


def load_word2vec_embeddings(vocab, word2vec_path, binary=True):
    """
    Load pre-trained Word2Vec embeddings untuk vocabulary.

    Args:
        vocab (dict): word2idx dictionary.
        word2vec_path (str): path ke file Word2Vec.
        binary (bool): apakah file dalam format binary.
    Returns:
        embedding_matrix: numpy array (vocab_size, embed_dim).
    """
    # Implementasi word2vec loading memerlukan library tambahan
    # Disini kita gunakan gensim jika ada
    try:
        from gensim.models import KeyedVectors
        print(f"[Word2Vec] Loading from {word2vec_path}...")
        w2v = KeyedVectors.load_word2vec_format(word2vec_path, binary=binary)

        vocab_size = len(vocab)
        embed_dim = w2v.vector_size
        embedding_matrix = np.random.randn(vocab_size, embed_dim).astype(np.float64) * 0.01

        embedding_matrix[0:4] = 0.0  # special tokens

        found = 0
        for word, idx in vocab.items():
            if word in w2v:
                embedding_matrix[idx] = w2v[word]
                found += 1

        print(f"[Word2Vec] Loaded {found}/{vocab_size} words")
        return embedding_matrix

    except ImportError:
        print("[Word2Vec] gensim tidak terinstall. "
              "Install dengan: pip install gensim")
        raise


# ============================================================================
# Embedding Utilities
# ============================================================================

def embedding_similarity(embedding_layer, idx1, idx2, metric='cosine'):
    """
    Hitung similarity antara dua embedding vectors.

    Args:
        embedding_layer: instance Embedding.
        idx1, idx2: integer indices token.
        metric (str): 'cosine' atau 'euclidean'.
    Returns:
        similarity: float.
    """
    vec1 = embedding_layer.weights[int(idx1)]
    vec2 = embedding_layer.weights[int(idx2)]

    if metric == 'cosine':
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-10)
    elif metric == 'euclidean':
        return -np.linalg.norm(vec1 - vec2)
    else:
        raise ValueError(f"Unknown metric: {metric}")


def find_nearest_neighbors(embedding_layer, idx, k=10):
    """
    Temukan k token terdekat dengan embedding vector tertentu.

    Args:
        embedding_layer: instance Embedding.
        idx: integer index token query.
        k (int): jumlah neighbors.
    Returns:
        neighbors: list of (token_idx, similarity) tuples.
    """
    query_vec = embedding_layer.weights[int(idx)]
    all_vecs = embedding_layer.weights

    # Cosine similarity
    norms = np.linalg.norm(all_vecs, axis=1)
    query_norm = np.linalg.norm(query_vec)
    similarities = np.dot(all_vecs, query_vec) / (norms * query_norm + 1e-10)

    # Top-k (excluding self)
    similarities[int(idx)] = -1
    top_k_idx = np.argsort(similarities)[-k:][::-1]

    neighbors = [(int(i), float(similarities[i])) for i in top_k_idx]
    return neighbors


def create_embedding_matrix_from_vocab(vocab, embed_dim, std=0.01):
    """
    Helper: buat embedding matrix dari vocabulary dictionary.

    Args:
        vocab (dict): word2idx.
        embed_dim (int): dimensi embedding.
        std (float): std dev untuk random init.
    Returns:
        embedding_matrix: numpy array (vocab_size, embed_dim).
    """
    vocab_size = len(vocab)
    embedding = np.random.randn(vocab_size, embed_dim).astype(np.float64) * std
    embedding[0:4] = 0.0  # special tokens
    return embedding
