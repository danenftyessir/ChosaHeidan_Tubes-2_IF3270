"""
Init-Inject Architecture untuk LSTM dari nol.
Arsitektur alternatif di mana image feature digabungkan SETELAH LSTM
memproses caption prefix (bukan pre-inject / prepend ke sequence).
Referensi: Tanti et al., 2017 — "Where to put the image in an image captioning model".
Bisa dipakai untuk batch inference dan dibandingkan BLEU-nya dengan pre-inject.
"""

import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))
from dense import Dense
from embedding import Embedding
from lstm_cell import LSTMCell, StackedLSTMCell
from caption_preprocess import END_IDX, START_IDX


# ============================================================================
# LSTM Init-Inject Model
# ============================================================================

class LSTMInitInject:
    """
    LSTM Decoder dengan Init-Inject Architecture.

    Perbedaan dengan Pre-Inject:
        Pre-Inject:  CNN_feature → Dense → x_{-1} → di-prepend ke sequence
                     Input: [x_{-1}, emb_0, emb_1, ..., emb_{T-1}]
        Init-Inject: LSTM memproses sequence embedding TERLEBIH DAHULU
                    SETELAH LSTM selesai, output hidden state di-concatenate
                    dengan CNN_feature → Dense → softmax

    Arsitektur Init-Inject:
        1. x_t = Embedding(token_t) (embed_dim)
        2. LSTM forward: h_T = LSTM(x_0, x_1, ..., x_{T-1}) — TANPA CNN_feature
        3. CNN_feature di-project ke hidden_dim: cnn_proj = Dense(CNN_feature)
        4. combined = [h_T; cnn_proj] — concatenation
        5. output = Dense(combined) → softmax → vocab_size

    Args:
        vocab_size (int): ukuran vocabulary.
        embed_dim (int): dimensi embedding. Default 256.
        hidden_dim (int): dimensi hidden state. Default 512.
        num_layers (int): jumlah layer LSTM. Default 1.
        feature_dim (int): dimensi CNN feature vector. Default 2048.
    """

    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512,
                 num_layers=1, feature_dim=2048):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.feature_dim = feature_dim

        # Layers
        self.embedding = None
        self.lstm = None
        self.cnn_projection = None  # Dense: feature_dim → hidden_dim
        self.output_dense = None    # Dense: 2*hidden_dim → vocab_size

        self._is_built = False

    def build(self):
        """Build model: inisialisasi semua layer."""
        # Embedding layer
        self.embedding = Embedding(self.vocab_size, self.embed_dim)

        # LSTM layers
        if self.num_layers == 1:
            self.lstm = LSTMCell(
                input_dim=self.embed_dim,
                hidden_dim=self.hidden_dim,
                return_sequences=False,
                name="lstm_init_inject"
            )
        else:
            self.lstm = StackedLSTMCell(
                input_dim=self.embed_dim,
                hidden_dims=[self.hidden_dim] * self.num_layers,
                return_sequences=False
            )

        # CNN projection: feature_dim → hidden_dim
        self.cnn_projection = Dense(
            input_dim=self.feature_dim,
            units=self.hidden_dim,
            activation='linear',
            name='cnn_proj'
        )

        # Output: 2 * hidden_dim (LSTM output + CNN proj) → vocab_size
        self.output_dense = Dense(
            input_dim=2 * self.hidden_dim,
            units=self.vocab_size,
            activation='softmax',
            name='output'
        )

        self._is_built = True

    def forward(self, cnn_features, token_seq, training=True):
        """
        Forward pass (teacher forcing) dengan Init-Inject.

        Args:
            cnn_features: (batch_size, feature_dim)
            token_seq: (batch_size, seq_len) — token indices (tanpa <start> prepend)
            training (bool): jika True, return probabilities
        Returns:
            output: (batch_size, vocab_size)
        """
        if not self._is_built:
            self.build()

        batch_size = cnn_features.shape[0]

        # Step 1: Embed token sequence
        embedded = self.embedding.forward(token_seq)

        # Step 2: LSTM forward — TANPA CNN feature di sini
        if self.num_layers == 1:
            self.lstm.reset_cache()
            h_out, _ = self.lstm.forward_sequence(embedded)
        else:
            h_out, _ = self.lstm.forward_sequence(embedded)

        # Step 3: Project CNN feature → hidden_dim
        cnn_proj = self.cnn_projection.forward(cnn_features)

        # Step 4: Concatenate [h_T; cnn_proj]
        combined = np.concatenate([h_out, cnn_proj], axis=1)

        # Step 5: Output Dense → vocab_size
        output = self.output_dense.forward(combined)

        return output

    def forward_batch(self, cnn_features_batch, token_seq_batch, training=True):
        """
        Forward pass untuk batch (alias untuk forward).
        Batch inference: model menerima lebih dari satu input sekaligus.

        Args:
            cnn_features_batch: (batch_size, feature_dim)
            token_seq_batch: (batch_size, seq_len)
            training: bool
        Returns:
            output: (batch_size, vocab_size)
        """
        return self.forward(cnn_features_batch, token_seq_batch, training=training)

    def greedy_decode(self, cnn_feature, idx2word, max_length=30):
        """
        Greedy decoding untuk Init-Inject model.

        Args:
            cnn_feature: (feature_dim,) atau (1, feature_dim)
            idx2word: dict, index → word mapping
            max_length (int): panjang maksimum caption
        Returns:
            str: caption string
        """
        if cnn_feature.ndim == 1:
            cnn_feature = cnn_feature.reshape(1, -1)

        batch_size = 1

        # Project CNN feature ke hidden_dim (sekali, di awal)
        cnn_proj = self.cnn_projection.forward(cnn_feature)[0]

        # LSTM states
        h = np.zeros((batch_size, self.hidden_dim), dtype=np.float64)
        c = np.zeros((batch_size, self.hidden_dim), dtype=np.float64)

        tokens = []
        for step in range(max_length):
            if step == 0:
                prev_token = np.array([[START_IDX]], dtype=np.int32)
            else:
                prev_token = np.array([[tokens[-1]]], dtype=np.int32)

            x_t = self.embedding.forward(prev_token)[:, 0, :]

            # LSTM step
            xh = np.concatenate([x_t, h], axis=1)
            gates = xh @ self.lstm.kernel + h @ self.lstm.recurrent_kernel + self.lstm.bias

            f = 1 / (1 + np.exp(-gates[:, :self.hidden_dim]))
            i = 1 / (1 + np.exp(-gates[:, self.hidden_dim:2*self.hidden_dim]))
            o = 1 / (1 + np.exp(-gates[:, 2*self.hidden_dim:3*self.hidden_dim]))
            g = np.tanh(gates[:, 3*self.hidden_dim:])

            c = f * c + i * g
            h = o * np.tanh(c)

            # Combined [h; cnn_proj]
            combined = np.concatenate([h[0], cnn_proj], axis=0)

            # Output
            logits = combined @ self.output_dense.weights + self.output_dense.bias
            probs = self._softmax(logits)
            next_token = int(np.argmax(probs))

            if next_token == END_IDX:
                break

            tokens.append(next_token)

        words = []
        for token in tokens:
            word = idx2word.get(token, '<unk>')
            if word in ('<pad>', '<start>', '<end>'):
                continue
            words.append(word)

        return ' '.join(words)

    def greedy_decode_batch(self, cnn_features_batch, idx2word, max_length=30):
        """
        Greedy decoding untuk batch gambar sekaligus.

        Args:
            cnn_features_batch: (batch_size, feature_dim)
            idx2word: dict
            max_length (int): panjang maksimum
        Returns:
            list: list caption strings
        """
        if cnn_features_batch.ndim == 1:
            cnn_features_batch = cnn_features_batch.reshape(1, -1)

        batch_size = cnn_features_batch.shape[0]

        # Project CNN features ke hidden_dim
        cnn_projs = self.cnn_projection.forward(cnn_features_batch)

        # LSTM states
        h = np.zeros((batch_size, self.hidden_dim), dtype=np.float64)
        c = np.zeros((batch_size, self.hidden_dim), dtype=np.float64)

        tokens = [[] for _ in range(batch_size)]
        finished = np.zeros(batch_size, dtype=bool)

        for step in range(max_length):
            if step == 0:
                prev_tokens = np.full((batch_size, 1), START_IDX, dtype=np.int32)
            else:
                prev_tokens = np.array([[t[-1] if t else START_IDX]
                                       for t in tokens], dtype=np.int32)

            x_t = self.embedding.forward(prev_tokens)[:, 0, :]

            # LSTM step
            xh = np.concatenate([x_t, h], axis=1)
            gates = xh @ self.lstm.kernel + h @ self.lstm.recurrent_kernel + self.lstm.bias

            f = 1 / (1 + np.exp(-gates[:, :self.hidden_dim]))
            i = 1 / (1 + np.exp(-gates[:, self.hidden_dim:2*self.hidden_dim]))
            o = 1 / (1 + np.exp(-gates[:, 2*self.hidden_dim:3*self.hidden_dim]))
            g = np.tanh(gates[:, 3*self.hidden_dim:])

            c = f * c + i * g
            h = o * np.tanh(c)

            # Output per sample
            for b in range(batch_size):
                if not finished[b]:
                    combined = np.concatenate([h[b], cnn_projs[b]], axis=0)
                    logits = combined @ self.output_dense.weights + self.output_dense.bias
                    probs = self._softmax(logits)
                    next_token = int(np.argmax(probs))

                    if next_token == END_IDX:
                        finished[b] = True
                    else:
                        tokens[b].append(next_token)

            if np.all(finished):
                break

        captions = []
        for batch_tokens in tokens:
            words = []
            for token in batch_tokens:
                word = idx2word.get(token, '<unk>')
                if word in ('<pad>', '<start>', '<end>'):
                    continue
                words.append(word)
            captions.append(' '.join(words))

        return captions

    def _softmax(self, x, axis=-1):
        """Softmax yang aman secara numerik."""
        x_shifted = x - np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    def load_weights_from_h5(self, h5_path):
        """
        Load bobot dari file .h5 hasil pelatihan Keras (Init-Inject).

        Keras weight format untuk LSTM:
            kernel: (input_dim + hidden_dim, 4*hidden_dim) — concatenated gates
            recurrent_kernel: (hidden_dim, 4*hidden_dim)
            bias: (4*hidden_dim,)

        Args:
            h5_path: path ke file .h5
        """
        import h5py

        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"File tidak ditemukan: {h5_path}")

        with h5py.File(h5_path, 'r') as f:
            layer_weights = {}
            for key in f.keys():
                if isinstance(f[key], h5py.Group):
                    layer_weights[key] = {}
                    for subkey in f[key].keys():
                        layer_weights[key][subkey] = np.array(f[key][subkey])

        lstm_weights = None
        dense_weights = None
        cnn_proj_weights = None

        for key, weights in sorted(layer_weights.items()):
            w_keys = list(weights.keys())
            if 'kernel' in w_keys and 'recurrent_kernel' in w_keys:
                kernel = weights['kernel'][:]
                recurrent_kernel = weights['recurrent_kernel'][:]
                bias = weights['bias'][:]
                lstm_weights = (kernel, recurrent_kernel, bias)

            elif 'kernel' in w_keys and 'bias' in w_keys:
                kernel = weights['kernel'][:]
                bias = weights['bias'][:]
                if kernel.shape[0] == 2 * self.hidden_dim:
                    W_out = kernel.T
                    b_out = bias
                    dense_weights = (W_out, b_out)
                elif kernel.shape[1] == self.hidden_dim:
                    W_proj = kernel.T
                    b_proj = bias
                    cnn_proj_weights = (W_proj, b_proj)

        if lstm_weights is not None:
            self.lstm.set_weights(*lstm_weights)

        if dense_weights is not None:
            self.output_dense.set_weights(*dense_weights)

        if cnn_proj_weights is not None:
            self.cnn_projection.set_weights(*cnn_proj_weights)

        self._load_embedding_weights(h5_path)
        print(f"Bobot Init-Inject berhasil dimuat dari: {h5_path}")

    def _load_embedding_weights(self, h5_path):
        """Load embedding weights dari Keras."""
        import h5py

        with h5py.File(h5_path, 'r') as f:
            for key in f.keys():
                if isinstance(f[key], h5py.Group):
                    subkeys = list(f[key].keys())
                    if 'embeddings' in subkeys:
                        embeddings = np.array(f[key]['embeddings'][:])
                        if embeddings.shape == (self.vocab_size, self.embed_dim):
                            self.embedding.set_weights(embeddings)

    def get_config(self):
        """Return konfigurasi model."""
        return {
            'vocab_size': self.vocab_size,
            'embed_dim': self.embed_dim,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'feature_dim': self.feature_dim,
            'architecture': 'init_inject',
        }

    def summary(self):
        """Cetak arsitektur model."""
        config = self.get_config()
        print("LSTMInitInject Architecture (Init-Inject — Tanti et al., 2017):")
        print("=" * 60)
        print(f"  Vocab Size:   {config['vocab_size']}")
        print(f"  Embed Dim:    {config['embed_dim']}")
        print(f"  Hidden Dim:   {config['hidden_dim']}")
        print(f"  Num Layers:   {config['num_layers']}")
        print(f"  Feature Dim:  {config['feature_dim']}")
        print(f"  Architecture: {config['architecture']}")
        print("=" * 60)
        print("  Arsitektur:")
        print("    1. Embedding(token_seq) → LSTM → h_T")
        print("    2. CNN_feature → Dense_proj → hidden_dim")
        print("    3. [h_T; cnn_proj] → Dense → vocab_size")
        print("=" * 60)
        if self.embedding:
            print(f"  Embedding:      {self.embedding.summary()}")
            print(f"  LSTM:           {self.lstm.summary()}")
            print(f"  CNN Projection: {self.cnn_projection.summary()}")
            print(f"  Output Dense:   {self.output_dense.summary()}")
        print("=" * 60)


# ============================================================================
# Helper Functions
# ============================================================================

def build_lstm_initinject_from_config(vocab_size, embed_dim, hidden_dim, num_layers,
                                     feature_dim=2048):
    """
    Helper: bangun LSTMInitInject model dari konfigurasi.

    Args:
        vocab_size (int): ukuran vocabulary
        embed_dim (int): dimensi embedding
        hidden_dim (int): dimensi hidden state
        num_layers (int): jumlah layer LSTM
        feature_dim (int): dimensi CNN feature
    Returns:
        LSTMInitInject instance
    """
    model = LSTMInitInject(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        feature_dim=feature_dim
    )
    model.build()
    return model


def compare_lstm_preinject_vs_initinject(preinject_model, initinject_model,
                                        cnn_features, idx2word,
                                        max_length=30, n_samples=10):
    """
    Bandingkan caption hasil Pre-Inject vs Init-Inject (LSTM).

    Args:
        preinject_model: LSTMScratch instance
        initinject_model: LSTMInitInject instance
        cnn_features: (N, feature_dim)
        idx2word: dict, vocabulary
        max_length (int): panjang maksimum caption
        n_samples (int): jumlah sample untuk dibandingkan
    Returns:
        dict: {preinject_captions, initinject_captions}
    """
    n_samples = min(n_samples, len(cnn_features))
    preinject_captions = []
    initinject_captions = []

    for i in range(n_samples):
        feat = cnn_features[i:i+1]

        pre_caption = preinject_model.greedy_decode(feat, idx2word, max_length)
        init_caption = initinject_model.greedy_decode(feat, idx2word, max_length)

        preinject_captions.append(pre_caption)
        initinject_captions.append(init_caption)

        if i < 5:
            print(f"\n[Gambar {i+1}]")
            print(f"  Pre-Inject:  {pre_caption}")
            print(f"  Init-Inject: {init_caption}")

    return {
        'preinject': preinject_captions,
        'initinject': initinject_captions,
    }


if __name__ == '__main__':
    print("[Test] Membangun LSTMInitInject model...")

    vocab_size = 10000
    embed_dim = 256
    hidden_dim = 512
    feature_dim = 2048

    model = LSTMInitInject(vocab_size, embed_dim, hidden_dim, 1, feature_dim)
    model.build()
    model.summary()

    # Test forward
    batch_size = 4
    seq_len = 10
    cnn_features = np.random.randn(batch_size, feature_dim).astype(np.float64)
    token_seq = np.random.randint(0, vocab_size, size=(batch_size, seq_len))

    output = model.forward(cnn_features, token_seq)
    print(f"\n[Forward Test] Output shape: {output.shape}")
    print(f"  Expected: ({batch_size}, {vocab_size})")

    # Test greedy decode
    single_feature = np.random.randn(feature_dim).astype(np.float64)
    caption = model.greedy_decode(single_feature, {i: f'word{i}' for i in range(vocab_size)})
    print(f"\n[Greedy Decode Test] Caption: {caption[:50]}...")

    print("\n[Test] Selesai.")
