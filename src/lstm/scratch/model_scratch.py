"""
Model Image Captioning dengan LSTM dari nol.
Arsitektur pre-inject (Show and Tell — Vinyals et al., 2015).
Mendukung forward pass, greedy decoding, batch inference, dan load bobot dari Keras.
"""

import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))
from dense import Dense
from embedding import Embedding
from lstm_cell import LSTMCell, StackedLSTMCell


# ============================================================================
# Konstanta
# ============================================================================

START_IDX = 1
END_IDX = 2


# ============================================================================
# LSTMScratch Model
# ============================================================================

class LSTMScratch:
    """
    Model Image Captioning dengan LSTM (pre-inject architecture).

    Arsitektur pre-inject (Show and Tell):
        1. CNN_feature → Dense_proj → x_{-1} (embed_dim)
        2. x_t = Embedding(token_t) (embed_dim)
        3. Sequence input: [x_{-1}, x_0, x_1, ..., x_{T-1}]
        4. LSTM forward: gates → cell state → hidden state
        5. Output: h_{T-1} → Dense → softmax → vocab_size

    Teacher forcing (training):
        Input:  [x_{-1}, emb(<start>), emb(S_0), ..., emb(S_{N-1})]
        Output: probabilities over vocabulary

    Args:
        vocab_size (int): ukuran vocabulary.
        embed_dim (int): dimensi embedding. Default 256.
        hidden_dim (int): dimensi hidden state. Default 512.
        num_layers (int): jumlah layer LSTM. Default 1.
        feature_dim (int): dimensi CNN feature vector (default 2048 untuk InceptionV3).
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
        self.projection = None    # Dense: CNN_feature → embed_dim
        self.lstm = None           # LSTM atau StackedLSTM
        self.output_dense = None   # Dense: hidden_dim → vocab_size

        self._is_built = False

    def build(self):
        """Build model: inisialisasi semua layer."""
        self.embedding = Embedding(self.vocab_size, self.embed_dim)

        self.projection = Dense(
            input_dim=self.feature_dim,
            units=self.embed_dim,
            activation='linear'
        )

        if self.num_layers == 1:
            self.lstm = LSTMCell(
                input_dim=self.embed_dim,
                hidden_dim=self.hidden_dim,
                return_sequences=False,
                name="lstm"
            )
        else:
            self.lstm = StackedLSTMCell(
                input_dim=self.embed_dim,
                hidden_dims=[self.hidden_dim] * self.num_layers,
                return_sequences=False
            )

        self.output_dense = Dense(
            input_dim=self.hidden_dim,
            units=self.vocab_size,
            activation='softmax'
        )

        self._is_built = True

    def forward(self, cnn_features, token_seq, training=True):
        """
        Forward pass (teacher forcing).

        Args:
            cnn_features: (batch_size, feature_dim).
            token_seq: (batch_size, seq_len). Token indices dengan <start> di posisi 0.
            training (bool): jika True, return probabilities.
        Returns:
            output: (batch_size, vocab_size), distribusi probabilitas.
        """
        if not self._is_built:
            self.build()

        batch_size = cnn_features.shape[0]

        # Step 1: Project CNN feature → x_{-1}
        x_start = self.projection.forward(cnn_features)

        # Step 2: Embed token sequence
        embedded = self.embedding.forward(token_seq)

        # Step 3: Prepend x_{-1}
        x_start_expanded = x_start[:, np.newaxis, :]
        combined = np.concatenate([x_start_expanded, embedded], axis=1)

        # Step 4: LSTM forward
        if self.num_layers == 1:
            self.lstm.reset_cache()
            h_out, _ = self.lstm.forward_sequence(combined)
        else:
            h_out, _ = self.lstm.forward_sequence(combined)

        # Step 5: Output Dense → vocab_size
        output = self.output_dense.forward(h_out)

        return output

    def forward_batch(self, cnn_features, token_seq, training=True):
        """
        Forward pass untuk batch.
        Batch inference: model menerima lebih dari satu input sekaligus.

        Args:
            cnn_features: (batch_size, feature_dim)
            token_seq: (batch_size, seq_len)
            training: bool
        Returns:
            output: (batch_size, vocab_size)
        """
        return self.forward(cnn_features, token_seq, training=training)

    def greedy_decode(self, cnn_feature, idx2word, max_length=30):
        """
        Greedy decoding: generate caption satu token per step.

        Args:
            cnn_feature: (feature_dim,) atau (1, feature_dim).
            idx2word: dict, index → word mapping.
            max_length (int): panjang maksimum caption.
        Returns:
            str: caption string.
        """
        if cnn_feature.ndim == 1:
            cnn_feature = cnn_feature.reshape(1, -1)

        batch_size = 1

        # Project CNN feature → x_{-1}
        x_start = self.projection.forward(cnn_feature)

        # Initial states
        h = np.zeros((batch_size, self.hidden_dim), dtype=np.float64)
        c = np.zeros((batch_size, self.hidden_dim), dtype=np.float64)

        tokens = []
        for step in range(max_length):
            if step == 0:
                x_t = x_start
            else:
                prev_token = np.array([[tokens[-1]]], dtype=np.int32)
                x_t = self.embedding.forward(prev_token)[:, 0, :]

            # LSTM gates
            xh = np.concatenate([x_t, h], axis=1)
            gates = xh @ self.lstm.kernel + h @ self.lstm.recurrent_kernel + self.lstm.bias

            f = 1 / (1 + np.exp(-gates[:, :self.hidden_dim]))
            i = 1 / (1 + np.exp(-gates[:, self.hidden_dim:2*self.hidden_dim]))
            o = 1 / (1 + np.exp(-gates[:, 2*self.hidden_dim:3*self.hidden_dim]))
            g = np.tanh(gates[:, 3*self.hidden_dim:])

            c = f * c + i * g
            h = o * np.tanh(c)

            # Output
            logits = h @ self.output_dense.weights + self.output_dense.bias
            probs = self._softmax(logits)
            next_token = int(np.argmax(probs, axis=1)[0])

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

    def greedy_decode_batch(self, cnn_features, idx2word, max_length=30):
        """
        Greedy decoding untuk batch gambar sekaligus.

        Args:
            cnn_features: (batch_size, feature_dim).
            idx2word: dict.
            max_length (int): panjang maksimum.
        Returns:
            list: list caption strings.
        """
        if cnn_features.ndim == 1:
            cnn_features = cnn_features.reshape(1, -1)

        batch_size = cnn_features.shape[0]

        # Project CNN features
        x_start = self.projection.forward(cnn_features)

        # Initial states
        h = np.zeros((batch_size, self.hidden_dim), dtype=np.float64)
        c = np.zeros((batch_size, self.hidden_dim), dtype=np.float64)

        tokens = [[] for _ in range(batch_size)]
        finished = np.zeros(batch_size, dtype=bool)

        for step in range(max_length):
            if step == 0:
                x_t = x_start
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

            # Output
            logits = h @ self.output_dense.weights + self.output_dense.bias
            probs = self._softmax(logits)
            next_tokens = np.argmax(probs, axis=1)

            for b in range(batch_size):
                if not finished[b]:
                    nt = int(next_tokens[b])
                    if nt == END_IDX:
                        finished[b] = True
                    else:
                        tokens[b].append(nt)

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

    def step_forward(self, cnn_feature, prev_token, h_prev, c_prev):
        """
        Single decoding step (untuk beam search).

        Args:
            cnn_feature: (feature_dim,) atau (1, feature_dim)
            prev_token: integer token index
            h_prev: (hidden_dim,) atau (1, hidden_dim)
            c_prev: (hidden_dim,) atau (1, hidden_dim)
        Returns:
            probs: (vocab_size,)
            h_new: (hidden_dim,)
            c_new: (hidden_dim,)
        """
        if cnn_feature.ndim == 1:
            cnn_feature = cnn_feature.reshape(1, -1)
        if h_prev.ndim == 1:
            h_prev = h_prev.reshape(1, -1)
        if c_prev.ndim == 1:
            c_prev = c_prev.reshape(1, -1)

        x_start = self.projection.forward(cnn_feature)
        x_t = self.embedding.forward(np.array([[prev_token]]))[:, 0, :]

        xh = np.concatenate([x_t, h_prev], axis=1)
        gates = xh @ self.lstm.kernel + h_prev @ self.lstm.recurrent_kernel + self.lstm.bias

        f = 1 / (1 + np.exp(-gates[:, :self.hidden_dim]))
        i = 1 / (1 + np.exp(-gates[:, self.hidden_dim:2*self.hidden_dim]))
        o = 1 / (1 + np.exp(-gates[:, 2*self.hidden_dim:3*self.hidden_dim]))
        g = np.tanh(gates[:, 3*self.hidden_dim:])

        c_new = f * c_prev + i * g
        h_new = o * np.tanh(c_new)

        logits = h_new @ self.output_dense.weights + self.output_dense.bias
        probs = self._softmax(logits)[0]

        return probs, h_new[0], c_new[0]

    def load_weights_from_h5(self, h5_path):
        """
        Load bobot dari file .h5 hasil pelatihan Keras.

        Keras LSTM weight format:
            kernel: (input_dim + hidden_dim, 4*hidden_dim) — concatenated gates
            recurrent_kernel: (hidden_dim, 4*hidden_dim) — concatenated gates
            bias: (4*hidden_dim,) — concatenated gate biases

        Args:
            h5_path: path ke file .h5.
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
        proj_weights = None

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
                if kernel.shape[0] == self.hidden_dim:
                    W_out = kernel.T
                    b_out = bias
                    dense_weights = (W_out, b_out)
                elif kernel.shape[1] == self.hidden_dim:
                    W_proj = kernel.T
                    b_proj = bias
                    proj_weights = (W_proj, b_proj)

        if lstm_weights is not None:
            if self.num_layers == 1:
                self.lstm.set_weights(*lstm_weights)
            else:
                self._load_stacked_lstm_weights(h5_path)

        if dense_weights is not None:
            self.output_dense.set_weights(*dense_weights)

        if proj_weights is not None:
            self.projection.set_weights(*proj_weights)

        self._load_embedding_weights(h5_path)
        print(f"Bobot berhasil dimuat dari: {h5_path}")

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

    def _load_stacked_lstm_weights(self, h5_path):
        """Load weights untuk stacked LSTM dari Keras."""
        import h5py

        with h5py.File(h5_path, 'r') as f:
            layer_weights = {}
            for key in sorted(f.keys()):
                if isinstance(f[key], h5py.Group):
                    layer_weights[key] = {}
                    for subkey in f[key].keys():
                        layer_weights[key][subkey] = np.array(f[key][subkey])

        lstm_keys = sorted([k for k in layer_weights.keys()
                          if 'kernel' in layer_weights[k]
                          and 'recurrent_kernel' in layer_weights[k]])

        for i, key in enumerate(lstm_keys[:self.num_layers]):
            w = layer_weights[key]
            kernel = w['kernel'][:]
            recurrent_kernel = w['recurrent_kernel'][:]
            bias = w['bias'][:]
            if i < len(self.lstm.layers):
                self.lstm.layers[i].set_weights(kernel, recurrent_kernel, bias)

    def get_config(self):
        """Return konfigurasi model."""
        return {
            'vocab_size': self.vocab_size,
            'embed_dim': self.embed_dim,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'feature_dim': self.feature_dim,
        }

    def summary(self):
        """Cetak arsitektur model."""
        config = self.get_config()
        print("LSTMScratch Architecture (Pre-Inject — Show and Tell):")
        print("=" * 60)
        print(f"  Vocab Size:   {config['vocab_size']}")
        print(f"  Embed Dim:    {config['embed_dim']}")
        print(f"  Hidden Dim:   {config['hidden_dim']}")
        print(f"  Num Layers:   {config['num_layers']}")
        print(f"  Feature Dim:  {config['feature_dim']}")
        print("=" * 60)
        if self.embedding:
            print(f"  Embedding:    {self.embedding.summary()}")
            print(f"  Projection:   {self.projection.summary()}")
            print(f"  LSTM:         {self.lstm.summary()}")
            print(f"  Output Dense: {self.output_dense.summary()}")
        print("=" * 60)


# ============================================================================
# Helper Functions
# ============================================================================

def build_lstm_from_config(vocab_size, embed_dim, hidden_dim, num_layers,
                          feature_dim=2048):
    """
    Helper: bangun LSTMScratch model dari konfigurasi.

    Args:
        vocab_size (int): ukuran vocabulary.
        embed_dim (int): dimensi embedding.
        hidden_dim (int): dimensi hidden state.
        num_layers (int): jumlah layer LSTM.
        feature_dim (int): dimensi CNN feature.
    Returns:
        LSTMScratch instance.
    """
    model = LSTMScratch(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        feature_dim=feature_dim
    )
    model.build()
    return model