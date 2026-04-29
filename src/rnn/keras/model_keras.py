"""
Arsitektur decoder RNN menggunakan Keras (pre-inject + init-inject).
Arsitektur encoder-decoder untuk Image Captioning.
Pre-inject: CNN feature di-inject sebelum sequence (Show and Tell).
Init-inject: CNN feature di-inject setelah RNN selesai memproses sequence.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))
from caption_preprocess import START_IDX, END_IDX


# ============================================================================
# Pre-Inject Architecture (Baseline)
# ============================================================================

def build_rnn_decoder_preinject(vocab_size, embed_dim=256, hidden_dim=512,
                                  num_layers=1, feature_dim=2048,
                                  seq_max_length=40, dropout=0.3):
    """
    Bangun decoder RNN dengan arsitektur Pre-Inject.

    Arsitektur Pre-Inject (Show and Tell — Vinyals et al., 2015):
        - CNN_feature → Dense(projection) → embed_dim → x_{-1}
        - x_t = Embedding(token_t) → embed_dim
        - Sequence: [x_{-1}, emb(<start>), emb(S_0), ..., emb(S_{N-1})]
        - RNN forward: h_t = tanh(W_xh @ x_t + W_hh @ h_{t-1} + b)
        - Output: h_{T-1} → Dense → softmax → vocab_size

    Keras Functional API:
        1. Input: CNN_features (feature_dim,) dan caption_tokens (seq_len,)
        2. Projection: Dense(feature_dim → embed_dim)
        3. Embedding: token → embed_dim vector
        4. Concatenate: [x_{-1}, embedding_sequence]
        5. RNN: SimpleRNN dengan state from CNN
        6. Output: Dense(hidden_dim → vocab_size, softmax)

    Args:
        vocab_size (int): ukuran vocabulary.
        embed_dim (int): dimensi embedding. Default 256.
        hidden_dim (int): dimensi hidden state. Default 512.
        num_layers (int): jumlah layer RNN. Default 1.
        feature_dim (int): dimensi CNN feature vector. Default 2048.
        seq_max_length (int): panjang maksimal sequence. Default 40.
        dropout (float): dropout rate. Default 0.3.
    Returns:
        Keras Model instance.
    """
    try:
        import tensorflow as tf
        from tensorflow.keras import Model, Input
        from tensorflow.keras.layers import (
            Embedding, Dense, Concatenate, SimpleRNN,
            Dropout, LSTM, Bidirectional
        )
    except ImportError:
        raise ImportError("TensorFlow diperlukan untuk Keras RNN model")

    # Inputs
    cnn_input = Input(shape=(feature_dim,), name='cnn_features')
    caption_input = Input(shape=(seq_max_length,), name='caption_tokens')

    # Step 1: Project CNN feature → embed_dim
    # CNN_feature (feature_dim,) → Dense(embed_dim)
    x_start = Dense(embed_dim, activation='linear', name='cnn_projection')(cnn_input)
    x_start = Dropout(dropout)(x_start)

    # Step 2: Embedding untuk caption tokens
    # tokens (seq_len,) → embed vectors (seq_len, embed_dim)
    embeddings = Embedding(
        input_dim=vocab_size,
        output_dim=embed_dim,
        mask_zero=False,  # mask_zero=True tidak kompatibel dengan RNN custom
        name='token_embedding'
    )(caption_input)

    # Step 3: Prepend x_{-1} ke sequence embedding
    # embeddings[:, 0, :] = emb(<start>)
    # combined = [x_{-1}; emb(<start>); emb(S_0); ...]
    x_start_expanded = tf.expand_dims(x_start, axis=1)  # (batch, 1, embed_dim)
    combined = Concatenate(axis=1)([x_start_expanded, embeddings])
    # combined shape: (batch, seq_len + 1, embed_dim)

    # Step 4: RNN forward
    # Hidden state diinisialisasi dari CNN projection (seperti pre-inject standar)
    if num_layers == 1:
        rnn_out = SimpleRNN(
            hidden_dim,
            return_sequences=False,  # hanya last hidden state
            dropout=dropout,
            recurrent_dropout=dropout,
            name='rnn_decoder'
        )(combined)
    else:
        # Stacked RNN
        x = combined
        for i in range(num_layers):
            return_seq = (i < num_layers - 1)
            x = SimpleRNN(
                hidden_dim,
                return_sequences=return_seq,
                dropout=dropout,
                recurrent_dropout=dropout,
                name=f'rnn_layer_{i}'
            )(x)
        rnn_out = x

    rnn_out = Dropout(dropout)(rnn_out)

    # Step 5: Output Dense → vocab_size
    output = Dense(vocab_size, activation='softmax', name='output')(rnn_out)

    model = Model(inputs=[cnn_input, caption_input], outputs=output, name='rnn_preinject')
    return model


def build_rnn_decoder_preinject_v2(vocab_size, embed_dim=256, hidden_dim=512,
                                     num_layers=1, feature_dim=2048,
                                     seq_max_length=40, dropout=0.3):
    """
    Alternative pre-inject architecture dengan separate RNN initial state.

    Sama seperti build_rnn_decoder_preinject tapi dengan initial state
    dari CNN feature (bukan concatenate di level input).

    Args:
        vocab_size (int): ukuran vocabulary.
        embed_dim (int): dimensi embedding.
        hidden_dim (int): dimensi hidden state.
        num_layers (int): jumlah layer RNN.
        feature_dim (int): dimensi CNN feature.
        seq_max_length (int): panjang sequence.
        dropout (float): dropout rate.
    Returns:
        Keras Model instance.
    """
    try:
        from tensorflow.keras import Model, Input
        from tensorflow.keras.layers import (
            Embedding, Dense, Concatenate, SimpleRNN, Dropout
        )
    except ImportError:
        raise ImportError("TensorFlow diperlukan")

    # Inputs
    cnn_input = Input(shape=(feature_dim,), name='cnn_features')
    caption_input = Input(shape=(seq_max_length,), name='caption_tokens')

    # Embedding saja (tanpa prepend CNN feature)
    embeddings = Embedding(
        input_dim=vocab_size,
        output_dim=embed_dim,
        name='token_embedding'
    )(caption_input)

    # CNN feature → initial hidden state via Dense
    # h_0 = Dense(CNN_feature)
    initial_h = Dense(hidden_dim, activation='tanh', name='init_h')(cnn_input)

    # RNN dengan initial state
    if num_layers == 1:
        rnn_out = SimpleRNN(
            hidden_dim,
            return_sequences=False,
            dropout=dropout,
            recurrent_dropout=dropout,
            name='rnn_decoder'
        )(embeddings, initial_state=[initial_h, initial_h])
    else:
        # Stacked RNN
        x = embeddings
        h_state = initial_h
        for i in range(num_layers):
            return_seq = (i < num_layers - 1)
            if i == 0:
                x = SimpleRNN(
                    hidden_dim,
                    return_sequences=return_seq,
                    dropout=dropout,
                    recurrent_dropout=dropout,
                    name=f'rnn_layer_{i}'
                )(x, initial_state=[h_state, h_state])
            else:
                x = SimpleRNN(
                    hidden_dim,
                    return_sequences=return_seq,
                    dropout=dropout,
                    recurrent_dropout=dropout,
                    name=f'rnn_layer_{i}'
                )(x)
        rnn_out = x

    rnn_out = Dropout(dropout)(rnn_out)
    output = Dense(vocab_size, activation='softmax', name='output')(rnn_out)

    model = Model(inputs=[cnn_input, caption_input], outputs=output, name='rnn_preinject_v2')
    return model


# ============================================================================
# Init-Inject Architecture (Bonus)
# ============================================================================

def build_rnn_decoder_initinject(vocab_size, embed_dim=256, hidden_dim=512,
                                  num_layers=1, feature_dim=2048,
                                  seq_max_length=40, dropout=0.3):
    """
    Bangun decoder RNN dengan arsitektur Init-Inject.

    Arsitektur Init-Inject (Tanti et al., 2017):
        - RNN memproses caption sequence TERLEBIH DAHULU
        - CNN feature di-inject SETELAH RNN selesai (bukan sebelum)
        - Gabungan [h_T; CNN_feature] → Dense → vocab_size

    Perbedaan dengan Pre-Inject:
        Pre-Inject: CNN feature di concat SEBELUM sequence input (sebagai x_{-1})
        Init-Inject: RNN output di concat dengan CNN feature SETELAH forward

    Args:
        vocab_size (int): ukuran vocabulary.
        embed_dim (int): dimensi embedding.
        hidden_dim (int): dimensi hidden state.
        num_layers (int): jumlah layer RNN.
        feature_dim (int): dimensi CNN feature.
        seq_max_length (int): panjang sequence.
        dropout (float): dropout rate.
    Returns:
        Keras Model instance.
    """
    try:
        from tensorflow.keras import Model, Input
        from tensorflow.keras.layers import (
            Embedding, Dense, Concatenate, SimpleRNN, Dropout
        )
    except ImportError:
        raise ImportError("TensorFlow diperlukan")

    # Inputs
    cnn_input = Input(shape=(feature_dim,), name='cnn_features')
    caption_input = Input(shape=(seq_max_length,), name='caption_tokens')

    # Embedding untuk caption (TIDAK ada prepend CNN feature)
    embeddings = Embedding(
        input_dim=vocab_size,
        output_dim=embed_dim,
        name='token_embedding'
    )(caption_input)

    # RNN forward: h_T = RNN(emb_0, emb_1, ..., emb_{T-1})
    # Tidak ada CNN feature di sini
    if num_layers == 1:
        rnn_out = SimpleRNN(
            hidden_dim,
            return_sequences=False,
            dropout=dropout,
            recurrent_dropout=dropout,
            name='rnn_decoder'
        )(embeddings)
    else:
        x = embeddings
        for i in range(num_layers):
            return_seq = (i < num_layers - 1)
            x = SimpleRNN(
                hidden_dim,
                return_sequences=return_seq,
                dropout=dropout,
                recurrent_dropout=dropout,
                name=f'rnn_layer_{i}'
            )(x)
        rnn_out = x

    # Project CNN feature ke hidden_dim (untuk concatenation)
    cnn_projected = Dense(hidden_dim, activation='linear', name='cnn_proj')(cnn_input)

    # INIT-INJECT: Gabungan [h_T; CNN_feature]
    # Ini adalah perbedaan utama dari pre-inject
    combined = Concatenate()([rnn_out, cnn_projected])

    # Combined → Dense → vocab_size
    combined = Dropout(dropout)(combined)
    output = Dense(vocab_size, activation='softmax', name='output')(combined)

    model = Model(inputs=[cnn_input, caption_input], outputs=output, name='rnn_initinject')
    return model


# ============================================================================
# Bidirectional RNN (Optional)
# ============================================================================

def build_rnn_decoder_bidirectional(vocab_size, embed_dim=256, hidden_dim=256,
                                     num_layers=1, feature_dim=2048,
                                     seq_max_length=40, dropout=0.3):
    """
    Bangun decoder RNN dengan Bidirectional SimpleRNN.
    Tidak standar untuk image captioning, tapi sebagai eksperimen.

    Args:
        vocab_size (int): ukuran vocabulary.
        embed_dim (int): dimensi embedding.
        hidden_dim (int): dimensi hidden state per direction.
        num_layers (int): jumlah layer RNN.
        feature_dim (int): dimensi CNN feature.
        seq_max_length (int): panjang sequence.
        dropout (float): dropout rate.
    Returns:
        Keras Model instance.
    """
    try:
        from tensorflow.keras import Model, Input
        from tensorflow.keras.layers import (
            Embedding, Dense, Concatenate, SimpleRNN, Dropout, Bidirectional
        )
    except ImportError:
        raise ImportError("TensorFlow diperlukan")

    cnn_input = Input(shape=(feature_dim,), name='cnn_features')
    caption_input = Input(shape=(seq_max_length,), name='caption_tokens')

    x_start = Dense(embed_dim, activation='linear', name='cnn_projection')(cnn_input)
    x_start_expanded = tf.expand_dims(x_start, axis=1)

    embeddings = Embedding(
        input_dim=vocab_size,
        output_dim=embed_dim,
        name='token_embedding'
    )(caption_input)

    combined = Concatenate(axis=1)([x_start_expanded, embeddings])

    if num_layers == 1:
        rnn_out = Bidirectional(
            SimpleRNN(hidden_dim, return_sequences=False, dropout=dropout),
            name='bidirectional_rnn'
        )(combined)
        rnn_dim = hidden_dim * 2
    else:
        x = combined
        for i in range(num_layers):
            return_seq = (i < num_layers - 1)
            x = Bidirectional(
                SimpleRNN(hidden_dim, return_sequences=return_seq, dropout=dropout),
                name=f'bidir_layer_{i}'
            )(x)
        rnn_out = x
        rnn_dim = hidden_dim * 2

    rnn_out = Dropout(dropout)(rnn_out)
    output = Dense(vocab_size, activation='softmax', name='output')(rnn_out)

    model = Model(inputs=[cnn_input, caption_input], outputs=output, name='rnn_bidirectional')
    return model


# ============================================================================
# Factory Function
# ============================================================================

def build_rnn_model(vocab_size, embed_dim=256, hidden_dim=512,
                    num_layers=1, feature_dim=2048,
                    seq_max_length=40, architecture='preinject',
                    dropout=0.3):
    """
    Factory function untuk membangun RNN decoder dengan berbagai arsitektur.

    Args:
        vocab_size (int): ukuran vocabulary.
        embed_dim (int): dimensi embedding.
        hidden_dim (int): dimensi hidden state.
        num_layers (int): jumlah layer RNN.
        feature_dim (int): dimensi CNN feature.
        seq_max_length (int): panjang sequence.
        architecture (str): 'preinject', 'preinject_v2', atau 'initinject'.
        dropout (float): dropout rate.
    Returns:
        Keras Model instance.
    """
    if architecture == 'preinject':
        return build_rnn_decoder_preinject(
            vocab_size, embed_dim, hidden_dim, num_layers,
            feature_dim, seq_max_length, dropout
        )
    elif architecture == 'preinject_v2':
        return build_rnn_decoder_preinject_v2(
            vocab_size, embed_dim, hidden_dim, num_layers,
            feature_dim, seq_max_length, dropout
        )
    elif architecture == 'initinject':
        return build_rnn_decoder_initinject(
            vocab_size, embed_dim, hidden_dim, num_layers,
            feature_dim, seq_max_length, dropout
        )
    elif architecture == 'bidirectional':
        return build_rnn_decoder_bidirectional(
            vocab_size, embed_dim, hidden_dim, num_layers,
            feature_dim, seq_max_length, dropout
        )
    else:
        raise ValueError(f"Unknown architecture: {architecture}. "
                        "Pilih: 'preinject', 'preinject_v2', 'initinject', 'bidirectional'")


def print_model_summary(model, name="RNN Decoder"):
    """Cetak ringkasan model."""
    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"{'='*60}")
    model.summary()
    print(f"{'='*60}\n")


# ============================================================================
# Preprocessing Helpers
# ============================================================================

def prepare_rnn_inputs(cnn_features, token_sequences, start_idx=START_IDX,
                       end_idx=END_IDX, pad_idx=0):
    """
    Siapkan input untuk training RNN decoder.

    Format:
        Input[0]: CNN features (N, feature_dim)
        Input[1]: Token sequences dengan start token di depan (N, seq_len)

    Args:
        cnn_features (np.ndarray): CNN feature vectors, shape (N, feature_dim).
        token_sequences (np.ndarray): Token sequences, shape (N, seq_len).
            Sudah termasuk <start> token di depan.
        start_idx (int): index untuk <start> token.
        end_idx (int): index untuk <end> token.
        pad_idx (int): index untuk <pad> token.
    Returns:
        tuple: ([cnn_features, input_tokens], output_tokens)
    """
    import tensorflow as tf

    # Input tokens: token sequences (sudah dengan <start> di posisi 0)
    input_tokens = token_sequences

    # Output tokens: shifted right (target adalah token berikutnya)
    # Teacher forcing: output adalah [S_0, S_1, ..., S_N]
    # Di Keras, output biasanya adalah token_sequences[:, 1:] (shifted)
    # Tapi karena sudah ada <start> di [0], kita shift:
    # input[0] = <start>, output[0] = S_0
    output_tokens = token_sequences  # Keras handle shift internally dengan SparseCE

    return [cnn_features, input_tokens], output_tokens


# ============================================================================
# Test Model
# ============================================================================

if __name__ == '__main__':
    # Test model building
    print("[Test] Membangun RNN decoder models...")

    vocab_size = 10000
    embed_dim = 256
    hidden_dim = 512
    feature_dim = 2048
    seq_max = 40

    # Pre-inject
    model_pre = build_rnn_decoder_preinject(vocab_size, embed_dim, hidden_dim, 1, feature_dim, seq_max)
    print_model_summary(model_pre, "RNN Pre-Inject")

    # Init-inject
    model_init = build_rnn_decoder_initinject(vocab_size, embed_dim, hidden_dim, 1, feature_dim, seq_max)
    print_model_summary(model_init, "RNN Init-Inject")

    print("[Test] Selesai. Model dapat dicompile dengan:")
    print("  model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')")