"""
Arsitektur decoder LSTM menggunakan Keras (pre-inject + init-inject).
Arsitektur encoder-decoder untuk Image Captioning.
Pre-inject: CNN feature di-inject sebelum sequence (Show and Tell).
Init-inject: CNN feature di-inject setelah LSTM selesai memproses sequence.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))
from caption_preprocess import START_IDX, END_IDX


# ============================================================================
# Pre-Inject Architecture (Baseline)
# ============================================================================

def build_lstm_decoder_preinject(vocab_size, embed_dim=256, hidden_dim=512,
                                  num_layers=1, feature_dim=2048,
                                  seq_max_length=40, dropout=0.3):
    """
    Bangun decoder LSTM dengan arsitektur Pre-Inject.

    Arsitektur Pre-Inject (Show and Tell — Vinyals et al., 2015):
        - CNN_feature → Dense(projection) → embed_dim → x_{-1}
        - x_t = Embedding(token_t) → embed_dim
        - Sequence: [x_{-1}, emb(<start>), emb(S_0), ..., emb(S_{N-1})]
        - LSTM forward: gates → cell state → hidden state
        - Output: h_{T-1} → Dense → softmax → vocab_size

    Keras Functional API:
        1. Input: CNN_features dan caption_tokens
        2. Projection: Dense(feature_dim → embed_dim)
        3. Embedding: token → embed_dim
        4. Concatenate: [x_{-1}, embedding_sequence]
        5. LSTM: hidden_dim
        6. Output: Dense(hidden_dim → vocab_size, softmax)

    Args:
        vocab_size (int): ukuran vocabulary.
        embed_dim (int): dimensi embedding. Default 256.
        hidden_dim (int): dimensi hidden state. Default 512.
        num_layers (int): jumlah layer LSTM. Default 1.
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
            Embedding, Dense, Concatenate, LSTM, Dropout
        )
    except ImportError:
        raise ImportError("TensorFlow diperlukan untuk Keras LSTM model")

    # Inputs
    cnn_input = Input(shape=(feature_dim,), name='cnn_features')
    caption_input = Input(shape=(seq_max_length,), name='caption_tokens')

    # Step 1: Project CNN feature → embed_dim
    x_start = Dense(embed_dim, activation='linear', name='cnn_projection')(cnn_input)
    x_start = Dropout(dropout)(x_start)

    # Step 2: Embedding untuk caption tokens
    embeddings = Embedding(
        input_dim=vocab_size,
        output_dim=embed_dim,
        mask_zero=False,
        name='token_embedding'
    )(caption_input)

    # Step 3: Prepend x_{-1} ke sequence embedding
    x_start_expanded = tf.expand_dims(x_start, axis=1)
    combined = Concatenate(axis=1)([x_start_expanded, embeddings])
    # combined shape: (batch, seq_len + 1, embed_dim)

    # Step 4: LSTM forward
    if num_layers == 1:
        lstm_out = LSTM(
            hidden_dim,
            return_sequences=False,
            dropout=dropout,
            recurrent_dropout=dropout,
            name='lstm_decoder'
        )(combined)
    else:
        x = combined
        for i in range(num_layers):
            return_seq = (i < num_layers - 1)
            x = LSTM(
                hidden_dim,
                return_sequences=return_seq,
                dropout=dropout,
                recurrent_dropout=dropout,
                name=f'lstm_layer_{i}'
            )(x)
        lstm_out = x

    lstm_out = Dropout(dropout)(lstm_out)

    # Step 5: Output Dense → vocab_size
    output = Dense(vocab_size, activation='softmax', name='output')(lstm_out)

    model = Model(inputs=[cnn_input, caption_input], outputs=output, name='lstm_preinject')
    return model


def build_lstm_decoder_preinject_v2(vocab_size, embed_dim=256, hidden_dim=512,
                                     num_layers=1, feature_dim=2048,
                                     seq_max_length=40, dropout=0.3):
    """
    Alternative pre-inject architecture dengan separate LSTM initial state.

    Args:
        vocab_size (int): ukuran vocabulary.
        embed_dim (int): dimensi embedding.
        hidden_dim (int): dimensi hidden state.
        num_layers (int): jumlah layer LSTM.
        feature_dim (int): dimensi CNN feature.
        seq_max_length (int): panjang sequence.
        dropout (float): dropout rate.
    Returns:
        Keras Model instance.
    """
    try:
        from tensorflow.keras import Model, Input
        from tensorflow.keras.layers import (
            Embedding, Dense, LSTM, Dropout
        )
    except ImportError:
        raise ImportError("TensorFlow diperlukan")

    cnn_input = Input(shape=(feature_dim,), name='cnn_features')
    caption_input = Input(shape=(seq_max_length,), name='caption_tokens')

    # Embedding saja (tanpa prepend CNN feature)
    embeddings = Embedding(
        input_dim=vocab_size,
        output_dim=embed_dim,
        name='token_embedding'
    )(caption_input)

    # CNN feature → initial hidden state
    initial_h = Dense(hidden_dim, activation='tanh', name='init_h')(cnn_input)
    initial_c = Dense(hidden_dim, activation='tanh', name='init_c')(cnn_input)

    # LSTM dengan initial state
    if num_layers == 1:
        lstm_out = LSTM(
            hidden_dim,
            return_sequences=False,
            dropout=dropout,
            recurrent_dropout=dropout,
            name='lstm_decoder'
        )(embeddings, initial_state=[initial_h, initial_c])
    else:
        x = embeddings
        h_state = initial_h
        c_state = initial_c
        for i in range(num_layers):
            return_seq = (i < num_layers - 1)
            if i == 0:
                x = LSTM(
                    hidden_dim,
                    return_sequences=return_seq,
                    dropout=dropout,
                    recurrent_dropout=dropout,
                    name=f'lstm_layer_{i}'
                )(x, initial_state=[h_state, c_state])
            else:
                x = LSTM(
                    hidden_dim,
                    return_sequences=return_seq,
                    dropout=dropout,
                    recurrent_dropout=dropout,
                    name=f'lstm_layer_{i}'
                )(x)
        lstm_out = x

    lstm_out = Dropout(dropout)(lstm_out)
    output = Dense(vocab_size, activation='softmax', name='output')(lstm_out)

    model = Model(inputs=[cnn_input, caption_input], outputs=output, name='lstm_preinject_v2')
    return model


# ============================================================================
# Init-Inject Architecture (Bonus)
# ============================================================================

def build_lstm_decoder_initinject(vocab_size, embed_dim=256, hidden_dim=512,
                                  num_layers=1, feature_dim=2048,
                                  seq_max_length=40, dropout=0.3):
    """
    Bangun decoder LSTM dengan arsitektur Init-Inject.

    Arsitektur Init-Inject (Tanti et al., 2017):
        - LSTM memproses caption sequence TERLEBIH DAHULU
        - CNN feature di-inject SETELAH LSTM selesai
        - Gabungan [h_T; CNN_feature] → Dense → vocab_size

    Perbedaan dengan Pre-Inject:
        Pre-Inject: CNN feature di concat SEBELUM sequence input
        Init-Inject: LSTM output di concat dengan CNN feature SETELAH forward

    Args:
        vocab_size (int): ukuran vocabulary.
        embed_dim (int): dimensi embedding.
        hidden_dim (int): dimensi hidden state.
        num_layers (int): jumlah layer LSTM.
        feature_dim (int): dimensi CNN feature.
        seq_max_length (int): panjang sequence.
        dropout (float): dropout rate.
    Returns:
        Keras Model instance.
    """
    try:
        from tensorflow.keras import Model, Input
        from tensorflow.keras.layers import (
            Embedding, Dense, Concatenate, LSTM, Dropout
        )
    except ImportError:
        raise ImportError("TensorFlow diperlukan")

    cnn_input = Input(shape=(feature_dim,), name='cnn_features')
    caption_input = Input(shape=(seq_max_length,), name='caption_tokens')

    # Embedding untuk caption (TIDAK ada prepend CNN feature)
    embeddings = Embedding(
        input_dim=vocab_size,
        output_dim=embed_dim,
        name='token_embedding'
    )(caption_input)

    # LSTM forward: h_T = LSTM(emb_0, emb_1, ..., emb_{T-1})
    # Tidak ada CNN feature di sini
    if num_layers == 1:
        lstm_out = LSTM(
            hidden_dim,
            return_sequences=False,
            dropout=dropout,
            recurrent_dropout=dropout,
            name='lstm_decoder'
        )(embeddings)
    else:
        x = embeddings
        for i in range(num_layers):
            return_seq = (i < num_layers - 1)
            x = LSTM(
                hidden_dim,
                return_sequences=return_seq,
                dropout=dropout,
                recurrent_dropout=dropout,
                name=f'lstm_layer_{i}'
            )(x)
        lstm_out = x

    # Project CNN feature ke hidden_dim
    cnn_projected = Dense(hidden_dim, activation='linear', name='cnn_proj')(cnn_input)

    # INIT-INJECT: Gabungan [h_T; CNN_feature]
    combined = Concatenate()([lstm_out, cnn_projected])

    combined = Dropout(dropout)(combined)
    output = Dense(vocab_size, activation='softmax', name='output')(combined)

    model = Model(inputs=[cnn_input, caption_input], outputs=output, name='lstm_initinject')
    return model


# ============================================================================
# Bidirectional LSTM (Optional)
# ============================================================================

def build_lstm_decoder_bidirectional(vocab_size, embed_dim=256, hidden_dim=256,
                                     num_layers=1, feature_dim=2048,
                                     seq_max_length=40, dropout=0.3):
    """
    Bangun decoder LSTM dengan Bidirectional LSTM.

    Args:
        vocab_size (int): ukuran vocabulary.
        embed_dim (int): dimensi embedding.
        hidden_dim (int): dimensi hidden state per direction.
        num_layers (int): jumlah layer LSTM.
        feature_dim (int): dimensi CNN feature.
        seq_max_length (int): panjang sequence.
        dropout (float): dropout rate.
    Returns:
        Keras Model instance.
    """
    try:
        import tensorflow as tf
        from tensorflow.keras import Model, Input
        from tensorflow.keras.layers import (
            Embedding, Dense, Concatenate, LSTM, Dropout, Bidirectional
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
        lstm_out = Bidirectional(
            LSTM(hidden_dim, return_sequences=False, dropout=dropout),
            name='bidirectional_lstm'
        )(combined)
        lstm_dim = hidden_dim * 2
    else:
        x = combined
        for i in range(num_layers):
            return_seq = (i < num_layers - 1)
            x = Bidirectional(
                LSTM(hidden_dim, return_sequences=return_seq, dropout=dropout),
                name=f'bidir_layer_{i}'
            )(x)
        lstm_out = x
        lstm_dim = hidden_dim * 2

    lstm_out = Dropout(dropout)(lstm_out)
    output = Dense(vocab_size, activation='softmax', name='output')(lstm_out)

    model = Model(inputs=[cnn_input, caption_input], outputs=output, name='lstm_bidirectional')
    return model


# ============================================================================
# Factory Function
# ============================================================================

def build_lstm_model(vocab_size, embed_dim=256, hidden_dim=512,
                     num_layers=1, feature_dim=2048,
                     seq_max_length=40, architecture='preinject',
                     dropout=0.3):
    """
    Factory function untuk membangun LSTM decoder dengan berbagai arsitektur.

    Args:
        vocab_size (int): ukuran vocabulary.
        embed_dim (int): dimensi embedding.
        hidden_dim (int): dimensi hidden state.
        num_layers (int): jumlah layer LSTM.
        feature_dim (int): dimensi CNN feature.
        seq_max_length (int): panjang sequence.
        architecture (str): 'preinject', 'preinject_v2', 'initinject', 'bidirectional'.
        dropout (float): dropout rate.
    Returns:
        Keras Model instance.
    """
    if architecture == 'preinject':
        return build_lstm_decoder_preinject(
            vocab_size, embed_dim, hidden_dim, num_layers,
            feature_dim, seq_max_length, dropout
        )
    elif architecture == 'preinject_v2':
        return build_lstm_decoder_preinject_v2(
            vocab_size, embed_dim, hidden_dim, num_layers,
            feature_dim, seq_max_length, dropout
        )
    elif architecture == 'initinject':
        return build_lstm_decoder_initinject(
            vocab_size, embed_dim, hidden_dim, num_layers,
            feature_dim, seq_max_length, dropout
        )
    elif architecture == 'bidirectional':
        return build_lstm_decoder_bidirectional(
            vocab_size, embed_dim, hidden_dim, num_layers,
            feature_dim, seq_max_length, dropout
        )
    else:
        raise ValueError(f"Unknown architecture: {architecture}. "
                        "Pilih: 'preinject', 'preinject_v2', 'initinject', 'bidirectional'")


def print_model_summary(model, name="LSTM Decoder"):
    """Cetak ringkasan model."""
    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"{'='*60}")
    model.summary()
    print(f"{'='*60}\n")


# ============================================================================
# Test Model
# ============================================================================

if __name__ == '__main__':
    print("[Test] Membangun LSTM decoder models...")

    vocab_size = 10000
    embed_dim = 256
    hidden_dim = 512
    feature_dim = 2048
    seq_max = 40

    # Pre-inject
    model_pre = build_lstm_decoder_preinject(vocab_size, embed_dim, hidden_dim, 1, feature_dim, seq_max)
    print_model_summary(model_pre, "LSTM Pre-Inject")

    # Init-inject
    model_init = build_lstm_decoder_initinject(vocab_size, embed_dim, hidden_dim, 1, feature_dim, seq_max)
    print_model_summary(model_init, "LSTM Init-Inject")

    print("[Test] Selesai. Model dapat dicompile dengan:")
    print("  model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')")