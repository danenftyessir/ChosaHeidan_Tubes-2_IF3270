"""
Backward Propagation (BPTT) untuk LSTM dari nol.
Implementasi lengkap BPTT untuk training LSTM model dari nol.
Termasuk gradient checker, optimizer (SGD, Adam), dan full training loop.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))
from caption_preprocess import START_IDX, END_IDX


# ============================================================================
# Loss Functions
# ============================================================================

def cross_entropy_loss_lstm(logits, targets, epsilon=1e-10):
    """
    Cross-entropy loss untuk klasifikasi multi-class (vocabulary).

    Args:
        logits: (batch_size, vocab_size) — output dari softmax
        targets: (batch_size,) — ground truth token indices
        epsilon (float): small value untuk numerical stability
    Returns:
        loss: scalar, average cross-entropy loss
        dout: (batch_size, vocab_size) — gradient dL/dlogits
    """
    batch_size = logits.shape[0]
    logits = np.clip(logits, epsilon, 1.0 - epsilon)

    targets_one_hot = np.zeros_like(logits)
    for i, t in enumerate(targets):
        targets_one_hot[i, t] = 1.0

    loss = -np.sum(targets_one_hot * np.log(logits)) / batch_size
    dout = (logits - targets_one_hot) / batch_size

    return loss, dout


def cross_entropy_loss_sequence_lstm(logits_seq, targets_seq, epsilon=1e-10):
    """
    Cross-entropy loss untuk sequence prediction (teacher forcing).

    Args:
        logits_seq: (batch_size, seq_len, vocab_size) — output probabilities
        targets_seq: (batch_size, seq_len) — ground truth token indices
        epsilon (float): numerical stability
    Returns:
        loss: scalar
        dout_seq: gradient
    """
    batch_size, seq_len, vocab_size = logits_seq.shape
    logits_seq = np.clip(logits_seq, epsilon, 1.0 - epsilon)

    targets_one_hot = np.zeros_like(logits_seq)
    for i in range(batch_size):
        for t in range(seq_len):
            targets_one_hot[i, t, targets_seq[i, t]] = 1.0

    loss = -np.sum(targets_one_hot * np.log(logits_seq)) / (batch_size * seq_len)
    dout_seq = (logits_seq - targets_one_hot) / (batch_size * seq_len)

    return loss, dout_seq


# ============================================================================
# BPTT for LSTM Model
# ============================================================================

def backward_pass_lstm(model, cnn_features, token_seq, targets,
                    output_type='loss', verbose=False):
    """
    Full backward pass (BPTT) untuk LSTMScratch.

    Args:
        model: LSTMScratch instance
        cnn_features: (batch_size, feature_dim)
        token_seq: (batch_size, seq_len) — input token indices
        targets: (batch_size,) — target token index (next token prediction)
        output_type (str): 'loss' atau 'logits'
        verbose (bool): cetak debug info
    Returns:
        loss: scalar cross-entropy loss
        gradients: dict {layer_name: {param_name: grad_array}}
    """
    batch_size = cnn_features.shape[0]

    if output_type == 'loss':
        # Forward pass
        probs = model.forward(cnn_features, token_seq, training=True)
        loss, dout = cross_entropy_loss_lstm(probs, targets)

        # Gradient terhadap hidden state terakhir: dL/dh_T = dout @ W_out^T
        W_out = model.output_dense.weights
        d_h_final = dout @ W_out.T

        if verbose:
            print(f"  Loss: {loss:.4f}")
            print(f"  d_h_final shape: {d_h_final.shape}")

        # Backward pass LSTM
        if model.num_layers == 1:
            model.lstm.reset_cache()

            # Re-run forward untuk cache
            x_start = model.projection.forward(cnn_features)
            embedded = model.embedding.forward(token_seq)
            x_start_expanded = x_start[:, np.newaxis, :]
            combined = np.concatenate([x_start_expanded, embedded], axis=1)
            h_out, c_out = model.lstm.forward_sequence(combined)

            # BPTT
            dx_seq = model.lstm.backward_sequence(None, dh_final=d_h_final)

            # Gradient untuk embedding layer
            d_embed = dx_seq[:, 1:, :]
            model.embedding.backward(d_embed.reshape(-1, model.embed_dim))

        # Gradient untuk projection layer
        d_x_start = dx_seq[:, 0, :]
        d_cnn = model.projection.backward(d_x_start)

        # Gradient untuk output dense
        d_W_out, d_b_out = model.output_dense.get_grad_weights()

        # Collect gradients
        dkernel, drec_kernel, dbias = model.lstm.get_grad_weights()

        gradients = {
            'lstm': {
                'kernel': dkernel,
                'recurrent_kernel': drec_kernel,
                'bias': dbias,
            },
            'output_dense': {
                'weights': d_W_out,
                'bias': d_b_out,
            },
            'projection': {
                'weights': model.projection.grad_weights,
                'bias': model.projection.grad_bias,
            },
            'embedding': {
                'weights': model.embedding.grad_weights,
            }
        }

        return loss, gradients

    else:
        raise NotImplementedError("output_type='logits' not yet implemented")


def compute_loss_gradient_lstm(logits, labels, loss_type='cross_entropy'):
    """
    Hitung gradient loss terhadap logits (LSTM).

    Args:
        logits: (batch_size, vocab_size) — output probabilities
        labels: (batch_size,) — ground truth indices
        loss_type (str): 'cross_entropy'
    Returns:
        loss: scalar
        dout: (batch_size, vocab_size)
    """
    if loss_type == 'cross_entropy':
        return cross_entropy_loss_lstm(logits, labels)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")


# ============================================================================
# Gradient Checker
# ============================================================================

def gradient_checker_lstm(model, cnn_feature_sample, token_sample, label_sample,
                         epsilon=1e-5, verbose=True):
    """
    Verifikasi BPTT gradient LSTM dengan numerical gradient checking.

    Args:
        model: LSTMScratch instance
        cnn_feature_sample: (1, feature_dim) — single image feature
        token_sample: (1, seq_len) — single token sequence
        label_sample: (1,) — single target label
        epsilon (float): perturbation size
        verbose (bool): cetak hasil
    Returns:
        max_relative_error: float
    """
    if verbose:
        print("[Gradient Checker LSTM] Memeriksa gradient LSTM...")

    # Forward + Backward analytical
    probs = model.forward(cnn_feature_sample, token_sample, training=True)
    loss, dout = compute_loss_gradient_lstm(probs, label_sample)

    W_out = model.output_dense.weights
    d_h_final = dout @ W_out.T

    model.lstm.reset_cache()
    x_start = model.projection.forward(cnn_feature_sample)
    embedded = model.embedding.forward(token_sample)
    x_start_expanded = x_start[:, np.newaxis, :]
    combined = np.concatenate([x_start_expanded, embedded], axis=1)
    h_out, c_out = model.lstm.forward_sequence(combined)
    dx_seq = model.lstm.backward_sequence(None, dh_final=d_h_final)

    grad_analytical_kernel = model.lstm._dkernel

    # Numerical gradient untuk kernel
    kernel_orig = model.lstm.kernel.copy()
    grad_numerical_kernel = np.zeros_like(kernel_orig)

    def compute_loss_given_kernel(kernel_new):
        """Helper: hitung loss dengan kernel dimodifikasi."""
        model.lstm.kernel = kernel_new
        probs_new = model.forward(cnn_feature_sample, token_sample, training=True)
        loss_new, _ = compute_loss_gradient_lstm(probs_new, label_sample)
        return loss_new

    # Sample-based check (tidak semua element)
    max_check = min(50, kernel_orig.shape[0] * kernel_orig.shape[1])
    import itertools
    indices = list(itertools.product(range(kernel_orig.shape[0]),
                                     range(kernel_orig.shape[1])))
    np.random.seed(42)
    sample_idx = np.random.choice(len(indices), min(max_check, len(indices)), replace=False)

    for idx in sample_idx:
        i, j = idx
        # +
        kernel_plus = kernel_orig.copy()
        kernel_plus[i, j] += epsilon
        loss_plus = compute_loss_given_kernel(kernel_plus)

        # -
        kernel_minus = kernel_orig.copy()
        kernel_minus[i, j] -= epsilon
        loss_minus = compute_loss_given_kernel(kernel_minus)

        # Central difference
        grad_numerical_kernel[i, j] = (loss_plus - loss_minus) / (2 * epsilon)

    # Restore
    model.lstm.kernel = kernel_orig

    # Relative error hanya untuk checked elements
    checked_grad_analytical = grad_analytical_kernel[sample_idx[:, 0], sample_idx[:, 1]]
    checked_grad_numerical = grad_numerical_kernel[sample_idx[:, 0], sample_idx[:, 1]]

    numerator = np.abs(checked_grad_analytical - checked_grad_numerical)
    denominator = np.abs(checked_grad_analytical) + np.abs(checked_grad_numerical) + 1e-8
    relative_error = np.max(numerator / denominator)

    if verbose:
        print(f"  Max relative error (sampled): {relative_error:.6f}")
        if relative_error < 1e-4:
            print("  [OK] Gradient LSTM implementasi BENAR")
        else:
            print("  [WARNING] Error lebih besar dari toleransi 1e-4")

    return relative_error


# ============================================================================
# Optimizers
# ============================================================================

def sgd_update_lstm(layer, lr=0.01):
    """
    Update bobot LSTM layer dengan SGD.

    Args:
        layer: LSTMCell instance
        lr (float): learning rate
    """
    if hasattr(layer, '_dkernel'):
        layer.kernel -= lr * layer._dkernel
        layer.recurrent_kernel -= lr * layer._drecurrent_kernel
        layer.bias -= lr * layer._dbias


def sgd_update_dense_lstm(layer, lr=0.01):
    """Update bobot Dense layer dengan SGD."""
    if hasattr(layer, 'grad_weights'):
        layer.weights -= lr * layer.grad_weights
        layer.bias -= lr * layer.grad_bias


def sgd_update_embedding_lstm(layer, lr=0.01):
    """Update bobot Embedding layer dengan SGD."""
    if hasattr(layer, 'grad_weights'):
        layer.weights -= lr * layer.grad_weights


def adam_update_lstm(layer, t, m, v, beta1=0.9, beta2=0.999,
                    epsilon=1e-8, lr=0.001):
    """
    Update bobot LSTM layer dengan Adam optimizer.

    Args:
        layer: LSTMCell instance
        t (int): timestep
        m (dict): first moment estimates
        v (dict): second moment estimates
        beta1 (float): exponential decay rate for first moment
        beta2 (float): exponential decay rate for second moment
        epsilon (float): small constant
        lr (float): learning rate
    Returns:
        m, v: updated moments
    """
    if not hasattr(layer, '_dkernel'):
        return m, v

    params = ['kernel', 'recurrent_kernel', 'bias']
    grads = [layer._dkernel, layer._drecurrent_kernel, layer._dbias]

    for param, grad in zip(params, grads):
        if param not in m:
            m[param] = np.zeros_like(grad)
            v[param] = np.zeros_like(grad)

        m[param] = beta1 * m[param] + (1 - beta1) * grad
        v[param] = beta2 * v[param] + (1 - beta2) * (grad ** 2)
        m_hat = m[param] / (1 - beta1 ** t)
        v_hat = v[param] / (1 - beta2 ** t)

        if param == 'kernel':
            layer.kernel -= lr * m_hat / (np.sqrt(v_hat) + epsilon)
        elif param == 'recurrent_kernel':
            layer.recurrent_kernel -= lr * m_hat / (np.sqrt(v_hat) + epsilon)
        else:
            layer.bias -= lr * m_hat / (np.sqrt(v_hat) + epsilon)

    return m, v


# ============================================================================
# Training Loop
# ============================================================================

def train_one_step_lstm(model, cnn_features, token_seq, targets,
                        optimizer='sgd', lr=0.001, t=None, adam_state=None,
                        verbose=False):
    """
    Satu training step untuk LSTM: forward + backward + update.

    Args:
        model: LSTMScratch instance
        cnn_features: (batch_size, feature_dim)
        token_seq: (batch_size, seq_len)
        targets: (batch_size,) — target label
        optimizer (str): 'sgd' atau 'adam'
        lr (float): learning rate
        t (int): timestep untuk Adam
        adam_state (dict): Adam moments
        verbose (bool): cetak progress
    Returns:
        loss: scalar loss
        adam_state: updated Adam state
    """
    loss, gradients = backward_pass_lstm(
        model, cnn_features, token_seq, targets,
        output_type='loss', verbose=verbose
    )

    if optimizer == 'sgd':
        sgd_update_lstm(model.lstm, lr=lr)
        sgd_update_dense_lstm(model.output_dense, lr=lr)
        sgd_update_dense_lstm(model.projection, lr=lr)
        sgd_update_embedding_lstm(model.embedding, lr=lr)

    elif optimizer == 'adam':
        if adam_state is None:
            adam_state = {
                'lstm': {'m': {}, 'v': {}},
            }

        t_eff = t if t is not None else 1
        m_lstm, v_lstm = adam_update_lstm(
            model.lstm, t_eff,
            adam_state['lstm']['m'], adam_state['lstm']['v'],
            lr=lr
        )
        adam_state['lstm']['m'] = m_lstm
        adam_state['lstm']['v'] = v_lstm

        # SGD untuk dense layers
        sgd_update_dense_lstm(model.output_dense, lr=lr)
        sgd_update_dense_lstm(model.projection, lr=lr)
        sgd_update_embedding_lstm(model.embedding, lr=lr)

    return loss, adam_state


def train_lstm_scratch(model, train_features, train_seqs, train_targets,
                       val_features=None, val_seqs=None, val_targets=None,
                       epochs=10, batch_size=32, lr=0.001, optimizer='adam',
                       verbose=True):
    """
    Training loop lengkap untuk LSTM dari nol dengan BPTT.

    Args:
        model: LSTMScratch instance
        train_features: (N_train, feature_dim)
        train_seqs: (N_train, seq_len)
        train_targets: (N_train,) — target labels
        val_features: validation features (optional)
        val_seqs: validation sequences (optional)
        val_targets: validation targets (optional)
        epochs (int): jumlah epoch
        batch_size (int): ukuran batch
        lr (float): learning rate
        optimizer (str): 'sgd' atau 'adam'
        verbose (bool): cetak progress
    Returns:
        history: dict {train_loss: [], val_loss: []}
    """
    history = {'train_loss': [], 'val_loss': []}
    adam_state = None

    N = len(train_features)
    num_batches = (N + batch_size - 1) // batch_size

    for epoch in range(epochs):
        if verbose:
            print(f"\n[Epoch {epoch + 1}/{epochs}]")

        epoch_loss = 0.0

        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, N)

            batch_cnn = train_features[start:end]
            batch_seq = train_seqs[start:end]
            batch_target = train_targets[start:end]

            loss, adam_state = train_one_step_lstm(
                model, batch_cnn, batch_seq, batch_target,
                optimizer=optimizer, lr=lr, t=epoch * num_batches + batch_idx + 1,
                adam_state=adam_state, verbose=False
            )

            epoch_loss += loss

        avg_train_loss = epoch_loss / num_batches
        history['train_loss'].append(avg_train_loss)

        if verbose:
            print(f"  Train Loss: {avg_train_loss:.4f}")

        # Validation
        if val_features is not None and val_seqs is not None:
            val_loss = 0.0
            num_val_batches = (len(val_features) + batch_size - 1) // batch_size

            for vb in range(num_val_batches):
                v_start = vb * batch_size
                v_end = min(v_start + batch_size, len(val_features))

                v_cnn = val_features[v_start:v_end]
                v_seq = val_seqs[v_start:v_end]
                v_target = val_targets[v_start:v_end]

                v_probs = model.forward(v_cnn, v_seq, training=True)
                v_loss, _ = compute_loss_gradient_lstm(v_probs, v_target)
                val_loss += v_loss

            avg_val_loss = val_loss / num_val_batches
            history['val_loss'].append(avg_val_loss)

            if verbose:
                print(f"  Val Loss:   {avg_val_loss:.4f}")

    return history


if __name__ == '__main__':
    print("[Test] BPTT untuk LSTM dari nol...")

    print("\n[Test] Contoh usage:")
    print("  from bonus_backward import train_lstm_scratch, gradient_checker_lstm")
    print("  history = train_lstm_scratch(model, train_feat, train_seq, train_tgt, epochs=10)")
    print("  error = gradient_checker_lstm(model, feat_sample, seq_sample, label_sample)")

    print("\n[Test] Selesai.")
