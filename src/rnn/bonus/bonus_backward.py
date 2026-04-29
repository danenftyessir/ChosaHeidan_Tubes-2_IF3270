"""
Backward Propagation (BPTT) untuk SimpleRNN dari nol.
Implementasi lengkap BPTT untuk training RNN model dari nol.
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

def cross_entropy_loss(logits, targets, epsilon=1e-10):
    """
    Cross-entropy loss untuk klasifikasi multi-class (vocabulary).

    Args:
        logits: (batch_size, vocab_size) — output dari softmax (bukan logit asli)
        targets: (batch_size,) — ground truth token indices
        epsilon (float): small value untuk numerical stability
    Returns:
        loss: scalar, average cross-entropy loss
        dout: (batch_size, vocab_size) — gradient dL/dlogits
    """
    batch_size = logits.shape[0]

    # Clamp logits
    logits = np.clip(logits, epsilon, 1.0 - epsilon)

    # Loss: -sum(target * log(pred))
    # Targets: one-hot encoding
    targets_one_hot = np.zeros_like(logits)
    for i, t in enumerate(targets):
        targets_one_hot[i, t] = 1.0

    # Cross-entropy
    loss = -np.sum(targets_one_hot * np.log(logits)) / batch_size

    # Gradient dL/dlogits = (pred - target) / batch_size
    dout = (logits - targets_one_hot) / batch_size

    return loss, dout


def cross_entropy_loss_sequence(logits_seq, targets_seq, epsilon=1e-10):
    """
    Cross-entropy loss untuk sequence prediction (teacher forcing).

    Args:
        logits_seq: (batch_size, seq_len, vocab_size) — output probabilities
        targets_seq: (batch_size, seq_len) — ground truth token indices
        epsilon (float): numerical stability
    Returns:
        loss: scalar, average cross-entropy loss
        dout_seq: (batch_size, seq_len, vocab_size) — gradient
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
# BPTT for RNN Model
# ============================================================================

def backward_pass_rnn(model, cnn_features, token_seq, targets,
                     output_type='loss', verbose=False):
    """
    Full backward pass (BPTT) untuk RNNScratch.

    Args:
        model: RNNScratch instance
        cnn_features: (batch_size, feature_dim)
        token_seq: (batch_size, seq_len) — input token indices
        targets: (batch_size, seq_len) — target token indices (untuk teacher forcing)
        output_type (str): 'loss' atau 'logits'
        verbose (bool): cetak debug info
    Returns:
        loss: scalar cross-entropy loss
        gradients: dict {layer_name: {param_name: grad_array}}
    """
    batch_size = cnn_features.shape[0]

    # Forward pass
    if output_type == 'loss':
        # Teacher forcing: forward dengan sequence
        probs = model.forward(cnn_features, token_seq, training=True)
        loss, dout = cross_entropy_loss(probs, targets[:, -1])  # Prediksi next token

        # Gradient terhadap hidden state terakhir: dL/dh_T = dout @ W_out^T
        W_out = model.output_dense.weights
        d_h_final = dout @ W_out.T

        if verbose:
            print(f"  Loss: {loss:.4f}")
            print(f"  d_h_final shape: {d_h_final.shape}")

        # Backward pass RNN
        if model.num_layers == 1:
            model.rnn.reset_cache()

            # Re-run forward untuk cache
            x_start = model.projection.forward(cnn_features)
            embedded = model.embedding.forward(token_seq)
            x_start_expanded = x_start[:, np.newaxis, :]
            combined = np.concatenate([x_start_expanded, embedded], axis=1)
            h_out, _ = model.rnn.forward_sequence(combined)

            # Backward sequence
            dx_seq = model.rnn.backward_sequence(None, dh_final=d_h_final)

            # Gradient untuk embedding layer
            d_embed = dx_seq[:, 1:, :]
            model.embedding.backward(d_embed.reshape(-1, model.embed_dim))

        # Gradient untuk projection layer
        d_x_start = dx_seq[:, 0, :]
        d_cnn = model.projection.backward(d_x_start)

        # Gradient untuk output dense
        d_W_out, d_b_out = model.output_dense.get_grad_weights()

        # Collect gradients
        gradients = {
            'rnn': {
                'W_xh': model.rnn._dW_xh,
                'W_hh': model.rnn._dW_hh,
                'b_h': model.rnn._db,
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


def compute_loss_gradient(logits, labels, loss_type='cross_entropy'):
    """
    Hitung gradient loss terhadap logits.

    Args:
        logits: (batch_size, vocab_size) — output probabilities
        labels: (batch_size,) — ground truth indices
        loss_type (str): 'cross_entropy'
    Returns:
        loss: scalar
        dout: (batch_size, vocab_size)
    """
    if loss_type == 'cross_entropy':
        return cross_entropy_loss(logits, labels)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")


# ============================================================================
# Gradient Checker
# ============================================================================

def gradient_checker_rnn(model, cnn_feature_sample, token_sample, label_sample,
                         epsilon=1e-5, verbose=True):
    """
    Verifikasi BPTT gradient dengan numerical gradient checking.

    Args:
        model: RNNScratch instance
        cnn_feature_sample: (1, feature_dim) — single image feature
        token_sample: (1, seq_len) — single token sequence
        label_sample: (1,) — single target label
        epsilon (float): perturbation size
        verbose (bool): cetak hasil
    Returns:
        max_relative_error: float
    """
    if verbose:
        print("[Gradient Checker] Memeriksa gradient RNN...")

    # Forward + Backward analytical
    probs = model.forward(cnn_feature_sample, token_sample, training=True)
    loss, dout = compute_loss_gradient(probs, label_sample)

    W_out = model.output_dense.weights
    d_h_final = dout @ W_out.T

    model.rnn.reset_cache()
    x_start = model.projection.forward(cnn_feature_sample)
    embedded = model.embedding.forward(token_sample)
    x_start_expanded = x_start[:, np.newaxis, :]
    combined = np.concatenate([x_start_expanded, embedded], axis=1)
    h_out, _ = model.rnn.forward_sequence(combined)
    dx_seq = model.rnn.backward_sequence(None, dh_final=d_h_final)

    grad_analytical_W_xh = model.rnn._dW_xh

    # Numerical gradient untuk W_xh
    W_xh_orig = model.rnn.W_xh.copy()
    grad_numerical_W_xh = np.zeros_like(W_xh_orig)

    def compute_loss_given_W_xh(W_xh_new):
        """Helper: hitung loss dengan W_xh dimodifikasi."""
        model.rnn.W_xh = W_xh_new
        probs_new = model.forward(cnn_feature_sample, token_sample, training=True)
        loss_new, _ = compute_loss_gradient(probs_new, label_sample)
        return loss_new

    for i in range(W_xh_orig.shape[0]):
        for j in range(W_xh_orig.shape[1]):
            # +
            W_xh_plus = W_xh_orig.copy()
            W_xh_plus[i, j] += epsilon
            loss_plus = compute_loss_given_W_xh(W_xh_plus)

            # -
            W_xh_minus = W_xh_orig.copy()
            W_xh_minus[i, j] -= epsilon
            loss_minus = compute_loss_given_W_xh(W_xh_minus)

            # Central difference
            grad_numerical_W_xh[i, j] = (loss_plus - loss_minus) / (2 * epsilon)

    # Restore
    model.rnn.W_xh = W_xh_orig

    # Relative error
    numerator = np.abs(grad_analytical_W_xh - grad_numerical_W_xh)
    denominator = np.abs(grad_analytical_W_xh) + np.abs(grad_numerical_W_xh) + 1e-8
    relative_error = np.max(numerator / denominator)

    if verbose:
        print(f"  Max relative error: {relative_error:.6f}")
        if relative_error < 1e-4:
            print("  [OK] Gradient implementasi BENAR")
        else:
            print("  [WARNING] Error lebih besar dari toleransi 1e-4")
            print(f"  Analytical grad norm: {np.linalg.norm(grad_analytical_W_xh):.6f}")
            print(f"  Numerical grad norm: {np.linalg.norm(grad_numerical_W_xh):.6f}")

    return relative_error


# ============================================================================
# Optimizers
# ============================================================================

def sgd_update_rnn(layer, lr=0.01):
    """
    Update bobot RNN layer dengan SGD.

    Args:
        layer: SimpleRNNCell instance
        lr (float): learning rate
    """
    if hasattr(layer, '_dW_xh'):
        layer.W_xh -= lr * layer._dW_xh
        layer.W_hh -= lr * layer._dW_hh
        layer.b_h -= lr * layer._db


def sgd_update_dense(layer, lr=0.01):
    """
    Update bobot Dense layer dengan SGD.

    Args:
        layer: Dense instance
        lr (float): learning rate
    """
    if hasattr(layer, 'grad_weights'):
        layer.weights -= lr * layer.grad_weights
        layer.bias -= lr * layer.grad_bias


def sgd_update_embedding(layer, lr=0.01):
    """
    Update bobot Embedding layer dengan SGD.

    Args:
        layer: Embedding instance
        lr (float): learning rate
    """
    if hasattr(layer, 'grad_weights'):
        layer.weights -= lr * layer.grad_weights


def adam_update_rnn(layer, t, m, v, beta1=0.9, beta2=0.999, epsilon=1e-8, lr=0.001):
    """
    Update bobot RNN layer dengan Adam optimizer.

    Args:
        layer: SimpleRNNCell instance
        t (int): timestep
        m (dict): first moment estimates {param: m_array}
        v (dict): second moment estimates {param: v_array}
        beta1 (float): exponential decay rate for first moment
        beta2 (float): exponential decay rate for second moment
        epsilon (float): small constant
        lr (float): learning rate
    Returns:
        m, v: updated moments
    """
    if not hasattr(layer, '_dW_xh'):
        return m, v

    params = ['W_xh', 'W_hh', 'b_h']
    grads = [layer._dW_xh, layer._dW_hh, layer._db]

    for param, grad in zip(params, grads):
        if param not in m:
            m[param] = np.zeros_like(grad)
            v[param] = np.zeros_like(grad)

        m[param] = beta1 * m[param] + (1 - beta1) * grad
        v[param] = beta2 * v[param] + (1 - beta2) * (grad ** 2)
        m_hat = m[param] / (1 - beta1 ** t)
        v_hat = v[param] / (1 - beta2 ** t)

        if param == 'W_xh':
            layer.W_xh -= lr * m_hat / (np.sqrt(v_hat) + epsilon)
        elif param == 'W_hh':
            layer.W_hh -= lr * m_hat / (np.sqrt(v_hat) + epsilon)
        else:
            layer.b_h -= lr * m_hat / (np.sqrt(v_hat) + epsilon)

    return m, v


# ============================================================================
# Training Loop
# ============================================================================

def train_one_step(model, cnn_features, token_seq, targets,
                   optimizer='sgd', lr=0.001, t=None, adam_state=None,
                   verbose=False):
    """
    Satu training step: forward + backward + update.

    Args:
        model: RNNScratch instance
        cnn_features: (batch_size, feature_dim)
        token_seq: (batch_size, seq_len)
        targets: (batch_size,) atau (batch_size, seq_len)
        optimizer (str): 'sgd' atau 'adam'
        lr (float): learning rate
        t (int): timestep untuk Adam
        adam_state (dict): Adam moments (jika optimizer='adam')
        verbose (bool): cetak progress
    Returns:
        loss: scalar loss
        adam_state: updated Adam state (jika Adam)
    """
    # Forward + Backward
    loss, gradients = backward_pass_rnn(
        model, cnn_features, token_seq, targets,
        output_type='loss', verbose=verbose
    )

    # Update parameters
    if optimizer == 'sgd':
        sgd_update_rnn(model.rnn, lr=lr)
        sgd_update_dense(model.output_dense, lr=lr)
        sgd_update_dense(model.projection, lr=lr)
        sgd_update_embedding(model.embedding, lr=lr)

    elif optimizer == 'adam':
        if adam_state is None:
            adam_state = {
                'rnn': {'m': {}, 'v': {}},
            }

        t_eff = t if t is not None else 1
        m_rnn, v_rnn = adam_update_rnn(
            model.rnn, t_eff,
            adam_state['rnn']['m'], adam_state['rnn']['v'],
            lr=lr
        )
        adam_state['rnn']['m'] = m_rnn
        adam_state['rnn']['v'] = v_rnn

        # SGD untuk dense layers
        sgd_update_dense(model.output_dense, lr=lr)
        sgd_update_dense(model.projection, lr=lr)
        sgd_update_embedding(model.embedding, lr=lr)

    return loss, adam_state


def train_rnn_scratch(model, train_features, train_seqs, train_targets,
                      val_features=None, val_seqs=None, val_targets=None,
                      epochs=10, batch_size=32, lr=0.001, optimizer='adam',
                      verbose=True):
    """
    Training loop lengkap untuk RNN dari nol dengan BPTT.

    Args:
        model: RNNScratch instance
        train_features: (N_train, feature_dim)
        train_seqs: (N_train, seq_len)
        train_targets: (N_train,) atau (N_train, seq_len)
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

            loss, adam_state = train_one_step(
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
                v_loss, _ = compute_loss_gradient(v_probs, v_target[:, -1])
                val_loss += v_loss

            avg_val_loss = val_loss / num_val_batches
            history['val_loss'].append(avg_val_loss)

            if verbose:
                print(f"  Val Loss:   {avg_val_loss:.4f}")

    return history


if __name__ == '__main__':
    print("[Test] BPTT untuk RNN dari nol...")

    print("\n[Test] Contoh usage:")
    print("  from bonus_backward import train_rnn_scratch, gradient_checker_rnn")
    print("  history = train_rnn_scratch(model, train_feat, train_seq, train_tgt, epochs=10)")
    print("  error = gradient_checker_rnn(model, feat_sample, seq_sample, label_sample)")

    print("\n[Test] Selesai.")
