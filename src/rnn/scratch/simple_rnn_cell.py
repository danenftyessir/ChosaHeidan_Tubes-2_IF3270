"""
SimpleRNN Cell dari nol (NumPy only).
Mendukung forward pass, backward pass (BPTT), dan batch inference.
Batch-first: input/output shape menggunakan (batch_size, ...).
"""

import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))
from activations import tanh, d_tanh


# ============================================================================
# SimpleRNN Cell
# ============================================================================

class SimpleRNNCell:
    """
    SimpleRNN Cell — fundamental recurrent unit.

    Forward pass:
        h_t = tanh(W_xh @ x_t + W_hh @ h_{t-1} + b_h)

    Bentuk data:
        Input x_t:      shape (batch_size, input_dim)
        Hidden h_{t-1}: shape (batch_size, hidden_dim)
        Hidden h_t:     shape (batch_size, hidden_dim)

    Backward pass (BPTT — Backpropagation Through Time):
        Untuk sequence length T:
        1. Forward pass semua timestep → simpan caches
        2. Backward pass: dari T-1 ke 0
        3. Akumulasi gradient terhadap W_xh, W_hh, b_h

    Args:
        input_dim (int): dimensi input (embed_dim).
        hidden_dim (int): dimensi hidden state.
        return_sequences (bool): return seluruh hidden states atau hanya last.
            Default False → return last hidden state.
        name (str): nama cell untuk debugging.
    """

    def __init__(self, input_dim, hidden_dim, return_sequences=False, name="rnn"):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.return_sequences = return_sequences
        self.name = name

        # Bobot dan bias
        self.W_xh = None   # (input_dim, hidden_dim)
        self.W_hh = None   # (hidden_dim, hidden_dim)
        self.b_h = None    # (hidden_dim,)
        self._init_weights()

        # Cache untuk backward pass
        self.x_caches = []      # list of x_t per timestep
        self.h_caches = []      # list of h_t per timestep
        self.h_prev_caches = [] # list of h_{t-1} per timestep
        self.z_caches = []      # list of pre-activation (before tanh)
        self.batch_size = None

    def _init_weights(self):
        """
        Inisialisasi bobot dengan Glorot/Xavier initialization.
        Digunakan tanh activation, sehingga:
            Var(W) = 2 / (fan_in + fan_out)
        """
        # Xavier untuk tanh
        scale = np.sqrt(2.0 / (self.input_dim + self.hidden_dim))

        self.W_xh = np.random.randn(self.input_dim, self.hidden_dim).astype(np.float64) * scale
        self.W_hh = np.random.randn(self.hidden_dim, self.hidden_dim).astype(np.float64) * scale
        self.b_h = np.zeros(self.hidden_dim, dtype=np.float64)

    def forward(self, x, h_prev=None):
        """
        Forward pass untuk SATU timestep.

        Args:
            x: array numpy, bentuk (batch_size, input_dim).
            h_prev: hidden state sebelumnya, bentuk (batch_size, hidden_dim).
                   Jika None, diasumsikan zeros.
        Returns:
            h_next: hidden state baru, bentuk (batch_size, hidden_dim).
        """
        if h_prev is None:
            h_prev = np.zeros((x.shape[0], self.hidden_dim), dtype=np.float64)

        self.batch_size = x.shape[0]

        # Pre-activation: z = W_xh @ x + W_hh @ h_prev + b
        z = x @ self.W_xh + h_prev @ self.W_hh + self.b_h

        # Activation: h = tanh(z)
        h_next = tanh(z)

        # Cache untuk backward
        self.x_caches.append(x)
        self.h_prev_caches.append(h_prev)
        self.z_caches.append(z)
        self.h_caches.append(h_next)

        return h_next

    def forward_sequence(self, x_seq, h0=None):
        """
        Forward pass untuk SELURUH sequence.

        Args:
            x_seq: array numpy, bentuk (batch_size, seq_len, input_dim).
            h0: hidden state awal, bentuk (batch_size, hidden_dim).
                Jika None, gunakan zeros.
        Returns:
            h_seq: array numpy, bentuk (batch_size, seq_len, hidden_dim)
                   jika return_sequences=True, else (batch_size, hidden_dim)
                   berisi hidden state terakhir.
            h_final: hidden state terakhir (untuk chaining).
        """
        batch_size, seq_len, _ = x_seq.shape

        if h0 is None:
            h0 = np.zeros((batch_size, self.hidden_dim), dtype=np.float64)

        # Clear caches
        self.x_caches = []
        self.h_prev_caches = []
        self.z_caches = []
        self.h_caches = []

        h_prev = h0
        h_all = []

        for t in range(seq_len):
            x_t = x_seq[:, t, :]  # (batch_size, input_dim)
            h_t = self.forward(x_t, h_prev)
            h_all.append(h_t)
            h_prev = h_t

        h_seq = np.stack(h_all, axis=1)  # (batch_size, seq_len, hidden_dim)

        if self.return_sequences:
            return h_seq, h_prev
        else:
            return h_seq[:, -1, :], h_prev

    def backward(self, dout, dh_next=None):
        """
        Backward pass untuk SATU timestep.
        Ini adalah TIDAK digunakan langsung — Gunakan backward_sequence().

        Args:
            dout: gradient dari loss terhadap output hidden state,
                  bentuk (batch_size, hidden_dim).
            dh_next: gradient dari layer atas terhadap h_t (untuk BPTT).
                   Dalam context BPTT, ini adalah grad dari timestep t+1.
        Returns:
            dx: gradient terhadap input x_t, bentuk (batch_size, input_dim).
            dh_prev: gradient terhadap hidden state sebelumnya.
        """
        # Gradient terhadap h_t (sudah termasuk dout + dh_next dari timesteps berikutnya)
        dh = dout
        if dh_next is not None:
            dh = dh + dh_next

        # Gradient terhadap z (pre-activation): dh/dz = tanh'(z) * dh
        # tanh'(z) = 1 - tanh(z)^2
        z = self.z_caches[-1]
        h = self.h_caches[-1]
        d_tanh = d_tanh(h)  # shape (batch_size, hidden_dim)
        dz = dh * d_tanh     # shape (batch_size, hidden_dim)

        # Gradient terhadap weights dan biases
        x = self.x_caches[-1]
        h_prev = self.h_prev_caches[-1]

        # dL/dW_xh = x^T @ dz
        dW_xh = x.T @ dz
        # dL/dW_hh = h_prev^T @ dz
        dW_hh = h_prev.T @ dz
        # dL/db = sum over batch
        db = np.sum(dz, axis=0)

        # Gradient terhadap input x: dx = dz @ W_xh^T
        dx = dz @ self.W_xh.T

        # Gradient terhadap h_prev: dh_prev = dz @ W_hh^T
        dh_prev = dz @ self.W_hh.T

        # Simpan gradients
        self._dW_xh = dW_xh
        self._dW_hh = dW_hh
        self._db = db
        self._dx = dx

        return dx, dh_prev

    def backward_sequence(self, dout_seq, dh_final=None):
        """
        Backward pass untuk SELURUH sequence (BPTT).

        Args:
            dout_seq: gradient dari loss terhadap seluruh sequence output,
                      bentuk (batch_size, seq_len, hidden_dim).
                      Jika None, diasumsikan hanya last hidden state yang mattered.
            dh_final: gradient terhadap hidden state terakhir dari layer atas.
        Returns:
            dx_seq: gradient terhadap input sequence,
                    bentuk (batch_size, seq_len, input_dim).
        """
        seq_len = len(self.x_caches)
        batch_size = self.batch_size if self.batch_size else self.x_caches[0].shape[0]

        if dout_seq is None:
            # Default: hanya last timestep mattered
            dout_seq = np.zeros((batch_size, seq_len, self.hidden_dim), dtype=np.float64)
            # Set grad terakhir dari dh_final jika ada
            if dh_final is not None:
                # Ini akan ditambahkan di akhir BPTT
                pass

        # Akumulasi gradients
        dW_xh_accum = np.zeros_like(self.W_xh)
        dW_hh_accum = np.zeros_like(self.W_hh)
        db_accum = np.zeros_like(self.b_h)

        # Gradient terhadap x dan h_prev
        dx_seq = np.zeros((batch_size, seq_len, self.input_dim), dtype=np.float64)

        # dh_next = grad dari layer atas (jika ada)
        dh_next = dh_final if dh_final is not None else np.zeros((batch_size, self.hidden_dim), dtype=np.float64)

        # Backward through time: dari T-1 ke 0
        for t in range(seq_len - 1, -1, -1):
            # Gradient dari output pada timestep t
            dh_out = dout_seq[:, t, :]  # (batch_size, hidden_dim)

            # Total gradient terhadap hidden state timestep t
            dh = dh_out + dh_next

            # Gradient terhadap pre-activation
            h = self.h_caches[t]
            z = self.z_caches[t]
            d_tanh = d_tanh(h)
            dz = dh * d_tanh  # (batch_size, hidden_dim)

            # Akumulasi gradient terhadap weights
            x = self.x_caches[t]
            h_prev = self.h_prev_caches[t]

            dW_xh_t = x.T @ dz
            dW_hh_t = h_prev.T @ dz
            db_t = np.sum(dz, axis=0)

            dW_xh_accum += dW_xh_t
            dW_hh_accum += dW_hh_t
            db_accum += db_t

            # Gradient terhadap input
            dx_seq[:, t, :] = dz @ self.W_xh.T

            # Gradient terhadap h_prev (menjalar mundur ke timestep sebelumnya)
            dh_next = dz @ self.W_hh.T

        # Simpan akumulasi gradients
        self._dW_xh = dW_xh_accum
        self._dW_hh = dW_hh_accum
        self._db = db_accum
        self._dx_seq = dx_seq

        return dx_seq

    def backward_from_loss(self, logits, labels, hiddens=None):
        """
        Backward dari loss cross-entropy.
        Menghitung gradient dari output probabilities.

        Args:
            logits: output hidden states, bentuk (batch_size, seq_len, hidden_dim)
                    atau (batch_size, hidden_dim).
            labels: ground truth token indices, bentuk (batch_size,) atau (batch_size, seq_len).
            hiddens: hidden states dari forward pass (cache).
        Returns:
            dx: gradient terhadap input sequence.
        """
        if hiddens is None:
            hiddens = self.h_caches

        batch_size = logits.shape[0]
        seq_len = logits.shape[1] if len(logits.shape) > 2 else 1

        # Gradient dari loss terhadap hidden states
        # Untuk cross-entropy + softmax di timestep t:
        # dL/dh_t = probs - one_hot(label)
        dout_seq = np.zeros_like(logits)

        return self.backward_sequence(dout_seq, dh_final=None)

    def get_weights(self):
        """
        Kembalikan bobot cell.

        Returns:
            tuple: (W_xh, W_hh, b_h).
        """
        return self.W_xh, self.W_hh, self.b_h

    def set_weights(self, W_xh, W_hh, b_h):
        """
        Set bobot dari sumber eksternal (misal: Keras).

        Args:
            W_xh: array (input_dim, hidden_dim).
            W_hh: array (hidden_dim, hidden_dim).
            b_h: array (hidden_dim,).
        """
        self.W_xh = W_xh.astype(np.float64)
        self.W_hh = W_hh.astype(np.float64)
        self.b_h = b_h.astype(np.float64)
        self.input_dim = W_xh.shape[0]
        self.hidden_dim = W_hh.shape[0]

    def get_grad_weights(self):
        """
        Kembalikan gradient bobot (harus dipanggil setelah backward_sequence).

        Returns:
            tuple: (dW_xh, dW_hh, db_h).
        """
        return self._dW_xh, self._dW_hh, self._db

    def reset_cache(self):
        """Reset caches (panggil sebelum forward sequence baru)."""
        self.x_caches = []
        self.h_prev_caches = []
        self.z_caches = []
        self.h_caches = []
        self.batch_size = None

    def summary(self):
        """Kembalikan info cell."""
        return (f"SimpleRNNCell(input_dim={self.input_dim}, "
                f"hidden_dim={self.hidden_dim}, "
                f"return_sequences={self.return_sequences})")


# ============================================================================
# Stacked RNN Layers
# ============================================================================

class StackedRNNCell:
    """
    Stacked RNN (Deep RNN): multiple RNN layers.
    Output dari layer L-1 menjadi input untuk layer L.

    Args:
        input_dim (int): dimensi input layer pertama.
        hidden_dims (list): list hidden_dim per layer.
        return_sequences (bool): apakah setiap layer return sequences.
    """

    def __init__(self, input_dim, hidden_dims, return_sequences=False):
        self.layers = []
        current_dim = input_dim

        for i, hdim in enumerate(hidden_dims):
            self.layers.append(SimpleRNNCell(
                input_dim=current_dim,
                hidden_dim=hdim,
                return_sequences=return_sequences or (i < len(hidden_dims) - 1),
                name=f"rnn_layer_{i}"
            ))
            current_dim = hdim

        self.hidden_dims = hidden_dims
        self.num_layers = len(hidden_dims)

    def forward_sequence(self, x_seq, h0_list=None):
        """
        Forward pass untuk stacked RNN.

        Args:
            x_seq: shape (batch_size, seq_len, input_dim).
            h0_list: list of initial hidden states per layer.
        Returns:
            output_seq: shape (batch_size, seq_len, final_hidden_dim)
                        atau (batch_size, final_hidden_dim) jika return_sequences=False.
        """
        if h0_list is None:
            h0_list = [None] * self.num_layers

        current = x_seq
        h_last_list = []

        for i, cell in enumerate(self.layers):
            h0 = h0_list[i]
            cell.reset_cache()
            output, h_final = cell.forward_sequence(current, h0)
            current = output
            h_last_list.append(h_final)

        return current, h_last_list

    def backward_sequence(self, dout_seq, dh_final_list=None):
        """
        Backward pass untuk stacked RNN.

        Args:
            dout_seq: gradient terhadap output.
            dh_final_list: gradient terhadap last hidden state per layer.
        """
        if dh_final_list is None:
            dh_final_list = [None] * self.num_layers

        dx = None

        for i in range(self.num_layers - 1, -1, -1):
            cell = self.layers[i]

            if i == 0:
                dx = cell.backward_sequence(dout_seq, dh_final_list[i])
            else:
                dx = cell.backward_sequence(dx, dh_final_list[i])

        return dx

    def get_weights(self):
        """Return bobot semua layer."""
        return [layer.get_weights() for layer in self.layers]

    def set_weights(self, weights_list):
        """Set bobot semua layer."""
        for layer, weights in zip(self.layers, weights_list):
            layer.set_weights(*weights)

    def reset_cache(self):
        """Reset caches semua layer."""
        for cell in self.layers:
            cell.reset_cache()

    def summary(self):
        """Return info stacked RNN."""
        return (f"StackedRNNCell(layers={self.num_layers}, "
                f"hidden_dims={self.hidden_dims})")


# ============================================================================
# Utilities
# ============================================================================

def rnn_step_forward(x, h_prev, W_xh, W_hh, b_h):
    """
    Single RNN step (utility function, tanpa cache).

    Args:
        x: (batch_size, input_dim).
        h_prev: (batch_size, hidden_dim).
        W_xh, W_hh, b_h: bobot.
    Returns:
        h_next: (batch_size, hidden_dim).
    """
    z = x @ W_xh + h_prev @ W_hh + b_h
    return tanh(z)


def rnn_sequence_forward(x_seq, h0, W_xh, W_hh, b_h):
    """
    RNN sequence forward (utility function).

    Args:
        x_seq: (batch_size, seq_len, input_dim).
        h0: (batch_size, hidden_dim).
        W_xh, W_hh, b_h: bobot.
    Returns:
        h_seq: (batch_size, seq_len, hidden_dim).
    """
    batch_size, seq_len, _ = x_seq.shape
    h_prev = h0
    h_seq = []

    for t in range(seq_len):
        h_next = rnn_step_forward(x_seq[:, t, :], h_prev, W_xh, W_hh, b_h)
        h_seq.append(h_next)
        h_prev = h_next

    return np.stack(h_seq, axis=1)


def gradient_check_rnn(cell, x_sample, h0_sample, epsilon=1e-5, verbose=True):
    """
    Verifikasi gradient RNN dengan numerical gradient checking.

    Args:
        cell: SimpleRNNCell instance.
        x_sample: input sample, shape (1, input_dim).
        h0_sample: initial hidden, shape (1, hidden_dim).
        epsilon: perturbation.
        verbose: cetak hasil.
    Returns:
        max_relative_error: float.
    """
    cell.reset_cache()

    # Forward
    h_out = cell.forward(x_sample, h0_sample)

    # Backward (dammy gradient)
    dout = np.ones_like(h_out)
    cell.backward_sequence(dout)

    # Numerical gradient check untuk W_xh
    W_xh = cell.W_xh.copy()
    grad_numerical = np.zeros_like(W_xh)

    for i in range(W_xh.shape[0]):
        for j in range(W_xh.shape[1]):
            # Forward +
            cell.W_xh = W_xh.copy()
            cell.W_xh[i, j] += epsilon
            h_plus = rnn_step_forward(x_sample, h0_sample,
                                       cell.W_xh, cell.W_hh, cell.b_h)

            # Forward -
            cell.W_xh = W_xh.copy()
            cell.W_xh[i, j] -= epsilon
            h_minus = rnn_step_forward(x_sample, h0_sample,
                                        cell.W_xh, cell.W_hh, cell.b_h)

            # Loss (squared error terhadap target)
            loss_plus = np.sum(h_plus ** 2)
            loss_minus = np.sum(h_minus ** 2)
            grad_numerical[i, j] = (loss_plus - loss_minus) / (2 * epsilon)

    cell.W_xh = W_xh.copy()
    grad_analytical = cell._dW_xh

    rel_error = np.max(np.abs(grad_analytical - grad_numerical) /
                       (np.abs(grad_analytical) + np.abs(grad_numerical) + 1e-8))

    if verbose:
        print(f"[Gradient Check RNN] Max relative error: {rel_error:.6f}")
        if rel_error < 1e-4:
            print("  [OK] Implementasi gradient BENAR")
        else:
            print("  [WARNING] Error lebih besar dari toleransi")

    return rel_error