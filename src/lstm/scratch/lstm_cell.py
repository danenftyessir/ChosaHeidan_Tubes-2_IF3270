"""
LSTM Cell dari nol (NumPy only).
Mendukung forward pass, backward pass (BPTT), dan batch inference.
Batch-first: input/output shape menggunakan (batch_size, ...).
Arsitektur LSTM standar dengan 4 gate: forget, input, output, cell candidate.
"""

import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))
from activations import sigmoid, d_sigmoid, tanh, d_tanh


# ============================================================================
# LSTM Cell
# ============================================================================

class LSTMCell:
    """
    LSTM Cell — Long Short-Term Memory unit.

    Forward pass (per timestep):
        Gates computation (concatenate [x_t, h_{t-1}]):
            f_t = sigmoid(W_f @ [x_t, h_{t-1}] + b_f)     # forget gate
            i_t = sigmoid(W_i @ [x_t, h_{t-1}] + b_i)     # input gate
            o_t = sigmoid(W_o @ [x_t, h_{t-1}] + b_o)     # output gate
            g_t = tanh(W_g @ [x_t, h_{t-1}] + b_g)        # cell candidate

        State update:
            c_t = f_t * c_{t-1} + i_t * g_t                # cell state
            h_t = o_t * tanh(c_t)                          # hidden state

    Bentuk data (batch-first):
        Input x_t:      shape (batch_size, input_dim)
        Hidden h_{t-1}: shape (batch_size, hidden_dim)
        Cell c_{t-1}:   shape (batch_size, hidden_dim)
        Hidden h_t:     shape (batch_size, hidden_dim)
        Cell c_t:       shape (batch_size, hidden_dim)

    Keras weight format:
        kernel: concat([W_f, W_i, W_o, W_g]) → shape (input_dim + hidden_dim, 4*hidden_dim)
        recurrent_kernel: concat([U_f, U_i, U_o, U_g]) → shape (hidden_dim, 4*hidden_dim)
        bias: [b_f, b_i, b_o, b_g] → shape (4*hidden_dim,)

    Backward pass (BPTT):
        - dL/dW_f, dL/dW_i, dL/dW_o, dL/dW_g via backprop through time
        - Gradient terhadap input dan previous hidden/cell state

    Args:
        input_dim (int): dimensi input (embed_dim).
        hidden_dim (int): dimensi hidden state.
        return_sequences (bool): return seluruh hidden states atau hanya last.
        name (str): nama cell untuk debugging.
    """

    def __init__(self, input_dim, hidden_dim, return_sequences=False, name="lstm"):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.return_sequences = return_sequences
        self.name = name

        # Bobot: concatenated gates
        # Kernel: [W_f, W_i, W_o, W_g] → shape (input_dim, 4*hidden_dim)
        # Recurrent: [U_f, U_i, U_o, U_g] → shape (hidden_dim, 4*hidden_dim)
        # Bias: [b_f, b_i, b_o, b_g] → shape (4*hidden_dim,)
        self.kernel = None       # (input_dim, 4*hidden_dim)
        self.recurrent_kernel = None  # (hidden_dim, 4*hidden_dim)
        self.bias = None          # (4*hidden_dim,)
        self._init_weights()

        # Cache untuk backward pass
        self.x_caches = []        # x_t per timestep
        self.h_prev_caches = []   # h_{t-1} per timestep
        self.c_prev_caches = []   # c_{t-1} per timestep
        self.f_caches = []        # forget gate output
        self.i_caches = []        # input gate output
        self.o_caches = []        # output gate output
        self.g_caches = []        # cell candidate output
        self.c_caches = []        # cell state
        self.h_caches = []        # hidden state
        self.xh_caches = []       # concatenated [x_t, h_{t-1}]
        self.batch_size = None

    def _init_weights(self):
        """
        Inisialisasi bobot dengan Glorot initialization.
        """
        # Total input = input_dim + hidden_dim
        fan_in = self.input_dim + self.hidden_dim
        fan_out = self.hidden_dim
        scale = np.sqrt(2.0 / fan_in)

        # Kernel: (input_dim, 4*hidden_dim)
        self.kernel = np.random.randn(self.input_dim, 4 * self.hidden_dim).astype(np.float64) * scale

        # Recurrent kernel: (hidden_dim, 4*hidden_dim)
        scale_rec = np.sqrt(2.0 / (self.hidden_dim + self.hidden_dim))
        self.recurrent_kernel = np.random.randn(self.hidden_dim, 4 * self.hidden_dim).astype(np.float64) * scale_rec

        # Bias: (4*hidden_dim,) — forget gate bias = 1 untuk initialize "remember"
        self.bias = np.zeros(4 * self.hidden_dim, dtype=np.float64)
        # Set forget gate bias = 1 (standard LSTM initialization)
        self.bias[:self.hidden_dim] = 1.0

    def _split_gates(self, gates):
        """
        Split concatenated gates output.

        Args:
            gates: (batch_size, 4*hidden_dim)
        Returns:
            (f, i, o, g) masing-masing (batch_size, hidden_dim)
        """
        f = sigmoid(gates[:, :self.hidden_dim])
        i = sigmoid(gates[:, self.hidden_dim:2*self.hidden_dim])
        o = sigmoid(gates[:, 2*self.hidden_dim:3*self.hidden_dim])
        g = tanh(gates[:, 3*self.hidden_dim:])
        return f, i, o, g

    def forward(self, x, h_prev=None, c_prev=None):
        """
        Forward pass untuk SATU timestep.

        Args:
            x: array numpy, bentuk (batch_size, input_dim).
            h_prev: hidden state sebelumnya, bentuk (batch_size, hidden_dim).
                   Jika None, gunakan zeros.
            c_prev: cell state sebelumnya, bentuk (batch_size, hidden_dim).
                   Jika None, gunakan zeros.
        Returns:
            (h_next, c_next): hidden state dan cell state baru.
        """
        batch_size = x.shape[0]
        self.batch_size = batch_size

        if h_prev is None:
            h_prev = np.zeros((batch_size, self.hidden_dim), dtype=np.float64)
        if c_prev is None:
            c_prev = np.zeros((batch_size, self.hidden_dim), dtype=np.float64)

        # Concatenate [x_t, h_{t-1}]
        xh = np.concatenate([x, h_prev], axis=1)  # (batch_size, input_dim + hidden_dim)

        # Gates: [f, i, o, g] = W @ xh + U @ h_prev + b
        gates = xh @ self.kernel + h_prev @ self.recurrent_kernel + self.bias

        # Split dan activate
        f, i, o, g = self._split_gates(gates)

        # Cell state update
        c_next = f * c_prev + i * g

        # Hidden state
        h_next = o * tanh(c_next)

        # Cache untuk backward
        self.x_caches.append(x)
        self.h_prev_caches.append(h_prev)
        self.c_prev_caches.append(c_prev)
        self.xh_caches.append(xh)
        self.f_caches.append(f)
        self.i_caches.append(i)
        self.o_caches.append(o)
        self.g_caches.append(g)
        self.c_caches.append(c_next)
        self.h_caches.append(h_next)

        return h_next, c_next

    def forward_sequence(self, x_seq, h0=None, c0=None):
        """
        Forward pass untuk SELURUH sequence.

        Args:
            x_seq: array numpy, bentuk (batch_size, seq_len, input_dim).
            h0: hidden state awal, bentuk (batch_size, hidden_dim).
            c0: cell state awal, bentuk (batch_size, hidden_dim).
        Returns:
            h_seq: (batch_size, seq_len, hidden_dim) jika return_sequences
                   else (batch_size, hidden_dim) last hidden state.
            c_final: (batch_size, hidden_dim) last cell state.
        """
        batch_size, seq_len, _ = x_seq.shape

        if h0 is None:
            h0 = np.zeros((batch_size, self.hidden_dim), dtype=np.float64)
        if c0 is None:
            c0 = np.zeros((batch_size, self.hidden_dim), dtype=np.float64)

        self.reset_cache()
        h_prev = h0
        c_prev = c0
        h_all = []

        for t in range(seq_len):
            x_t = x_seq[:, t, :]
            h_t, c_t = self.forward(x_t, h_prev, c_prev)
            h_all.append(h_t)
            h_prev = h_t
            c_prev = c_t

        h_seq = np.stack(h_all, axis=1)

        if self.return_sequences:
            return h_seq, c_prev
        else:
            return h_seq[:, -1, :], c_prev

    def backward(self, dout, dc_next=None):
        """
        Backward pass untuk SATU timestep (tidak dipakai langsung).

        Args:
            dout: gradient terhadap output hidden state, (batch_size, hidden_dim).
            dc_next: gradient terhadap cell state dari timestep berikutnya.
        Returns:
            dx: gradient terhadap input x_t.
        """
        # Implementation untuk single step (dipanggil oleh backward_sequence)
        return np.zeros((self.batch_size, self.input_dim)), \
               np.zeros((self.batch_size, self.hidden_dim)), \
               np.zeros((self.batch_size, self.hidden_dim))

    def backward_sequence(self, dout_seq, dh_final=None, dc_final=None):
        """
        Backward pass untuk SELURUH sequence (BPTT).

        Args:
            dout_seq: gradient terhadap hidden states output,
                      bentuk (batch_size, seq_len, hidden_dim).
            dh_final: gradient terhadap hidden state terakhir dari layer atas.
            dc_final: gradient terhadap cell state terakhir dari layer atas.
        Returns:
            dx_seq: gradient terhadap input sequence,
                    bentuk (batch_size, seq_len, input_dim).
        """
        seq_len = len(self.x_caches)
        batch_size = self.batch_size

        if dout_seq is None:
            dout_seq = np.zeros((batch_size, seq_len, self.hidden_dim), dtype=np.float64)

        # Akumulasi gradients
        dkernel_accum = np.zeros_like(self.kernel)
        drecurrent_kernel_accum = np.zeros_like(self.recurrent_kernel)
        dbias_accum = np.zeros_like(self.bias)

        dx_seq = np.zeros((batch_size, seq_len, self.input_dim), dtype=np.float64)

        # Gradient propagation ke belakang
        dh_next = dh_final if dh_final is not None else np.zeros((batch_size, self.hidden_dim), dtype=np.float64)
        dc_next = dc_final if dc_final is not None else np.zeros((batch_size, self.hidden_dim), dtype=np.float64)

        # Backward through time
        for t in range(seq_len - 1, -1, -1):
            f = self.f_caches[t]
            i = self.i_caches[t]
            o = self.o_caches[t]
            g = self.g_caches[t]
            c = self.c_caches[t]
            c_prev = self.c_prev_caches[t]
            h_prev = self.h_prev_caches[t]
            xh = self.xh_caches[t]

            # Total gradient terhadap hidden state pada timestep t
            dh = dout_seq[:, t, :] + dh_next

            # Gradient terhadap cell state
            # dh/do = tanh(c) * do/dh
            do = dh * tanh(c)

            # Gradient terhadap output gate sigmoid
            d_o_pre = do * d_sigmoid(o)

            # Gradient terhadap cell state: dc = f*dc_prev + i*g
            # dh/dc = o * d_tanh(c)
            dc_from_h = dh * o * d_tanh(c)
            dc = dc_from_h + dc_next

            # Gradient terhadap gates
            d_f_pre = dc * c_prev * d_sigmoid(f)  # forget gate
            d_i_pre = dc * g * d_sigmoid(i)        # input gate
            d_g_pre = dc * i * d_tanh(g)             # cell candidate

            # Concatenate gradients untuk gates
            d_gates_pre = np.concatenate([d_f_pre, d_i_pre, d_o_pre, d_g_pre], axis=1)

            # Gradient terhadap kernel
            dkernel_t = xh.T @ d_gates_pre

            # Gradient terhadap recurrent_kernel
            drec_t = h_prev.T @ d_gates_pre

            # Gradient terhadap bias
            dbias_t = np.sum(d_gates_pre, axis=0)

            # Akumulasi
            dkernel_accum += dkernel_t
            drecurrent_kernel_accum += drec_t
            dbias_accum += dbias_t

            # Gradient terhadap xh: dxh = d_gates_pre @ kernel^T
            dxh = d_gates_pre @ self.kernel.T

            # Split: dx dan dh_prev
            dx_t = dxh[:, :self.input_dim]
            dh_prev = dxh[:, self.input_dim:]

            # Gradient terhadap c_prev: dari forget gate
            dc_prev = dc * f

            # Gradient terhadap h_prev: dari recurrent connection
            dh_prev_rec = d_gates_pre @ self.recurrent_kernel.T
            dh_prev = dh_prev[:, self.input_dim:] + dh_prev_rec

            dx_seq[:, t, :] = dx_t
            dh_next = dh_prev
            dc_next = dc_prev

        self._dkernel = dkernel_accum
        self._drecurrent_kernel = drecurrent_kernel_accum
        self._dbias = dbias_accum
        self._dx_seq = dx_seq

        return dx_seq

    def get_weights(self):
        """
        Kembalikan bobot cell (format Keras: kernel, recurrent_kernel, bias).

        Returns:
            tuple: (kernel, recurrent_kernel, bias).
        """
        return self.kernel, self.recurrent_kernel, self.bias

    def set_weights(self, kernel, recurrent_kernel, bias):
        """
        Set bobot dari sumber eksternal (Keras).
        Args:
            kernel: (input_dim, 4*hidden_dim)
            recurrent_kernel: (hidden_dim, 4*hidden_dim)
            bias: (4*hidden_dim,)
        """
        self.kernel = kernel.astype(np.float64)
        self.recurrent_kernel = recurrent_kernel.astype(np.float64)
        self.bias = bias.astype(np.float64)
        self.input_dim = kernel.shape[0]
        self.hidden_dim = recurrent_kernel.shape[0]

    def get_grad_weights(self):
        """
        Kembalikan gradient bobot (harus dipanggil setelah backward_sequence).

        Returns:
            tuple: (dkernel, drecurrent_kernel, dbias).
        """
        return self._dkernel, self._drecurrent_kernel, self._dbias

    def reset_cache(self):
        """Reset caches (panggil sebelum forward sequence baru)."""
        self.x_caches = []
        self.h_prev_caches = []
        self.c_prev_caches = []
        self.f_caches = []
        self.i_caches = []
        self.o_caches = []
        self.g_caches = []
        self.c_caches = []
        self.h_caches = []
        self.xh_caches = []
        self.batch_size = None

    def summary(self):
        """Kembalikan info cell."""
        return (f"LSTMCell(input_dim={self.input_dim}, "
                f"hidden_dim={self.hidden_dim}, "
                f"return_sequences={self.return_sequences})")


# ============================================================================
# Stacked LSTM Layers
# ============================================================================

class StackedLSTMCell:
    """
    Stacked LSTM (Deep LSTM): multiple LSTM layers.
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
            self.layers.append(LSTMCell(
                input_dim=current_dim,
                hidden_dim=hdim,
                return_sequences=return_sequences or (i < len(hidden_dims) - 1),
                name=f"lstm_layer_{i}"
            ))
            current_dim = hdim

        self.hidden_dims = hidden_dims
        self.num_layers = len(hidden_dims)

    def forward_sequence(self, x_seq, h0_list=None, c0_list=None):
        """
        Forward pass untuk stacked LSTM.

        Args:
            x_seq: shape (batch_size, seq_len, input_dim).
            h0_list: list of initial hidden states per layer.
            c0_list: list of initial cell states per layer.
        Returns:
            output_seq: shape (batch_size, seq_len, final_hidden_dim)
                        atau (batch_size, final_hidden_dim).
        """
        if h0_list is None:
            h0_list = [None] * self.num_layers
        if c0_list is None:
            c0_list = [None] * self.num_layers

        current = x_seq
        h_final_list = []
        c_final_list = []

        for i, cell in enumerate(self.layers):
            cell.reset_cache()
            output, c_out = cell.forward_sequence(current, h0_list[i], c0_list[i])
            current = output if cell.return_sequences else output[:, np.newaxis, :]
            h_final_list.append(output[:, -1, :] if output.ndim == 3 else output)
            c_final_list.append(c_out)

        return current, c_final_list[-1]

    def backward_sequence(self, dout_seq, dh_final_list=None, dc_final_list=None):
        """Backward pass untuk stacked LSTM."""
        if dh_final_list is None:
            dh_final_list = [None] * self.num_layers
        if dc_final_list is None:
            dc_final_list = [None] * self.num_layers

        dx = None

        for i in range(self.num_layers - 1, -1, -1):
            cell = self.layers[i]
            if i == 0:
                dx = cell.backward_sequence(dout_seq, dh_final_list[i], dc_final_list[i])
            else:
                dx = cell.backward_sequence(dx, dh_final_list[i], dc_final_list[i])

        return dx

    def get_weights(self):
        return [layer.get_weights() for layer in self.layers]

    def set_weights(self, weights_list):
        for layer, weights in zip(self.layers, weights_list):
            layer.set_weights(*weights)

    def reset_cache(self):
        for cell in self.layers:
            cell.reset_cache()

    def summary(self):
        return (f"StackedLSTMCell(layers={self.num_layers}, "
                f"hidden_dims={self.hidden_dims})")


# ============================================================================
# Utilities
# ============================================================================

def lstm_step_forward(x, h_prev, c_prev, kernel, recurrent_kernel, bias):
    """
    Single LSTM step (utility function, tanpa cache).

    Args:
        x: (batch_size, input_dim).
        h_prev: (batch_size, hidden_dim).
        c_prev: (batch_size, hidden_dim).
        kernel, recurrent_kernel, bias: bobot LSTM.
    Returns:
        (h_next, c_next): hidden dan cell state.
    """
    hidden_dim = recurrent_kernel.shape[0]
    xh = np.concatenate([x, h_prev], axis=1)
    gates = xh @ kernel + h_prev @ recurrent_kernel + bias

    f = sigmoid(gates[:, :hidden_dim])
    i = sigmoid(gates[:, hidden_dim:2*hidden_dim])
    o = sigmoid(gates[:, 2*hidden_dim:3*hidden_dim])
    g = tanh(gates[:, 3*hidden_dim:])

    c_next = f * c_prev + i * g
    h_next = o * tanh(c_next)

    return h_next, c_next


def gradient_check_lstm(cell, x_sample, h0_sample, c0_sample,
                       epsilon=1e-5, verbose=True):
    """
    Verifikasi gradient LSTM dengan numerical gradient checking.

    Args:
        cell: LSTMCell instance.
        x_sample: input sample, shape (1, input_dim).
        h0_sample: initial hidden, shape (1, hidden_dim).
        c0_sample: initial cell, shape (1, hidden_dim).
        epsilon: perturbation.
        verbose: cetak hasil.
    Returns:
        max_relative_error: float.
    """
    cell.reset_cache()
    h_out, c_out = cell.forward(x_sample, h0_sample, c0_sample)

    # Dummy gradient
    dout = np.ones_like(h_out)
    cell.backward_sequence(dout)

    # Check gradient kernel
    kernel = cell.kernel.copy()
    grad_numerical = np.zeros_like(kernel)

    for i in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            cell.kernel = kernel.copy()
            cell.kernel[i, j] += epsilon
            h_plus, _ = lstm_step_forward(x_sample, h0_sample, c0_sample,
                                          cell.kernel, cell.recurrent_kernel, cell.bias)

            cell.kernel = kernel.copy()
            cell.kernel[i, j] -= epsilon
            h_minus, _ = lstm_step_forward(x_sample, h0_sample, c0_sample,
                                         cell.kernel, cell.recurrent_kernel, cell.bias)

            loss_plus = np.sum(h_plus ** 2)
            loss_minus = np.sum(h_minus ** 2)
            grad_numerical[i, j] = (loss_plus - loss_minus) / (2 * epsilon)

    cell.kernel = kernel.copy()
    grad_analytical = cell._dkernel

    rel_error = np.max(np.abs(grad_analytical - grad_numerical) /
                       (np.abs(grad_analytical) + np.abs(grad_numerical) + 1e-8))

    if verbose:
        print(f"[Gradient Check LSTM] Max relative error: {rel_error:.6f}")
        if rel_error < 1e-4:
            print("  [OK] Implementasi gradient BENAR")
        else:
            print("  [WARNING] Error lebih besar dari toleransi")

    return rel_error
