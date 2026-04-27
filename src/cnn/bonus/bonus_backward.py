"""
Backward Propagation untuk seluruh layer CNN dari nol.
Melakukan traverse mundur melalui seluruh lapisan model,
memanggil method backward masing-masing layer secara berurutan.

Digunakan untuk menghitung gradient bobot dan input untuk training.
"""

import numpy as np


def backward_pass(model, loss_gradient, verbose=False):
    """
    Lakukan backward pass untuk seluruh model.

    Args:
        model: instance CNNScratch yang telah di-build.
        loss_gradient: gradient dari loss function, bentuk (batch_size, num_classes).
            Untuk cross-entropy + softmax: ini adalah (probs - labels).
        verbose (bool): cetak info tiap layer.
    Returns:
        dict_gradient: dictionary berisi gradient untuk setiap layer.
    """
    gradients = {}
    current_grad = loss_gradient

    if verbose:
        print("\n[Backward Pass] Memulai traverse mundur...")

    for i in range(len(model.layers) - 1, -1, -1):
        layer = model.layers[i]
        layer_name = layer.__class__.__name__
        layer_key = f'layer_{i}'

        if verbose:
            print(f"  Layer {i} {layer_name}: backward pass...")

        current_grad = layer.backward(current_grad)

        if hasattr(layer, 'get_grad_weights'):
            grad_weights = layer.get_grad_weights()
            if grad_weights is not None:
                if layer_name in ('Conv2D',):
                    gradients[layer_key] = {
                        'kernel': grad_weights[0],
                        'bias': grad_weights[1],
                    }
                elif layer_name == 'LocallyConnected2D':
                    gradients[layer_key] = {
                        'weights': grad_weights[0],
                        'bias': grad_weights[1],
                    }
                elif layer_name == 'Dense':
                    gradients[layer_key] = {
                        'weights': grad_weights[0],
                        'bias': grad_weights[1],
                    }

        if verbose:
            print(f"    -> gradient shape: {current_grad.shape}")

    return gradients


def compute_loss_gradient(logits, labels, loss_type='cross_entropy'):
    """
    Hitung gradient loss terhadap output logits.

    Args:
        logits: output model (sebelum softmax), bentuk (batch_size, num_classes).
        labels: ground truth (integer atau one-hot), bentuk (batch_size,) atau (batch_size, num_classes).
        loss_type (str): 'cross_entropy' (default) atau 'mse'.
    Returns:
        grad_logits: gradient terhadap logits, bentuk (batch_size, num_classes).
    """
    if loss_type == 'cross_entropy':
        probs = _stable_softmax(logits, axis=-1)

        if labels.ndim == 1:
            num_classes = logits.shape[-1]
            one_hot = np.eye(num_classes)[labels.astype(int)]
        else:
            one_hot = np.asarray(labels)

        grad_logits = probs - one_hot
        return grad_logits

    elif loss_type == 'mse':
        probs = _stable_softmax(logits, axis=-1)
        grad_logits = (probs - labels) / labels.shape[0]
        return grad_logits

    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def _stable_softmax(x, axis=-1):
    """Softmax yang aman secara numerik."""
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def gradient_checker(model, X_sample, label_sample, epsilon=1e-5, verbose=True):
    """
    Verifikasi implementasi backward pass menggunakan numerical gradient checking.

    Algoritma:
        Untuk setiap bobot W:
            grad_numerical[i] = (L(W+eps[i]) - L(W-eps[i])) / (2 * epsilon)
        Bandingkan dengan grad_analytical dari backward pass.

    Args:
        model: instance CNNScratch.
        X_sample: sample input, bentuk (1, H, W, C).
        label_sample: label integer.
        epsilon: small perturbation.
        verbose: cetak hasil per layer.
    Returns:
        max_relative_error: error maksimum antara numerical dan analytical gradient.
    """
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))

    output = model.forward(X_sample)
    loss_grad = compute_loss_gradient(output, np.array([label_sample]),
                                       loss_type='cross_entropy')
    gradients = backward_pass(model, loss_grad)

    max_error = 0.0

    for i, layer in enumerate(model.layers):
        layer_name = layer.__class__.__name__
        key = f'layer_{i}'

        if layer_name == 'Conv2D' and key in gradients:
            grad_k = gradients[key]['kernel']
            W = layer.kernel

            grad_numerical = np.zeros_like(W)
            flat_idx = 0
            for f1 in range(W.shape[0]):
                for f2 in range(W.shape[1]):
                    for f3 in range(W.shape[2]):
                        for f4 in range(W.shape[3]):
                            W_plus = W.copy()
                            W_minus = W.copy()
                            W_plus.flat[flat_idx] += epsilon
                            W_minus.flat[flat_idx] -= epsilon

                            layer.kernel = W_plus
                            out_plus = model.forward(X_sample)
                            loss_plus = -np.sum(out_plus[0] * np.eye(model.num_classes)[label_sample])

                            layer.kernel = W_minus
                            out_minus = model.forward(X_sample)
                            loss_minus = -np.sum(out_minus[0] * np.eye(model.num_classes)[label_sample])

                            grad_numerical.flat[flat_idx] = (loss_plus - loss_minus) / (2 * epsilon)
                            flat_idx += 1

            rel_error = np.max(np.abs(grad_k - grad_numerical) /
                               (np.abs(grad_k) + np.abs(grad_numerical) + 1e-8))
            max_error = max(max_error, rel_error)

            if verbose:
                print(f"  Layer {i} Conv2D kernel - Max relative error: {rel_error:.6f}")

            layer.kernel = W

        elif layer_name == 'Dense' and key in gradients:
            grad_W = gradients[key]['weights']
            W = layer.weights

            grad_numerical = np.zeros_like(W)
            for r in range(W.shape[0]):
                for c in range(W.shape[1]):
                    W_plus = W.copy()
                    W_minus = W.copy()
                    W_plus[r, c] += epsilon
                    W_minus[r, c] -= epsilon

                    layer.weights = W_plus
                    out_plus = model.forward(X_sample)
                    loss_plus = -np.sum(out_plus[0] * np.eye(model.num_classes)[label_sample])

                    layer.weights = W_minus
                    out_minus = model.forward(X_sample)
                    loss_minus = -np.sum(out_minus[0] * np.eye(model.num_classes)[label_sample])

                    grad_numerical[r, c] = (loss_plus - loss_minus) / (2 * epsilon)

            rel_error = np.max(np.abs(grad_W - grad_numerical) /
                               (np.abs(grad_W) + np.abs(grad_numerical) + 1e-8))
            max_error = max(max_error, rel_error)

            if verbose:
                print(f"  Layer {i} Dense weights - Max relative error: {rel_error:.6f}")

            layer.weights = W

    if verbose:
        print(f"\nGradient Checker: Max relative error = {max_error:.6f}")
        if max_error < 1e-4:
            print("  [OK] Implementasi backward pass BENAR")
        else:
            print("  [WARNING] Error lebih besar dari toleransi. Periksa implementasi!")

    return max_error


def sgd_update(layer, learning_rate=0.01):
    """
    Update bobot satu layer menggunakan Stochastic Gradient Descent (SGD).

    Args:
        layer: objek layer (Conv2D, Dense, dst).
        learning_rate: laju pembelajaran.
    """
    layer_name = layer.__class__.__name__

    if layer_name == 'Conv2D':
        if hasattr(layer, '_dkernel'):
            layer.kernel -= learning_rate * layer._dkernel
            layer.bias -= learning_rate * layer._dbias

    elif layer_name == 'LocallyConnected2D':
        if hasattr(layer, '_dweights'):
            layer.weights -= learning_rate * layer._dweights
            layer.bias -= learning_rate * layer._dbias

    elif layer_name == 'Dense':
        if hasattr(layer, '_grad_weights'):
            layer.weights -= learning_rate * layer._grad_weights
            layer.bias -= learning_rate * layer._grad_bias


def adam_update(layer, t, m_w, v_w, m_b, v_b,
                beta1=0.9, beta2=0.999, epsilon=1e-8, lr=0.001):
    """
    Update bobot satu layer menggunakan Adam optimizer.

    Args:
        layer: objek layer.
        t: timestep (batch ke berapa).
        m_w, v_w: momentum dan velocity untuk weights.
        m_b, v_b: momentum dan velocity untuk bias.
        beta1, beta2: momentum coefficients.
        epsilon: numerical stability.
        lr: learning rate.
    Returns:
        updated m_w, v_w, m_b, v_b
    """
    layer_name = layer.__class__.__name__

    if layer_name == 'Conv2D':
        g_w = layer._dkernel
        g_b = layer._dbias
    elif layer_name == 'Dense':
        g_w = layer._grad_weights
        g_b = layer._grad_bias
    else:
        return m_w, v_w, m_b, v_b

    m_w = beta1 * m_w + (1 - beta1) * g_w
    v_w = beta2 * v_w + (1 - beta2) * (g_w ** 2)
    m_b = beta1 * m_b + (1 - beta1) * g_b
    v_b = beta2 * v_b + (1 - beta2) * (g_b ** 2)

    m_w_hat = m_w / (1 - beta1 ** t)
    v_w_hat = v_w / (1 - beta2 ** t)
    m_b_hat = m_b / (1 - beta1 ** t)
    v_b_hat = v_b / (1 - beta2 ** t)

    if layer_name == 'Conv2D':
        layer.kernel -= lr * m_w_hat / (np.sqrt(v_w_hat) + epsilon)
        layer.bias -= lr * m_b_hat / (np.sqrt(v_b_hat) + epsilon)
    elif layer_name == 'Dense':
        layer.weights -= lr * m_w_hat / (np.sqrt(v_w_hat) + epsilon)
        layer.bias -= lr * m_b_hat / (np.sqrt(v_b_hat) + epsilon)

    return m_w, v_w, m_b, v_b


def train_step(model, X_batch, y_batch, optimizer='sgd',
               learning_rate=0.01, **optimizer_kwargs):
    """
    Satu langkah training (forward + backward + update).

    Args:
        model: instance CNNScratch.
        X_batch: batch input, bentuk (batch_size, H, W, C).
        y_batch: batch label (integer), bentuk (batch_size,).
        optimizer: 'sgd' atau 'adam'.
        learning_rate: learning rate.
        **optimizer_kwargs: argumen tambahan untuk optimizer.
    Returns:
        loss: nilai loss untuk batch ini.
    """
    output = model.forward(X_batch)

    num_classes = model.num_classes
    probs = _stable_softmax(output, axis=-1)
    labels_onehot = np.eye(num_classes)[y_batch.astype(int)]

    loss = -np.mean(np.sum(labels_onehot * np.log(probs + 1e-12), axis=1))

    loss_grad = probs - labels_onehot
    gradients = backward_pass(model, loss_grad)

    if optimizer == 'sgd':
        for layer in model.layers:
            sgd_update(layer, learning_rate)
    elif optimizer == 'adam':
        t = optimizer_kwargs.get('t', 1)
        for i, layer in enumerate(model.layers):
            m_w = optimizer_kwargs.get(f'm_w_{i}')
            sgd_update(layer, learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")

    return loss