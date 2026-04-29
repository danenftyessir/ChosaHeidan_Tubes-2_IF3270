"""
Training loop untuk decoder LSTM menggunakan Keras.
Mendukung variasi hyperparameter: jumlah layer dan hidden state.
Save bobot ke file .h5 untuk setiap variasi.
Sama strukturnya dengan RNN train.py.
"""

import os
import sys
import json
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))
from caption_preprocess import START_IDX, END_IDX, PAD_IDX


def build_model(vocab_size, embed_dim, hidden_dim, num_layers,
               feature_dim, seq_max_length, architecture='preinject', dropout=0.3):
    """
    Build LSTM decoder model sesuai konfigurasi.

    Args:
        vocab_size (int): ukuran vocabulary.
        embed_dim (int): dimensi embedding.
        hidden_dim (int): dimensi hidden state.
        num_layers (int): jumlah layer LSTM.
        feature_dim (int): dimensi CNN feature.
        seq_max_length (int): panjang maksimal sequence.
        architecture (str): arsitektur ('preinject', 'initinject').
        dropout (float): dropout rate.
    Returns:
        Keras Model instance.
    """
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    from model_keras import build_lstm_model

    model = build_lstm_model(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        feature_dim=feature_dim,
        seq_max_length=seq_max_length,
        architecture=architecture,
        dropout=dropout
    )
    return model


def compile_model(model, learning_rate=0.001):
    """Compile model dengan optimizer dan loss."""
    from tensorflow.keras.optimizers import Adam

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def create_model_name(num_layers, hidden_dim, architecture='preinject'):
    """Buat nama model untuk file saving."""
    return f"lstm_l{num_layers}_h{hidden_dim}_{architecture}"


def train_single_model(cnn_features, train_seq, train_labels,
                      val_cnn_features, val_seq, val_labels,
                      vocab_size, embed_dim, hidden_dim, num_layers,
                      feature_dim=2048, seq_max_length=40,
                      epochs=30, batch_size=64, lr=0.001,
                      model_name=None, weights_dir='weights/lstm',
                      architecture='preinject', dropout=0.3,
                      verbose=1):
    """
    Train satu konfigurasi LSTM decoder.

    Args:
        cnn_features (np.ndarray): CNN features untuk training.
        train_seq (np.ndarray): token sequences untuk training.
        train_labels (np.ndarray): labels untuk training.
        val_cnn_features: CNN features untuk validation.
        val_seq: token sequences untuk validation.
        val_labels: labels untuk validation.
        vocab_size, embed_dim, hidden_dim, num_layers: hyperparameter.
        feature_dim (int): dimensi CNN feature.
        seq_max_length (int): panjang maksimal sequence.
        epochs (int): jumlah epoch.
        batch_size (int): ukuran batch.
        lr (float): learning rate.
        model_name (str): nama model.
        weights_dir (str): direktori untuk menyimpan bobot.
        architecture (str): arsitektur model.
        dropout (float): dropout rate.
        verbose (int): verbose level.
    Returns:
        tuple: (trained_model, history).
    """
    if model_name is None:
        model_name = create_model_name(num_layers, hidden_dim, architecture)

    print(f"\n{'='*60}")
    print(f"[Training] {model_name}")
    print(f"  embed_dim={embed_dim}, hidden_dim={hidden_dim}, "
          f"num_layers={num_layers}, epochs={epochs}, batch_size={batch_size}")
    print(f"{'='*60}")

    model = build_model(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        feature_dim=feature_dim,
        seq_max_length=seq_max_length,
        architecture=architecture,
        dropout=dropout
    )
    model = compile_model(model, lr)

    start_time = time.time()

    history = model.fit(
        [cnn_features, train_seq],
        train_labels,
        validation_data=([val_cnn_features, val_seq], val_labels),
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
        callbacks=[
            create_early_stopping(patience=5, min_delta=0.001),
            create_model_checkpoint(weights_dir, model_name)
        ]
    )

    elapsed = time.time() - start_time
    print(f"\n[Training] {model_name} selesai dalam {elapsed:.1f} detik")

    os.makedirs(weights_dir, exist_ok=True)
    weights_path = os.path.join(weights_dir, f"{model_name}.h5")
    model.save_weights(weights_path)
    print(f"[Training] Bobot disimpan ke: {weights_path}")

    history_path = os.path.join(weights_dir, f"{model_name}_history.json")
    save_history(history, history_path)

    return model, history


def create_early_stopping(patience=5, min_delta=0.001):
    """Buat early stopping callback."""
    try:
        from tensorflow.keras.callbacks import EarlyStopping
        return EarlyStopping(
            monitor='val_loss',
            patience=patience,
            min_delta=min_delta,
            restore_best_weights=True,
            verbose=1
        )
    except ImportError:
        return None


def create_model_checkpoint(weights_dir, model_name):
    """Buat model checkpoint callback."""
    try:
        from tensorflow.keras.callbacks import ModelCheckpoint
        os.makedirs(weights_dir, exist_ok=True)
        path = os.path.join(weights_dir, f"{model_name}_best.h5")
        return ModelCheckpoint(
            path,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True,
            verbose=0
        )
    except ImportError:
        return None


def save_history(history, path):
    """Simpan training history ke JSON."""
    hist_dict = {}
    for key, values in history.history.items():
        hist_dict[key] = [float(v) for v in values]

    with open(path, 'w') as f:
        json.dump(hist_dict, f, indent=2)
    print(f"[Training] History disimpan ke: {path}")


def train_with_variations(cnn_features_train, train_seq, train_labels,
                        val_cnn_features, val_seq, val_labels,
                        vocab_size, embed_dim=256,
                        layer_variations=None, hidden_variations=None,
                        feature_dim=2048, seq_max_length=40,
                        epochs=30, batch_size=64, lr=0.001,
                        weights_dir='weights/lstm',
                        architecture='preinject', dropout=0.3,
                        verbose=1):
    """
    Train LSTM decoder dengan variasi hyperparameter.

    Variasi:
        - Jumlah layer: 1, 2, 3
        - Hidden state: 128, 512

    Total: 3 layer variations × 2 hidden variations = 6 variasi per architecture.

    Args:
        cnn_features_train (np.ndarray): CNN features training.
        train_seq (np.ndarray): sequences training.
        train_labels (np.ndarray): labels training.
        val_cnn_features: CNN features validation.
        val_seq: sequences validation.
        val_labels: labels validation.
        vocab_size (int): ukuran vocabulary.
        embed_dim (int): dimensi embedding.
        layer_variations (list): list jumlah layer. Default [1, 2, 3].
        hidden_variations (list): list hidden dimensions. Default [128, 512].
        feature_dim (int): dimensi CNN feature.
        seq_max_length (int): panjang sequence.
        epochs (int): jumlah epoch per variasi.
        batch_size (int): ukuran batch.
        lr (float): learning rate.
        weights_dir (str): direktori menyimpan bobot.
        architecture (str): arsitektur model.
        dropout (float): dropout rate.
        verbose (int): verbose level.
    Returns:
        dict: {model_name: (model, history)}.
    """
    if layer_variations is None:
        layer_variations = [1, 2, 3]
    if hidden_variations is None:
        hidden_variations = [128, 512]

    results = {}

    print(f"\n{'='*60}")
    print(f"[Training Variations] {architecture}")
    print(f"  Layer variations: {layer_variations}")
    print(f"  Hidden variations: {hidden_variations}")
    print(f"  Total models: {len(layer_variations) * len(hidden_variations)}")
    print(f"{'='*60}")

    for num_layers in layer_variations:
        for hidden_dim in hidden_variations:
            model_name = create_model_name(num_layers, hidden_dim, architecture)

            model, history = train_single_model(
                cnn_features=cnn_features_train,
                train_seq=train_seq,
                train_labels=train_labels,
                val_cnn_features=val_cnn_features,
                val_seq=val_seq,
                val_labels=val_labels,
                vocab_size=vocab_size,
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                feature_dim=feature_dim,
                seq_max_length=seq_max_length,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                model_name=model_name,
                weights_dir=weights_dir,
                architecture=architecture,
                dropout=dropout,
                verbose=verbose
            )

            results[model_name] = {
                'model': model,
                'history': history,
                'config': {
                    'embed_dim': embed_dim,
                    'hidden_dim': hidden_dim,
                    'num_layers': num_layers,
                    'feature_dim': feature_dim,
                    'seq_max_length': seq_max_length,
                    'architecture': architecture,
                }
            }

    print(f"\n[Training Variations] Selesai! {len(results)} model dilatih.")

    return results


def compare_training_results(results_dict, save_path=None):
    """Bandingkan training results dari beberapa variasi model."""
    comparison = {}

    for model_name, result in results_dict.items():
        history = result.get('history')
        config = result.get('config', {})

        if history is None:
            continue

        train_loss = history.history.get('loss', [])
        val_loss = history.history.get('val_loss', [])
        train_acc = history.history.get('accuracy', [])
        val_acc = history.history.get('val_accuracy', [])

        comparison[model_name] = {
            'final_train_loss': train_loss[-1] if train_loss else None,
            'final_val_loss': val_loss[-1] if val_loss else None,
            'final_train_acc': train_acc[-1] if train_acc else None,
            'final_val_acc': val_acc[-1] if val_acc else None,
            'best_val_loss': min(val_loss) if val_loss else None,
            'epochs_trained': len(train_loss),
            'hidden_dim': config.get('hidden_dim'),
            'num_layers': config.get('num_layers'),
        }

    print(f"\n{'='*80}")
    print(f"{'Model':<30} {'Layers':>6} {'Hidden':>6} {'TrainLoss':>10} "
          f"{'ValLoss':>10} {'BestVal':>10}")
    print(f"{'-'*80}")
    for name, data in sorted(comparison.items()):
        print(f"{name:<30} {data['num_layers']:>6} {data['hidden_dim']:>6} "
              f"{data['final_train_loss'] or 0:>10.4f} "
              f"{data['final_val_loss'] or 0:>10.4f} "
              f"{data['best_val_loss'] or 0:>10.4f}")
    print(f"{'='*80}\n")

    if save_path:
        with open(save_path, 'w') as f:
            json.dump(comparison, f, indent=2)
        print(f"Comparison disimpan ke: {save_path}")

    return comparison


def main():
    """Contoh training pipeline untuk LSTM decoder."""
    print("[Main] Siapkan data sebelum training:")
    print("  1. Extract CNN features dengan feature_extract.py")
    print("  2. Build vocabulary dengan caption_preprocess.py")
    print("  3. Load data dan prepare sequences")
    print("  4. Jalankan train_with_variations()")


if __name__ == '__main__':
    main()