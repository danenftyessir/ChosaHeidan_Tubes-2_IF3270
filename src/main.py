"""
Tubes 2 IF3270 — Entry Point CLI
CNN, RNN, dan LSTM dari scratch (NumPy) + implementasi Keras.
Mendukung training, evaluasi, demo, dan benchmark dari command line.

Usage:
    python main.py train --dataset intel --arch conv2d --epochs 20
    python main.py train --dataset flickr8k --model lstm --epochs 30
    python main.py evaluate --dataset intel --weights weights/cnn/conv2d.h5
    python main.py demo --dataset flickr8k
    python main.py benchmark --dataset intel
    python main.py --help
"""

import argparse
import os
import sys


# ============================================================================
# Project root setup
# ============================================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')

# Add src/ to path
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, os.path.join(SRC_DIR, 'shared'))


# ============================================================================
# Banner / Quick-start teaching mode
# ============================================================================

BANNER = r"""
=======================================================================
  Tubes 2 IF3270 — CNN, RNN & LSTM dari Scratch (NumPy) + Keras
=======================================================================
  Available datasets : intel (Image Classification), flickr8k (Captioning)
  Available models   : cnn (Conv2D, LocallyConnected), rnn, lstm
  Sub-commands      : train, evaluate, demo, benchmark
=======================================================================
"""


def print_banner():
    print(BANNER)


# ============================================================================
# Train Command
# ============================================================================

def cmd_train(args):
    """Train a model (CNN, RNN, or LSTM)."""
    if args.dataset == 'intel':
        _train_cnn(args)
    elif args.dataset == 'flickr8k':
        _train_caption(args)
    else:
        print(f"[Error] Dataset '{args.dataset}' tidak dikenali.")
        print("Pilih: intel, flickr8k")
        sys.exit(1)


def _train_cnn(args):
    """Train CNN on Intel Image dataset."""
    from src.cnn.keras.train import train_single_cnn
    from src.shared.intel_preprocess import INTEL_CLASSES

    data_dir = args.data_dir or _get_default_data_dir('intel')

    if not os.path.exists(data_dir):
        print(f"[Error] Direktori data tidak ditemukan: {data_dir}")
        print("Gunakan --data-dir untuk menentukan path dataset Intel.")
        sys.exit(1)

    arch = getattr(args, 'arch', None) or 'conv2d'
    epochs = args.epochs or 30
    batch_size = args.batch_size or 32
    num_conv = args.num_conv_layers or 3
    num_filters = args.num_filters or 64
    kernel_size = tuple(args.kernel_size) if args.kernel_size else (3, 3)
    pooling = args.pooling_type or 'max'

    print(f"\n[Train CNN] Dataset: intel | Arch: {arch} | Epochs: {epochs}")
    print(f"  Konfigurasi: conv_layers={num_conv}, filters={num_filters}, "
          f"kernel={kernel_size}, pooling={pooling}")

    model, history, config = train_single_cnn(
        data_dir=data_dir,
        arch_type=arch,
        num_conv_layers=num_conv,
        num_filters=num_filters,
        kernel_size=kernel_size,
        pooling_type=pooling,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    print(f"\n[Train CNN] Selesai! Model: {config.get('name', 'unknown')}")
    return model, history


def _train_caption(args):
    """Train RNN/LSTM caption decoder on Flickr8k dataset."""
    if args.model == 'rnn':
        from src.rnn.keras.train import train_single_model as _train_fn
        from src.rnn.keras.model_keras import build_model as _build_fn
        weights_dir = os.path.join(PROJECT_ROOT, 'weights', 'rnn')
    elif args.model == 'lstm':
        from src.lstm.keras.train import train_single_model as _train_fn
        from src.lstm.keras.model_keras import build_model as _build_fn
        weights_dir = os.path.join(PROJECT_ROOT, 'weights', 'lstm')
    else:
        print(f"[Error] Model '{args.model}' tidak dikenali untuk dataset flickr8k.")
        print("Pilih: rnn, lstm")
        sys.exit(1)

    data_dir = args.data_dir or _get_default_data_dir('flickr8k')

    if not os.path.exists(data_dir):
        print(f"[Error] Direktori data tidak ditemukan: {data_dir}")
        print("Gunakan --data-dir untuk menentukan path dataset Flickr8k.")
        sys.exit(1)

    print(f"\n[Train Caption] Model: {args.model.upper()} | Dataset: flickr8k "
          f"| Epochs: {args.epochs}")

    # Load preprocessed data
    try:
        from src.shared.caption_preprocess import (
            load_flickr8k_dataset, build_vocabulary,
            create_train_val_split, create_batches
        )
    except ImportError as e:
        print(f"[Error] Gagal import caption_preprocess: {e}")
        sys.exit(1)

    print("[Train Caption] Memuat dataset Flickr8k...")
    images_df, captions_df = load_flickr8k_dataset(data_dir)

    print("[Train Caption] Membangun vocabulary...")
    vocab = build_vocabulary(captions_df, min_word_freq=args.min_word_freq or 5)
    vocab_size = len(vocab)

    print(f"[Train Caption] Vocabulary size: {vocab_size}")

    # Extract CNN features if not already cached
    from src.shared.feature_extract import extract_cnn_features_batch
    features_path = os.path.join(data_dir, 'cnn_features.npz')

    if os.path.exists(features_path):
        print("[Train Caption] Memuat CNN features dari cache...")
        data = dict(np.load(features_path, allow_pickle=True))
        cnn_features_train = data['train_features']
        cnn_features_val = data['val_features']
    else:
        print("[Train Caption] Mengekstrak CNN features (akan disimpan ke cache)...")
        print("  [Note] Jalankan feature_extract.py dulu untuk generate features.")
        cnn_features_train = np.zeros((len(images_df), 2048))
        cnn_features_val = np.zeros((len(images_df) // 5, 2048))

    # Create sequences
    seq_max_length = args.seq_max_length or 40
    train_captions = create_captions_tokenized(images_df, vocab, seq_max_length)
    train_seq, train_labels, val_seq, val_labels, val_features = _create_split(
        cnn_features_train, train_captions, split_ratio=0.2
    )

    # Train
    model, history = _train_fn(
        cnn_features=cnn_features_train[:len(train_seq)],
        train_seq=train_seq,
        train_labels=train_labels,
        val_cnn_features=val_features,
        val_seq=val_seq,
        val_labels=val_labels,
        vocab_size=vocab_size,
        embed_dim=args.embed_dim or 256,
        hidden_dim=args.hidden_dim or 512,
        num_layers=args.num_layers or 2,
        feature_dim=2048,
        seq_max_length=seq_max_length,
        epochs=args.epochs or 30,
        batch_size=args.batch_size or 64,
        lr=args.lr or 0.001,
        model_name=None,
        weights_dir=weights_dir,
        architecture=args.architecture or 'preinject',
        dropout=args.dropout or 0.3,
        verbose=1
    )
    print(f"\n[Train Caption] Selesai! Model: {args.model.upper()}")


def create_captions_tokenized(images_df, vocab, seq_max_length):
    """Tokenize captions for training."""
    from src.shared.caption_preprocess import START_IDX, END_IDX, PAD_IDX
    import pandas as pd

    captions = []
    for _, row in images_df.iterrows():
        tokens = [START_IDX]
        for cap in row.get('captions', []):
            tokens.extend([vocab.get(w, PAD_IDX) for w in cap.split()])
        tokens.append(END_IDX)
        if len(tokens) < seq_max_length:
            tokens += [PAD_IDX] * (seq_max_length - len(tokens))
        else:
            tokens = tokens[:seq_max_length]
        captions.append(tokens)
    return np.array(captions)


def _create_split(cnn_features, captions, split_ratio=0.2):
    """Split data into train/val."""
    n = len(captions)
    idx = np.random.permutation(n)
    split = int(n * (1 - split_ratio))
    train_idx, val_idx = idx[:split], idx[split:]
    return (captions[train_idx], captions[train_idx],
            captions[val_idx], captions[val_idx],
            cnn_features[val_idx])


# ============================================================================
# Evaluate Command
# ============================================================================

def cmd_evaluate(args):
    """Evaluate a trained model."""
    if args.dataset == 'intel':
        _evaluate_cnn(args)
    elif args.dataset == 'flickr8k':
        _evaluate_caption(args)
    else:
        print(f"[Error] Dataset '{args.dataset}' tidak dikenali.")
        sys.exit(1)


def _evaluate_cnn(args):
    """Evaluate CNN on Intel Image dataset."""
    from src.cnn.keras.evaluate import (
        evaluate_from_weights, evaluate_cnn_keras,
        load_model_weights
    )
    from src.cnn.keras.model_keras import build_cnn_conv2d, build_cnn_locallyconnected
    from src.shared.intel_preprocess import INTEL_CLASSES

    data_dir = args.data_dir or _get_default_data_dir('intel')
    weights_path = args.weights

    if not weights_path:
        # Auto-detect latest weights
        weights_path = _find_latest_weights('cnn')
        if not weights_path:
            print("[Error] Tidak ada bobot ditemukan di weights/cnn/. "
                  "Latih dulu dengan 'python main.py train --dataset intel'.")
            sys.exit(1)

    if not os.path.exists(data_dir):
        print(f"[Error] Direktori data tidak ditemukan: {data_dir}")
        sys.exit(1)
    if not os.path.exists(weights_path):
        print(f"[Error] File bobot tidak ditemukan: {weights_path}")
        sys.exit(1)

    # Detect arch from filename
    arch = 'conv2d'
    if 'locallyconnected' in weights_path.lower() or 'local' in weights_path.lower():
        arch = 'locallyconnected'

    print(f"\n[Evaluate CNN] Bobot: {os.path.basename(weights_path)}")
    print(f"  Dataset: {data_dir} | Split: {args.split}")

    # Build model
    if arch == 'conv2d':
        model, _ = build_cnn_conv2d(num_classes=6)
    else:
        model, _ = build_cnn_locallyconnected(num_classes=6)

    results = evaluate_from_weights(
        model=model,
        weights_path=weights_path,
        data_dir=data_dir,
        split=args.split or 'test',
        batch_size=args.batch_size or 32,
        class_names=INTEL_CLASSES,
        verbose=True
    )
    print(f"\n[Evaluate CNN] Macro F1: {results['macro_f1']:.4f}")
    return results


def _evaluate_caption(args):
    """Evaluate RNN/LSTM caption model on Flickr8k."""
    if args.model not in ('rnn', 'lstm'):
        print(f"[Error] Model '{args.model}' tidak dikenali.")
        sys.exit(1)

    if args.model == 'rnn':
        from src.rnn.keras.evaluate import evaluate_caption_model
    else:
        from src.lstm.keras.evaluate import evaluate_caption_model

    data_dir = args.data_dir or _get_default_data_dir('flickr8k')
    weights_path = args.weights

    if not weights_path:
        weights_path = _find_latest_weights(args.model)
        if not weights_path:
            print(f"[Error] Tidak ada bobot ditemukan di weights/{args.model}/.")
            sys.exit(1)

    if not os.path.exists(data_dir):
        print(f"[Error] Direktori data tidak ditemukan: {data_dir}")
        sys.exit(1)

    print(f"\n[Evaluate Caption] Model: {args.model.upper()} | Bobot: {os.path.basename(weights_path)}")

    results = evaluate_caption_model(
        weights_path=weights_path,
        data_dir=data_dir,
        split=args.split or 'test',
        batch_size=args.batch_size or 64,
        verbose=True
    )
    return results


# ============================================================================
# Demo Command
# ============================================================================

def cmd_demo(args):
    """Run interactive demo."""
    if args.dataset == 'intel':
        _demo_intel(args)
    elif args.dataset == 'flickr8k':
        _demo_flickr8k(args)
    else:
        print(f"[Error] Dataset '{args.dataset}' tidak dikenali.")
        sys.exit(1)


def _demo_intel(args):
    """Demo: classify an image with the trained CNN."""
    from src.cnn.keras.evaluate import load_model_weights
    from src.cnn.keras.model_keras import build_cnn_conv2d
    from src.shared.intel_preprocess import INTEL_CLASSES
    import matplotlib.image as mpimg

    weights_path = args.weights or _find_latest_weights('cnn')
    image_path = args.image

    if not weights_path:
        print("[Error] Tidak ada bobot ditemukan. Latih dulu model.")
        sys.exit(1)

    if not image_path or not os.path.exists(image_path):
        print("[Error] Path gambar tidak valid. Gunakan --image <path>.")
        sys.exit(1)

    # Load model
    model, _ = build_cnn_conv2d(num_classes=6)
    model = load_model_weights(model, weights_path)
    model.compile(optimizer='adam', loss='categorical_crossentropy')

    # Preprocess image
    from src.shared.preprocessing import preprocess_image
    img = preprocess_image(image_path, target_size=(150, 150))

    # Predict
    probs = model.predict(img[np.newaxis, ...], verbose=0)[0]
    pred_idx = np.argmax(probs)

    print(f"\n[Demo Intel] Gambar: {os.path.basename(image_path)}")
    print(f"  Prediksi: {INTEL_CLASSES[pred_idx]} (prob={probs[pred_idx]:.3f})")
    print("\n  Semua kelas:")
    for i, cls in enumerate(INTEL_CLASSES):
        bar = '#' * int(probs[i] * 30)
        print(f"    {cls:12s} [{bar:30s}] {probs[i]:.3f}")


def _demo_flickr8k(args):
    """Demo: generate caption for an image using RNN/LSTM."""
    if args.model not in ('rnn', 'lstm'):
        print(f"[Error] Model '{args.model}' tidak dikenali.")
        sys.exit(1)

    from src.lstm.keras.model_keras import build_model as build_lstm
    from src.rnn.keras.model_keras import build_model as build_rnn
    from src.shared.caption_preprocess import START_IDX, END_IDX, idx_to_word
    import matplotlib.image as mpimg

    weights_path = args.weights or _find_latest_weights(args.model)
    image_path = args.image

    if not weights_path:
        print(f"[Error] Tidak ada bobot ditemukan untuk model {args.model}.")
        sys.exit(1)

    if not image_path or not os.path.exists(image_path):
        print("[Error] Path gambar tidak valid. Gunakan --image <path>.")
        sys.exit(1)

    # Load model
    print(f"[Demo Caption] Memuat model {args.model.upper()} dari: {weights_path}")
    if args.model == 'lstm':
        model, _ = build_lstm(
            vocab_size=10000, embed_dim=256, hidden_dim=512,
            num_layers=2, feature_dim=2048,
            seq_max_length=40, architecture='preinject'
        )
    else:
        model, _ = build_rnn(
            vocab_size=10000, embed_dim=256, hidden_dim=512,
            num_layers=2, feature_dim=2048,
            seq_max_length=40, architecture='preinject'
        )

    # Load weights
    from tensorflow.keras.models import load_model
    try:
        model.load_weights(weights_path)
    except Exception:
        print("[Warning] Gagal load bobot penuh. Caption generation mungkin tidak akurat.")

    # Extract image feature
    print("[Demo Caption] Mengekstrak fitur gambar...")
    from src.shared.feature_extract import extract_cnn_features_batch
    features = extract_cnn_features_batch([image_path])

    # Greedy decode
    seq = [START_IDX]
    for _ in range(40):
        seq_input = np.array([seq + [0] * (40 - len(seq))])
        feat_input = features
        probs = model.predict([feat_input, seq_input], verbose=0)[0]
        next_word_idx = np.argmax(probs[len(seq) - 1])
        if next_word_idx == END_IDX:
            break
        seq.append(next_word_idx)

    caption = ' '.join(idx_to_word(w) for w in seq[1:] if w not in (START_IDX, END_IDX, 0))
    print(f"\n[Demo Caption] Caption generated:")
    print(f'  "{caption}"')

    # Display image
    if args.show:
        try:
            import matplotlib.pyplot as plt
            img = mpimg.imread(image_path)
            plt.figure(figsize=(8, 6))
            plt.imshow(img)
            plt.axis('off')
            plt.title(caption, fontsize=12)
            plt.show()
        except Exception:
            print("  (matplotlib display tidak tersedia)")


# ============================================================================
# Benchmark Command
# ============================================================================

def cmd_benchmark(args):
    """Run inference speed benchmark."""
    if args.dataset == 'intel':
        _benchmark_cnn(args)
    elif args.dataset == 'flickr8k':
        _benchmark_caption(args)
    else:
        print(f"[Error] Dataset '{args.dataset}' tidak dikenali.")
        sys.exit(1)


def _benchmark_cnn(args):
    """Benchmark CNN inference speed."""
    from src.cnn.keras.evaluate import benchmark_inference_speed
    from src.cnn.keras.model_keras import build_cnn_conv2d
    from src.shared.intel_preprocess import IntelImagePreprocessor

    data_dir = args.data_dir or _get_default_data_dir('intel')
    weights_path = args.weights or _find_latest_weights('cnn')

    if not os.path.exists(data_dir):
        print(f"[Error] Direktori data tidak ditemukan: {data_dir}")
        sys.exit(1)

    print(f"\n[Benchmark CNN] Dataset: {data_dir}")

    # Build model
    model, _ = build_cnn_conv2d(num_classes=6)

    if weights_path and os.path.exists(weights_path):
        from src.cnn.keras.evaluate import load_model_weights
        print(f"[Benchmark CNN] Memuat bobot: {weights_path}")
        model = load_model_weights(model, weights_path)

    # Load sample images
    preprocessor = IntelImagePreprocessor(data_dir, target_size=(150, 150))
    preprocessor.load_data()
    image_paths = (preprocessor.test_paths if hasattr(preprocessor, 'test_paths')
                   else preprocessor.val_paths)[:500]

    print(f"[Benchmark CNN] Menjalankan benchmark pada {len(image_paths)} gambar...")

    batch_sizes = [1, 8, 16, 32]
    results = benchmark_inference_speed(
        model, image_paths,
        batch_sizes=batch_sizes,
        num_runs=args.num_runs or 3
    )

    print(f"\n{'=' * 50}")
    print(f"{'Batch':>8} | {'Time/batch (ms)':>18} | {'Images/sec':>12}")
    print(f"{'-'* 50}")
    for bs, (t, ips) in results.items():
        print(f"{bs:>8} | {t*1000:>18.2f} | {ips:>12.1f}")
    print(f"{'=' * 50}")

    return results


def _benchmark_caption(args):
    """Benchmark RNN/LSTM caption inference speed."""
    if args.model not in ('rnn', 'lstm'):
        print(f"[Error] Model '{args.model}' tidak dikenali.")
        sys.exit(1)

    if args.model == 'rnn':
        from src.rnn.keras.evaluate import benchmark_caption_speed
    else:
        from src.lstm.keras.evaluate import benchmark_caption_speed

    weights_path = args.weights or _find_latest_weights(args.model)

    if not weights_path:
        print(f"[Error] Tidak ada bobot ditemukan untuk model {args.model}.")
        sys.exit(1)

    print(f"\n[Benchmark Caption] Model: {args.model.upper()}")

    results = benchmark_caption_speed(
        weights_path=weights_path,
        num_samples=args.num_samples or 100,
        verbose=True
    )
    return results


# ============================================================================
# Utility Functions
# ============================================================================

def _get_default_data_dir(dataset):
    """Get default data directory based on dataset type."""
    if dataset == 'intel':
        # Try common paths
        for path in [
            os.path.join(PROJECT_ROOT, 'data', 'intel'),
            os.path.join(PROJECT_ROOT, 'data', 'Intel-Images'),
            os.path.join(PROJECT_ROOT, 'data', 'seg_train'),
        ]:
            if os.path.exists(path):
                return path
        return os.path.join(PROJECT_ROOT, 'data', 'intel')

    elif dataset == 'flickr8k':
        for path in [
            os.path.join(PROJECT_ROOT, 'data', 'flickr8k'),
            os.path.join(PROJECT_ROOT, 'data', 'Flickr8k'),
        ]:
            if os.path.exists(path):
                return path
        return os.path.join(PROJECT_ROOT, 'data', 'flickr8k')

    return os.path.join(PROJECT_ROOT, 'data', dataset)


def _find_latest_weights(model_type):
    """Find the latest weights file for a given model type."""
    weights_dir = os.path.join(PROJECT_ROOT, 'weights', model_type)

    if not os.path.exists(weights_dir):
        return None

    weight_files = []
    for fname in os.listdir(weights_dir):
        if fname.endswith('.h5'):
            fpath = os.path.join(weights_dir, fname)
            weight_files.append((fpath, os.path.getmtime(fpath)))

    if not weight_files:
        return None

    weight_files.sort(key=lambda x: x[1], reverse=True)
    return weight_files[0][0]


# ============================================================================
# CLI Argument Parsers
# ============================================================================

def _build_train_parser(sub):
    """Build 'train' subcommand parser."""
    sub.add_argument('--dataset', '-d', required=True,
                     choices=['intel', 'flickr8k'],
                     help='Dataset: intel (image classification) atau flickr8k (captioning)')
    sub.add_argument('--data-dir',
                     help='Path ke direktori dataset (default: auto-detect dari data/)')
    sub.add_argument('--epochs', '-e', type=int,
                     help='Jumlah epoch training')
    sub.add_argument('--batch-size', '-b', type=int,
                     help='Batch size')
    sub.add_argument('--lr', type=float,
                     help='Learning rate (untuk RNN/LSTM)')

    # CNN-specific
    sub.add_argument('--arch',
                     choices=['conv2d', 'locallyconnected'],
                     help='Arsitektur CNN (untuk dataset intel)')
    sub.add_argument('--num-conv-layers', type=int,
                     help='Jumlah convolutional layers')
    sub.add_argument('--num-filters', type=int,
                     help='Jumlah filter base')
    sub.add_argument('--kernel-size', nargs=2, type=int,
                     help='Ukuran kernel (contoh: 3 3)')
    sub.add_argument('--pooling-type',
                     choices=['max', 'average'],
                     help='Tipe pooling')

    # RNN/LSTM-specific
    sub.add_argument('--model', '-m',
                     choices=['rnn', 'lstm'],
                     help='Model decoder (untuk dataset flickr8k)')
    sub.add_argument('--embed-dim', type=int,
                     help='Embedding dimension')
    sub.add_argument('--hidden-dim', type=int,
                     help='Hidden state dimension')
    sub.add_argument('--num-layers', type=int,
                     help='Jumlah layer RNN/LSTM')
    sub.add_argument('--seq-max-length', type=int,
                     help='Panjang maksimal sequence')
    sub.add_argument('--architecture',
                     choices=['preinject', 'initinject'],
                     help='Arsitektur: preinject atau initinject')
    sub.add_argument('--dropout', type=float,
                     help='Dropout rate')
    sub.add_argument('--min-word-freq', type=int,
                     help='Minimum word frequency untuk vocabulary')


def _build_evaluate_parser(sub):
    """Build 'evaluate' subcommand parser."""
    sub.add_argument('--dataset', '-d', required=True,
                     choices=['intel', 'flickr8k'],
                     help='Dataset')
    sub.add_argument('--data-dir',
                     help='Path ke direktori dataset')
    sub.add_argument('--weights', '-w',
                     help='Path ke file bobot .h5 (default: auto-detect terbaru)')
    sub.add_argument('--model', '-m',
                     choices=['rnn', 'lstm'],
                     help='Model (untuk flickr8k)')
    sub.add_argument('--split',
                     choices=['train', 'val', 'test'],
                     help='Data split untuk evaluasi')
    sub.add_argument('--batch-size', '-b', type=int,
                     help='Batch size')


def _build_demo_parser(sub):
    """Build 'demo' subcommand parser."""
    sub.add_argument('--dataset', '-d', required=True,
                     choices=['intel', 'flickr8k'],
                     help='Dataset')
    sub.add_argument('--image', '-i',
                     help='Path ke gambar untuk demo')
    sub.add_argument('--weights', '-w',
                     help='Path ke file bobot .h5 (default: auto-detect terbaru)')
    sub.add_argument('--model', '-m',
                     choices=['rnn', 'lstm'],
                     help='Model decoder (untuk flickr8k)')
    sub.add_argument('--show', action='store_true',
                     help='Tampilkan gambar dengan matplotlib (flickr8k)')


def _build_benchmark_parser(sub):
    """Build 'benchmark' subcommand parser."""
    sub.add_argument('--dataset', '-d', required=True,
                     choices=['intel', 'flickr8k'],
                     help='Dataset')
    sub.add_argument('--data-dir',
                     help='Path ke direktori dataset')
    sub.add_argument('--weights', '-w',
                     help='Path ke file bobot .h5 (default: auto-detect terbaru)')
    sub.add_argument('--model', '-m',
                     choices=['rnn', 'lstm'],
                     help='Model (untuk flickr8k)')
    sub.add_argument('--num-runs', type=int,
                     help='Jumlah runs untuk averaging (default: 3)')
    sub.add_argument('--num-samples', type=int,
                     help='Jumlah sample untuk benchmark (default: 100)')


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Tubes 2 IF3270 — CNN, RNN & LSTM dari Scratch + Keras',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python main.py train --dataset intel --arch conv2d --epochs 20
  python main.py train --dataset flickr8k --model lstm --epochs 30
  python main.py evaluate --dataset intel --weights weights/cnn/conv2d.h5
  python main.py demo --dataset flickr8k --image foto.jpg --show
  python main.py benchmark --dataset intel --num-runs 5
        '''
    )
    parser.add_argument('--version', action='version', version='%(prog)s 1.0')

    sub = parser.add_subparsers(dest='command', help='Sub-command')

    # train
    p_train = sub.add_parser('train', help='Latih model (CNN / RNN / LSTM)')
    _build_train_parser(p_train)

    # evaluate
    p_eval = sub.add_parser('evaluate', help='Evaluasi model terlatih')
    _build_evaluate_parser(p_eval)

    # demo
    p_demo = sub.add_parser('demo', help='Demo prediksi / caption generation')
    _build_demo_parser(p_demo)

    # benchmark
    p_bench = sub.add_parser('benchmark', help='Benchmark kecepatan inferensi')
    _build_benchmark_parser(p_bench)

    args = parser.parse_args()

    # No sub-command: print teaching/banner mode
    if args.command is None:
        print_banner()
        parser.print_help()
        print("\n[Tip] Jalankan salah satu sub-command di atas, contoh:")
        print("  python main.py train --dataset intel --arch conv2d --epochs 1")
        print("  python main.py demo --dataset flickr8k --image foto.jpg")
        return

    # Dispatch
    if args.command == 'train':
        cmd_train(args)
    elif args.command == 'evaluate':
        cmd_evaluate(args)
    elif args.command == 'demo':
        cmd_demo(args)
    elif args.command == 'benchmark':
        cmd_benchmark(args)


if __name__ == '__main__':
    main()
