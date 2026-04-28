"""
Plotting utilities untuk training curves dan evaluasi.
Mendukung plotting: training/validation loss, BLEU comparison, qualitative analysis.
Dapat dipakai oleh CNN, RNN, dan LSTM.
"""

import os
import numpy as np


# ============================================================================
# Plot Training History
# ============================================================================

def plot_training_history(history, save_path=None, title="Training", figsize=(10, 6)):
    """
    Plot training dan validation loss per epoch.

    Args:
        history (dict): dictionary dengan keys 'train_loss' dan 'val_loss'
                       (masing-masing list of floats).
        save_path (str): path untuk menyimpan gambar. Jika None, hanya display.
        title (str): judul plot.
        figsize (tuple): ukuran figure (width, height).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARNING] matplotlib tidak terinstall. "
              "Plot tidak dapat ditampilkan.")
        return

    train_loss = history.get('train_loss', [])
    val_loss = history.get('val_loss', [])

    if not train_loss and not val_loss:
        print("[WARNING] history kosong, tidak ada yang diplot.")
        return

    epochs = range(1, max(len(train_loss), len(val_loss)) + 1)

    plt.figure(figsize=figsize)

    if train_loss:
        plt.plot(epochs[:len(train_loss)], train_loss,
                 'b-o', label='Training Loss', linewidth=2, markersize=4)
    if val_loss:
        plt.plot(epochs[:len(val_loss)], val_loss,
                 'r-s', label='Validation Loss', linewidth=2, markersize=4)

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Plot] Training history disimpan ke: {save_path}")

    try:
        plt.show()
    except Exception:
        pass

    plt.close()


def plot_multi_history(histories_dict, save_path=None, title="Training Comparison",
                       figsize=(12, 5)):
    """
    Plot multiple training histories dalam satu figure.

    Args:
        histories_dict (dict): {name: history_dict}.
        save_path (str): path untuk menyimpan gambar.
        title (str): judul plot.
        figsize (tuple): ukuran figure.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARNING] matplotlib tidak terinstall.")
        return

    num_models = len(histories_dict)
    fig, axes = plt.subplots(1, num_models, figsize=figsize)

    if num_models == 1:
        axes = [axes]

    colors = ['b', 'r', 'g', 'm', 'c', 'orange', 'purple', 'brown']

    for idx, (name, history) in enumerate(histories_dict.items()):
        ax = axes[idx]
        train_loss = history.get('train_loss', [])
        val_loss = history.get('val_loss', [])

        if train_loss:
            ax.plot(range(1, len(train_loss)+1), train_loss,
                    color=colors[idx % len(colors)], label='Train', linewidth=2)
        if val_loss:
            ax.plot(range(1, len(val_loss)+1), val_loss,
                    color=colors[idx % len(colors)], linestyle='--',
                    label='Val', linewidth=2)

        ax.set_title(name, fontsize=12)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Plot] Multi history disimpan ke: {save_path}")

    try:
        plt.show()
    except Exception:
        pass

    plt.close()


# ============================================================================
# BLEU Comparison
# ============================================================================

def plot_bleu_comparison(results_dict, save_path=None, title="BLEU Score Comparison",
                         figsize=(12, 6)):
    """
    Plot perbandingan BLEU score antar model.

    Args:
        results_dict (dict): {model_name: {bleu1, bleu2, bleu3, bleu4, meteor}}.
        save_path (str): path untuk menyimpan gambar.
        title (str): judul plot.
        figsize (tuple): ukuran figure.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARNING] matplotlib tidak terinstall.")
        return

    models = list(results_dict.keys())
    metrics = ['bleu1', 'bleu2', 'bleu3', 'bleu4']
    n_metrics = len(metrics)

    # Prepare data
    data = np.array([[results_dict[m].get(metric, 0) for metric in metrics]
                     for m in models])

    x = np.arange(len(models))
    width = 0.2

    fig, ax = plt.subplots(figsize=figsize)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        offset = (i - n_metrics/2 + 0.5) * width
        bars = ax.bar(x + offset, data[:, i], width * 0.9,
                     label=metric.upper(), color=color, alpha=0.8)

        # Value label di atas bar
        for bar, val in zip(bars, data[:, i]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=7)

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('BLEU Score', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.legend(fontsize=10)
    ax.set_ylim(0, max(data.max() * 1.2, 0.1))
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Plot] BLEU comparison disimpan ke: {save_path}")

    try:
        plt.show()
    except Exception:
        pass

    plt.close()


def plot_rnn_lstm_comparison(rnn_results, lstm_results, save_path=None,
                             title="RNN vs LSTM Comparison", figsize=(10, 6)):
    """
    Plot perbandingan RNN vs LSTM untuk berbagai konfigurasi.

    Args:
        rnn_results (dict): {config_name: {bleu1..bleu4, meteor}}.
        lstm_results (dict): {config_name: {bleu1..bleu4, meteor}}.
        save_path (str): path untuk menyimpan.
        title (str): judul.
        figsize (tuple): ukuran figure.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    metrics = ['bleu1', 'bleu2', 'bleu3', 'bleu4']
    configs = list(set(rnn_results.keys()) | set(lstm_results.keys()))

    x = np.arange(len(configs))
    width = 0.35

    fig, axes = plt.subplots(1, len(metrics), figsize=figsize)
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        rnn_vals = [rnn_results.get(c, {}).get(metric, 0) for c in configs]
        lstm_vals = [lstm_results.get(c, {}).get(metric, 0) for c in configs]

        ax.bar(x - width/2, rnn_vals, width, label='RNN', color='#1f77b4', alpha=0.8)
        ax.bar(x + width/2, lstm_vals, width, label='LSTM', color='#ff7f0e', alpha=0.8)

        ax.set_title(f'{metric.upper()}', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(configs, fontsize=8, rotation=30, ha='right')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Plot] RNN vs LSTM comparison disimpan ke: {save_path}")

    try:
        plt.show()
    except Exception:
        pass

    plt.close()


# ============================================================================
# Confusion Matrix
# ============================================================================

def plot_confusion_matrix(y_true, y_pred, classes, save_path=None,
                          normalize=False, cmap='Blues', figsize=(8, 6)):
    """
    Plot confusion matrix.

    Args:
        y_true (array): ground truth labels.
        y_pred (array): predicted labels.
        classes (list): list nama kelas.
        save_path (str): path untuk menyimpan.
        normalize (bool): normalisasi confusion matrix.
        cmap (str): colormap.
        figsize (tuple): ukuran figure.
    """
    try:
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix
    except ImportError:
        print("[WARNING] sklearn/matplotlib tidak tersedia.")
        return

    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.get_cmap(cmap))
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           ylabel='True label', xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Annotate cells
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Plot] Confusion matrix disimpan ke: {save_path}")

    try:
        plt.show()
    except Exception:
        pass

    plt.close()


# ============================================================================
# Caption Samples (Qualitative Analysis)
# ============================================================================

def plot_caption_samples(image_paths, gt_captions, pred_captions,
                        save_path=None, n=10, figsize=(15, n * 2.5)):
    """
    Plot sample gambar dengan caption ground truth dan prediksi.
    Untuk qualitative analysis RNN vs LSTM.

    Args:
        image_paths (list): list path ke gambar.
        gt_captions (list): list ground truth captions.
        pred_captions (list): list predicted captions.
        save_path (str): path untuk menyimpan.
        n (int): jumlah sample yang ditampilkan.
        figsize (tuple): ukuran figure.
    """
    try:
        import matplotlib.pyplot as plt
        from PIL import Image
    except ImportError:
        print("[WARNING] matplotlib/PIL tidak tersedia.")
        return

    n = min(n, len(image_paths))

    fig, axes = plt.subplots(n, 1, figsize=figsize)

    if n == 1:
        axes = [axes]

    for idx, ax in enumerate(axes):
        img_path = image_paths[idx]
        gt = gt_captions[idx] if idx < len(gt_captions) else ''
        pred = pred_captions[idx] if idx < len(pred_captions) else ''

        if os.path.exists(img_path):
            img = Image.open(img_path)
            ax.imshow(img)
            ax.axis('off')
        else:
            ax.text(0.5, 0.5, f'[Image not found: {img_path}]',
                   ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')

        # Caption text
        caption_text = f"GT: {gt}\nPred: {pred}"
        ax.set_title(caption_text, fontsize=10, loc='left',
                    fontfamily='monospace')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Plot] Caption samples disimpan ke: {save_path}")

    try:
        plt.show()
    except Exception:
        pass

    plt.close()


def plot_beam_search_comparison(image_paths, gt_captions, greedy_captions,
                                 beam_captions, save_path=None, n=10,
                                 figsize=(15, n * 2.5)):
    """
    Plot perbandingan greedy vs beam search captioning.

    Args:
        image_paths (list): list path ke gambar.
        gt_captions (list): ground truth captions.
        greedy_captions (list): greedy decode captions.
        beam_captions (list): beam search captions.
        save_path (str): path untuk menyimpan.
        n (int): jumlah sample.
        figsize (tuple): ukuran figure.
    """
    try:
        import matplotlib.pyplot as plt
        from PIL import Image
    except ImportError:
        print("[WARNING] matplotlib/PIL tidak tersedia.")
        return

    n = min(n, len(image_paths))
    fig, axes = plt.subplots(n, 1, figsize=figsize)

    if n == 1:
        axes = [axes]

    for idx, ax in enumerate(axes):
        img_path = image_paths[idx]
        gt = gt_captions[idx] if idx < len(gt_captions) else ''
        greedy = greedy_captions[idx] if idx < len(greedy_captions) else ''
        beam = beam_captions[idx] if idx < len(beam_captions) else ''

        if os.path.exists(img_path):
            img = Image.open(img_path)
            ax.imshow(img)
            ax.axis('off')
        else:
            ax.text(0.5, 0.5, f'[Image not found]',
                   ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')

        caption_text = (f"GT: {gt}\n"
                        f"Greedy: {greedy}\n"
                        f"Beam(k=5): {beam}")
        ax.set_title(caption_text, fontsize=9, loc='left', fontfamily='monospace')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Plot] Beam search comparison disimpan ke: {save_path}")

    try:
        plt.show()
    except Exception:
        pass

    plt.close()


# ============================================================================
# Metrics Table
# ============================================================================

def save_metrics_table(metrics_dict, save_path, title="Model Comparison"):
    """
    Simpan metrics comparison sebagai text/CSV table.

    Args:
        metrics_dict (dict): {model_name: {metric: value}}.
        save_path (str): path untuk menyimpan.
        title (str): judul tabel.
    """
    if not metrics_dict:
        return

    # Collect all metrics
    all_metrics = set()
    for m in metrics_dict.values():
        all_metrics.update(m.keys())
    all_metrics = sorted(all_metrics)

    lines = []
    lines.append(title)
    lines.append("=" * 80)

    # Header
    header = f"{'Model':<25}"
    for metric in all_metrics:
        header += f"{metric.upper():>12}"
    lines.append(header)
    lines.append("-" * 80)

    # Rows
    for model, metrics in metrics_dict.items():
        row = f"{model:<25}"
        for metric in all_metrics:
            val = metrics.get(metric, 0)
            if isinstance(val, float):
                row += f"{val:>12.4f}"
            else:
                row += f"{str(val):>12}"
        lines.append(row)

    lines.append("=" * 80)

    with open(save_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"[Metrics Table] Disimpan ke: {save_path}")


# ============================================================================
# Heatmap
# ============================================================================

def plot_metrics_heatmap(results_dict, metric='bleu4', save_path=None,
                          title="BLEU-4 Heatmap", figsize=(8, 6)):
    """
    Plot heatmap perbandingan metric antar konfigurasi.

    Args:
        results_dict (dict): {row_label: {col_label: score}}.
        metric (str): metric yang akan diplot.
        save_path (str): path untuk menyimpan.
        title (str): judul.
        figsize (tuple): ukuran figure.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    rows = list(results_dict.keys())
    cols = sorted(set(k for v in results_dict.values() for k in v.keys()))
    data = np.array([[results_dict[r].get(c, {}).get(metric, 0)
                      for c in cols] for r in rows])

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(data, cmap='YlOrRd', aspect='auto')

    ax.set_xticks(np.arange(len(cols)))
    ax.set_yticks(np.arange(len(rows)))
    ax.set_xticklabels(cols, fontsize=10)
    ax.set_yticklabels(rows, fontsize=10)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Annotate
    for i in range(len(rows)):
        for j in range(len(cols)):
            text = ax.text(j, i, f'{data[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=9)

    ax.set_title(title, fontsize=14, fontweight='bold')
    fig.colorbar(im, ax=ax)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Plot] Heatmap disimpan ke: {save_path}")

    try:
        plt.show()
    except Exception:
        pass

    plt.close()
