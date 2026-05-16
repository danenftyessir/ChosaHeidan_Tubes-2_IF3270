# Vision-Language Lab from Scratch: CNN, RNN & LSTM Inference Engine with Keras Parity

**NumPy-only implementations of CNN, RNN, and LSTM verified against Keras — covering image classification and image captioning.**

## Highlights
- CNN forward propagation from scratch with Conv2D and LocallyConnected2D
- Simple RNN and LSTM decoder implementation from scratch
- CNN encoder + RNN/LSTM image captioning pipeline
- Keras vs NumPy inference comparison
- Macro F1, BLEU-4, METEOR, runtime, and qualitative caption analysis

---

## Table of Contents

- [Project Description](#project-description)
- [Repository Structure](#repository-structure)
- [Architecture & Implementation](#architecture--implementation)
  - [CNN — Intel Image Classification](#cnn--intel-image-classification)
  - [RNN/LSTM — Flickr8k Image Captioning](#rnnlstm--flickr8k-image-captioning)
- [Experiment Results](#experiment-results)
  - [CNN Results](#cnn-results)
  - [RNN/LSTM Results](#rnnlstm-results)
- [Bonus Features](#bonus-features)
- [How to Run](#how-to-run)
- [Dependencies](#dependencies)
- [Contributors](#contributors)

---

## Project Description

This project implements three fundamental deep learning architectures **from scratch using NumPy only**, then verifies their correctness by comparing outputs against Keras implementations:

| Component | Task | Dataset |
|---|---|---|
| **CNN** | Image classification (6 classes) | Intel Image Classification (~14K train / 3K test) |
| **RNN** | Image captioning (generate text from images) | Flickr8k (6K train / 1K val / 1K test) |
| **LSTM** | Image captioning (generate text from images) | Flickr8k (6K train / 1K val / 1K test) |

The primary goal is not merely to use a library, but to **understand the internal mechanics** of each layer through explicit mathematical implementation — including forward passes, Keras weight loading, and backward propagation.

---

## Architecture & Implementation

### CNN — Intel Image Classification

#### Layers Implemented from Scratch (NumPy)

| Layer | Implementation Details |
|---|---|
| **Conv2D** | Weight sharing, im2col-style matmul, `same`/`valid` padding, configurable stride, ReLU activation |
| **LocallyConnected2D** | **No weight sharing** — each spatial position (i,j) owns a unique weight matrix; parameter count = `H_out × W_out × kH × kW × C_in × C_out` |
| **MaxPooling2D** | Sliding window with argmax tracking for backward pass |
| **AveragePooling2D** | Sliding window with spatial averaging |
| **GlobalAveragePooling2D** | Mean over all spatial positions per channel |
| **Flatten** | Reshape to 1D vector |
| **Dense** | Matmul + bias + activation, weights loaded from Keras |

#### Training Pipeline (Keras)

Hyperparameter sweep:
- **Conv2D layers**: 2, 3, 4
- **Filters**: 32, 64, 128
- **Kernel size**: 3×3, 5×5
- **Pooling type**: Max, Average

Total: **16 Conv2D configurations** + **3 LocallyConnected2D configurations**

Loss: `SparseCategoricalCrossentropy` | Optimizer: `Adam` | Epochs: 25–30 per run

---

### RNN/LSTM — Flickr8k Image Captioning

#### Encoder-Decoder Architecture (Pre-inject)

```
[Image]
   │
[InceptionV3 (frozen)] ──► 2048-dim feature vector
   │
[Dense projection] ──► embed_dim
   │
   ▼
[<start>] → [RNN/LSTM Cell] → [Dense + Softmax] → first word
                │
                ▼
          [RNN/LSTM Cell] → second word → ... → [<end>]
```

The CNN feature is *injected* as the initial hidden state before the `<start>` token (**pre-inject method**).

#### Layers Implemented from Scratch (NumPy)

**SimpleRNNCell:**
```
h_t = tanh(W_xh @ x_t + W_hh @ h_{t-1} + b_h)
```

**LSTMCell (4 gates):**
```
f_t = σ(W_f @ [h_{t-1}, x_t] + b_f)    # Forget gate
i_t = σ(W_i @ [h_{t-1}, x_t] + b_i)    # Input gate
g_t = tanh(W_g @ [h_{t-1}, x_t] + b_g) # Cell candidate
o_t = σ(W_o @ [h_{t-1}, x_t] + b_o)    # Output gate
c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t        # Cell state update
h_t = o_t ⊙ tanh(c_t)                   # Hidden state output
```

Weights are loaded from Keras format: `kernel` (input+hidden, 4×hidden), `recurrent_kernel`, `bias` (4×hidden).

#### Hyperparameter Sweep

| Hyperparameter | Values |
|---|---|
| Number of layers | 1, 2, 3 |
| Hidden size | 128, 512 |
| Inject method | pre-inject (primary), init-inject (bonus) |

Total: **6 RNN configurations** + **6 LSTM configurations** (pre-inject) + 2 init-inject variants

Loss: `SparseCategoricalCrossentropy` | Optimizer: `Adam` | Max epochs: 30 (with early stopping)

---

## Experiment Results

### CNN Results

#### Conv2D — All Configurations (ranked by Validation F1)

| Model | Layers | Filters | Kernel | Pooling | Val F1 | Val Acc | Params |
|---|---|---|---|---|---|---|---|
| `conv2d_L4_F32_K5_average` | 4 | 32 | 5×5 | Average | **0.9047** | **90.47%** | ~37K |
| `conv2d_L4_F64_K5_average` | 4 | 64 | 5×5 | Average | 0.897 | ~89% | ~142K |
| `conv2d_L3_F128_K3_max` | 3 | 128 | 3×3 | Max | 0.881 | ~88% | ~232K |
| ... | ... | ... | ... | ... | ... | ... | ... |

**Key findings:**
- Models with **4 layers, 5×5 kernel, average pooling** consistently deliver the best F1
- Increasing filter count (32 → 128) does not reliably improve performance and slows training
- Overfitting grows significantly in models with many filters and large kernels

#### Conv2D vs LocallyConnected2D

| Aspect | Conv2D (best) | LC2D (best) |
|---|---|---|
| Val F1 | **0.9047** | 0.6582 |
| Val Accuracy | **90.47%** | 65.82% |
| Parameter Count | ~37K | ~4.7M (125× larger) |
| Training Speed | Fast | Much slower |
| Conclusion | More efficient & accurate | Severe overfitting due to parameter explosion |

**Analysis:** LocallyConnected2D uses 125× more parameters than Conv2D yet achieves 24 F1 points lower. This empirically confirms that **weight sharing in Conv2D is precisely the right inductive bias for image data** — translational invariance is an inherent property of natural images and should be baked into the architecture rather than learned at great cost.

---

### RNN/LSTM Results

#### All Configurations — Best Validation Loss

**RNN:**

| Model | Layers | Hidden | Best Val Loss | Epochs |
|---|---|---|---|---|
| `rnn_l1_h512_preinject` | 1 | 512 | **1.0285** | 22 |
| `rnn_l1_h128_preinject` | 1 | 128 | 1.0448 | 30 |
| `rnn_l2_h512_preinject` | 2 | 512 | 1.0496 | 30 |
| `rnn_l2_h128_preinject` | 2 | 128 | 1.0607 | 30 |
| `rnn_l3_h512_preinject` | 3 | 512 | 1.0954 | 30 |
| `rnn_l3_h128_preinject` | 3 | 128 | 1.0997 | 30 |

**LSTM:**

| Model | Layers | Hidden | Best Val Loss | Epochs |
|---|---|---|---|---|
| `lstm_l1_h512_preinject` | 1 | 512 | **0.9427** | 21 |
| `lstm_l2_h512_preinject` | 2 | 512 | 0.9513 | 24 |
| `lstm_l3_h512_preinject` | 3 | 512 | 0.9728 | 29 |
| `lstm_l1_h128_preinject` | 1 | 128 | 0.9917 | 30 |
| `lstm_l2_h128_preinject` | 2 | 128 | 0.9911 | 30 |
| `lstm_l3_h128_preinject` | 3 | 128 | 1.0078 | 30 |

#### BLEU & METEOR Scores — Best Models on Test Set

| Metric | RNN (l1_h512) | LSTM (l1_h512) | Δ |
|---|---|---|---|
| **BLEU-1** | 0.2679 | 0.2404 | LSTM −2.75 |
| **BLEU-2** | 0.1337 | 0.1326 | LSTM −0.11 |
| **BLEU-3** | 0.0703 | 0.0766 | LSTM **+0.63** |
| **BLEU-4** | 0.0402 | **0.0455** | LSTM **+0.53** |
| **METEOR** | 0.000378 | **0.002388** | LSTM **+0.20** |

**Key findings:**
- **LSTM outperforms RNN** on BLEU-3, BLEU-4, and METEOR, confirming that the gating mechanism better captures long-range dependencies in caption generation
- **1 layer with hidden size 512** beats deeper models — adding layers does not help on a medium-sized dataset and risks vanishing gradients in RNN stacks
- **LSTM converges faster** (21 epochs) and reaches a lower val loss (0.943 vs 1.028), demonstrating that the cell state pathway effectively combats vanishing gradients
- Low absolute BLEU scores are expected for image captioning without end-to-end encoder fine-tuning or data augmentation

#### Sample Generated Captions

```
[Image: Dog running in a field]

Ground Truth : "a dog runs through the grass"
RNN Output   : "a dog is running in a field"
LSTM Output  : "a brown dog is running through a field of grass"
```

---

## Bonus Features

### CNN Bonus

| Feature | File | Description |
|---|---|---|
| **Backward Propagation** | `src/cnn/bonus/bonus_backward.py` | Full gradient flow through Conv2D, Pooling, and Dense. Includes a numerical gradient checker (finite difference) to verify analytical gradient correctness |
| **Feature Map Visualization** | `src/cnn/bonus/bonus_feature_maps.py` | Visualizes intermediate activation maps from each Conv2D layer to interpret what each filter has learned |
| **GradCAM** | `src/cnn/bonus/bonus_gradcam.py` | Gradient-weighted Class Activation Mapping — generates a heatmap over the input image highlighting the regions most responsible for the model's classification decision |
| **Batch Inference Benchmark** | `src/cnn/bonus/bonus_batch_inference.py` | Measures throughput (images/sec) and latency across various batch sizes |

### RNN/LSTM Bonus

| Feature | File | Description |
|---|---|---|
| **BPTT** | `src/rnn/bonus/bonus_backward.py`, `src/lstm/bonus/bonus_backward.py` | Backpropagation Through Time — manual gradient computation across all timesteps in the unrolled sequence |
| **Beam Search** | `src/rnn/bonus/bonus_beam_search.py`, `src/lstm/bonus/bonus_beam_search.py` | Decoding with beam width k=3 and k=5 as an alternative to greedy decoding, producing more fluent and globally coherent captions |
| **Init-inject Architecture** | `src/rnn/bonus/bonus_init_inject.py`, `src/lstm/bonus/bonus_init_inject.py` | Alternative injection strategy where the CNN feature is concatenated at each timestep after the recurrent cell, rather than used as the initial hidden state |
| **Batch Inference** | `src/rnn/bonus/bonus_batch_inference.py`, `src/lstm/bonus/bonus_batch_inference.py` | Batched caption generation for efficient multi-image inference |

---

## How to Run

### 1. Setup Environment

```bash
python -m venv .venv
source .venv/bin/activate       # Linux/Mac
# or
.venv\Scripts\activate          # Windows

pip install -r requirements.txt
```

### 2. Run via Jupyter Notebook

```bash
# CNN
jupyter notebook src/01_CNN_Intel_Image_Classification.ipynb

# RNN/LSTM
jupyter notebook src/02_RNN_LSTM_Flickr8k_Image_Captioning.ipynb
```

### 3. Data

- **Intel Image Classification**: Download from [Kaggle — Intel Image Classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification), place under `data/intel_image_classification/`
- **Flickr8k**: Download from [Kaggle — Flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k), place under `data/flickr8k/`
- Pre-extracted features (`flickr8k_inception.npy`) are generated automatically on the first notebook run

---

## Dependencies

```
tensorflow==2.10.0   # Keras backend with GPU support (last Windows-native version)
numpy<2              # Core engine for all from-scratch implementations
pillow               # Image loading and preprocessing
nltk                 # Caption tokenization
evaluate             # BLEU and METEOR scoring
matplotlib           # Plotting and visualization
scikit-learn         # F1 score and confusion matrix
```

> **Note:** TensorFlow 2.10.0 is the last release with native GPU support on Windows. On Linux/Mac, a newer version can be used without issue.

---

## Contributors

| Name | Student ID |
|---|---|
| Danendra Shafi Athallah | 13523136 |
| Muhammad Raihaan Perdana | 13523124 |
| M. Abizzar Gamadrian | 13523155 |

---

*Institut Teknologi Bandung*
