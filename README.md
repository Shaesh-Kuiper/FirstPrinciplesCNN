# FirstPrinciplesCNN

A convolutional neural network built entirely from scratch in C++, no PyTorch, no TensorFlow, no ML libraries. Every operation (convolution, backprop, Adam, dropout) is hand-implemented using only the C++ standard library and OpenMP.

Trains on Fashion MNIST and achieves **93%+ test accuracy**.

---

## Why

The goal was to understand exactly how CNNs work internally — not just use them. That means implementing every piece of the math yourself: im2col for efficient convolution, manual backward passes through every layer, the Adam update rule with bias correction, and multi-threaded data parallelism with gradient accumulation.

The only external libraries used are [`stb_image`](https://github.com/nothings/stb) and `stb_image_resize` — solely for loading and resizing arbitrary images at inference time.

---

## Architecture

**TinyVGG** — a lightweight VGG-style network:

```
Input (1 × 28 × 28)
  → Conv2D(1→16, 3×3) → ReLU
  → Conv2D(16→16, 3×3) → ReLU
  → MaxPool2D(2×2)                    [28×28 → 14×14]

  → Conv2D(16→32, 3×3) → ReLU
  → Conv2D(32→32, 3×3) → ReLU
  → MaxPool2D(2×2)                    [14×14 → 7×7]

  → Flatten (32 × 7 × 7 = 1568)
  → Dropout(0.3)
  → Dense(1568 → 10)

Output: 10 class logits
```

~32,000 parameters total.

---

## Key Implementation Details

### Convolution via im2col + GEMM

Convolution is not implemented as a naive nested loop. Instead:

1. **`im2col`** — extracts all kernel-sized patches from the input and arranges them as a matrix
2. **`gemm`** — performs a single matrix multiply of the weight matrix against the patch matrix
3. **`col2im`** — accumulates gradients back to input during the backward pass

This makes convolution a pure matrix multiplication problem, which is cache-friendly and fast.

### Backpropagation

Every layer (Conv2D, ReLU, MaxPool, Dropout, Dense) implements its own backward pass:

- **Conv2D**: weight gradients via `gemm(grad_out, col^T)`, input gradients via `col2im(W^T × grad_out)`
- **MaxPool**: routes gradients only through the positions that held the max value (stored during forward pass)
- **ReLU**: gates gradients — zero where input was negative
- **Dropout**: applies the same binary mask and inverted scaling as the forward pass
- **Dense**: outer product for weight gradients, transposed multiply for input gradients

### Optimizer

Adam with L2 weight decay:
- `β1 = 0.9`, `β2 = 0.999`, `ε = 1e-8`, `λ = 1e-4`
- Per-parameter first and second moment estimates with bias correction

### Multi-threaded Training

Data parallelism using OpenMP:
- Each thread holds its own model copy and processes a subset of the batch
- Gradients are accumulated into the master thread after each batch
- Master performs the weight update; all threads sync before the next batch

### Weight Initialization

Kaiming (He) initialization:
- Conv layers: `stddev = sqrt(2 / (in_channels × k × k))`
- Dense layers: `stddev = sqrt(2 / in_features)`

---

## Training Config

| Setting | Value |
|---|---|
| Epochs | 50 |
| Batch size | 64 |
| Initial LR | 0.0028 |
| LR schedule | ×0.5 every 6 epochs (starting epoch 6) |
| Train/test split | 85% / 15% |
| Augmentation | Random horizontal flip (50%) |
| Normalization | mean=0.286, std=0.353 (Fashion MNIST) |

---

## Results

Trained on a Ryzen 7 5800HS (16 threads via OpenMP):

| | |
|---|---|
| Test accuracy | **93%+** |
| Dataset | Fashion MNIST (60,000 training images) |
| Parameters | ~32,000 |

---

## File Structure

```
├── train.cpp              # Training entry point
├── predict.cpp            # Inference entry point
├── train_model.hpp        # TinyVGG model (forward + backward)
├── infer_model.hpp        # TinyVGG model (forward only)
├── train_layers.hpp       # Layer implementations with backprop + Adam
├── infer_layers.hpp       # Layer implementations, inference only
├── train_math.hpp         # im2col, col2im, gemm, gemm variants
├── infer_math.hpp         # im2col, gemm for inference
├── dataset.hpp            # MNIST loader, cross-entropy loss, augmentation
├── stb_image.h            # Image loading (STB, header-only)
├── stb_image_resize.h     # Image resizing (STB, header-only)
└── tinyvgg_fashion_mnist.bin  # Pre-trained weights (included)
```

---

## Build & Run

**Requirements:** C++11 compiler, OpenMP

```bash
# Train from scratch
# Requires Fashion MNIST binary files in the working directory:
#   train-images-idx3-ubyte
#   train-labels-idx1-ubyte
# Download from: http://fashion-mnist.s3-website.eu-west-1.amazonaws.com/

g++ -O3 -march=native -ffast-math -fopenmp train.cpp -o train_vgg
./train
```

```bash
# Run inference with the included pre-trained weights
g++ -O3 -march=native -ffast-math -fopenmp predict.cpp -o predict
./predict
```

The inference binary prompts for an image path, preprocesses it to 28×28 grayscale, and outputs the predicted class. It auto-detects light-background images and inverts them to match Fashion MNIST convention.

---

## Fashion MNIST Classes

| ID | Class |
|---|---|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |
