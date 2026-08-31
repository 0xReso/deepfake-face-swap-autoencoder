<div align="center">

# Deepfake Face Swapping with a Shared-Encoder Autoencoder

**An end-to-end deepfake generation pipeline — from raw video to a swapped-face video, with quantitative quality evaluation.**

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-D00000?style=flat-square&logo=keras&logoColor=white)](https://keras.io/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white)](https://opencv.org/)
[![dlib](https://img.shields.io/badge/dlib-68%20landmarks-008000?style=flat-square)](http://dlib.net/)
[![Kaggle](https://img.shields.io/badge/Kaggle-2x%20Tesla%20T4-20BEFF?style=flat-square&logo=kaggle&logoColor=white)](https://www.kaggle.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

</div>

---

## Overview

This project reimplements and optimises the classical **shared-encoder / dual-decoder** deepfake architecture. A single encoder learns a person-agnostic latent representation of a face, while two separate decoders learn to reconstruct **Person A** and **Person B** respectively. At inference time the decoders are swapped — encoding Person A's face and decoding it with Person B's decoder produces Person A's pose and expression rendered with Person B's identity.

The generated face is then blended back into the original video frame using **dlib 68-point landmarks**, **Procrustes alignment**, a **feathered convex-hull mask**, and **histogram-free colour correction**, before the frames are reassembled into a video.

The pipeline is split into four Kaggle notebooks that chain together, each consuming the previous notebook's output as a mounted input.

## Pipeline

```
┌──────────────────────┐    ┌──────────────────────┐    ┌──────────────────────┐    ┌──────────────────────┐
│  1. PREPROCESSING    │───▶│  2. MODEL TRAINING   │───▶│  3. FACE SWAPPING    │───▶│  4. EVALUATION       │
│                      │    │                      │    │                      │    │                      │
│  Video → frames      │    │  Shared encoder      │    │  Cross-decode        │    │  SSIM / PSNR / MSE   │
│  MTCNN detection     │    │  Dual decoders       │    │  enc_A + dec_B       │    │  Face-level +        │
│  64×64 face crops    │    │  VGG16 perceptual    │    │  dlib landmark warp  │    │  frame-level         │
│                      │    │  loss + mixed FP16   │    │  Poisson-free blend  │    │  Plots + report      │
│  → results_1/        │    │  → *.weights.h5      │    │  → deepfake.avi      │    │  → report.txt        │
│  → results_2/        │    │                      │    │                      │    │                      │
└──────────────────────┘    └──────────────────────┘    └──────────────────────┘    └──────────────────────┘
```

| # | Stage | Script | Kaggle Notebook |
|:-:|:------|:-------|:----------------|
| 1 | Preprocessing — frame extraction & MTCNN face cropping | [`01_preprocessing.py`](01_preprocessing.py) | [![Open in Kaggle](https://img.shields.io/badge/Kaggle-deepfakes--preprocessing-20BEFF?style=flat-square&logo=kaggle&logoColor=white)](https://www.kaggle.com/code/ayushragh/deepfakes-preprocessing) |
| 2 | Model training — shared encoder + dual decoders | [`02_model_training.py`](02_model_training.py) | [![Open in Kaggle](https://img.shields.io/badge/Kaggle-deepfakes--model--training-20BEFF?style=flat-square&logo=kaggle&logoColor=white)](https://www.kaggle.com/code/ayushragh/deepfakes-model-training) |
| 3 | Face swapping — cross-decode, blend & video rebuild | [`03_face_swapping.py`](03_face_swapping.py) | [![Open in Kaggle](https://img.shields.io/badge/Kaggle-deepfake--face--swapping-20BEFF?style=flat-square&logo=kaggle&logoColor=white)](https://www.kaggle.com/code/ayushragh/deepfake-face-swapping) |
| 4 | Evaluation — SSIM / PSNR / MSE metrics | [`04_evaluation.py`](04_evaluation.py) | [![Open in Kaggle](https://img.shields.io/badge/Kaggle-evaluation-20BEFF?style=flat-square&logo=kaggle&logoColor=white)](KAGGLE_NOTEBOOK_4_URL) |

> **Dataset:** [`ayushragh/presidents-data`](https://www.kaggle.com/datasets/ayushragh/presidents-data) — four source videos (2× Person A, 2× Person B) yielding **6,215 raw frames**.
> **Landmark model:** [`shape_predictor_68_face_landmarks.dat`](https://www.kaggle.com/datasets/sergiovirahonda/shape-predictor-68-face-landmarksdat)

---

## Architecture

### Shared encoder, dual decoders

The core idea is **weight sharing**. Because both autoencoders share one encoder, the latent space is forced to represent *pose, expression, and lighting* — attributes common to all faces — while identity-specific detail is pushed into the decoder weights.

```
                            ┌──────────────┐
   Face A ──┐               │              │──▶ Decoder A ──▶ Reconstructed A   (training)
            ├──▶ SHARED ENCODER (latent)   │
   Face B ──┘               │              │──▶ Decoder B ──▶ Reconstructed B   (training)
                            └──────────────┘
                                   │
   Face A ─────────────────────────┴────────▶ Decoder B ──▶ ★ A's pose, B's identity  (inference)
```

**Encoder** — `64×64×3` → `4×4×512`

| Layer | Filters | Kernel | Stride | Activation |
|:------|:--------|:-------|:-------|:-----------|
| Conv2D | 128 | 5×5 | 2 | LeakyReLU (α=0.1) |
| Conv2D | 256 | 5×5 | 2 | LeakyReLU (α=0.1) |
| Conv2D | 512 | 5×5 | 2 | LeakyReLU (α=0.1) |
| Conv2D | 1024 | 5×5 | 2 | LeakyReLU (α=0.1) |
| Flatten → Dense | 1024 (latent) | — | — | linear |
| Dense → Reshape | 4·4·512 | — | — | → `(4, 4, 512)` |

**Decoder** (×2, one per identity) — `4×4×512` → `64×64×3`

| Layer | Filters | Kernel | Stride | Activation |
|:------|:--------|:-------|:-------|:-----------|
| Conv2DTranspose | 512 | 5×5 | 2 | LeakyReLU (α=0.1) |
| Conv2DTranspose | 256 | 5×5 | 2 | LeakyReLU (α=0.1) |
| Conv2DTranspose | 128 | 5×5 | 2 | LeakyReLU (α=0.1) |
| Conv2DTranspose | 3 | 5×5 | 2 | Sigmoid (float32) |

### Perceptual loss

Plain L1/L2 reconstruction loss produces blurry faces because it averages over plausible outputs. This implementation adds a **VGG16 feature-space term** taken from `block2_conv2` of an ImageNet-pretrained, frozen VGG16:

```
L = ‖y − ŷ‖₁  +  0.5 · ‖ φ(y) − φ(ŷ) ‖₁          where φ = VGG16 block2_conv2
```

The feature term penalises structural/textural differences rather than raw pixel distance, which measurably sharpens eyes, teeth, and hair boundaries.

### Training strategy

| Setting | Value | Rationale |
|:--------|:------|:----------|
| Optimiser | Adam, `lr=5e-5`, `β₁=0.5`, `β₂=0.999` | Low `β₁` stabilises generative training |
| Precision | `mixed_float16` | ~2× throughput on Tesla T4 tensor cores |
| Batch size | 32 | |
| Max epochs | 300 (alternating A/B each epoch) | Keeps the shared encoder balanced between identities |
| Train/val split | 85 / 15 | |
| LR schedule | Halved after every 15 stagnant epochs | Manual plateau decay |
| Early stopping | Patience 40, best-weights checkpointing per decoder | Each autoencoder checkpoints independently |
| Output head | Sigmoid forced to `float32` | Prevents FP16 overflow in the final activation |

**Why alternate instead of joint training?** Training A then B within the same epoch means the shared encoder receives gradient updates from both identities at equal frequency. Training one to convergence first would bias the latent space toward that identity.

### Blending back into the frame

Naïvely pasting the generated `64×64` face produces obvious seams. Stage 3 instead:

1. Detects **68 facial landmarks** with dlib on both the generated face and the original frame.
2. Solves the **Procrustes problem via SVD** for the optimal similarity transform (rotation + uniform scale + translation) aligning eyes, brows, nose, and mouth.
3. Builds a **feathered convex-hull mask** over two landmark groups (eyes+brows, nose+mouth), Gaussian-blurred twice for a soft alpha edge.
4. Applies **colour correction** by dividing the warped face by its own Gaussian blur and multiplying by the target's — a lightweight alternative to Poisson blending that matches local illumination.
5. Alpha-composites and reassembles all frames into a 25 fps video, falling back to the original frame whenever landmark detection fails.

---

## Evaluation

Quality is measured on two levels, because a single number is misleading for face swapping:

- **Face level** (`results/` vs `transformed/`) — how much pose and structure the autoencoder preserves while *deliberately* changing identity. SSIM is expected to be moderate here.
- **Frame level** (`frames/` vs `final/`) — how invisibly the new face integrates into an otherwise untouched frame. SSIM should be high, since only a small region changes.

| Level | MSE ↓ | PSNR (dB) ↑ | SSIM ↑ |
|:------|:------|:------------|:-------|
| Face-level (generation quality) | `TBD` | `TBD` | `TBD` |
| Frame-level (blending quality) | `TBD` | `TBD` | `TBD` |

Stage 4 also emits `evaluation_report.txt` and a six-panel `evaluation_metrics_plot.png` showing per-frame metric traces, which surface the frames where landmark detection or blending degraded.

---

## Getting started

### Requirements

```bash
pip install -r requirements.txt
```

`dlib` requires CMake and a C++ toolchain:

```bash
# macOS
brew install cmake

# Ubuntu / Debian
sudo apt-get install -y build-essential cmake
```

### Running the pipeline

Each script expects the previous stage's output. Update the `/kaggle/input/...` paths at the top of each file to your local directories before running outside Kaggle.

```bash
python 01_preprocessing.py    # videos  → results_1/, results_2/  (cropped 64×64 faces)
python 02_model_training.py   # faces   → encoder/decoder_a/decoder_b .weights.h5
python 03_face_swapping.py    # weights → deepfake.avi
python 04_evaluation.py       # frames  → evaluation_report.txt + metrics plot
```

Download the dlib landmark model before stage 3:

```bash
curl -LO http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 && bunzip2 shape_predictor_68_face_landmarks.dat.bz2
```

### Repository layout

```
.
├── 01_preprocessing.py     # Frame extraction + MTCNN face detection & cropping
├── 02_model_training.py    # Shared encoder, dual decoders, VGG16 perceptual loss
├── 03_face_swapping.py     # Cross-decoding, dlib landmark warp, blending, video rebuild
├── 04_evaluation.py        # SSIM / PSNR / MSE at face and frame level
├── requirements.txt
└── README.md
```

---

## Engineering notes

Choices made while adapting the reference implementation for a constrained 2×T4 Kaggle session:

- **`FACE_SIZE` reduced from 120 → 64.** Cuts activation memory roughly 3.5× and made 300-epoch training feasible within the session limit. This is the main quality trade-off in the project.
- **Mixed precision with an explicit float32 output layer.** `mixed_float16` alone causes the sigmoid head to saturate; forcing `dtype='float32'` on the final activation keeps gradients well-scaled.
- **Batched inference in stage 3.** Predicting 64 faces at a time instead of one-by-one avoids per-call graph overhead across thousands of frames.
- **Fail-open frame handling.** Landmark detection fails on profile views and motion blur; every such frame falls back to the untouched original so the output video never drops frames or stutters.
- **Independent checkpointing per decoder.** Autoencoders A and B converge at different rates, so each saves its own best weights rather than sharing a single early-stopping criterion.

## Ethics

This is an academic project demonstrating generative modelling and computer-vision techniques. Deepfakes carry real potential for misuse — non-consensual imagery, disinformation, and fraud. The code is published for education and for research into **detection**, and should only ever be used on footage you have the right to use, with clear disclosure that the output is synthetic.

## Acknowledgements

Architecture and blending approach adapted from Sergio Virahonda's Kaggle deepfake series and Matthew Earl's landmark-based face-swap method. Extended here with VGG16 perceptual loss, mixed-precision training, an alternating training schedule with adaptive LR decay, and a quantitative evaluation stage.

## License

[MIT](LICENSE)
