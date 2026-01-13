# Image Inpainting via Mask-Aware Multi-Scale Structure Encoding (MSSE)

## 1. Project Overview

This project implements a **coarse-to-fine deep learning image inpainting framework** that explicitly integrates **structure awareness**, **mask-aware attention**, and **boundary-sensitive losses**. The goal is to reconstruct visually plausible content in missing regions while preserving:

* Global semantic consistency
* Local texture realism
* Edge and corner continuity
* Boundary color smoothness

The system is designed for **research-oriented experimentation**, with configurable architectures, loss terms, and training regimes.

---

## 2. Problem Definition

Given:

* An RGB image ( I \in \mathbb{R}^{3 \times H \times W} )
* A binary mask ( M \in {0,1}^{1 \times H \times W} )

  * (M=1): missing region
  * (M=0): valid region

The objective is to predict a completed image ( \hat{I} ) such that:

[
\hat{I} = I \odot (1 - M) + G(I, M) \odot M
]

where (G) is a learnable inpainting network.

---

## 3. Overall Architecture

### 3.1 High-Level Pipeline

```
Input Image + Mask
        │
        ▼
Context Encoder (Partial Conv)
        │
        ▼
Coarse Decoder ──► Coarse Output
        │
        ▼
Progressive Refinement (3 Stages)
        │   ├─ Structure-Level Attention
        │   ├─ Mid-Level Attention
        │   └─ Texture-Level Attention
        │
        ▼
Refinement Decoder
        │
        ▼
Final Inpainted Output
```

---

## 4. Module-Level Design (Layer-by-Layer)

### 4.1 Context Encoder (Mask-Aware Feature Extraction)

**Purpose:** Extract reliable multi-scale context features while avoiding contamination from masked pixels.

**Design Choice:** Partial Convolution

For each convolution window:
[
X' = \frac{\sum (W \cdot X \cdot (1-M))}{\sum (1-M) + \epsilon} + b
]

* Only valid pixels contribute
* Mask is updated and propagated

**Why?**

* Prevents feature bias from missing regions
* Stabilizes early training

---

### 4.2 Coarse Decoder (Global Structure Completion)

**Purpose:** Recover rough structure and layout

* Symmetric upsampling path
* Skip connections from encoder
* Produces (I_{coarse})

**Loss Supervision:** Coarse L1 loss
[
\mathcal{L}*{coarse} = | I*{coarse} - I |_1
]

This forces global semantic correctness before fine detail learning.

---

### 4.3 Multi-Scale Structure Encoder (MSSE)

**Enabled via:** `--use_structure_encoder`

**Motivation:** Standard attention treats all spatial locations equally, but structure (edges, corners) is more important in inpainting.

**Implementation:**

* Multi-scale pooling (3×3, 7×7, 15×15)
* Mask-weighted aggregation
* Outputs structure confidence map (S \in [0,1]^{H\times W})

[
S = \sigma\left( \sum_k f_k(X \odot M) \right)
]

**Used as:** Attention bias

---

### 4.4 Progressive Refinement with Mask-Aware Attention

Three refinement stages:

| Stage | Semantic Focus | Attention Scope |
| ----- | -------------- | --------------- |
| 1     | Structure      | Long-range      |
| 2     | Shape          | Medium-range    |
| 3     | Texture        | Local           |

**Attention formulation:**
[
\text{Attn}(Q,K,V) = \text{Softmax}\left( \frac{QK^T}{\sqrt{d}} + B_{structure} \right)V
]

Where:

* (B_{structure}) comes from MSSE
* Masked tokens attend more strongly to valid boundaries

---

### 4.5 Refinement Decoder (Residual Learning)

Instead of predicting full image:
[
I_{final} = I_{coarse} + \Delta I
]

**Benefits:**

* Easier optimization
* Preserves coarse structure

---

## 5. Loss Functions (Mathematical Details)

### 5.1 Total Loss

[
\mathcal{L} = \lambda_c \mathcal{L}*{coarse} + \lambda_r \mathcal{L}*{refine} + \lambda_e \mathcal{L}*{edge} + \lambda*{corner} \mathcal{L}*{corner} + \lambda*{color} \mathcal{L}_{color}
]

---

### 5.2 Edge Loss (Sobel)

[
\mathcal{L}*{edge} = | \nabla I*{pred} - \nabla I_{gt} |_1
]

Encourages sharp edge continuity.

---

### 5.3 Corner Loss

Corners detected using Sobel + Laplacian responses.

Loss emphasizes high-curvature regions:
[
\mathcal{L}*{corner} = | C(I*{pred}) - C(I_{gt}) |_1
]

---

### 5.4 Color Consistency Loss

Applied near mask boundaries to prevent color bleeding:
[
\mathcal{L}*{color} = | I*{pred}^{boundary} - I_{gt}^{boundary} |_1
]

---

## 6. Training Strategy

* Optimizer: **AdamW**
* Scheduler: **Cosine Annealing LR**
* Gradient Clipping: ( |g|_2 \le 1.0 )

**Progressive supervision:**

* Early epochs focus on coarse loss
* Later epochs emphasize refinement & structure

---

## 7. Evaluation Metrics

Computed only on masked regions:

* PSNR (Mask)
* SSIM (Mask)
* PSNR (Boundary)
* LPIPS (Optional)

---

## 8. Visualization Outputs

Saved automatically:

* `loss.png` – training loss curves
* `valmetrics.png` – PSNR / SSIM curves
* `val_epoch_xxxx.png` – qualitative validation results
* `test_results.png` – test comparison grid
* Attention maps (optional)

---

## 9. Scripts Overview

| Script           | Purpose                    |
| ---------------- | -------------------------- |
| `train.py`       | Full training pipeline     |
| `test.py`        | Batch testing & inference  |
| `quick_train.py` | Config-based training      |
| `utils.py`       | Visualization, checkpoints |

---

## 10. Key Design Contributions

1. Mask-aware multi-stage attention
2. Explicit structure bias via MSSE
3. Corner-preserving supervision
4. Boundary-focused color consistency
5. Fully modular research framework

---

## 11. Intended Use

* Final Year Project (FYP)
* Research prototype for structure-aware inpainting
* Ablation studies on attention & loss design

---

## 12. Conclusion

This project demonstrates that **explicit structural reasoning**, when tightly integrated into attention and loss design, significantly improves inpainting quality—especially for edges, corners, and large missing regions.

The framework is extensible and suitable for further research in **vision restoration**, **transformer-based refinement**, and **structure-guided generative models**.
