# Architecture steps

> **Format:** `(C, H, W) → k=kernel, s=stride, p=padding, σ=activation → (C′, H′, W′)`.  
> **ConvT** = transposed convolution · **Skip** = concat with encoder feature at same resolution.  
> All main networks take input **(3, 32, 32)** after terrain downsampler + concat.

---

## Common input (all architectures)

| Step | Description |
|------|--------------|
| Input | Terrain (1,32,32) + depth (1,32,32) + BC (1,32,32) |
| Downsampler | (1,32,32) → k=3 s=2, LeakyReLU → (20,15,15) → AlternatingStrideConv k=3 → (40,…) LeakyReLU → k=2 s=2 → (1,…) LeakyReLU |
| Main net in | **(3, 32, 32)** |

---

## Non-downsampling Convolutions with Self-Attention

*Main proposed architecture. Activation σ (e.g. LeakyReLU), number of self-attention modules, and number of layers in between were hyperparameters during tuning.*

```
(  3,32,32)       → k=3, s=1, p=1, σ  →       ( 32,32,32)
( 32,32,32)       → k=3, s=1, p=1, σ  →       ( 32,32,32)   repeated n times (hyperparameter)
( 32,32,32)       →  Self-attention   →       ( 32,32,32)
( 32,32,32)       → k=3, s=1, p=1, σ  →       ( 32,32,32)   repeated n times (hyperparameter)
( 32,32,32)       →  Self-attention    →       ( 32,32,32)
( 32,32,32)       → k=3, s=1, p=1, σ  →       ( 32,32,32)   repeated n times (hyperparameter)
( 32,32,32)       → k=3, s=1, p=1, σ  →       (  1,32,32)
```

---

## Non-downsampling Convolutions

*Ablation (no self-attention). Hyperparameters that worked for Non-downsampling Convolutions with Self-Attention were chosen for this architecture, not the other way around.*

```
(  3,32,32)       → k=3, s=1, p=1, σ  →       ( 32,32,32)
( 32,32,32)       → k=3, s=1, p=1, σ  →       ( 32,32,32)   repeated N times (hyperparameter)
( 32,32,32)       → k=3, s=1, p=1, σ  →       (  1,32,32)
```

---

**Encoder**

```
(  3,32,32)    → k=4, s=2, p=1, LeakyReLU → ( 32,16,16)
( 32,16,16)    → k=4, s=2, p=1, LeakyReLU → ( 64, 8, 8)
( 64, 8, 8)    → k=4, s=2, p=1, LeakyReLU → (128, 4, 4)
(128, 4, 4)    → k=4, s=2, p=1, LeakyReLU → (256, 2, 2)
(256, 2, 2)    → k=4, s=2, p=1, LeakyReLU → (256, 1, 1)
```

**Decoder**

```
(256, 1, 1)    → ConvT k=4 s=2 → (256, 2, 2)
concat skip    → (512, 2, 2)     → ConvT k=4 s=2 → (128, 4, 4)
concat skip    → (256, 4, 4)    → ConvT k=4 s=2 → ( 64, 8, 8)
concat skip    → (128, 8, 8)    → ConvT k=4 s=2 → ( 32,16,16)
concat skip    → ( 64,16,16)    → ConvT k=4 s=2 → (  1,32,32)  LeakyReLU
```

---

## Classic UNet

**Encoder**

```
(  3,32,32)    → k=3, p=1, LeakyReLU ×2 → ( 64,32,32)   → MaxPool 2×2 → ( 64,16,16)
( 64,16,16)    → k=3, p=1, LeakyReLU ×2 → (128,16,16)  → MaxPool 2×2 → (128, 8, 8)
(128, 8, 8)    → k=3, p=1, LeakyReLU ×2 → (256, 8, 8)  → MaxPool 2×2 → (256, 4, 4)
(256, 4, 4)    → k=3, p=1, LeakyReLU ×2 → (512, 4, 4)  → MaxPool 2×2 → (512, 2, 2)
(512, 2, 2)    → k=3, p=1, LeakyReLU ×2 → (1024, 2, 2)
```

**Decoder**

```
(1024, 2, 2)   → ConvT k=2 s=2 → (512, 4, 4)
concat skip    → (1024, 4, 4)   → k=3, p=1, LeakyReLU ×2 → (512, 4, 4)
( 512, 4, 4)   → ConvT k=2 s=2 → (256, 8, 8)
concat skip    → ( 512, 8, 8)   → k=3, p=1, LeakyReLU ×2 → (256, 8, 8)
( 256, 8, 8)   → ConvT k=2 s=2 → (128,16,16)
concat skip    → ( 256,16,16)   → k=3, p=1, LeakyReLU ×2 → (128,16,16)
( 128,16,16)   → ConvT k=2 s=2 → ( 64,32,32)
concat skip    → ( 128,32,32)   → k=3, p=1, LeakyReLU ×2 → ( 64,32,32)
(  64,32,32)   → k=1, LeakyReLU → (  1,32,32)
```

---

## Encoder–Decoder with Self-Attention

**Encoder**

```
(  3,32,32)    → k=3, s=1, p=1, LeakyReLU → ( 32,32,32)
( 32,32,32)    → k=5, s=1, p=2, LeakyReLU → ( 48,32,32)
( 48,32,32)    → k=5, s=2, p=2, LeakyReLU → ( 64,16,16)
( 64,16,16)    → k=3, s=2, p=1, LeakyReLU → ( 96, 8, 8)
( 96, 8, 8)    → k=3, s=2, p=1, LeakyReLU → (128, 4, 4)
                Self-attention residual         → (128, 4, 4)
```

**Decoder**

```
( 128, 4, 4)   → ConvT k=3 s=2 → ( 69, 8, 8)
(  69, 8, 8)   → ConvT k=3 s=2 → ( 64,16,16)
(  64,16,16)   → ConvT k=4 s=2 → ( 48,32,32)
(  48,32,32)   → k=5, s=1, p=2, LeakyReLU → ( 32,32,32)
(  32,32,32)   → k=3, s=1, p=1, LeakyReLU → (  6,32,32)
(   6,32,32)   → k=1, LeakyReLU → (  1,32,32)
```

---

## Modified UNet with ResNet

**Encoder**

```
(  3,32,32)    → ResBlock k=3 s=1, SELU → ( 15,32,32)
( 15,32,32)    → ResBlock k=3 s=2, SELU → ( 30,16,16)
( 30,16,16)    → ResBlock k=3 s=2, SELU → ( 60, 8, 8)
( 60, 8, 8)    → ResBlock k=3 s=2, SELU → (120, 4, 4)
```

**Decoder**

```
( 120, 4, 4)   → Bicubic ↑2    → ( 120, 8, 8)
( 120, 8, 8)   → ResBlock k=3 s=1, SELU → ( 60, 8, 8)
(  60, 8, 8)   → Bicubic ↑2    → (  60,16,16)
(  60,16,16)   → ResBlock k=3 s=1, SELU → ( 30,16,16)
(  30,16,16)   → Bicubic ↑2    → (  30,32,32)
(  30,32,32)   → ResBlock k=3 s=1, SELU → ( 15,32,32)
(  15,32,32)   → ResBlock k=3 s=1, SELU → ( 15,32,32)
(  15,32,32)   → k=1, LeakyReLU → (  1,32,32)
```

*ResBlock:* two Conv 3×3 + 1×1 shortcut, add residual, SELU.

---

## Encoder–Decoder with Large Convolutions

**Encoder**

```
(  3,32,32)    → k=6, s=1, p=0, PReLU  → (  4,27,27)
(  4,27,27)    → k=6, s=1, p=0, Tanh   → ( 32,22,22)
( 32,22,22)    → k=11, s=11, p=0, Tanh → (256, 2, 2)
```

**Decoder**

```
( 256, 2, 2)   → ConvT k=11 s=11, Tanh → ( 32,…)
(  32,…)       → ConvT k=8 s=1 p=1, Tanh → (  3,…)
(   3,…)       → ConvT k=6 s=1 p=0, Tanh → (  1,32,32)
```

---

## Self-attention block

*Used in: Non-downsampling Convolutions with Self-Attention, Encoder–Decoder with Self-Attention.*

```
(C,H,W)   → reshape (C, N), N = H×W
           → Q/K/V: Conv1d k=1, C→C each
           → Attention = softmax(Q^T K / √C)
           → out_attn = V · Attention
           → Conv1d C→C, reshape → (C,H,W)
           → out = γ·attn + input   (γ learnable, init 0)
```
