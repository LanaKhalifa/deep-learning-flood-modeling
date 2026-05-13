# Architecture steps

> **Format:** `(C, H, W) → k=kernel, s=stride, p=padding, σ=activation → (C′, H′, W′)`. Here *C* and *C′* are the input and output channel counts; *H*, *W* and *H′*, *W′* are the height and width of the input and output feature maps. 
> **Conv** (with trailing space to align with ConvT) = convolution · **ConvT** = transposed convolution · **Skip** = concat with encoder feature at same resolution.  
> All main networks take input **(3, 32, 32)** after terrain downsampler + concat.


## Non-downsampling Convolutions with Self-Attention

This is the novel proposed architecture. Activation function σ (e.g., LeakyReLU), the number of self-attention modules, the number of conv layers between them (n), and the number of channels (C) were treated as tunable hyperparameters. In the implementation, attention layers are placed at evenly spaced depths. Here’s an example with 2 self-attention modules and n conv layers between them:

```
(  3,32,32)           →   Conv  k=3, s=1, p=1, σ        →   (  C,32,32)
(  C,32,32)           →   Conv  k=3, s=1, p=1, σ        →   (  C,32,32)   [repeat n times]
(  C,32,32)           →   Self-attention               →   (  C,32,32)
(  C,32,32)           →   Conv  k=3, s=1, p=1, σ        →   (  C,32,32)   [repeat n times]
(  C,32,32)           →   Self-attention               →   (  C,32,32)
(  C,32,32)           →   Conv  k=3, s=1, p=1, σ        →   (  C,32,32)   [repeat n times]
(  C,32,32)           →   Conv  k=3, s=1, p=1, σ        →   (  1,32,32)
```

**In short:** Input (3,32,32) → one conv to C channels → (conv×n → attention → conv×n → attention → conv×n) → one conv to (1,32,32).

---

## Non-downsampling Convolutions

This is an ablation with self-attention removed. The tuning results (hyperparameters) from the architecture above were used here (not vice versa). The middle block is repeated so that the total number of conv layers matches the architecture above: in the example above there are three segments of n conv layers each (3n middle conv layers), so the ablation middle is repeated 3n times:

```
(  3,32,32)           →   Conv  k=3, s=1, p=1, σ        →   ( 32,32,32)
( 32,32,32)           →   Conv  k=3, s=1, p=1, σ        →   ( 32,32,32)   [repeat 3n times]
( 32,32,32)           →   Conv  k=3, s=1, p=1, σ        →   (  1,32,32)
```

**In short:** Input (3,32,32) → one conv to C channels (C from tuning) → 3n conv blocks at C channels → one conv to (1,32,32). Same total conv depth as the architecture above.

---

## Simplified UNet

*Lightweight encoder–decoder with 4×4 strided convolutions and transposed convolutions; skip connections at each resolution (Alhada-Lahbabi et al., 2023; ferroelectric phase-field modelling).*

**Activations:** σ₁ = LeakyReLU(α=0.2), σ₂ = ReLU. Encoder uses σ₁; decoder uses σ₂ on all but the last layer, which uses σ₁ in our implementation. *Original architecture:* the final decoder output used **tanh** (and the original included batch normalization and dropout). σ₁ is used instead of tanh on the final layer because the predicted water depth must not be constrained to a bounded range; batch normalization and dropout are omitted for simplicity.

**Encoder**

```
(  3,32,32)           →   Conv  k=4, s=2, p=1, σ₁       →   ( 32,16,16)
( 32,16,16)           →   Conv  k=4, s=2, p=1, σ₁       →   ( 64, 8, 8)
( 64, 8, 8)           →   Conv  k=4, s=2, p=1, σ₁       →   (128, 4, 4)
(128, 4, 4)           →   Conv  k=4, s=2, p=1, σ₁       →   (256, 2, 2)
(256, 2, 2)           →   Conv  k=4, s=2, p=1, σ₁       →   (256, 1, 1)
```

**Decoder**

```
(256, 1, 1)           →   ConvT k=4, s=2, p=1, σ₂       →   (256, 2, 2)
(512, 2, 2)           →   ConvT k=4, s=2, p=1, σ₂       →   (128, 4, 4)   [concat skip]
(256, 4, 4)           →   ConvT k=4, s=2, p=1, σ₂       →   ( 64, 8, 8)   [concat skip]
(128, 8, 8)           →   ConvT k=4, s=2, p=1, σ₂       →   ( 32,16,16)   [concat skip]
( 64,16,16)           →   ConvT k=4, s=2, p=1, σ₁       →   (  1,32,32)   [concat skip]
```

---

## Classic UNet

*Original (Ronneberger et al., 2015) uses ReLU; our implementation uses LeakyReLU. Same channel progression and structure; we use padding=1 so skip concat needs no crop.*

**Encoder**

```
(  3,32,32)           →   Conv  k=3, s=1, p=1, LeakyReLU ×2   →   ( 64,32,32)   →   MaxPool 2×2   →   ( 64,16,16)
( 64,16,16)           →   Conv  k=3, s=1, p=1, LeakyReLU ×2   →   (128,16,16)   →   MaxPool 2×2   →   (128, 8, 8)
(128, 8, 8)           →   Conv  k=3, s=1, p=1, LeakyReLU ×2   →   (256, 8, 8)   →   MaxPool 2×2   →   (256, 4, 4)
(256, 4, 4)           →   Conv  k=3, s=1, p=1, LeakyReLU ×2   →   (512, 4, 4)   →   MaxPool 2×2   →   (512, 2, 2)
(512, 2, 2)           →   Conv  k=3, s=1, p=1, LeakyReLU ×2   →   (1024, 2, 2)
```

**Decoder**

```
(1024, 2, 2)          →   ConvT k=2, s=2, p=0          →   (512, 4, 4)
(1024, 4, 4)          →   Conv  k=3, s=1, p=1, LeakyReLU ×2  →   (512, 4, 4)   [concat skip]
( 512, 4, 4)          →   ConvT k=2, s=2, p=0          →   (256, 8, 8)
( 512, 8, 8)          →   Conv  k=3, s=1, p=1, LeakyReLU ×2  →   (256, 8, 8)   [concat skip]
( 256, 8, 8)          →   ConvT k=2, s=2, p=0          →   (128,16,16)
( 256,16,16)          →   Conv  k=3, s=1, p=1, LeakyReLU ×2  →   (128,16,16)   [concat skip]
( 128,16,16)          →   ConvT k=2, s=2, p=0          →   ( 64,32,32)
( 128,32,32)          →   Conv  k=3, s=1, p=1, LeakyReLU ×2  →   ( 64,32,32)   [concat skip]
(  64,32,32)          →   Conv  k=1, s=1, p=0, LeakyReLU     →   (  1,32,32)
```

---

## Encoder–Decoder with Self-Attention

**Encoder**

```
(  3,32,32)           →   Conv  k=3, s=1, p=1, LeakyReLU  →   ( 32,32,32)
( 32,32,32)           →   Conv  k=5, s=1, p=2, LeakyReLU  →   ( 48,32,32)
( 48,32,32)           →   Conv  k=5, s=2, p=2, LeakyReLU  →   ( 64,16,16)
( 64,16,16)           →   Conv  k=3, s=2, p=1, LeakyReLU  →   ( 96, 8, 8)
( 96, 8, 8)           →   Conv  k=3, s=2, p=1, LeakyReLU  →   (128, 4, 4)
                       →   Self-attention residual       →   (128, 4, 4)
```

**Decoder**

```
( 128, 4, 4)          →   ConvT k=3, s=2, p=0          →   ( 69, 8, 8)
(  69, 8, 8)          →   ConvT k=3, s=2, p=0          →   ( 64,16,16)
(  64,16,16)          →   ConvT k=4, s=2, p=0          →   ( 48,32,32)
(  48,32,32)          →   Conv  k=5, s=1, p=2, LeakyReLU  →   ( 32,32,32)
(  32,32,32)          →   Conv  k=3, s=1, p=1, LeakyReLU  →   (  6,32,32)
(   6,32,32)          →   Conv  k=1, s=1, p=0, LeakyReLU  →   (  1,32,32)
```

---

## Modified UNet with ResNet

**Encoder**

```
(  3,32,32)           →   ResBlock k=3 s=1, SELU       →   ( 15,32,32)
( 15,32,32)           →   ResBlock k=3 s=2, SELU       →   ( 30,16,16)
( 30,16,16)           →   ResBlock k=3 s=2, SELU       →   ( 60, 8, 8)
( 60, 8, 8)           →   ResBlock k=3 s=2, SELU       →   (120, 4, 4)
```

**Decoder**

```
( 120, 4, 4)          →   Bicubic ↑2                    →   ( 120, 8, 8)
( 120, 8, 8)          →   ResBlock k=3 s=1, SELU       →   ( 60, 8, 8)
(  60, 8, 8)          →   Bicubic ↑2                    →   (  60,16,16)
(  60,16,16)          →   ResBlock k=3 s=1, SELU       →   ( 30,16,16)
(  30,16,16)          →   Bicubic ↑2                    →   (  30,32,32)
(  30,32,32)          →   ResBlock k=3 s=1, SELU       →   ( 15,32,32)
(  15,32,32)          →   ResBlock k=3 s=1, SELU       →   ( 15,32,32)
(  15,32,32)          →   Conv  k=1, s=1, p=0, LeakyReLU  →   (  1,32,32)
```

*ResBlock:* two Conv 3×3 + 1×1 shortcut, add residual, SELU.

---

## Encoder–Decoder with Large Convolutions

**Encoder**

```
(  3,32,32)           →   Conv  k=6, s=1, p=0, PReLU    →   (  4,27,27)
(  4,27,27)           →   Conv  k=6, s=1, p=0, Tanh     →   ( 32,22,22)
( 32,22,22)           →   Conv  k=11, s=11, p=0, Tanh   →   (256, 2, 2)
```

**Decoder**

```
( 256, 2, 2)          →   ConvT k=11, s=11, p=0, Tanh   →   ( 32,…)
(  32,…)              →   ConvT k=8, s=1, p=1, Tanh     →   (  3,…)
(   3,…)              →   ConvT k=6, s=1, p=0, Tanh     →   (  1,32,32)
```

---

## Self-attention block

*Used in: Non-downsampling Convolutions with Self-Attention, Encoder–Decoder with Self-Attention.*

```
(C,H,W)                →   reshape (C, N), N = H×W
                         →   Q/K/V: Conv1d k=1, C→C each
                         →   Attention = softmax(Q^T K / √C)
                         →   out_attn = V · Attention
                         →   Conv1d C→C, reshape       →   (C,H,W)
                         →   out = γ·attn + input   (γ learnable, init 0)
```
