# Architecture steps (simple format)

Each architecture is described as a sequence of steps: **in** (channels, height, width) → **layer** (kernel, stride, channels) → **out** (channels, height, width) → **activation**.  
Spatial size is written as H×W when unchanged; when it changes we write the new size (e.g. H/2×W/2).

---

## Common input (all architectures)

- **in:** terrain (1, H, W), depth (1, H, W), BC (1, H, W) → concatenated after terrain downsampling.
- **Terrain downsampler:**  
  in (1, H, W) → Conv 3×3, s=2 → (20, H/2, W/2) → LeakyReLU  
  → AlternatingStrideConv 3×3 → (40, H′, W′) → LeakyReLU  
  → Conv 2×2, s=2 → (1, H″, W″) → LeakyReLU  
- **Main network in:** (3, H″, W″) — 3 channels (downsampled terrain + depth + BC).

---

## Arch_02 — Non-downsampling Convolutions

in (3, H, W)

Conv 3×3, s=1, p=1, 3 → 32  
out (32, H, W)  
LeakyReLU

× 10 more: Conv 3×3, s=1, p=1, 32 → 32 → out (32, H, W) → LeakyReLU

Conv 3×3, s=1, p=1, 32 → 1  
out (1, H, W)  
LeakyReLU

---

## Arch_03 — Simplified UNet

in (3, H, W)

**Encoder**  
Conv 4×4, s=2, p=1, 3 → 32 → out (32, H/2, W/2) → LeakyReLU  
Conv 4×4, s=2, p=1, 32 → 64 → out (64, H/4, W/4) → LeakyReLU  
Conv 4×4, s=2, p=1, 64 → 128 → out (128, H/8, W/8) → LeakyReLU  
Conv 4×4, s=2, p=1, 128 → 256 → out (256, H/16, W/16) → LeakyReLU  
Conv 4×4, s=2, p=1, 256 → 256 → out (256, H/32, W/32) → LeakyReLU  

**Decoder** (each step: upconv then concat with skip)  
ConvT 4×4, s=2, p=1, 256 → 256 → out (256, H/16, W/16) → ReLU  
concat skip from encoder → (512, H/16, W/16)  
ConvT 4×4, s=2, p=1, 512 → 128 → out (128, H/8, W/8) → ReLU  
concat skip → (256, H/8, W/8)  
ConvT 4×4, s=2, p=1, 256 → 64 → out (64, H/4, W/4) → ReLU  
concat skip → (128, H/4, W/4)  
ConvT 4×4, s=2, p=1, 128 → 32 → out (32, H/2, W/2) → ReLU  
concat skip → (64, H/2, W/2)  
ConvT 4×4, s=2, p=1, 64 → 1 → out (1, H, W) → LeakyReLU

---

## Arch_04 — Non-downsampling Convolutions with Self-Attention

in (3, H, W)

Conv 3×3, s=1, p=1, 3 → 32 → out (32, H, W) → LeakyReLU  

(Conv 3×3, s=1, 32→32 → LeakyReLU) × 3  
Self-attention (32 → 32, spatial, residual) → out (32, H, W)  
(Conv 3×3, s=1, 32→32 → LeakyReLU) × 4  
Self-attention (32 → 32, spatial, residual) → out (32, H, W)  
(Conv 3×3, s=1, 32→32 → LeakyReLU) × 3  

Conv 3×3, s=1, p=1, 32 → 1 → out (1, H, W) → LeakyReLU

---

## Arch_05 — Classic UNet

in (3, H, W)

**Encoder**  
Conv 3×3, p=1, 3 → 64 → out (64, H, W) → LeakyReLU  
Conv 3×3, p=1, 64 → 64 → out (64, H, W) → LeakyReLU  
MaxPool 2×2 → out (64, H/2, W/2)  

Conv 3×3, p=1, 64 → 128 → LeakyReLU  
Conv 3×3, p=1, 128 → 128 → LeakyReLU  
MaxPool 2×2 → out (128, H/4, W/4)  

Conv 3×3, p=1, 128 → 256 → LeakyReLU  
Conv 3×3, p=1, 256 → 256 → LeakyReLU  
MaxPool 2×2 → out (256, H/8, W/8)  

Conv 3×3, p=1, 256 → 512 → LeakyReLU  
Conv 3×3, p=1, 512 → 512 → LeakyReLU  
MaxPool 2×2 → out (512, H/16, W/16)  

Conv 3×3, p=1, 512 → 1024 → LeakyReLU  
Conv 3×3, p=1, 1024 → 1024 → out (1024, H/16, W/16) → LeakyReLU  

**Decoder**  
ConvT 2×2, s=2, 1024 → 512 → out (512, H/8, W/8)  
concat skip → (1024, H/8, W/8)  
Conv 3×3, p=1, 1024 → 512 → LeakyReLU  
Conv 3×3, p=1, 512 → 512 → out (512, H/8, W/8) → LeakyReLU  

ConvT 2×2, s=2, 512 → 256 → out (256, H/4, W/4)  
concat skip → (512, H/4, W/4)  
Conv 3×3, p=1, 512 → 256 → LeakyReLU  
Conv 3×3, p=1, 256 → 256 → out (256, H/4, W/4) → LeakyReLU  

ConvT 2×2, s=2, 256 → 128 → out (128, H/2, W/2)  
concat skip → (256, H/2, W/2)  
Conv 3×3, p=1, 256 → 128 → LeakyReLU  
Conv 3×3, p=1, 128 → 128 → out (128, H/2, W/2) → LeakyReLU  

ConvT 2×2, s=2, 128 → 64 → out (64, H, W)  
concat skip → (128, H, W)  
Conv 3×3, p=1, 128 → 64 → LeakyReLU  
Conv 3×3, p=1, 64 → 64 → out (64, H, W) → LeakyReLU  

Conv 1×1, 64 → 1 → out (1, H, W) → LeakyReLU

---

## Arch_07 — Encoder–Decoder with Self-Attention

in (3, H, W)

**Encoder**  
Conv 3×3, s=1, p=1, 3 → 32 → out (32, H, W) → LeakyReLU  
Conv 5×5, s=1, p=2, 32 → 48 → out (48, H, W) → LeakyReLU  
Conv 5×5, s=2, p=2, 48 → 64 → out (64, H/2, W/2) → LeakyReLU  
Conv 3×3, s=2, p=1, 64 → 96 → out (96, H/4, W/4) → LeakyReLU  
Conv 3×3, s=2, p=1, 96 → 128 → out (128, H/8, W/8) → LeakyReLU  

Self-attention (128 → 128, spatial, residual) → out (128, H/8, W/8)

**Decoder**  
ConvT 3×3, s=2, p=1, 128 → 69 → out (69, H/4, W/4) → LeakyReLU  
ConvT 3×3, s=2, 69 → 64 → out (64, H/2, W/2) → LeakyReLU  
ConvT 4×4, s=2, 64 → 48 → out (48, H, W) → LeakyReLU  
Conv 5×5, s=1, p=2, 48 → 32 → out (32, H, W) → LeakyReLU  
Conv 3×3, s=1, p=1, 32 → 6 → out (6, H, W) → LeakyReLU  
Conv 1×1, 6 → 1 → out (1, H, W) → LeakyReLU

---

## Arch_08 — Modified UNet with ResNet

in (3, H, W)

**Encoder**  
ResBlock 3×3, s=1, 3 → 15 → out (15, H, W) → SELU  
ResBlock 3×3, s=2, 15 → 30 → out (30, H/2, W/2) → SELU  
ResBlock 3×3, s=2, 30 → 60 → out (60, H/4, W/4) → SELU  
ResBlock 3×3, s=2, 60 → 120 → out (120, H/8, W/8) → SELU  

**Decoder**  
Bicubic ↑2 → out (120, H/4, W/4)  
ResBlock 3×3, s=1, 120 → 60 → out (60, H/4, W/4) → SELU  
Bicubic ↑2 → out (60, H/2, W/2)  
ResBlock 3×3, s=1, 60 → 30 → out (30, H/2, W/2) → SELU  
Bicubic ↑2 → out (30, H, W)  
ResBlock 3×3, s=1, 30 → 15 → out (15, H, W) → SELU  
ResBlock 3×3, s=1, 15 → 15 → out (15, H, W) → SELU  

Conv 1×1, 15 → 1 → out (1, H, W) → LeakyReLU  

*(ResBlock: two Conv 3×3 + 1×1 shortcut; residual add; SELU.)*

---

## Arch_09 — Encoder–Decoder with Large Convolutions

in (3, H, W)

**Encoder**  
Conv 6×6, s=1, p=0, 3 → 4 → out (4, H−5, W−5) → PReLU  
Conv 6×6, s=1, p=0, 4 → 32 → out (32, H−10, W−10) → Tanh  
Conv 11×11, s=11, p=0, 32 → 256 → out (256, ⌊(H−10)/11⌋, ⌊(W−10)/11⌋) → Tanh  

**Decoder**  
ConvT 11×11, s=11, p=0, 256 → 32 → out (32, …) → Tanh  
ConvT 8×8, s=1, p=1, 32 → 3 → out (3, …) → Tanh  
ConvT 6×6, s=1, p=0, 3 → 1 → out (1, H, W) → Tanh

---

## Self-attention block (used in Arch_04 and Arch_07)

in (C, H, W)  
Reshape to (C, N), N = H×W  
Query: Conv1d kernel 1, C → C  
Key:   Conv1d kernel 1, C → C  
Value: Conv1d kernel 1, C → C  
Attention = softmax(Q^T K / √C)  
out_attn = V · Attention  
Projection: Conv1d, C → C  
Reshape to (C, H, W)  
out = γ · out_attn + input  (γ learnable, init 0)

---

*Notation: s = stride, p = padding. ConvT = transposed convolution. Skip = concatenation with encoder feature map at same resolution.*
