# Architecture steps (simple format)

**Format:** (C, H, W) → k=kernel, s=stride, p=padding, activation → (C′, H′, W′). ConvT = transposed conv. Skip = concat with encoder feature at same resolution. All main networks take input (3, 32, 32) after terrain downsampler + concat.

---

## Common input (all)

Terrain (1,32,32) + depth (1,32,32) + BC (1,32,32) → terrain downsampler → concat → **(3, 32, 32)**.  
Terrain downsampler: (1,32,32) → k=3 s=2, LeakyReLU → (20,15,15) → AlternatingStrideConv k=3 → (40,…) LeakyReLU → k=2 s=2 → (1,…) LeakyReLU.

---

## Arch_02 — Non-downsampling Convolutions

(3,32,32) → k=3, s=1, p=1, LeakyReLU → (32,32,32)  
×10: (32,32,32) → k=3, s=1, p=1, LeakyReLU → (32,32,32)  
(32,32,32) → k=3, s=1, p=1, LeakyReLU → (1,32,32)

---

## Arch_03 — Simplified UNet

**(3,32,32)**  
Encoder: → k=4 s=2 p=1, LeakyReLU → (32,16,16) → (64,8,8) → (128,4,4) → (256,2,2) → (256,1,1)  
Decoder: (256,1,1) → ConvT k=4 s=2 → (256,2,2); concat skip → (512,2,2) → ConvT → (128,4,4); concat → (256,4,4) → ConvT → (64,8,8); concat → (128,8,8) → ConvT → (32,16,16); concat → (64,16,16) → ConvT → (1,32,32) LeakyReLU

---

## Arch_04 — Non-downsampling Convolutions with Self-Attention

(3,32,32) → k=3, s=1, p=1, LeakyReLU → (32,32,32)  
×3 k=3 s=1, LeakyReLU → (32,32,32) | Self-attn residual → (32,32,32) | ×4 k=3 s=1, LeakyReLU → (32,32,32) | Self-attn residual → (32,32,32) | ×3 k=3 s=1, LeakyReLU → (32,32,32)  
(32,32,32) → k=3, s=1, p=1, LeakyReLU → (1,32,32)

---

## Arch_05 — Classic UNet

**(3,32,32)**  
Encoder: → k=3 p=1, LeakyReLU ×2 → (64,32,32) MaxPool 2×2 → (64,16,16); → k=3 p=1 ×2 → (128,16,16) MaxPool → (128,8,8); → k=3 p=1 ×2 → (256,8,8) MaxPool → (256,4,4); → k=3 p=1 ×2 → (512,4,4) MaxPool → (512,2,2); → k=3 p=1 ×2 → (1024,2,2)  
Decoder: (1024,2,2) → ConvT k=2 s=2 → (512,4,4); concat skip → (1024,4,4) → k=3 p=1 ×2 LeakyReLU → (512,4,4); ConvT → (256,8,8); concat → (512,8,8) → k=3 p=1 ×2 → (256,8,8); ConvT → (128,16,16); concat → (256,16,16) → k=3 p=1 ×2 → (128,16,16); ConvT → (64,32,32); concat → (128,32,32) → k=3 p=1 ×2 → (64,32,32)  
(64,32,32) → k=1, LeakyReLU → (1,32,32)

---

## Arch_07 — Encoder–Decoder with Self-Attention

(3,32,32) → k=3 s=1 p=1, LeakyReLU → (32,32,32) → k=5 s=1 p=2 → (48,32,32) → k=5 s=2 p=2 → (64,16,16) → k=3 s=2 p=1 → (96,8,8) → k=3 s=2 p=1 → (128,4,4) LeakyReLU  
Self-attention residual → (128,4,4)  
(128,4,4) → ConvT k=3 s=2 → (69,8,8) → ConvT → (64,16,16) → ConvT → (48,32,32) → k=5 s=1 p=2 → (32,32,32) → k=3 s=1 p=1 → (6,32,32) → k=1 → (1,32,32) LeakyReLU

---

## Arch_08 — Modified UNet with ResNet

(3,32,32) → ResBlock k=3 s=1, SELU → (15,32,32) → ResBlock k=3 s=2 → (30,16,16) → ResBlock k=3 s=2 → (60,8,8) → ResBlock k=3 s=2 → (120,4,4)  
(120,4,4) → Bicubic ↑2 → (120,8,8) → ResBlock k=3 s=1 → (60,8,8) → Bicubic ↑2 → (60,16,16) → ResBlock → (30,16,16) → Bicubic ↑2 → (30,32,32) → ResBlock ×2 → (15,32,32)  
(15,32,32) → k=1, LeakyReLU → (1,32,32)  
*ResBlock: two Conv 3×3 + 1×1 shortcut, add residual, SELU.*

---

## Arch_09 — Encoder–Decoder with Large Convolutions

(3,32,32) → k=6 s=1 p=0, PReLU → (4,27,27) → k=6 s=1 p=0, Tanh → (32,22,22) → k=11 s=11 p=0, Tanh → (256,2,2)  
(256,2,2) → ConvT k=11 s=11 → (32,…) Tanh → ConvT k=8 s=1 p=1 → (3,…) Tanh → ConvT k=6 s=1 p=0 → (1,32,32) Tanh

---

## Self-attention block (Arch_04, Arch_07)

(C,H,W) → reshape (C,N); Q/K/V Conv1d k=1 C→C; Attention = softmax(Q^T K/√C); out = V·Attention; project C→C; reshape → (C,H,W); out = γ·attn + input (γ learnable, init 0).
