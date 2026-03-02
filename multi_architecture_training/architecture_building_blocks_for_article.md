# Architecture steps (simple format)

**Format:** in (C, H, W) → layer (kernel, stride, in→out) → out (C, H, W) → activation. H×W = spatial size (unchanged unless noted). s = stride, p = padding. ConvT = transposed conv. Skip = concat with encoder feature at same resolution.

---

## Common input (all)

Terrain (1,H,W) + depth (1,H,W) + BC (1,H,W) → terrain downsampler → concat → **in (3, H″, W″)**.  
Terrain downsampler: (1,H,W) → Conv 3×3 s=2 → (20, H/2,W/2) LeakyReLU → AlternatingStrideConv 3×3 → (40,H′,W′) LeakyReLU → Conv 2×2 s=2 → (1,H″,W″) LeakyReLU.

---

## Arch_02 — Non-downsampling Convolutions

in (3,H,W)  
Conv 3×3 s=1 p=1, 3→32 → (32,H,W) LeakyReLU  
×10: Conv 3×3 s=1 p=1, 32→32 → (32,H,W) LeakyReLU  
Conv 3×3 s=1 p=1, 32→1 → (1,H,W) LeakyReLU

---

## Arch_03 — Simplified UNet

in (3,H,W)  
**Encoder:** Conv 4×4 s=2 p=1: 3→32 (32,H/2,W/2) LeakyReLU | 32→64 (64,H/4,W/4) LeakyReLU | 64→128 (128,H/8,W/8) LeakyReLU | 128→256 (256,H/16,W/16) LeakyReLU | 256→256 (256,H/32,W/32) LeakyReLU  
**Decoder:** ConvT 4×4 s=2 p=1 256→256 (256,H/16,W/16) ReLU; concat skip → (512,H/16,W/16); ConvT 512→128 (128,H/8,W/8) ReLU; concat → (256,H/8,W/8); ConvT 256→64 (64,H/4,W/4) ReLU; concat → (128,H/4,W/4); ConvT 128→32 (32,H/2,W/2) ReLU; concat → (64,H/2,W/2); ConvT 64→1 (1,H,W) LeakyReLU

---

## Arch_04 — Non-downsampling Convolutions with Self-Attention

in (3,H,W)  
Conv 3×3 s=1 p=1, 3→32 → (32,H,W) LeakyReLU  
×3 Conv 3×3 s=1 32→32 LeakyReLU | Self-attn 32→32 (32,H,W) residual | ×4 Conv 3×3 s=1 32→32 LeakyReLU | Self-attn 32→32 (32,H,W) residual | ×3 Conv 3×3 s=1 32→32 LeakyReLU  
Conv 3×3 s=1 p=1, 32→1 → (1,H,W) LeakyReLU

---

## Arch_05 — Classic UNet

in (3,H,W)  
**Encoder:** [Conv 3×3 p=1 3→64, Conv 3×3 p=1 64→64 LeakyReLU] MaxPool 2×2 → (64,H/2,W/2); [Conv 3×3 p=1 64→128, 128→128 LeakyReLU] MaxPool 2×2 → (128,H/4,W/4); [Conv 3×3 p=1 128→256, 256→256 LeakyReLU] MaxPool 2×2 → (256,H/8,W/8); [Conv 3×3 p=1 256→512, 512→512 LeakyReLU] MaxPool 2×2 → (512,H/16,W/16); Conv 3×3 p=1 512→1024, 1024→1024 LeakyReLU → (1024,H/16,W/16)  
**Decoder:** ConvT 2×2 s=2 1024→512 (512,H/8,W/8); concat skip → (1024,H/8,W/8); Conv 3×3 p=1 1024→512, 512→512 LeakyReLU; ConvT 2×2 s=2 512→256 (256,H/4,W/4); concat → (512,H/4,W/4); Conv 3×3 p=1 512→256, 256→256 LeakyReLU; ConvT 2×2 s=2 256→128 (128,H/2,W/2); concat → (256,H/2,W/2); Conv 3×3 p=1 256→128, 128→128 LeakyReLU; ConvT 2×2 s=2 128→64 (64,H,W); concat → (128,H,W); Conv 3×3 p=1 128→64, 64→64 LeakyReLU  
Conv 1×1 64→1 → (1,H,W) LeakyReLU

---

## Arch_07 — Encoder–Decoder with Self-Attention

in (3,H,W)  
**Encoder:** Conv 3×3 s=1 p=1 3→32 (32,H,W) LeakyReLU | Conv 5×5 s=1 p=2 32→48 (48,H,W) LeakyReLU | Conv 5×5 s=2 p=2 48→64 (64,H/2,W/2) LeakyReLU | Conv 3×3 s=2 p=1 64→96 (96,H/4,W/4) LeakyReLU | Conv 3×3 s=2 p=1 96→128 (128,H/8,W/8) LeakyReLU  
Self-attention 128→128 (128,H/8,W/8) residual  
**Decoder:** ConvT 3×3 s=2 p=1 128→69 (69,H/4,W/4) LeakyReLU | ConvT 3×3 s=2 69→64 (64,H/2,W/2) LeakyReLU | ConvT 4×4 s=2 64→48 (48,H,W) LeakyReLU | Conv 5×5 s=1 p=2 48→32 (32,H,W) LeakyReLU | Conv 3×3 s=1 p=1 32→6 (6,H,W) LeakyReLU | Conv 1×1 6→1 (1,H,W) LeakyReLU

---

## Arch_08 — Modified UNet with ResNet

in (3,H,W)  
**Encoder:** ResBlock 3×3 s=1 3→15 (15,H,W) SELU | ResBlock 3×3 s=2 15→30 (30,H/2,W/2) SELU | ResBlock 3×3 s=2 30→60 (60,H/4,W/4) SELU | ResBlock 3×3 s=2 60→120 (120,H/8,W/8) SELU  
**Decoder:** Bicubic ↑2 → (120,H/4,W/4); ResBlock 3×3 s=1 120→60 (60,H/4,W/4) SELU; Bicubic ↑2 → (60,H/2,W/2); ResBlock 3×3 s=1 60→30 (30,H/2,W/2) SELU; Bicubic ↑2 → (30,H,W); ResBlock 3×3 s=1 30→15, 15→15 (15,H,W) SELU  
Conv 1×1 15→1 (1,H,W) LeakyReLU  
*(ResBlock: two Conv 3×3 + 1×1 shortcut, add residual, SELU.)*

---

## Arch_09 — Encoder–Decoder with Large Convolutions

in (3,H,W)  
**Encoder:** Conv 6×6 s=1 p=0 3→4 (4,H−5,W−5) PReLU | Conv 6×6 s=1 p=0 4→32 (32,H−10,W−10) Tanh | Conv 11×11 s=11 p=0 32→256 (256,⌊(H−10)/11⌋,⌊(W−10)/11⌋) Tanh  
**Decoder:** ConvT 11×11 s=11 256→32 Tanh | ConvT 8×8 s=1 p=1 32→3 Tanh | ConvT 6×6 s=1 p=0 3→1 (1,H,W) Tanh

---

## Self-attention block (Arch_04, Arch_07)

in (C,H,W) → reshape (C,N), N=H×W; Query/Key/Value: Conv1d k=1 C→C each; Attention = softmax(Q^T K/√C); out_attn = V·Attention; Conv1d C→C → reshape (C,H,W); out = γ·out_attn + input (γ learnable, init 0).
