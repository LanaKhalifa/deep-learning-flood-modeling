# Classic UNet: literature vs implementation vs schematic

## 1. Literature (Ronneberger et al., 2015, U-Net)

- **Contracting path:** Repeated 3×3 convolutions, ReLU, 2×2 max-pooling. Channel progression 64 → 128 → 256 → 512 → 1024 (doubling after each pool). Five “levels”; bottleneck has 1024 channels.
- **Expanding path:** 2×2 up-convolutions, concatenation with **cropped** feature maps from the contracting path, then 3×3 convolutions + ReLU. Symmetric channel decrease. Final 1×1 convolution for output.
- **Details:** Paper uses unpadded 3×3 convs (spatial size decreases); skip connection uses **crop** of encoder feature to match decoder size. ReLU everywhere.

## 2. Implementation (`models/classic_unet.py`)

- **Encoder:** 3→64 (e11, e12), pool → 64→128 (e21, e22), pool → 128→256 (e31, e32), pool → 256→512 (e41, e42), pool → 512→1024 (e51, e52). All convs 3×3, **padding=1** (so spatial size unchanged until pool). **LeakyReLU** (not ReLU). MaxPool2d(2,2) after each block except the last.
- **Decoder:** ConvTranspose2d kernel_size=2, stride=2 (no padding). After each upconv, **concat** with encoder feature (no crop needed because encoder used padding=1). Then two 3×3 convs (padding=1) with LeakyReLU. Final: Conv2d 64→1, kernel_size=1, then LeakyReLU.
- **Skip connections:** concat [xu1, xe42], [xu2, xe32], [xu3, xe22], [xu4, xe12] — matches symmetric levels.
- **Unused:** `self.tanh = nn.Tanh()` is never used; output uses `self.act` (LeakyReLU).

## 3. Schematic (`architecture_schematics.md`)

- Encoder: (3,32,32) → Conv k=3, s=1, p=1, LeakyReLU ×2 → (64,32,32) → MaxPool 2×2 → (64,16,16); same pattern for 64→128→256→512→1024. Bottleneck (1024, 2, 2). ✓
- Decoder: ConvT k=2, s=2, p=0; then (1024,4,4) after concat → Conv ×2 → (512,4,4); same for 512→256→128→64; final Conv k=1, LeakyReLU → (1,32,32). ✓

## Agreement

| Item | Literature | Implementation | Schematic |
|------|------------|----------------|-----------|
| Encoder channels | 64, 128, 256, 512, 1024 | ✓ same | ✓ same |
| 3×3 conv, 2×2 max-pool | ✓ | ✓ | ✓ |
| Decoder 2×2 up-conv | ✓ | ✓ (ConvT k=2 s=2) | ✓ |
| Skip connections (concat) | ✓ (with crop) | ✓ (no crop, p=1) | ✓ |
| Two convs per block | ✓ | ✓ | ✓ |
| Final 1×1 conv | ✓ | ✓ | ✓ |
| Activation | **ReLU** | **LeakyReLU** | **LeakyReLU** |
| Padding | Unpadded (paper) | p=1 (same) | p=1 |

**Conclusion:** Implementation and schematic **agree** with each other. Both differ from the **literature** in two ways: (1) **LeakyReLU** instead of ReLU; (2) **padding=1** on convs (so no crop on skip). The unused `tanh` in the code can be removed.
