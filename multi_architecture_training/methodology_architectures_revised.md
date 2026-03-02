# Methodology: Deep Learning Architectures (Revised for Reviewer Response)

## Draft text for the article

**Opening paragraph (completed):**

Seven architectures were implemented: four from the literature on machine-learning emulators of numerical simulations, drawn from different application domains—flood depth prediction (Simplified UNet, Alhada-Lahbabi et al., 2023), time-dependent PDE simulation (Encoder–Decoder with Self-Attention, Chen et al., 2023), flood or hydrological simulation emulation (Encoder–Decoder with Large Convolutions, Obiols-Sales et al., 2020), and [application domain for Santos] (Modified UNet with ResNet, Santos et al., 2020); the Classic UNet (Ronneberger et al., 2015) as benchmark against the Simplified UNet; one novel design (Non-downsampling Convolutions with Self-Attention); and one ablation (Non-downsampling Convolutions) to isolate the effect of self-attention. They differ in (i) encoder–decoder vs. full spatial resolution, (ii) pooling vs. strided downsampling, (iii) presence and placement of self-attention, (iv) residual connections, and (v) kernel size and depth. Table SX summarizes these attributes; Supplementary Information provides schematic diagrams.

**[Number check:** Your config has 7 architectures: Arch_02 (Non-downsampling, ablation), Arch_03 (Simplified UNet), Arch_04 (Non-downsampling + Attention, novel), Arch_05 (Classic UNet, benchmark), Arch_07 (Encoder–Decoder + Attention), Arch_08 (Modified UNet ResNet), Arch_09 (Encoder–Decoder Large Convolutions). So: 4 from sim-emulation literature + 1 classic benchmark + 1 novel + 1 for isolating effect = 7. If you prefer to say “six architectures” and treat the ablation separately, we can change to: “Six architectures were implemented for comparison … and a seventh (Non-downsampling Convolutions) was implemented to isolate the effect of self-attention.” **Field check:** Chen et al. (2023) = “Implicit Neural Spatial Representations for Time-dependent PDEs” (ICML 2023), i.e. **time-dependent PDE simulation** (computational physics; elasticity, turbulent fluids). **Obiols-Sales et al. (2020):** exact paper not found in search; “flood or hydrological simulation emulation” is a placeholder—please confirm from your reference. **Santos et al. (2020):** application domain still to be confirmed.]**

**Second paragraph – Core components and differences:**

All models share a common input setup: terrain elevation is passed through a configurable downsampler and concatenated with initial water depth and boundary-condition channels to form a single input tensor. The architectures then differ as follows.

The **Classic UNet** (Arch_05; Ronneberger et al., 2015) provides a well-established baseline. It uses a symmetric encoder–decoder with four levels of max-pooling (2×2), 3×3 convolutions, and skip connections between encoder and decoder. The bottleneck has 1024 channels. This design is known to preserve fine spatial detail via skip connections and to perform well in segmentation tasks, at the cost of higher parameter count and memory use.

The **Simplified UNet** (Arch_03; Alhada-Lahbabi et al., 2023) is a lighter UNet variant for flood depth prediction. Downsampling and upsampling are performed with 4×4 strided convolutions and transposed convolutions rather than max-pooling, with skip connections at each level. It uses fewer channels and layers than the Classic UNet, reducing computational cost while retaining the encoder–decoder and multi-scale structure.

The **Encoder–Decoder with Self-Attention** (Arch_07; Chen et al., 2023) combines an encoder (strided 3×3 and 5×5 convolutions), a single self-attention block at the bottleneck, and a decoder based on transposed convolutions. The self-attention block allows the model to capture long-range dependencies at the most compressed representation. Unlike the UNets, this design does not use skip connections in our implementation, so spatial detail is recovered only through the decoder.

The **Non-downsampling Convolutions with Self-Attention** (Arch_04) omits the encoder–decoder structure entirely. It consists of several 3×3 convolutional layers with unit stride (no spatial downsampling), with self-attention modules inserted at evenly spaced positions along the depth of the network. Spatial resolution is preserved throughout, which aligns with the cell-wise, grid-based nature of solvers such as HEC-RAS. The self-attention adds non-local context while keeping the overall structure simple.

The **Non-downsampling Convolutions** (Arch_02) is the same as Arch_04 but without any attention layers. It serves as an ablation to isolate the contribution of the convolutional stack alone, with full spatial resolution and minimal inductive bias.

The **Modified UNet with ResNet** (Arch_08; Santos et al., 2020) replaces standard encoder–decoder blocks with ResNet-style residual blocks (SELU activation, 3×3 convolutions). Downsampling uses strided residual blocks; upsampling uses bicubic interpolation followed by residual blocks. This design emphasizes deeper feature learning and gradient flow through residual connections, at the cost of greater model size.

The **Encoder–Decoder with Large Convolutions** (Arch_09; Obiols-Sales et al., 2020) uses few but large-kernel convolutions (e.g., 11×11 with stride 11) to achieve strong downsampling and a large receptive field with minimal depth. The decoder uses transposed convolutions to restore resolution. This design is efficient in parameters and compute but may be less flexible for fine-scale spatial structure.

**Third paragraph – Known advantages and limitations:**

Prior work suggests the following. Classic UNet is robust and widely used but can overfit on smaller datasets and is relatively heavy. Simplified UNet trades some capacity for efficiency and has been used successfully in flood emulation. Encoder–decoder models with attention (e.g., Arch_07) are suited to capturing global context but can lose fine detail without skip connections. Non-downsampling designs (Arch_02, Arch_04) preserve spatial structure and are conceptually close to cell-wise numerical schemes; adding self-attention (Arch_04) is expected to improve handling of non-local flow interactions. ResNet-based UNet variants (Arch_08) improve gradient flow and representational capacity but increase training cost. Large-kernel encoder–decoder (Arch_09) offers a compact alternative with a large receptive field but may be less suited to highly localized flow features. All architectures were implemented as specified in the code repository (GitHub [URL]) and trained under the same protocol (e.g., loss, optimizer, epochs) for fair comparison.

---

## Suggested Supplementary Table (Table SX): Summary of architectures

| Designation | Architecture name | Encoder–decoder | Spatial downsampling | Self-attention | Residual connections | Kernel size / depth | Primary reference |
|-------------|-------------------|-----------------|----------------------|----------------|----------------------|----------------------|-------------------|
| Arch_02 | Non-downsampling Convolutions | No | No | No | No | 3×3, shallow stack | Ablation (this work) |
| Arch_03 | Simplified UNet | Yes | Yes (strided conv) | No | No (skip only) | 4×4, 5 levels | Alhada-Lahbabi et al. (2023) |
| Arch_04 | Non-downsampling Convolutions with Self-Attention | No | No | Yes (mid-network) | No | 3×3, shallow stack | This work (variant of Chen et al.) |
| Arch_05 | Classic UNet | Yes | Yes (max-pool) | No | No (skip only) | 3×3, 4 levels | Ronneberger et al. (2015) |
| Arch_07 | Encoder–Decoder with Self-Attention | Yes | Yes (strided conv) | Yes (bottleneck) | No | 3×3, 5×5 | Chen et al. (2023) |
| Arch_08 | Modified UNet with ResNet | Yes | Yes (strided res blocks) | No | Yes (within blocks) | 3×3, residual | Santos et al. (2020) |
| Arch_09 | Encoder–Decoder with Large Convolutions | Yes | Yes (large stride) | No | No | 6×6, 11×11 | Obiols-Sales et al. (2020) |

---

## Note for copy-editing

- Replace "Table SX" with the actual supplementary table number.
- Insert the GitHub repository URL where indicated.
- Adjust citation style (e.g., author–date vs. numbered) to match the journal.
- If the journal allows, add a sentence pointing to supplementary schematic diagrams for each architecture (e.g., "Schematic diagrams of each architecture are provided in Figures SX–SY.").
