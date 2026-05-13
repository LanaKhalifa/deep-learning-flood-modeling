# Methodology: Deep Learning Architectures (Full Draft, Literature-Supported)

*Replace Table SX with the actual supplementary table number and insert the GitHub repository URL where indicated. Schematic diagrams are in [architecture_schematics.md].*

---

## Paragraph 1 — Architectures and implementation reference

Seven deep learning architectures were implemented. Four were taken from the literature on machine-learning emulators of numerical simulations in different domains: Simplified UNet (ferroelectric phase-field modelling; Alhada-Lahbabi et al., 2023), Encoder–Decoder with Self-Attention (stress and fracture prediction in composites; Chen et al., 2023), Encoder–Decoder with Large Convolutions (Reynolds-averaged Navier–Stokes, flow around obstacles; Obiols-Sales et al., 2020), and Modified UNet with ResNet (Stokes 3D flow through porous media; Santos et al., 2020). These four were chosen to span different modules observed in the scanned literature. Classic UNet (medical image segmentation, Ronneberger et al., 2015) was included as a benchmark against the lighter Simplified UNet. For these five architectures, where necessary, kernel sizes, depth of network and channel-related design choices were modified from the cited work so that each network accepts 3×32×32 input and produces 1×32×32 output, with minimal modifications from the original published versions. One novel architecture, Non-downsampling Convolutions with Self-Attention, was designed to preserve full spatial resolution and to use self-attention, and one ablation, Non-downsampling Convolutions (same design without self-attention), was used to isolate the effect of self-attention. The architectures differ in (i) encoder–decoder vs. full spatial resolution, (ii) pooling vs. strided convolutions for downsampling, (iii) presence and placement of self-attention, (iv) residual connections, and (v) kernel size and depth. Table SX summarizes these attributes and reports parameter counts and forward-pass time; a compact schematic representation of each architecture (operations and tensor shapes) in architecture_schematics.md gives layer-by-layer descriptions and tensor dimensions for each architecture. Full implementation is available in the repository: [model definitions], and the [training pipeline] that uses these model implementations.

---

## Paragraph 2 — Common input

All models share the same input setup: terrain elevation is passed through a configurable downsampler and concatenated with initial water depth and boundary-condition channels to form a single input tensor of shape (3, 32, 32) fed to the main network. The architectures then differ as follows.

---

## Paragraph 3 — Core components and differences (per architecture)

**Classic UNet (Arch_05).** Ronneberger et al. (2015) introduced a symmetric encoder–decoder: a contracting path of repeated 3×3 convolutions and 2×2 max-pooling, and an expanding path where “high resolution features from the contracting path are combined with the upsampled output” via skip connections so that the network can achieve “precise localization” while using context. The design uses only convolutions (no fully connected layers) and preserves spatial detail through these skip connections. We use it as a well-established baseline and benchmark for the lighter Simplified UNet.

**Simplified UNet (Arch_03).** Alhada-Lahbabi et al. (2023) used “a version of the classical U-net with skip connections” as an encoder–decoder CNN to accelerate ferroelectric phase-field modelling. They progressively downsample with “convolutional layers (Conv2D) with a 4 × 4 kernel… and stride equal to 2” instead of max-pooling—“These convolution parameters are preferred over the commonly used MaxPool2D layer, as it gives better results in our study”—and upsample with deconvolutional layers (4×4 kernel, stride 2), with skip connections between encoder and decoder. Our implementation follows this design (4×4 stride-2 conv/deconv, channel progression, skip connections, LeakyReLU in the encoder and ReLU in intermediate decoder layers) but omits batch normalization and dropout and uses LeakyReLU on the final decoder layer instead of tanh. We adopt this lighter, strided-convolution UNet variant as a representative simulation emulator from the phase-field literature.

**Encoder–Decoder with Self-Attention (Arch_07).** Chen et al. (2023) used an encoder–decoder CNN with convolutions with stride (instead of pooling) and a single self-attention module at the bottleneck for full-field prediction of stress and fracture patterns in composites. They argued that “using encoder-decoder convolutional layers alone can be inefficient for capturing the long-range dependencies” across the domain, and that “introducing self-attention modules within convolutional architectures can significantly improve the ability to model the long-range dependencies whilst preserving computational efficiency”; the self-attention mechanism “can learn both local and global features” that are relevant to the output. In our implementation we do not use skip connections, so spatial detail is recovered only through the decoder.

**Encoder–Decoder with Large Convolutions (Arch_09).** Obiols-Sales et al. (2020) introduced CFDNet, a “six-layer encoder-decoder convolutional neural network” (three convolution and three deconvolution layers) to accelerate Reynolds-averaged Navier–Stokes simulations. They applied “a striding of the same size as the filter” in each layer, achieving strong downsampling and a large receptive field with few layers. Our implementation follows this idea with an encoder–decoder using large kernels (e.g. 6×6 and 11×11) and corresponding strides for efficient, large-receptive-field emulation of fluid-like systems.

**Modified UNet with ResNet (Arch_08).** Santos et al. (2020) presented PoreFlow-Net for predicting fluid flow through porous media, building on a “modification of the ResUnet” that combines residual units (He et al., 2016) with UNet-style skip connections. They state that residual connections “facilitate training” by “targeting this new referenced residual output, avoiding gradient vanishing or saturation,” and that the resulting network “prove[s] to be easy to train (compared to the U-Net that needed extensive data augmentation or a pre-trained model), with an efficient number of parameters and showed accurate results using a small training set.” Our implementation uses ResNet-style residual blocks (e.g. SELU, 3×3 convolutions) for the encoder–decoder, with strided residual blocks for downsampling and interpolation followed by residual blocks for upsampling.

**Non-downsampling Convolutions with Self-Attention (Arch_04).** We propose this architecture to align with the cell-wise, grid-based structure of solvers such as HEC-RAS, which do not reduce the number of cells when discretizing the equations. It keeps full spatial resolution throughout: several 3×3 convolutional layers with unit stride and self-attention modules at regular depth. Self-attention was added so that each cell can integrate information from the full domain during learning, rather than to mirror the local stencil of the numerical solver; the ablation (Arch_02) assesses whether this non-local mechanism improves predictions.

**Non-downsampling Convolutions (Arch_02).** This ablation is identical to Arch_04 but without self-attention. It isolates the contribution of the convolutional stack alone at full spatial resolution.

---

## Paragraph 4 — Known advantages and limitations

Prior work indicates the following. Classic UNet is robust and widely used but can overfit on smaller datasets and is relatively heavy. Simplified UNet trades some capacity for efficiency and has been used in flood emulation. Encoder–decoder models with self-attention (e.g. Arch_07) are suited to global context but can lose fine detail without skip connections. Non-downsampling designs (Arch_02, Arch_04) preserve spatial structure and are conceptually close to cell-wise schemes; adding self-attention (Arch_04) is intended to improve handling of non-local dependencies. ResNet-based UNet variants (Arch_08) improve gradient flow and representational capacity at the cost of greater size and training cost. Large-kernel encoder–decoder (Arch_09) offers a compact, large receptive-field alternative but may be less suited to highly localized features. All architectures were implemented as specified in the repository and trained under the same protocol (loss, optimizer, data, epochs) for fair comparison.

---

## Table SX — Summary of architectures

| Designation | Architecture | # Parameters | Forward pass (ms/batch) | Enc.–dec. | Downsampling | Self-attn | Residual | Kernel / depth | Reference |
|-------------|--------------|--------------|-------------------------|-----------|--------------|-----------|----------|----------------|-----------|
| Arch_02 | Non-downsampling Convolutions | *fill* | *optional* | No | No | No | No | 3×3, shallow | Ablation (this work) |
| Arch_03 | Simplified UNet | *fill* | *optional* | Yes | Strided 4×4 | No | No (skip only) | 4×4, 5 levels | Alhada-Lahbabi et al. (2023) |
| Arch_04 | Non-downsampling Convolutions with Self-Attention | *fill* | *optional* | No | No | Yes (mid-network) | No | 3×3, shallow | This work |
| Arch_05 | Classic UNet | *fill* | *optional* | Yes | Max-pool 2×2 | No | No (skip only) | 3×3, 4 levels | Ronneberger et al. (2015) |
| Arch_07 | Encoder–Decoder with Self-Attention | *fill* | *optional* | Yes | Strided conv | Yes (bottleneck) | No | 3×3, 5×5 | Chen et al. (2023) |
| Arch_08 | Modified UNet with ResNet | *fill* | *optional* | Yes | Strided res blocks | No | Yes (within blocks) | 3×3, residual | Santos et al. (2020) |
| Arch_09 | Encoder–Decoder with Large Convolutions | *fill* | *optional* | Yes | Large stride | No | No | 6×6, 11×11 | Obiols-Sales et al. (2020) |

*Note:* Fill “# Parameters” (and optionally forward-pass time per batch) from runs with the Stage B config in `config/model_configs.py`. Schematic step-by-step descriptions: `multi_architecture_training/architecture_building_blocks_for_article.md`.

---

## References used for methodology (from `references/`)

- **Ronneberger et al. (2015)** — U-Net: Convolutional Networks for Biomedical Image Segmentation (arXiv 1505.04597). Contracting path, symmetric expanding path, 3×3 conv, 2×2 max-pooling, skip connections (crop and concat), precise localization.
- **Santos et al. (2020)** — PoreFlow-Net: A 3D convolutional neural network to predict fluid flow through porous media. *Advances in Water Resources* 138, 103539. Underlying physics: Stokes flow (creeping flow in pore space). ResUnet modification; residual units and skip connections; gradient propagation, easier training, small training set.
- **Chen et al. (2023)** — Full-field prediction of stress and fracture patterns in composites using deep learning and self-attention. *Engineering Fracture Mechanics* 286, 109314. Encoder–decoder with self-attention at bottleneck; long-range dependencies; local and global features. *(If you instead cite the ICML 2023 paper “Implicit Neural Spatial Representations for Time-dependent PDEs”, adjust the application domain to time-dependent PDE simulation.)*
- **Obiols-Sales et al. (2020)** — CFDNet: A deep learning-based accelerator for fluid simulations. *ICS ’20: Proceedings of the 34th ACM International Conference on Supercomputing*. Six-layer encoder–decoder CNN; conv/deconv; stride equal to filter size; RANS/fluid acceleration.
- **Alhada-Lahbabi et al. (2023)** — Machine learning surrogate model for acceleration of ferroelectric phase-field modeling. *ACS Appl. Electron. Mater.* 5, 3894–3907. Encoder–decoder “version of the classical U-net with skip connections”; 4×4 strided convolutions (instead of MaxPool2D) and 4×4 transposed convolutions; phase-field simulation emulation.
