# Table draft: Non-downsampling architectures (for article)

Suggested columns and draft text for **Non-downsampling Convolutions with Self-Attention** and **Non-downsampling Convolutions** (ablation). Numeric cells are placeholders to fill from your runs.

---

## Supporting Information — Descriptive architecture table (recommended)

Expanded table for supplementary material. Includes structural columns (encoder–decoder, downsampling/upsampling) for quick comparison. Parameters and size exclude the downsampler for comparability; forward pass is with downsampler.

| Name | Description | Motive / strength | Encoder–decoder? | Downsampling / upsampling? | Parameters | Size (MB) | Forward pass (ms/batch) | Limitations |
|------|-------------|--------------------|------------------|----------------------------|-----------|-----------|--------------------------|-------------|
| **Non-downsampling Convolutions with Self-Attention** (Arch_04) | Constant spatial resolution with self-attention layers between convolutions; C and depth tunable. | Constant resolution emulates HEC-RAS solver; self-attention compensates for the narrow receptive field (each cell can depend on the full domain). | No | No (unit stride throughout) | 102,115 | 0.41 | *measure* | Self-attention adds compute; depth and C need tuning. |
| **Non-downsampling Convolutions** (Arch_02) | Constant spatial resolution with convolutions only (no self-attention); depth 3n to match Arch_04. | To isolate the effect of self-attention; uses same tuned parameters as the self-attention variant. | No | No (unit stride throughout) | 93,665 | 0.37 | *measure* | No non-local context; may underperform when long-range coupling matters. |
| **Simplified UNet** (Arch_03) | Mirrored 5-level encoder–decoder: strided 4×4 conv for encoding (no max-pooling), transposed 4×4 conv for decoding; skip connections at all levels. | Strided conv preferred over max-pooling (empirically better in original study); smaller model compared to classic UNet to preserve computational efficiency and generalization. | Yes | Yes: strided 4×4 conv (down), 4×4 transposed conv (up) | 4,165,313 | 16.66 | *measure* | Fewer parameters than Classic UNet; may sacrifice some capacity. |
| **Classic UNet** (Arch_05) | Symmetric encoder–decoder; encoder: 3×3 conv + 2×2 max-pool; decoder: 2×2 transposed conv + 3×3 conv after skip concat; skip connections at all levels. | Well-established baseline; precise localization with context. | Yes | Yes: 2×2 max-pool (down), 2×2 transposed conv (up) | 31,031,745 | 124.13 | *measure* | Heavier than Simplified UNet; can overfit on small data. |
| **Encoder–Decoder + Self-Attention** (Arch_07) | Encoder–decoder with strided conv and one self-attention at bottleneck; no skip connections. | Long-range dependencies; local and global features (Chen et al.). | Yes | Yes: strided conv (down), transposed conv (up) | 557,107 | 2.23 | *measure* | No skip connections; fine spatial detail can be lost. |
| **Modified UNet + ResNet** (Arch_08) | Encoder–decoder with ResBlock (3×3, SELU) and skip connections; Bicubic upsampling. | Residuals ease training and gradient flow; good with small training sets (Santos et al.). | Yes | Yes: strided residual blocks (down), Bicubic + residual (up) | 409,426 | 1.64 | *measure* | Larger and more expensive to train than lighter archs. |
| **Encoder–Decoder + Large Convolutions** (Arch_09) | Six-layer encoder–decoder; large kernels (6×6, 11×11) and stride = kernel size. | Large receptive field with few layers; RANS/fluid-style emulation (Obiols-Sales et al.). | Yes | Yes: large strided conv (down), transposed conv (up) | 1,994,085 | 7.98 | *measure* | May be less suited to highly localized features. |

*Note:* Parameters and Size (MB) are for the main network only (downsampler excluded). Forward pass is measured with downsampler included. Fill Forward pass using `measure_forward_pass.py` with Stage B config.

---

## Full table (all 7 architectures) — columns to fill

Use this structure for the main/supplementary table. Fill **Parameters**, **Size (MB)**, and **Forward pass** from your Stage B runs (param-count scripts + `measure_forward_pass.py` with dataloader). Size (MB) ≈ Parameters × 4 × 10⁻⁶ for float32.


| Name                                               | Description                                                                                                                                  | Motive / strength                                                                                                                                                               | Parameters | Size (MB) | Forward pass (ms/batch) | Limitations                                                              |
| -------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- | --------- | ----------------------- | ------------------------------------------------------------------------ |
| **Non-downsampling Convolutions** (Arch_02)        | Constant spatial resolution (unit stride, padding); 3×3 conv stack only, no attention; depth 3n to match Arch_04.                            | Ablation to isolate effect of self-attention; same tuned C and depth as Arch_04.                                                                                                | 93,665     | 0.37      | *measure*               | No non-local context; may underperform when long-range coupling matters. |
| **Simplified UNet** (Arch_03)                      | Mirrored 5-level encoder–decoder: strided conv for encoding (no max-pooling), transposed conv for decoding ; skip connections at all levels. | Strided conv preferred over max-pooling (empirically better in original study); smaller model compared to classic UNet to preserve computational efficiency and generalization. | 4,165,313  | 16.66     | *measure*               | Fewer parameters than Classic UNet; may sacrifice some capacity.         |
| **Non-downsampling + Self-Attention** (Arch_04)    | Constant spatial resolution with self-attention between conv blocks; C and depth tunable.                                                    | Constant resolution to emulate HEC-RAS; self-attention compensates for narrow receptive field.                                                                                  | 102,115    | 0.41      | *measure*               | Self-attention adds compute; depth and C need tuning.                    |
| **Classic UNet** (Arch_05)                         | Symmetric encoder–decoder; encoder: 3×3 conv + 2×2 max-pool; decoder: 2×2 transposed conv + 3×3 conv after skip concat; skip connections at all levels.                                                                        | Well-established baseline; precise localization with context.                                                                                                                   | 31,031,745 | 124.13    | *measure*               | Heavier than Simplified UNet; can overfit on small data.                 |
| **Encoder–Decoder + Self-Attention** (Arch_07)     | Encoder–decoder with stride conv and one self-attention at bottleneck; no skip connections.                                                  | Long-range dependencies; local and global features (Chen et al.).                                                                                                               | 557,107    | 2.23      | *measure*               | No skip connections; fine spatial detail can be lost.                    |
| **Modified UNet + ResNet** (Arch_08)               | Encoder–decoder with ResBlock (3×3, SELU) and skip connections; Bicubic upsampling.                                                          | Residuals ease training and gradient flow; good with small training sets (Santos et al.).                                                                                       | 409,426    | 1.64      | *measure*               | Larger and more expensive to train than lighter archs.                   |
| **Encoder–Decoder + Large Convolutions** (Arch_09) | Six-layer encoder–decoder; large kernels (6×6, 11×11) and stride = kernel size.                                                              | Large receptive field with few layers; RANS/fluid-style emulation (Obiols-Sales et al.).                                                                                        | 1,994,085  | 7.98      | *measure*               | May be less suited to highly localized features.                         |


**How to fill the remaining cells**

- **Parameters:** For each architecture, build the model from Stage B config (e.g. `config/model_configs.py`) and sum `p.numel()` over main network parameters (optionally exclude downsampler). You can adapt `count_params_arch02.py` / `count_params_arch04.py` for Arch_03, 05, 07, 08, 09.
- **Size (MB):** Parameters × 4 ÷ 10⁶ for float32 (or measure saved checkpoint size).
- **Forward pass (ms/batch):** Run `measure_forward_pass.py` with the same batch size and device as training (requires `small_train_loader.pt` or a dummy batch of the same shape).

---

## Column choices


| Column                      | Use? | Note                                               |
| --------------------------- | ---- | -------------------------------------------------- |
| **Name**                    | ✓    | Designation + short name                           |
| **Description**             | ✓    | One-line summary of structure                      |
| **Motive / strength**       | ✓    | Why it was chosen; main advantage (author or ours) |
| **# Parameters**            | ✓    | From model / config runs                           |
| **Size (MB)**               | ✓    | e.g. params × 4 bytes (float32) ÷ 10⁶; or measured |
| **Forward pass (ms/batch)** | ✓    | Measured with Stage B config                       |
| **Limitations / downsides** | ✓    | Keeps table balanced; helps reviewers              |


**Recommendation:** Include **Limitations** so the table is balanced and reviewers see you considered trade-offs.

---

## Draft table (two rows)


| Name                                                            | Description                                                                                                              | Motive / strength                                                                                                    | # Parameters | Size (MB) | Forward pass (ms/batch) | Limitations                                                               |
| --------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------- | ------------ | --------- | ----------------------- | ------------------------------------------------------------------------- |
| **Non-downsampling Convolutions with Self-Attention** (Arch_04) | Constant spatial resolution (unit stride, padding) with self-attention layers between convolutions; C and depth tunable. | Constant resolution to emulate HEC-RAS; self-attention to compensate for the narrow receptive field of convolutions. | *fill*       | *fill*    | *fill*                  | Self-attention adds compute; depth and C need tuning.                     |
| **Non-downsampling Convolutions** (Arch_02)                     | Same as above without self-attention; 3n middle conv layers, same total depth as Arch_04 for fair comparison.            | To isolate the effect of self-attention; uses same tuned parameters as the self-attention variant.                   | *fill*       | *fill*    | *fill*                  | No non-local context; may underperform on flows with long-range coupling. |


*Note:* Fill # Parameters, Size (MB), and Forward pass from Stage B runs (e.g. `config/model_configs.py`). Size (MB) ≈ # Parameters × 4 × 10⁻⁶ for float32.

---

## Shorter version (if space is tight)


| Name                                            | Description                                                                                     | Motive                                                                                             | # Params | Size (MB) | Fwd (ms) | Limitations                                                            |
| ----------------------------------------------- | ----------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- | -------- | --------- | -------- | ---------------------------------------------------------------------- |
| **Non-downsampling + Self-Attention** (Arch_04) | Constant spatial resolution (unit stride, padding); self-attention layers between convolutions. | Emulate HEC-RAS (constant resolution); self-attention compensates for narrow receptive field.      | *fill*   | *fill*    | *fill*   | Extra compute from attention; hyperparameters (depth, C) tuned.        |
| **Non-downsampling** (Arch_02)                  | Same conv stack as Arch_04, no attention; same depth (3n middle layers).                        | To isolate the effect of self-attention; uses same tuned parameters as the self-attention variant. | *fill*   | *fill*    | *fill*   | No non-local context; may be weaker where long-range coupling matters. |


---

## One-line “motive” only (for a very compact table)

- **Arch_04:** Constant resolution to emulate HEC-RAS; self-attention compensates for narrow receptive field.
- **Arch_02:** Ablation (no attention); same depth and tuning as Arch_04 to isolate effect of self-attention.

