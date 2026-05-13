# Parameter count: Non-downsampling Convolutions with Self-Attention (Arch_04)

**Config (Stage B):** `arch_num_layers=12`, `arch_num_c=32`, `arch_input_c=3`, `arch_num_attentions=2`.  
**Excluding:** downsampler (only `conv_net` counted).

---

## Convolutions

- **Conv2d(in, out, k=3, s=1, p=1):** params = `in × out × 9 + out` (weight + bias).

| Layer | in | out | Params |
|-------|----|-----|--------|
| First conv | 3 | 32 | 3×32×9 + 32 = **896** |
| Middle convs (×10) | 32 | 32 | 10 × (32×32×9 + 32) = 10 × 9248 = **92,480** |
| Final conv | 32 | 1 | 32×1×9 + 1 = **289** |
| **Conv total** | | | **93,665** |

---

## Self-attention (ConvSelfAttention)

Each module: 4 × Conv1d(32, 32, kernel_size=1) + 1 × gamma.

- **Conv1d(32, 32, 1):** 32×32×1 + 32 = **1,056** per conv.
- **Per attention module:** 4 × 1,056 + 1 = **4,225**.
- **Two modules:** 2 × 4,225 = **8,450**.

---

## Total (conv_net only, no downsampler)

**93,665 + 8,450 = 102,115 parameters.**

**Size (float32):** 102,115 × 4 bytes ≈ **0.39 MB**.

---

## Formula (generic)

For `arch_input_c=I`, `arch_num_c=C`, `arch_num_layers=L`, `arch_num_attentions=A`:

- First conv: `I×C×9 + C`
- Middle convs: `(L − 2) × (C×C×9 + C)`  (the loop runs `layer_idx in range(1, L-1)` so L−2 iterations)
- Final conv: `C×1×9 + 1`
- Attention: `A × (4×(C×C×1 + C) + 1)`

So:
- **Convs:** `9IC + C + (L−2)(9C² + C) + 9C + 1` = `9IC + (L−2)(9C²+C) + 10C + 1`
- **Attention:** `A × (4C² + 4C + 1)`

For I=3, C=32, L=12, A=2:
- Convs: 864+32 + 10×(9216+32) + 288+1 = 896 + 92480 + 289 = 93,665 ✓
- Attention: 2×(4096+128+1) = 2×4225 = 8,450 ✓

**Total = 102,115**
