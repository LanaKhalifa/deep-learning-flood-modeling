# multi_architecture_training/B_tune_one_arch_on_small_set/arch_04_tuned_config.py

from multi_architecture_training.models.terrain_downsampler_k11s10 import TerrainDownsampleK11S10
from multi_architecture_training.models.non_downsampling_convolutions_attention import NonDownsamplingConvolutionsWithAttention

# Final tuned configuration for Arch_04
final_config = {
    "arch_name": "Arch_04",
    "model_class": NonDownsamplingConvolutionsWithAttention,
    "params": {
        # TODO: Fill in final tuned params here
    },
    "downsampler_class": TerrainDownsampleK11S10,
    "downsampler_params": {
        "c_start": 1,
        "c1": 16,
        "c2": 8,
        "c_end": 1,
        "act": "leakyrelu"
    },
    "epochs": 150,      # Adjust if needed
    "lr": 0.0001        # Adjust if needed
}
