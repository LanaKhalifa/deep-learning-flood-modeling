# multi_architecture_training/C_train_best_three_on_big_set/architecture_configs.py

from multi_architecture_training.models.terrain_downsampler_k11s10 import TerrainDownsampleK11S10
from multi_architecture_training.models.classic_unet import ClassicUNet  # Arch_05
from multi_architecture_training.models.non_downsampling_convolutions_attention import NonDownsamplingConvolutionsWithAttention  # Arch_04
from multi_architecture_training.models.encoder_decoder_attention import EncoderDecoderWithAttention  # Arch_07

# Shared downsampler configuration
default_downsampler = {
    "downsampler_class": TerrainDownsampleK11S10,
    "downsampler_params": {
        "c_start": 1,
        "c1": 16,
        "c2": 8,
        "c_end": 1,
        "act": "leakyrelu"
    }
}

# Final settings from Stage 2
architectures = {
    "Arch_05": {
        "model_class": ClassicUNet,
        "params": {},
        **default_downsampler,
        "epochs": 150,
        "lr": 0.0001
    },
    "Arch_04": {
        "model_class": NonDownsamplingConvolutionsWithAttention,
        "params": {
            "arch_input_c": 3,
            "arch_num_layers": 6,
            "arch_num_c": 32,
            "arch_act": "leakyrelu",
            "arch_last_act": "leakyrelu",
            "arch_num_attentions": 1
        },
        **default_downsampler,
        "epochs": 150,
        "lr": 0.0001
    },
    "Arch_07": {
        "model_class": EncoderDecoderWithAttention,
        "params": {},
        **default_downsampler,
        "epochs": 150,
        "lr": 0.0001
    }
}
