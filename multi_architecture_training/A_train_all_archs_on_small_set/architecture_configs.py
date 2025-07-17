from multi_architecture_training.models.terrain_downsampler_k11s10 import TerrainDownsampleK11S10
from multi_architecture_training.models.terrain_downsampler_alternating import TerrainDownsampleAlternating
from multi_architecture_training.models.non_downsampling_convolutions import NonDownsamplingConvolutions  # Arch_02
from multi_architecture_training.models.simplified_unet import SimplifiedUNet  # Arch_03
from multi_architecture_training.models.non_downsampling_convolutions_attention import NonDownsamplingConvolutionsWithAttention  # Arch_04
from multi_architecture_training.models.classic_unet import ClassicUNet  # Arch_05
from multi_architecture_training.models.encoder_decoder_attention import EncoderDecoderWithAttention  # Arch_07
from multi_architecture_training.models.unet_resnet_modified import UNetResNetModified  # Arch_08
from multi_architecture_training.models.encoder_decoder_large_convolutions import EncoderDecoderWithLargeConvolutions  # Arch_09

# Shared downsampler params
default_downsampler = {
    "downsampler_class": TerrainDownsampleAlternating,
    "downsampler_params": {
        "c_start": 3,
        "c1": 20,
        "c2": 40,
        "c_end": 1
    }
}

architectures = {
    "Arch_02": {
        "model_class": NonDownsamplingConvolutions,
        "params": {
            "arch_input_c": 3,
            "arch_num_layers": 5,
            "arch_num_c": 32,
            "arch_act": "leakyrelu",
            "arch_last_act": "leakyrelu"
        },
        **default_downsampler,
        "epochs": 150,
        "lr": 0.0001
    },
    "Arch_03": {
        "model_class": SimplifiedUNet,
        "params": {
            "num_c_encoder": [3, 32, 64, 128, 256, 256, 256, 128, 64, 32, 1]
        },
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
    "Arch_05": {
        "model_class": ClassicUNet,
        "params": {},
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
    },
    "Arch_08": {
        "model_class": UNetResNetModified,
        "params": {},
        **default_downsampler,
        "epochs": 150,
        "lr": 0.0001
    },
    "Arch_09": {
        "model_class": EncoderDecoderWithLargeConvolutions,
        "params": {},
        **default_downsampler,
        "epochs": 150,
        "lr": 0.0001
    }
}
