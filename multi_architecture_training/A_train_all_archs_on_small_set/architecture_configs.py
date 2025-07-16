from models.terrain_downsampler_k11s10 import TerrainDownsampleModel as DownsamplerK11S10
from models.non_downsampling_convolutions import NonDownsamplingConvolutions  # Arch_02
from models.simplified_unet import UNet as SimplifiedUNet  # Arch_03
from models.non_downsampling_convolutions_attention import NonDownsamplingConvolutionsAttention  # Arch_04
from models.classic_unet import UNet as ClassicUNet  # Arch_05
from models.encoder_decoder_attention import EncoderDecoderWithAttention  # Arch_07
from models.unet_resnet_modified import UNetResNetModified  # Arch_08
from models.encoder_decoder_large_convolutions import EncoderDecoderLargeConv  # Arch_09

architectures = {
    "Arch_02": {
        "model_class": NonDownsamplingConvolutions,
        "downsampler_class": DownsamplerK11S10,
        "params": {
            "input_channels": 3,
            "num_layers": 5,
            "num_channels": 32,
            "act": "leakyrelu",
            "last_act": "leakyrelu",
        },
        "downsampler_params": {
            "c_start": 10,
            "c_end": 1,
            "which_type": 3,
            "norm": False,
            "act": "leakyrelu"
        },
        "epochs": 150,
        "lr": 0.0001
    },
    "Arch_03": {
        "model_class": SimplifiedUNet,
        "downsampler_class": DownsamplerK11S10,
        "params": {
            "num_c_encoder": [3, 32, 64, 128, 256, 256, 256, 128, 64, 32, 1]
        },
        "downsampler_params": {
            "c_start": 10,
            "c_end": 1,
            "which_type": 3,
            "norm": False,
            "act": "leakyrelu"
        },
        "epochs": 150,
        "lr": 0.0001
    },
    "Arch_04": {
        "model_class": NonDownsamplingConvolutionsAttention,
        "downsampler_class": DownsamplerK11S10,
        "params": {
            "input_channels": 3,
            "num_layers": 5,
            "num_channels": 32,
            "act": "leakyrelu",
            "last_act": "leakyrelu"
        },
        "downsampler_params": {
            "c_start": 10,
            "c_end": 1,
            "which_type": 3,
            "norm": False,
            "act": "leakyrelu"
        },
        "epochs": 150,
        "lr": 0.0001
    },
    "Arch_05": {
        "model_class": ClassicUNet,
        "downsampler_class": DownsamplerK11S10,
        "params": {
            "in_channels": 3,
            "out_channels": 1,
            "init_features": 32
        },
        "downsampler_params": {
            "c_start": 10,
            "c_end": 1,
            "which_type": 3,
            "norm": False,
            "act": "leakyrelu"
        },
        "epochs": 150,
        "lr": 0.0001
    },
    "Arch_07": {
        "model_class": EncoderDecoderWithAttention,
        "downsampler_class": DownsamplerK11S10,
        "params": {
            "input_channels": 3,
            "inner_channels": 32,
            "act": "leakyrelu"
        },
        "downsampler_params": {
            "c_start": 10,
            "c_end": 1,
            "which_type": 3,
            "norm": False,
            "act": "leakyrelu"
        },
        "epochs": 150,
        "lr": 0.0001
    },
    "Arch_08": {
        "model_class": UNetResNetModified,
        "downsampler_class": DownsamplerK11S10,
        "params": {
            "input_channels": 3,
            "base_channels": 32,
            "act": "leakyrelu"
        },
        "downsampler_params": {
            "c_start": 10,
            "c_end": 1,
            "which_type": 3,
            "norm": False,
            "act": "leakyrelu"
        },
        "epochs": 150,
        "lr": 0.0001
    },
    "Arch_09": {
        "model_class": EncoderDecoderLargeConv,
        "downsampler_class": DownsamplerK11S10,
        "params": {
            "input_channels": 3,
            "inner_channels": 32,
            "act": "leakyrelu"
        },
        "downsampler_params": {
            "c_start": 10,
            "c_end": 1,
            "which_type": 3,
            "norm": False,
            "act": "leakyrelu"
        },
        "epochs": 150,
        "lr": 0.0001
    }
}
