# config/model_configs.py
from multi_architecture_training.models.terrain_downsampler_k11s10 import TerrainDownsampleK11S10
from multi_architecture_training.models.terrain_downsampler_alternating import TerrainDownsampleAlternating
from multi_architecture_training.models.non_downsampling_convolutions import NonDownsamplingConvolutions
from multi_architecture_training.models.simplified_unet import SimplifiedUNet
from multi_architecture_training.models.non_downsampling_convolutions_attention import NonDownsamplingConvolutionsWithAttention
from multi_architecture_training.models.classic_unet import ClassicUNet
from multi_architecture_training.models.encoder_decoder_attention import EncoderDecoderWithAttention
from multi_architecture_training.models.unet_resnet_modified import UNetResNetModified
from multi_architecture_training.models.encoder_decoder_large_convolutions import EncoderDecoderWithLargeConvolutions

# ============================================================================
# STAGE A: Tune one architecture (Arch_04) on small set
# ============================================================================

def get_stage_A_config():
    """Configuration for tuning Arch_04 on small dataset"""
    
    final_config = {
        "arch_name": "Arch_04",
        "model_class": NonDownsamplingConvolutionsWithAttention,
        "params": {
            "arch_input_c": 3,        
            "arch_num_layers": 12,     
            "arch_num_c": 32,       
            "arch_act": "leakyrelu",
            "arch_last_act": "leakyrelu",
            "arch_num_attentions": 2
        },
        "downsampler_class": TerrainDownsampleAlternating,
        "downsampler_params": {
            "c_start": 1,
            "c1": 20,              
            "c2": 40,                 
            "c_end": 1,              
        }
    }
    
    
    return final_config

# ============================================================================
# STAGE B: Train all architectures on small set
# ============================================================================

def get_stage_B_configs():
    """Configuration for training all architectures on small dataset"""
        
    # Shared downsampler params for stage B
    default_downsampler = {
        "downsampler_class": TerrainDownsampleAlternating,
        "downsampler_params": {
            "c_start": 1,
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
                "arch_num_layers": 12,
                "arch_num_c": 32,
                "arch_act": "leakyrelu",
                "arch_last_act": "leakyrelu"
            },
            **default_downsampler
        },
        "Arch_03": {
            "model_class": SimplifiedUNet,
            "params": {
            },
            **default_downsampler
        },
        "Arch_04": {
            "model_class": NonDownsamplingConvolutionsWithAttention,
            "params": {
                "arch_input_c": 3,
                "arch_num_layers": 12,
                "arch_num_c": 32,
                "arch_act": "leakyrelu",
                "arch_last_act": "leakyrelu",
                "arch_num_attentions": 2
            },
            **default_downsampler
        },
        "Arch_05": {
            "model_class": ClassicUNet,
            "params": {},
            **default_downsampler
        },
        "Arch_07": {
            "model_class": EncoderDecoderWithAttention,
            "params": {},
            **default_downsampler
        },
        "Arch_08": {
            "model_class": UNetResNetModified,
            "params": {},
            **default_downsampler
        },
        "Arch_09": {
            "model_class": EncoderDecoderWithLargeConvolutions,
            "params": {},
            **default_downsampler
        }
    }
    
    return architectures

# ============================================================================
# STAGE C: Train best four architectures on big set
# ============================================================================

def get_stage_C_configs():
    """Configuration for training best four architectures on big dataset"""
    
    # Shared downsampler configuration for Arch_03, Arch_05, Arch_07
    default_downsampler = {
        "downsampler_class": TerrainDownsampleAlternating,
        "downsampler_params": {
            "c_start": 1,
            "c1": 20,
            "c2": 40,
            "c_end": 1
        }
    }
    
    # Commented out architectures (already trained):
    """
    "Arch_04": {
        "model_class": NonDownsamplingConvolutionsWithAttention,
        "params": {
            "arch_input_c": 3,        
            "arch_num_layers": 12,     
            "arch_num_c": 32,       
            "arch_act": "leakyrelu",
            "arch_last_act": "leakyrelu",
            "arch_num_attentions": 2
        },
        **default_downsampler
    },
    "Arch_05": {
        "model_class": ClassicUNet,
        "params": {},
        **default_downsampler
    },
    "Arch_02": {
    "model_class": NonDownsamplingConvolutions,
    "params": {
        "arch_input_c": 3,
        "arch_num_layers": 12,
        "arch_num_c": 32,
        "arch_act": "leakyrelu",
        "arch_last_act": "leakyrelu"
    },
    **default_downsampler
}
    """
    
    architectures = {
    "Arch_05": {
        "model_class": ClassicUNet,
        "params": {},
        **default_downsampler
    },
    "Arch_07": {
        "model_class": EncoderDecoderWithAttention,
        "params": {},
        **default_downsampler
    }}
    
    return architectures