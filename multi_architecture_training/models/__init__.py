# multi_architecture_training/models/__init__.py
# Model registry for all available architectures

from .classic_unet import ClassicUNet
from .simplified_unet import SimplifiedUNet
from .non_downsampling_convolutions import NonDownsamplingConvolutions
from .non_downsampling_convolutions_attention import NonDownsamplingConvolutionsWithAttention
from .encoder_decoder_attention import EncoderDecoderWithAttention
from .unet_resnet_modified import UNetResNetModified
from .encoder_decoder_large_convolutions import EncoderDecoderWithLargeConvolutions
from .terrain_downsampler_alternating import TerrainDownsampleAlternating
from .terrain_downsampler_k11s10 import TerrainDownsampleK11S10
from .terrain_downsampler_cubic import TerrainDownsampleCubic
from .attention import ConvSelfAttention as Attention
from .self_attention import ConvSelfAttention as SelfAttention
from .alternating_stride_conv2d import AlternatingStrideConv2d

# Model registry with descriptions
MODEL_REGISTRY = {
    # Core architectures
    'ClassicUNet': {
        'class': ClassicUNet,
        'description': 'Standard UNet implementation with skip connections',
        'complexity': 'High',
        'parameters': '~2M',
        'best_for': 'General flood modeling'
    },
    'SimplifiedUNet': {
        'class': SimplifiedUNet,
        'description': 'Lightweight UNet variant with reduced complexity',
        'complexity': 'Medium',
        'parameters': '~500K',
        'best_for': 'Fast inference, limited resources'
    },
    'NonDownsamplingConvolutions': {
        'class': NonDownsamplingConvolutions,
        'description': 'Basic convolutional network without downsampling',
        'complexity': 'Low',
        'parameters': '~100K',
        'best_for': 'Simple flood modeling tasks'
    },
    'NonDownsamplingConvolutionsWithAttention': {
        'class': NonDownsamplingConvolutionsWithAttention,
        'description': 'Convolutional network with self-attention mechanisms',
        'complexity': 'Medium',
        'parameters': '~200K',
        'best_for': 'Attention-enhanced flood modeling'
    },
    'EncoderDecoderWithAttention': {
        'class': EncoderDecoderWithAttention,
        'description': 'Encoder-decoder architecture with attention',
        'complexity': 'High',
        'parameters': '~1M',
        'best_for': 'Complex flood patterns'
    },
    'UNetResNetModified': {
        'class': UNetResNetModified,
        'description': 'UNet with ResNet-style residual connections',
        'complexity': 'High',
        'parameters': '~3M',
        'best_for': 'Deep feature learning'
    },
    'EncoderDecoderWithLargeConvolutions': {
        'class': EncoderDecoderWithLargeConvolutions,
        'description': 'Encoder-decoder with large kernel convolutions',
        'complexity': 'Medium',
        'parameters': '~800K',
        'best_for': 'Large-scale flood patterns'
    }
}

# Downsampler registry
DOWNSAMPLER_REGISTRY = {
    'TerrainDownsampleAlternating': {
        'class': TerrainDownsampleAlternating,
        'description': 'Alternating stride downsampling for terrain',
        'best_for': 'General terrain processing'
    },
    'TerrainDownsampleK11S10': {
        'class': TerrainDownsampleK11S10,
        'description': '11x11 kernel with stride 10 downsampling',
        'best_for': 'Aggressive terrain reduction'
    },
    'TerrainDownsampleCubic': {
        'class': TerrainDownsampleCubic,
        'description': 'Cubic interpolation downsampling',
        'best_for': 'Smooth terrain processing'
    }
}

# Attention mechanism registry
ATTENTION_REGISTRY = {
    'Attention': {
        'class': Attention,
        'description': 'General attention mechanism',
        'best_for': 'General attention applications'
    },
    'SelfAttention': {
        'class': SelfAttention,
        'description': 'Self-attention mechanism',
        'best_for': 'Self-attention in convolutions'
    }
}

def get_model_class(model_name):
    """Get model class by name"""
    if model_name in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_name]['class']
    else:
        raise ValueError(f"Unknown model: {model_name}")

def get_downsampler_class(downsampler_name):
    """Get downsampler class by name"""
    if downsampler_name in DOWNSAMPLER_REGISTRY:
        return DOWNSAMPLER_REGISTRY[downsampler_name]['class']
    else:
        raise ValueError(f"Unknown downsampler: {downsampler_name}")

def get_attention_class(attention_name):
    """Get attention class by name"""
    if attention_name in ATTENTION_REGISTRY:
        return ATTENTION_REGISTRY[attention_name]['class']
    else:
        raise ValueError(f"Unknown attention mechanism: {attention_name}")

def list_available_models():
    """List all available models with descriptions"""
    print("🏗️  Available Models:")
    print("=" * 60)
    for name, info in MODEL_REGISTRY.items():
        print(f"📋 {name}")
        print(f"   Description: {info['description']}")
        print(f"   Complexity: {info['complexity']}")
        print(f"   Parameters: {info['parameters']}")
        print(f"   Best for: {info['best_for']}")
        print()

def list_available_downsamplers():
    """List all available downsamplers"""
    print("🗜️  Available Downsamplers:")
    print("=" * 40)
    for name, info in DOWNSAMPLER_REGISTRY.items():
        print(f"📋 {name}")
        print(f"   Description: {info['description']}")
        print(f"   Best for: {info['best_for']}")
        print()

def list_available_attention():
    """List all available attention mechanisms"""
    print("👁️  Available Attention Mechanisms:")
    print("=" * 45)
    for name, info in ATTENTION_REGISTRY.items():
        print(f"📋 {name}")
        print(f"   Description: {info['description']}")
        print(f"   Best for: {info['best_for']}")
        print()

# Convenience imports for common models
__all__ = [
    'ClassicUNet',
    'SimplifiedUNet', 
    'NonDownsamplingConvolutions',
    'NonDownsamplingConvolutionsWithAttention',
    'EncoderDecoderWithAttention',
    'UNetResNetModified',
    'EncoderDecoderWithLargeConvolutions',
    'TerrainDownsampleAlternating',
    'TerrainDownsampleK11S10',
    'TerrainDownsampleCubic',
    'Attention',
    'SelfAttention',
    'AlternatingStrideConv2d',
    'MODEL_REGISTRY',
    'DOWNSAMPLER_REGISTRY',
    'ATTENTION_REGISTRY',
    'get_model_class',
    'get_downsampler_class',
    'get_attention_class',
    'list_available_models',
    'list_available_downsamplers',
    'list_available_attention'
]
