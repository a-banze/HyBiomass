# Source: https://github.com/AABNassim/spectral_earth

# Import VisionTransformer from timm
from timm.models.vision_transformer import VisionTransformer    
from functools import partial
from torch import nn
import torch
from typing import Union, Sequence
from typing import Tuple
from .spectral_adapter import SpectralAdapter



class SpecVisionTransformer(nn.Module):
    def __init__(self, token_patch_size=4, patch_size=128, embed_dim=768, reduced_channels=128, dynamic_img_size=False, depth=12, num_heads=6, mlp_ratio=4, **kwargs):
        super(SpecVisionTransformer, self).__init__()
        
        # 1x1 Convolution to reduce channel dimensions
        #self.channel_reducer = nn.Conv2d(in_channels, reduced_channels, kernel_size=1)

        self.spectral_adapter = SpectralAdapter()

        # Initialize Vision Transformer
        self.vit_core = VisionTransformer(
            img_size=patch_size, patch_size=token_patch_size, in_chans=reduced_channels, embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), dynamic_img_size=dynamic_img_size, **kwargs)
        
        self.num_features = self.vit_core.num_features
    
    def forward(self, x):
        x = self.spectral_adapter(x)
        
        # Pass through Vision Transformer
        x = self.vit_core(x)
        
        return x
    
    def _intermediate_layers(
            self,
            x: torch.Tensor,
            n: Union[int, Sequence] = 1,
    ):
        outputs, num_blocks = [], len(self.vit_core.blocks)
        take_indices = set(range(num_blocks - n, num_blocks) if isinstance(n, int) else n)

        # forward pass
        x = self.spectral_adapter(x)
        x = self.vit_core.patch_embed(x)
        x = self.vit_core._pos_embed(x)
        x = self.vit_core.patch_drop(x)
        x = self.vit_core.norm_pre(x)
        for i, blk in enumerate(self.vit_core.blocks):
            x = blk(x)
            if i in take_indices:
                outputs.append(x)

        return outputs
    
    def get_intermediate_layers(
            self,
            x: torch.Tensor,
            n: Union[int, Sequence] = 1,
            reshape: bool = False,
            return_prefix_tokens: bool = False,
            norm: bool = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        """ Intermediate layer accessor (NOTE: This is a WIP experiment).
        Inspired by DINO / DINOv2 interface
        """
        # take last n blocks if n is an int, if in is a sequence, select by matching indices
        outputs = self._intermediate_layers(x, n)
        if norm:
            outputs = [self.vit_core.norm(out) for out in outputs]
        prefix_tokens = [out[:, 0:self.vit_core.num_prefix_tokens] for out in outputs]
        outputs = [out[:, self.vit_core.num_prefix_tokens:] for out in outputs]

        if reshape:
            grid_size = self.vit_core.patch_embed.grid_size
            outputs = [
                out.reshape(x.shape[0], grid_size[0], grid_size[1], -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]

        if return_prefix_tokens:
            return tuple(zip(outputs, prefix_tokens))
        return tuple(outputs)
    
    def get_classifier(self):
        return self.vit_core.head


class SpecViTSmall(SpecVisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(
            embed_dim=384,
            depth=12,
            num_heads=6,
            mlp_ratio=4,
            **kwargs
        )

class SpecViTBase(SpecVisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            **kwargs
        )

class SpecViTLarge(SpecVisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(
            embed_dim=1024,
            depth=24,
            num_heads=16,
            mlp_ratio=4,
            **kwargs
        )

class SpecViTHuge(SpecVisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(
            embed_dim=1280,
            depth=32,
            num_heads=16,
            mlp_ratio=4,
            **kwargs
        )
        
class SpecViTGiant(SpecVisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(
            embed_dim=1536,
            depth=40,
            num_heads=24,
            mlp_ratio=4,
            **kwargs
        )

class SpectralAdapterProjection(nn.Module):
    def __init__(self, spectral_adapter, reduced_channels, embed_dim, token_patch_size):
        super(SpectralAdapterProjection, self).__init__()
        self.spectral_adapter = spectral_adapter
        self.conv2d = nn.Conv2d(
            in_channels=reduced_channels, 
            out_channels=embed_dim, 
            kernel_size=token_patch_size, 
            stride=token_patch_size
        )

    def forward(self, x):
        # Add channel dimension: [batch_size, 1, depth, height, width]
        x = self.spectral_adapter(x)

        # Pass through the final 2D convolution layer
        x = self.conv2d(x)
        return x
    