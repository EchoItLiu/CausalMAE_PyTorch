import math
from functools import partial
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from adaGRF import AdaptiveCNN_Encoder


    


class FinetuneGRF(nn.Module):
    """ 
    """
    def __init__(self, 
                 img_size=224, 
                 patch_size=16, 
                 in_chans=10,  # 
                 num_classes=2, #
                 # num_classes=2, #               
                 # num_classes=1, # 
                 embed_dim=768, #
                 depth=12,
                 num_heads=12, 
                 mlp_ratio=4., 
                 qkv_bias=False, 
                 qk_scale=None, 
                 drop_rate=0., 
                 attn_drop_rate=0.,
                 drop_path_rate=0., 
                 norm_layer=nn.LayerNorm, 
                 init_values=0.,
                 use_learnable_pos_emb=False, 
                 init_scale=0.,
                 use_mean_pooling=True,
                 **kwargs                 
                ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        
        
        self.encoder = AdaptiveCNN_Encoder(
            in_channels=10,
            depths=[3, 6, 40, 3],  # 
            dims=[96, 192, 384, 768]  # 
        )
        
        self.depths = [3, 6, 40, 3]
        

        #
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)

        
        trunc_normal_(self.head.weight, std=.02)
        self.apply(self._init_weights)

        self.head.weight.data.mul_(init_scale)
        self.head.bias.data.mul_(init_scale)
    


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def get_num_layers(self):
        return len(self.depths)
        

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}


    def forward(self, x, mask, is_pt):

        # [B, C1, L] -> [B, L, C1]  
        x = torch.transpose(x,1,2)
        # print ('x_ft_before_shape:', x.shape)
        #  [B, L, C1] -> [B, L, C2]        
        latent, _ = self.encoder(x, mask, is_pt)

        # RuntimeError: Given groups=1, weight of size [96, 10, 7], expected input[32, 101, 10] to have 10 channels, but got 101 channels instead

        # 
        latent_pooled = latent.mean(1)  # [B, C]   
        # 
        logits = self.head(latent_pooled) # [B, 1] / [B, 2]

        return logits


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
        **kwargs
    }


###
@register_model
def finetune_causalMAE(pretrained=False, **kwargs):
    model = FinetuneGRF(
        patch_size = 16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer = partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model




