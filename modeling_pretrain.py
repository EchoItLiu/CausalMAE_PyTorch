import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from adaGRF import AdaptiveCNN_Encoder
from vit_decoder import PretrainVisionTransformerDecoder
import numpy as np

from functools import partial
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_ as __call_trunc_normal_



def get_sinusoid_encoding_table(n_position, d_hid): 
    ''' Sinusoid position encoding table ''' 
    # TODO: make it with torch instead of numpy 
    def get_position_angle_vec(position): 
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)] 

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)]) 
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) # dim 2i 
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) # dim 2i+1 

    return torch.FloatTensor(sinusoid_table).unsqueeze(0) 

def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)



class PretrainGRF(nn.Module):
    """ 
    """
    def __init__(self,
                 img_size=224, 
                 patch_size=16, 
                 encoder_in_chans=10,  
                 encoder_num_classes=0, #
                 encoder_embed_dim=768, # 
                 encoder_depth=12,
                 encoder_num_heads=12, 
                 decoder_num_classes=768, 
                 decoder_embed_dim=512, 
                 decoder_depth=8,
                 decoder_num_heads=8, 
                 mlp_ratio=4., 
                 qkv_bias=False, 
                 qk_scale=None, 
                 drop_rate=0., 
                 attn_drop_rate=0.,
                 drop_path_rate=0., 
                 norm_layer=nn.LayerNorm, 
                 init_values=0.,
                 use_learnable_pos_emb=False,
                 num_classes=0, # avoid the error from create_fn in timm
                 in_chans=0, # avoid the error from create_fn in timm
                 **kwargs
                 ):
        super().__init__()
        
        
        self.encoder = AdaptiveCNN_Encoder(
            in_channels=10,
            depths=[3, 6, 40, 3],  
            dims=[96, 192, 384, 768] 
            
        )

        

        self.decoder = PretrainVisionTransformerDecoder(
            patch_size=patch_size, 
            num_classes=decoder_num_classes, 
            embed_dim=decoder_embed_dim, 
            depth=decoder_depth,
            num_heads=decoder_num_heads, 
            mlp_ratio=mlp_ratio, 
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale, 
            drop_rate=drop_rate, 
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate, 
            norm_layer=norm_layer, 
            init_values=init_values)
        
        # 
        self.encoder_to_decoder = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=False)

        
        # mask_token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        # 1 × 101  × 384
        pos_embed = get_sinusoid_encoding_table(101, decoder_embed_dim)
        self.register_buffer('pos_embed', pos_embed)
        self.pos_embed = pos_embed
        # initializing the mask token
        trunc_normal_(self.mask_token, std=.02)

        self.depths = [3, 6, 40, 3]
        #
        #
        
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()


    # initing
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.depths)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'mask_token'}   
    
    def forward(self, x, mask, is_pt):

        # [B, L, C]
        L = x.shape[1]
        # [B, L_visible/L, C]
        x_vis, _ = self.encoder(x, mask, is_pt) # 
        
        x_vis = self.encoder_to_decoder(x_vis) # [B, N_vis, C_d] 
        B, M, C = x_vis.shape



        x_dec_ordered = self.mask_token.expand(B, L, C).clone()

        
        batch_idx_vis, seq_idx_vis = (~mask).nonzero(as_tuple=True)
        batch_idx_mask, seq_idx_mask = mask.nonzero(as_tuple=True)
        
        
        x_vis_float = x_vis.reshape(-1, C).float()  

        
        x_dec_ordered[batch_idx_vis, seq_idx_vis, :] = x_vis_float


        pos_embed = self.pos_embed.to(x_dec_ordered.device)
        
        x = self.decoder(x_dec_ordered + pos_embed) #
        return x



def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
        **kwargs
    }



########################################################################
@register_model
def pretrain_causalMAE(pretrained=False, **kwargs):
    model = PretrainGRF(
        img_size=224,
        patch_size=16, 
        encoder_embed_dim=768, 
        encoder_depth=12, 
        encoder_num_heads=12,
        encoder_num_classes=0,
        decoder_num_classes=10,
        decoder_embed_dim=384,
        decoder_depth=4,
        decoder_num_heads=6,
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model
    

##################################################################


    