import torch
import torch.nn as nn
import torch.nn.functional as F

#
#
# 

class Permute(nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims
        
    def forward(self, x):
        return x.permute(*self.dims)

class DeformableConv1D(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, 
                 dilation=1, groups=1, offset_groups=4, norm_modulation=True):

        """
        Parameter description:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of the convolution kernel
        stride: Step size
        padding: Padding
        dilation: Expansion rate
        groups: Number of groups in the convolution (multi-group mechanism)
        offset_groups: Number of groups for offset prediction (usually less than or equal to groups)
        norm_modulation: Whether to normalize the modulation factor (core innovation of the paper) 
        """

        super().__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation       

   
        self.groups = groups
        self.offset_groups = offset_groups
        self.norm_modulation = norm_modulation



        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_size))


        self.bias = nn.Parameter(torch.empty(out_channels)) # [out_channels]

        
 
        self.offset_modulator = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=1, groups=offset_groups),
            nn.GELU(),  #
            nn.Conv1d(64, offset_groups * 3 * kernel_size, kernel_size=1, groups = offset_groups)
        )

        self._reset_parameters()


    def _reset_parameters(self):
        nn.init.trunc_normal_(self.weight, std=0.02)
        
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
        
        for m in self.offset_modulator.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.constant_(m.weight, 0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        #            
        nn.init.constant_(self.offset_modulator[-1].weight, 0)
        nn.init.constant_(self.offset_modulator[-1].bias, 0)


    def forward(self, x):
        """
        Forward propagation - Strictly implements the algorithm in the paper (5-dimensional interpolation can make your code explode~~)
        Input: x [B, C, L_in]
        Output: [B, out_channels, L_out]
        """
        B, C, L_in = x.shape
        device = x.device
        
        L_out = (L_in + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1

        
        grid_x = torch.linspace(-1, 1, L_out, device=device)  # [L_out]
        grid_x = grid_x.view(1, L_out, 1).expand(B, L_out, 1)  # [B, L_out, 1]
        grid_y = torch.zeros(1, L_out, 1, device=device).expand(B, L_out, 1)  # [B, W_out, 1] # 
        
        grid_base = torch.cat([grid_x, grid_y], dim=-1)  # [B, L_out, 2]
        grid_base = grid_base.unsqueeze(2)  # [B, L/W_out, 1, 2] -
        
        

        
        offset_modulator = self.offset_modulator(x)  # [B, 3*kernel_size, L_in]
        
        offset = offset_modulator[:, :2*self.kernel_size, :]  # [B, 2*kernel_size, L_in]
        modulator = offset_modulator[:, self.kernel_size:, :]  # [B, kernel_size, L_in]

        
        #  [B, 2*kernel_size, L_in] -> [B, kernel_size, 2, L_in]
        offset = offset.view(B, self.kernel_size, 2, L_in)   # L_in = L_out 
        offset[:, :, 1, :] = 0  # 
        
        
        if self.norm_modulation:
            modulator = F.softmax(modulator, dim=1)  # [B, kernel_size, L_in]
        else:
            modulator = torch.sigmoid(modulator)  # [0,1]

        
        kernel_pos = torch.linspace(-0.5, 0.5, self.kernel_size, device=device)  # [-0.5, 0.5]
        kernel_pos = kernel_pos.view(1, 1, 1, self.kernel_size, 1)  # [1, 1, 1, kernel_size, 1]
       
        
        output = torch.zeros(B, self.weight.size(0), L_out, device = device)

        
        for k in range(self.kernel_size):
            # [B, 2, L_in] -> [B, 1, 2, L_in] -> [B, 1, L_in, 2]
            offset_k = offset[:, k, :, :].unsqueeze(1)
            offset_k = torch.transpose(offset_k, 2,3) 
            
            offset_k = offset_k * (2.0 / L_in)  


            # [B, 1, L_in, 2]([B, 2, 1 ,L_in]) -> [B, 2, 1, L_out]
            offset_k = F.interpolate(offset_k.permute(0, 3, 1, 2), 
                                   size=(1, L_out), mode='bilinear', align_corners=True)
            offset_k = offset_k.permute(0, 2, 3, 1)  # [B, 2, 1 ,L_out] -> [B, 1, L_out, 2]

            
            # 
            grid_k = grid_base.clone() # [B, L_out, 1, 2]

            '''
            '''
            # p_0 + p_k + Δp_k  [B, L_out, 1, 2] + [1, 1, 1, 1] + [B, 1, L_out, 2]
            current_kernel_pos = kernel_pos[:, :, :, k, :]  # [1, 1, 1, 1]
            grid_k = grid_k + current_kernel_pos + offset_k  


            x_4d = x.unsqueeze(2)  

  
            
            
            sampled = F.grid_sample(
                x_4d,   # [B, C, 1, L_in]
                grid_k,  # [B, L_out, 1, 2]
                mode='bilinear',
                padding_mode='zeros',
                align_corners=True
                )  

            sampled  =  sampled[:, :, 0, :] # [B, C, L_out]

            # 
            modulator_k = modulator[:, k, :].unsqueeze(1)  # [B, 1, L_in] -> [B, 1, L_out]
         
            modulator_k = F.interpolate(modulator_k.unsqueeze(1), size=(1, L_out), mode='bilinear', align_corners=True)             
            modulator_k = modulator_k.squeeze(1)  # [B, L_out]
           
           
            sampled = sampled * modulator_k

            
            weight_k = self.weight[:, :, k].unsqueeze(-1)  # [out_channels, in_channels//groups, 1]

            
            conved = F.conv1d(
                sampled, 
                weight_k, 
                bias=None,
                stride=1,
                padding=0,
                dilation=1,
                groups=self.groups
            )

            
            # w_g * [m_kg * x(p_0 + p_k + Δp_kg)]
            # [B, C, L_out]
            output += conved
        
        if self.bias is not None:
            output += self.bias.view(1, -1, 1) # [1, C, 1]

        return output  # [B, out_channels, L_out]





class Block1d(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0.):
        super().__init__()
        LayerNorm = nn.LayerNorm
        self.norm1 = LayerNorm(dim)
        self.conv1 = DeformableConv1D(dim, dim, kernel_size=3, stride=1, padding=1, dilation=1, groups=4, offset_groups=2, norm_modulation=True )
        self.drop_path = nn.Dropout(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = LayerNorm(dim)
        
        # MLP
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv1d(dim, mlp_hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Conv1d(mlp_hidden_dim, dim, kernel_size=1),
            nn.Dropout(drop)
        )
    
    def forward(self, x):

        shortcut = x 
        x = self.norm1(x)
        # x: [B, L, C] -> [B, C, L]
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        # B × C  × L -> [B, L, C]
        x = x.permute(0, 2, 1)
        x = shortcut + self.drop_path(x)
        
        shortcut = x
        # [B, L, C]
        x = self.norm2(x)
        # x: [B, L, C] -> [B, C, L]
        x = x.permute(0, 2, 1)
        #
        x = self.mlp(x)
        # B × C  × L -> [B, L, C]
        x = x.permute(0, 2, 1)
        # [B, L, C]
        x = shortcut + self.drop_path(x)
        
        return x





class AdaptiveCNN_Encoder(nn.Module):
    def __init__(self, in_channels=10,  depths=[3, 6, 40, 3], dims=[96, 192, 384, 768]):
        super().__init__()

        LayerNorm = nn.LayerNorm
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, dims[0], kernel_size=7, stride=1, padding=3),
            
            Permute(0, 2, 1), # x: [B, C, L_visible/L ] -> [B, L_visible/L, C]
            LayerNorm(dims[0])
        )
        

        self.stages = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        self.dims = dims

        for i in range(len(depths)):
            if i > 0:
                downsample = nn.Sequential(
                    nn.Conv1d(dims[i-1], dims[i], kernel_size=3, stride=1, padding=1),                    
                    Permute(0, 2, 1),
                    LayerNorm(dims[i])
                )
            else:
                downsample = nn.Identity()
            self.downsample_layers.append(downsample)

            stage_blocks = []
            for j in range(depths[i]):
                stage_blocks.append(Block1d(dims[i]))
            self.stages.append(nn.Sequential(*stage_blocks))
            

    def forward(self, x, mask, is_pt):

        # 128 × 10 × 101  [B, L, C]
        B, _, C = x.shape

        if is_pt:
            x = x[~mask].reshape(B, -1, C) # [B, L_visible, C]
        
            
        # x: [B, L_visible/L, C] -> [B, C, L_visible/L]
        x = x.permute(0, 2, 1)

        
        # stem
        x = self.stem(x) # [B, L_visible/L, C] 
 
        
        features = []
        for i in range(len(self.stages)):
            if i > 0:
                x = x.permute(0, 2, 1)
                x = self.downsample_layers[i](x)
                x = x.permute(0, 2, 1)
            
            
            if x.dim() == 3 and x.size(1) != x.size(2): 
                B, dim1, dim2 = x.size()
                
                if dim1 == self.dims[i]:
                    x = x.permute(0, 2, 1)  # [B, C, L] -> [B, L, C]

    
            x = self.stages[i](x)
            
            features.append(x)
    
        return x, features