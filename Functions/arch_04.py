import torch.nn as nn 
import sys
import torch 

sys.path.append('/home/lana_k/Spyder_Projects/Inspect_HDF/Inspect_HDF/Functions')
from SelfAttention import ConvSelfAttention


class arch_04(nn.Module):
    def __init__(self, downsampler, arch_num_layers=6, arch_num_c=32, arch_input_c=3, arch_act='leakyrelu', arch_last_act='tanh', arch_num_attentions=1):
        super().__init__()
        self.nonlinearity = nn.LeakyReLU()
        self.downsampler = downsampler

        layers = []
        
        layers.append(nn.Conv2d(in_channels=arch_input_c, out_channels=arch_num_c, kernel_size=3, stride=1, padding=1))
        layers.append(self.nonlinearity)
        
        for _ in range(1, arch_num_layers // 2): # 1, 2
            layers.append(nn.Conv2d(in_channels=arch_num_c, out_channels=arch_num_c, kernel_size=3, stride=1, padding=1))
            layers.append(self.nonlinearity)
        
        layers.append(ConvSelfAttention(arch_num_c))
        
        for _ in range(arch_num_layers // 2, arch_num_layers - 1): # 3, 4
            layers.append(nn.Conv2d(in_channels=arch_num_c, out_channels=arch_num_c, kernel_size=3, stride=1, padding=1))
            layers.append(self.nonlinearity)
        
        layers.append(nn.Conv2d(in_channels=arch_num_c, out_channels=1, kernel_size=3, stride=1, padding=1))
        layers.append(self.nonlinearity)

        self.conv_net = nn.Sequential(*layers)        

    def forward(self, terrain, depths):
        if self.downsampler is not None: 
            terrain_out = self.downsampler(terrain)
            x = torch.cat([terrain_out, depths], dim=1)
        else:
            x = depths
        x = self.conv_net[:](x)  
        return x