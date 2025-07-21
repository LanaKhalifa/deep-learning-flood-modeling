import torch.nn as nn 
import sys
import torch 

sys.path.append('/home/lana_k/Spyder_Projects/Inspect_HDF/Inspect_HDF_thesis_final/Functions')

class arch_02(nn.Module):
    def __init__(self, downsampler, arch_num_layers=5, arch_num_c=32, arch_input_c=3, arch_act='leakyrelu', arch_last_act='leakyrelu'):
        super().__init__()
        if arch_act == 'leakyrelu':
            self.nonlinearity = nn.LeakyReLU()
            
        self.downsampler = downsampler
        
        layers = []
        layers.append(nn.Conv2d(in_channels=arch_input_c, out_channels=arch_num_c, kernel_size=3, stride=1, padding=1))
        layers.append(self.nonlinearity)
        for _ in range(1, arch_num_layers-1): # 1, 2, 3
            layers.append(nn.Conv2d(in_channels=arch_num_c, out_channels=arch_num_c, kernel_size=3, stride=1, padding=1))
            layers.append(nn.LeakyReLU())
        layers.append(nn.Conv2d(in_channels=arch_num_c, out_channels=1, kernel_size=3, stride=1, padding=1))
        layers.append(nn.LeakyReLU())
        
        self.conv_net = nn.Sequential(*layers)

    def forward(self, terrain, depths):
        terrain_out = self.downsampler(terrain)
        x = torch.cat([terrain_out, depths], dim=1)
        x = self.conv_net[:](x)  
        return x