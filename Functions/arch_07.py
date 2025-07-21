#%% Import libraries
import torch 
import torch.nn as nn 
import torch.nn.functional as F

class ConvSelfAttention(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1):
        super(ConvSelfAttention, self).__init__()
        self.query_conv = nn.Conv1d(in_channels, out_channels, kernel_size)
        self.key_conv = nn.Conv1d(in_channels, out_channels, kernel_size)
        self.value_conv = nn.Conv1d(in_channels, out_channels, kernel_size)
        self.get_back_C =  nn.Conv1d(out_channels, in_channels, kernel_size)
        self.gamma = nn.Parameter(torch.zeros(1))
        

    def forward(self, x):
        batch_size, C, width, height = x.size()
        N = width * height
        
        # Flatten spatial dimensions
        x_flat = x.view(batch_size, C, N)  # B x C x N
        
        # Generate queries, keys, and values
        queries = self.query_conv(x_flat)  # B x C' x N
        keys = self.key_conv(x_flat)       # B x C' x N
        values = self.value_conv(x_flat)   # B x C' x N
        
        attention = torch.bmm(queries.permute(0,2,1), keys)
        attention = F.softmax(attention / (C ** 0.5), dim=-1) # they wrote in the artoc;es attention = F.softmax(attention, dim=-1). ChatGPT is soooooo impressive
        
        # Apply attention to values
        out = torch.bmm(values, attention.permute(0, 2, 1))  # B x C' x N
        
        # return to original C: 
        out_with_C = self.get_back_C(out)
        out_with_C = out_with_C.view(batch_size, C, width, height)
        
        # Combine with input feature map
        out = self.gamma * out_with_C + x
        
        return out
  
class arch_07(nn.Module):
    def __init__(self, downsampler):
        super(arch_07, self).__init__()
        self.downsampler = downsampler
        self.nonlinearity = nn.LeakyReLU()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            self.nonlinearity,
            nn.Conv2d(in_channels=32, out_channels=48, kernel_size=5, stride=1, padding=2),
            self.nonlinearity,
            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=5, stride=2, padding=2),
            self.nonlinearity,
            nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, stride=2, padding=1),
            self.nonlinearity,
            nn.Conv2d(in_channels=96, out_channels=128, kernel_size=3, stride=2, padding=1),
            self.nonlinearity
        )

        # Self Attention Layer
        self.attention = ConvSelfAttention(in_channels=128, out_channels=128)

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=69, kernel_size=3, stride=2, padding=1),
            self.nonlinearity,
            nn.ConvTranspose2d(in_channels=69, out_channels=64, kernel_size=3, stride=2, padding=0),
            self.nonlinearity,
            nn.ConvTranspose2d(in_channels=64, out_channels=48, kernel_size=4, stride=2, padding=0),
            self.nonlinearity,
            nn.ConvTranspose2d(in_channels=48, out_channels=32, kernel_size=5, stride=1, padding=2),
            self.nonlinearity,
            nn.ConvTranspose2d(in_channels=32, out_channels=6, kernel_size=3, stride=1, padding=1),
            self.nonlinearity,
            nn.ConvTranspose2d(in_channels=6, out_channels=1, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU()
        )

    def forward(self, terrain, depths):
        terrain_out = self.downsampler(terrain)
        x = torch.cat([terrain_out, depths], dim=1)
        x = self.encoder(x)
        x = self.attention(x)
        x = self.decoder(x)
        return x
