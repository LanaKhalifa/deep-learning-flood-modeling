import torch.nn as nn 
import torch 

class down_conv(nn.Module):
    def __init__(self, in_c, out_c, last=False):
        super().__init__()
        self.down_conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2)
        )
        if last:
            self.down_conv = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.2)
            )

    def forward(self, x):
        x = self.down_conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self, in_c, out_c, first=False, second=False, last=False):
        super().__init__()
        if first or second:
            self.up_conv = nn.Sequential(
                nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1, output_padding=0, dilation=1),
                #nn.BatchNorm2d(out_c),
                nn.ReLU(),
                #nn.Dropout(0.5)
            )
        if last:
            self.up_conv = nn.Sequential(
                nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1, output_padding=0, dilation=1),
                nn.LeakyReLU()
            )
        else:
            self.up_conv = nn.Sequential(
                nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1, output_padding=0, dilation=1),
                nn.ReLU()
            )
    def forward(self, x):
        x = self.up_conv(x)
        return x

class arch_03(nn.Module):
    def __init__(self, downsampler, num_c_encoder=[3, 32, 64, 128, 256, 256, 256, 128, 64, 32, 1]):
        super().__init__()
        self.downsampler = downsampler
        
        self.down_conv_1 = down_conv(num_c_encoder[0], num_c_encoder[1])
        self.down_conv_2 = down_conv(num_c_encoder[1], num_c_encoder[2])
        self.down_conv_3 = down_conv(num_c_encoder[2], num_c_encoder[3])
        self.down_conv_4 = down_conv(num_c_encoder[3], num_c_encoder[4])
        self.down_conv_5 = down_conv(num_c_encoder[4], num_c_encoder[5], last=True)
        
        self.up_conv_1 = up_conv(num_c_encoder[5], num_c_encoder[6], first=True)
        self.up_conv_2 = up_conv(num_c_encoder[6] * 2, num_c_encoder[7], second=True)
        self.up_conv_3 = up_conv(num_c_encoder[7] * 2, num_c_encoder[8])
        self.up_conv_4 = up_conv(num_c_encoder[8] * 2, num_c_encoder[9])
        self.up_conv_5 = up_conv(num_c_encoder[9] * 2, num_c_encoder[10], last=True)
        
    def forward(self, terrain, depths):
        terrain_out = self.downsampler(terrain)
        x  = torch.concat([terrain_out, depths], dim=1)
        
        # Encoder
        x1 = self.down_conv_1(x)
        x2 = self.down_conv_2(x1)
        x3 = self.down_conv_3(x2)
        x4 = self.down_conv_4(x3)
        x5 = self.down_conv_5(x4)

        # Decoder
        x6 = self.up_conv_1(x5)
        x6 = torch.cat([x6, x4], dim=1)
        x7 = self.up_conv_2(x6)
        x7 = torch.cat([x7, x3], dim=1)
        x8 = self.up_conv_3(x7)
        x8 = torch.cat([x8, x2], dim=1)
        x9 = self.up_conv_4(x8)
        x9 = torch.cat([x9, x1], dim=1)
        x10 = self.up_conv_5(x9)
        return x10