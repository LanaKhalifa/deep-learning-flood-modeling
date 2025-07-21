import torch.nn as nn

class TerrainDownsample_k11s1p0(nn.Module):

    def __init__(self, c_start=1, c_end=1, c1=1, c2=1, act='leakyrelu'):
        super().__init__()
        self.c1 = nn.Conv2d(c_start, c1, kernel_size=11, stride=10, padding=0)
        self.c2 = nn.Conv2d(c1, c2, kernel_size=1, stride=1, padding=0)
        self.c3 = nn.Conv2d(c2, c_end, kernel_size=1, stride=1, padding=0)

        if act == 'leakyrelu':
            self.nonlinearity = nn.LeakyReLU()
        elif act == 'relu':
            self.nonlinearity = nn.ReLU()
        elif act == 'prelu':
            self.nonlinearity = nn.PReLU()
        elif act == 'relu':
            self.nonlinearity = nn.ReLU()
        
        self._initialize_weights()

    def forward(self, x):        
        x = self.c1(x)
        x = self.nonlinearity(x)

        x = self.c2(x)
        x = self.nonlinearity(x)
        
        x = self.c3(x)
        x = self.nonlinearity(x)
        return x    
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.2 if isinstance(self.nonlinearity, nn.LeakyReLU) else 0, mode='fan_in')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
