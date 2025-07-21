import torch.nn as nn 
import torch
import torch.nn.functional as F

class ConvSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(ConvSelfAttention, self).__init__()
        self.query_conv = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.value_conv = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.get_back_C =  nn.Conv1d(in_channels, in_channels, kernel_size=1)
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
if False: # from the internet
    class SelfAttention(nn.Module):
        """ Self attention Layer"""
        def __init__(self,in_dim):
            super(SelfAttention,self).__init__()
            self.chanel_in = in_dim
            self.activation = nn.LeakyReLU()
            
            self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
            self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
            self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
            self.gamma = nn.Parameter(torch.zeros(1))
    
            self.softmax  = nn.Softmax(dim=-1) 
            
        def forward(self,x):
            """
                inputs :
                    x : input feature maps( B X C X W X H)
                returns :
                    out : self attention value + input feature 
                    attention: B X N X N (N is Width*Height)
            """
            m_batchsize,C,width ,height = x.size()
            proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
            proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
            energy =  torch.bmm(proj_query,proj_key) # transpose check
            attention = self.softmax(energy) # BX (N) X (N) 
            proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N
    
            out = torch.bmm(proj_value,attention.permute(0,2,1) )
            out = out.view(m_batchsize,C,width,height)
            
            out = self.gamma*out + x
            return out