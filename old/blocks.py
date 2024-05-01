import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention


#Attention block for encoder and decoder
#used for better understanding of the features
class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
    
        self.groupnorm = nn.GroupNorm(32, channels)  
        self.attention = SelfAttention(1, channels)  
    
    def forward(self, x):
        #save
        residue = x  
        
        #reshape
        x = self.groupnorm(x) 
        n, c, h, w = x.shape
        x = x.view(n, c, h * w)  
        x = x.transpose(-1, -2)  
        
        #attention  
        x = self.attention(x) 
        
        #reshape
        x = x.transpose(-1, -2)  
        x = x.view(n, c, h, w)  
        
        #add residue
        x += residue 
        return x


# Residual block for encoder and decoder
# used for better understanding of the features
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # Match the dimensions of the residual 
        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
    
    def forward(self, x):

        #save
        residue = x

        #convolution 1
        x = self.groupnorm_1(x) 
        x = F.silu(x)
        x = self.conv_1(x)
        
        #convolution 2
        x = self.groupnorm_2(x) 
        x = F.silu(x)
        x = self.conv_2(x)
        
        #add residue
        return x + self.residual_layer(residue)


# Squeeze and Excitation block for encoder and decoder
# used for better understanding of the features
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                   stride=stride, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


# Squeeze and Excitation block for encoder and decoder
# used for better understanding of the features
class SqueezeAndExcitationBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)  # Global information squeezing
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        batch_size, num_channels, _, _ = x.size()
        y = self.squeeze(x).view(batch_size, num_channels)  # Squeeze
        y = self.excitation(y).view(batch_size, num_channels, 1, 1)  # Excite
        return x * y.expand_as(x)  # Scale input by channel-wise excitations
