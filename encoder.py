import torch
from torch import nn
from torch.nn import functional as F
from blocks import ResidualBlock, AttentionBlock,DepthwiseSeparableConv, SqueezeAndExcitationBlock

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            
            # Initial convolution
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # Depthwise separable convolution + Residual
            DepthwiseSeparableConv(64, 128, kernel_size=3, padding=1),
            ResidualBlock(128, 128),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),

            # Adaptive pooling for flexible dimension reduction
            nn.AdaptiveAvgPool2d((None, None)),  

            # Further processing with more depthwise convolutions and residuals
            DepthwiseSeparableConv(128, 256, kernel_size=3, padding=1, stride=2),
            ResidualBlock(256, 256),
            SqueezeAndExcitationBlock(256),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),

            # Incorporate attention mechanism
            AttentionBlock(256),
            DepthwiseSeparableConv(256, 512, kernel_size=3, padding=1, stride=2),
            ResidualBlock(512, 512),
            nn.InstanceNorm2d(512),
            nn.ReLU(inplace=True),

            # Final compression to desired latent size
            nn.Conv2d(512, 8, kernel_size=3, padding=1),
            nn.Conv2d(8, 8, kernel_size=1, padding=0),
        )

    def forward(self, x, noise):


        for module in self.layers:
            # Apply asymmetric padding for layers with stride 2 for downsampling
            if isinstance(module, nn.Conv2d) and module.stride == (2, 2):
                x = F.pad(x, (0, 1, 0, 1))  # Right and bottom padding
            
            x = module(x)
        
        # Split the output into mean and log variance for the latent representation
        mean, log_variance = torch.chunk(x, 2, dim=1)
        
        # Clamp the log variance to ensure numerical stability
        log_variance = torch.clamp(log_variance, -30, 20)
        
        # Convert log variance to standard deviation
        stdev = torch.exp(log_variance * 0.5)
        
        # Apply the reparameterization trick: scale the noise and add the mean
        latent = mean + noise * stdev
        
        # Scale the output 
        # Constant taken from: https://github.com/CompVis/stable-diffusion
        # /blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/configs/stable-diffusion/v1-inference
        # .yaml#L17C1-L17C1
        
        scaled_latent = latent * 0.18215
        
        return scaled_latent
