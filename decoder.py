import torch
from torch import nn
from blocks import ResidualBlock, AttentionBlock, DepthwiseSeparableConv, SqueezeAndExcitationBlock

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(4, 512, kernel_size=1, padding=0),  
            
            # Introduce Squeeze-and-Excitation early to recalibrate channel-wise feature responses
            SqueezeAndExcitationBlock(512),
            
            # Upsampling and depthwise separable convolutions for efficient feature expansion
            nn.Upsample(scale_factor=2),
            DepthwiseSeparableConv(512, 512, kernel_size=3, padding=1),
            
            # Incorporate attention mechanism for capturing long-range dependencies
            AttentionBlock(512),
            
            # Sequential Residual Blocks for enhanced feature processing
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            
            # Continue with upsampling and convolution to progressively increase spatial dimensions
            nn.Upsample(scale_factor=2),
            DepthwiseSeparableConv(512, 256, kernel_size=3, padding=1),
            
            # Further refine features with residual blocks at a lower channel dimension
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            
            # Final upsampling to reach the original image dimensions
            nn.Upsample(scale_factor=2),
            DepthwiseSeparableConv(256, 128, kernel_size=3, padding=1),
            
            # Final processing before reconstructing the image
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            
            nn.Conv2d(128, 3, kernel_size=3, padding=1),  # Output layer to produce the 3-channel image
            nn.Sigmoid()  # Ensure output values are in the [0, 1] range
        )
        
    def forward(self, x):
        # Reverse the scaling applied by the Encoder
        x /= 0.18215
        
        for module in self.layers:
            x = module(x)
        
        return x
