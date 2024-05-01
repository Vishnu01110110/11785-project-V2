import torch
from torch import nn
from torch.nn import functional as F
from decoder import VAEAttentionBlock, VAEResidualBlock


class VAEImageEncoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            # Input channels to 128 feature maps
            nn.Conv2d(3, 128, kernel_size=3, padding=1),

            # Processing with residual blocks without changing dimensions
            VAEResidualBlock(128, 128),
            VAEResidualBlock(128, 128),

            # Downsample the feature map by factor of 2
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),

            # Increase channels from 128 to 256, keeping size constant
            VAEResidualBlock(128, 256),
            VAEResidualBlock(256, 256),

            # Another downsampling step
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),

            # Expanding to 512 channels
            VAEResidualBlock(256, 512),
            VAEResidualBlock(512, 512),

            # Final downsampling reduces spatial dimensions further
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),

            # Stable features are processed in multiple residual and attention blocks
            VAEResidualBlock(512, 512),
            VAEResidualBlock(512, 512),
            VAEResidualBlock(512, 512),
            VAEAttentionBlock(512),
            VAEResidualBlock(512, 512),

            # Group normalization and activation
            nn.GroupNorm(32, 512),
            nn.SiLU(),

            # Compression to fewer channels and then to desired latent space dimensionality
            nn.Conv2d(512, 8, kernel_size=3, padding=1),
            nn.Conv2d(8, 8, kernel_size=1, padding=0),
        )

    def forward(self, input_tensor, random_noise):
        for layer in self:
            if getattr(layer, 'stride', None) == (2, 2):
                input_tensor = F.pad(input_tensor, (0, 1, 0, 1))

            input_tensor = layer(input_tensor)

        encoded_mean, encoded_log_var = torch.chunk(input_tensor, 2, dim=1)
        encoded_log_var = torch.clamp(encoded_log_var, -30, 20)
        encoded_variance = encoded_log_var.exp()
        encoded_std_dev = encoded_variance.sqrt()

        sampled_latent = encoded_mean + encoded_std_dev * random_noise
        sampled_latent *= 0.18215

        return sampled_latent

