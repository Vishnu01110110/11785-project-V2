import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention


class VAEAttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)

    def forward(self, input_tensor):
        # input_tensor: (Batch_Size, Features, Height, Width)

        residual_connection = input_tensor

        # Normalization step
        normalized_tensor = self.groupnorm(input_tensor)

        batch_size, feature_channels, height, width = normalized_tensor.shape

        # Reshape for attention
        reshaped_tensor = normalized_tensor.view((batch_size, feature_channels, height * width))

        # Transpose for attention operation
        transposed_tensor = reshaped_tensor.transpose(-1, -2)

        # Self-attention operation
        attended_tensor = self.attention(transposed_tensor)

        # Reverse transpose
        back_transposed_tensor = attended_tensor.transpose(-1, -2)

        # Reverse reshape
        final_tensor = back_transposed_tensor.view((batch_size, feature_channels, height, width))

        # Add the residual connection
        output_tensor = final_tensor + residual_connection

        return output_tensor


class VAEResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # Conditional residual layer
        self.residual_layer = nn.Conv2d(in_channels, out_channels,
                                        kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, input_tensor):
        # Save the original tensor for the residual connection
        residual = input_tensor

        # Apply normalization and activation
        normed_tensor = self.groupnorm_1(input_tensor)
        activated_tensor = F.silu(normed_tensor)

        # First convolution operation
        convoluted_tensor = self.conv_1(activated_tensor)

        # Second normalization and activation
        second_normed_tensor = self.groupnorm_2(convoluted_tensor)
        second_activated_tensor = F.silu(second_normed_tensor)

        # Second convolution
        final_convoluted_tensor = self.conv_2(second_activated_tensor)

        # Combine the residual
        combined_tensor = final_convoluted_tensor + self.residual_layer(residual)

        return combined_tensor


class VAEDecoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(4, 4, kernel_size=1, padding=0),
            nn.Conv2d(4, 512, kernel_size=3, padding=1),
            VAEResidualBlock(512, 512),
            VAEAttentionBlock(512),
            VAEResidualBlock(512, 512),
            VAEResidualBlock(512, 512),
            VAEResidualBlock(512, 512),
            VAEResidualBlock(512, 512),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            VAEResidualBlock(512, 512),
            VAEResidualBlock(512, 512),
            VAEResidualBlock(512, 512),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            VAEResidualBlock(512, 256),
            VAEResidualBlock(256, 256),
            VAEResidualBlock(256, 256),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            VAEResidualBlock(256, 128),
            VAEResidualBlock(128, 128),
            VAEResidualBlock(128, 128),
            nn.GroupNorm(32, 128),
            nn.SiLU(),
            nn.Conv2d(128, 3, kernel_size=3, padding=1),
        )

    def forward(self, input_tensor):
        # Scaling adjustment
        scaled_tensor = input_tensor / 0.18215

        for module in self:
            scaled_tensor = module(scaled_tensor)

        return scaled_tensor
