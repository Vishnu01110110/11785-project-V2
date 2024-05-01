import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention, CrossAttention

class TimeEmbedding(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        # Initialize first linear transformation with increased dimensionality
        self.linear_1 = nn.Linear(n_embd, 4 * n_embd)
        # Initialize second linear transformation to maintain the expanded dimensionality
        self.linear_2 = nn.Linear(4 * n_embd, 4 * n_embd)

    def forward(self, x):
        # Apply first linear transformation
        expanded_x = self.linear_1(x)
        # Apply SiLU activation function
        activated_x = F.silu(expanded_x)
        # Apply second linear transformation
        final_x = self.linear_2(activated_x)
        # Return the transformed input
        return final_x

class UNET_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_time=1280):
        super().__init__()
        # Group normalization for input features
        self.groupnorm_feature = nn.GroupNorm(32, in_channels)
        # Convolution layer to transform feature maps
        self.conv_feature = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        # Linear transformation for time embedding to match output channels
        self.linear_time = nn.Linear(n_time, out_channels)

        # Group normalization after merging feature and time
        self.groupnorm_merged = nn.GroupNorm(32, out_channels)
        # Convolution to further process the merged features
        self.conv_merged = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # Conditional assignment of the residual layer to either pass through or transform dimensions
        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, feature, time):
        # Preserve original features for residual connection
        residue = feature

        # Normalize, activate, and convolve the input features
        normalized_feature = self.groupnorm_feature(feature)
        activated_feature = F.silu(normalized_feature)
        convolved_feature = self.conv_feature(activated_feature)

        # Process time embedding
        activated_time = F.silu(time)
        time_transformed = self.linear_time(activated_time)
        # Add time dimension for broadcasting
        time_broadcast = time_transformed.unsqueeze(-1).unsqueeze(-1)

        # Merge feature maps with time and process further
        merged_features = convolved_feature + time_broadcast
        normalized_merged = self.groupnorm_merged(merged_features)
        activated_merged = F.silu(normalized_merged)
        convolved_merged = self.conv_merged(activated_merged)

        # Combine with the residual path and return the result
        final_output = convolved_merged + self.residual_layer(residue)
        return final_output


class UNET_AttentionBlock(nn.Module):
    def __init__(self, n_head: int, n_embd: int, d_context=768):
        super().__init__()
        channels = n_head * n_embd

        self.groupnorm = nn.GroupNorm(32, channels, eps=1e-6)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

        self.layernorm_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(n_head, channels, in_proj_bias=False)
        self.layernorm_2 = nn.LayerNorm(channels)
        self.attention_2 = CrossAttention(n_head, channels, d_context, in_proj_bias=False)
        self.layernorm_3 = nn.LayerNorm(channels)
        self.linear_geglu_1  = nn.Linear(channels, 4 * channels * 2)
        self.linear_geglu_2 = nn.Linear(4 * channels, channels)

        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

    def forward(self, i, context):


        residual = i

        i = self.groupnorm(i)
        i = self.conv_input(i)

        n, c, h, w = i.shape

        i = i.view((n, c, h * w))

        i = i.transpose(-1, -2)

        residue_short = i

        i = self.layernorm_1(i)

        i = self.attention_1(i)

        i += residue_short

        residue_short = i

        i = self.layernorm_2(i)

        i = self.attention_2(i, context)

        i += residue_short

        residue_short = i

        i = self.layernorm_3(i)

        i, gate = self.linear_geglu_1(i).chunk(2, dim=-1)

        i = i * F.gelu(gate)

        i = self.linear_geglu_2(i)

        i += residue_short

        i = i.transpose(-1, -2)

        i = i.view((n, c, h, w))

        return self.conv_output(i) + residual

class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)

class SwitchSequential(nn.Sequential):
    def forward(self, x, context, time):
        for layer in self:
            if isinstance(layer, UNET_AttentionBlock):
                x = layer(x, context)
            elif isinstance(layer, UNET_ResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)
        return x

class UNET(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoders = nn.ModuleList([
            SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),

            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),

            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),

            SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)),

            SwitchSequential(UNET_ResidualBlock(320, 640), UNET_AttentionBlock(8, 80)),

            SwitchSequential(UNET_ResidualBlock(640, 640), UNET_AttentionBlock(8, 80)),

            SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)),

            SwitchSequential(UNET_ResidualBlock(640, 1280), UNET_AttentionBlock(8, 160)),

            SwitchSequential(UNET_ResidualBlock(1280, 1280), UNET_AttentionBlock(8, 160)),

            SwitchSequential(nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)),

            SwitchSequential(UNET_ResidualBlock(1280, 1280)),

            SwitchSequential(UNET_ResidualBlock(1280, 1280)),
        ])

        self.bottleneck = SwitchSequential(
            UNET_ResidualBlock(1280, 1280),


            UNET_AttentionBlock(8, 160),

            UNET_ResidualBlock(1280, 1280),
        )

        self.decoders = nn.ModuleList([
            SwitchSequential(UNET_ResidualBlock(2560, 1280)),

            SwitchSequential(UNET_ResidualBlock(2560, 1280)),

            SwitchSequential(UNET_ResidualBlock(2560, 1280), Upsample(1280)),

            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),

            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),

            SwitchSequential(UNET_ResidualBlock(1920, 1280), UNET_AttentionBlock(8, 160), Upsample(1280)),

            SwitchSequential(UNET_ResidualBlock(1920, 640), UNET_AttentionBlock(8, 80)),

            SwitchSequential(UNET_ResidualBlock(1280, 640), UNET_AttentionBlock(8, 80)),

            SwitchSequential(UNET_ResidualBlock(960, 640), UNET_AttentionBlock(8, 80), Upsample(640)),

            SwitchSequential(UNET_ResidualBlock(960, 320), UNET_AttentionBlock(8, 40)),

            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)),

            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)),
        ])

    def forward(self, x, context, time):


        skip_connections = []
        for layers in self.encoders:
            x = layers(x, context, time)
            skip_connections.append(x)

        x = self.bottleneck(x, context, time)

        for layers in self.decoders:
            x = torch.cat((x, skip_connections.pop()), dim=1)
            x = layers(x, context, time)

        return x


class UNET_OutputLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNET_OutputLayer, self).__init__()
        self.groupnorm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, input_tensor):
        normalized_output = self.groupnorm(input_tensor)
        activated_output = F.silu(normalized_output)
        convoluted_output = self.conv(activated_output)
        return convoluted_output

class Diffusion(nn.Module):
    def __init__(self):
        super(Diffusion, self).__init__()
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNET()
        self.final = UNET_OutputLayer(320, 4)

    def forward(self, latent_representation, additional_context, temporal_data):
        embedded_time = self.time_embedding(temporal_data)
        unet_output = self.unet(latent_representation, additional_context, embedded_time)
        final_output = self.final(unet_output)
        return final_output
