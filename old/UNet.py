import torch
from torch import nn
from torch.nn import functional as F


class UNetBlock(nn.Module):

    def __init__(self, in_channels, out_channels) -> None:
        super(UNetBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)
    
class EncoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels) -> None:
        super(EncoderBlock).__init__()

        self.encode = nn.Sequential(
            nn.MaxPool2d(2),
            UNetBlock(in_channels, out_channels)
        )

    def forward(self, x):
        return self.encode(x)
    
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels) -> None:
        super(DecoderBlock, self).__init__()

        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(middle_channels, middle_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=2, stride=2)
        )

    def forward(self, x, skip):
        x = self.decode(x)
        x = torch.cat((x, skip), dim=1)  # Concatenate the skip connection
        return x

class UNet(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.encoder1 = UNetBlock(3, 64)
        self.encoder2 = EncoderBlock(64, 128)
        self.encoder3 = EncoderBlock(128, 256)
        self.encoder4 = EncoderBlock(256, 512)

        self.bottleneck = UNetBlock(512, 1024)

        self.decoder1 = DecoderBlock(1024, 512, 256)
        self.decoder2 = DecoderBlock(256 + 512, 256, 128)
        self.decoder3 = DecoderBlock(128 + 256, 128, 64)
        self.decoder4 = DecoderBlock(64 + 128, 64, 64)

        self.final_conv = nn.Conv2d(64 + 64, 3, kernel_size=1)

    def forward(self, x):
        # Encoder path
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)

        bottleneck = self.bottleneck(enc4)

        # Decoder path
        dec1 = self.decoder1(bottleneck, enc4)
        dec2 = self.decoder2(dec1, enc3)
        dec3 = self.decoder3(dec2, enc2)
        dec4 = self.decoder4(dec3, enc1)

        output = self.final_conv(dec4)
        return output












