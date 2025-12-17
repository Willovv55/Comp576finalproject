import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(Conv => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, channels=3):
        super().__init__()
        self.inc = DoubleConv(channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))

        self.bottleneck = DoubleConv(512, 1024)

        self.up3 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec3 = DoubleConv(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec2 = DoubleConv(512, 256)
        self.up1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec1 = DoubleConv(256, 128)
        self.final = nn.Conv2d(128, channels, 1)

    def forward(self, x, t):
        # timestep embedding
        t = t.view(-1, 1, 1, 1)
        x = x + t

        x1 = self.inc(x)         # 512x512
        x2 = self.down1(x1)      # 256x256
        x3 = self.down2(x2)      # 128x128
        x4 = self.down3(x3)      # 64x64
        x5 = self.bottleneck(x4) # 64x64

        x = self.up3(x5)         # 128x128
        x4 = F.interpolate(x4, size=x.shape[2:], mode='nearest') 
        x = torch.cat([x, x4], dim=1)
        x = self.dec3(x)

        x = self.up2(x)          # 256x256
        x3 = F.interpolate(x3, size=x.shape[2:], mode='nearest') 
        x = torch.cat([x, x3], dim=1)
        x = self.dec2(x)

        x = self.up1(x)          # 512x512
        x2 = F.interpolate(x2, size=x.shape[2:], mode='nearest') 
        x = torch.cat([x, x2], dim=1)
        x = self.dec1(x)

        out = self.final(x)
        return out
