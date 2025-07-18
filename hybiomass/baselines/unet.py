# U-Net Implementation for Pytorch: https://github.com/milesial/Pytorch-UNet/tree/master

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """2 x (Convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)
    

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
    

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_filters=32, bilinear=False):
        """Initialize a 'Unet' module.

        :param n_channels: The number of spectral bands of the input image.
        :param n_filters: The number of filters 
        :param bilinear: 
        """
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_filters= n_filters
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, n_filters))
        self.down1 = (Down(n_filters, n_filters*2))
        self.down2 = (Down(n_filters*2, n_filters*4))
        self.down3 = (Down(n_filters*4, n_filters*8))
        factor = 2 if bilinear else 1
        self.down4 = (Down(n_filters*8, n_filters*16 // factor))
        self.up1 = (Up(n_filters*16, n_filters*8 // factor, bilinear))
        self.up2 = (Up(n_filters*8, n_filters*4 // factor, bilinear))
        self.up3 = (Up(n_filters*4, n_filters*2 // factor, bilinear))
        self.up4 = (Up(n_filters*2, n_filters, bilinear))
        self.outc = (OutConv(n_filters, 1))

    def forward(self, x):
        """Perform a single forward pass through the model.
        
        :param x: The input tensor.
        :return: A tensor of predictions.
        """

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        outputs = self.outc(x)
        return outputs
    
if __name__ == "__main__":
    model = UNet(n_channels=202, n_filters=32)
    print(model)
    print("Number of parameters:", sum(p.numel() for p in model.parameters()))
    