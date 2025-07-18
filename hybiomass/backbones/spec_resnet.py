# Source: https://github.com/AABNassim/spectral_earth

import torch.nn as nn
import timm
from .spectral_adapter import SpectralAdapter


class CustomBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride, downsample=None):
        super(CustomBottleneck, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
    
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = self.relu(out)
        
        return out


class CustomResNet50(nn.Module):
    def __init__(self, num_classes=1000, replace_stride_with_dilation=[False, False, False, False], return_features=False):
        super(CustomResNet50, self).__init__()

        base_model = timm.create_model('resnet50', pretrained=False)
        self.return_features = return_features
        self.num_features = base_model.num_features

        downsample = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(256),
        )

        self.layer1 = nn.Sequential(
            CustomBottleneck(128, 256, stride=2, downsample=downsample),
            base_model.layer1[1],
            base_model.layer1[2]
        )

        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4

        layers = [self.layer1, self.layer2, self.layer3, self.layer4]

        dilation_value = 1  # Initial dilation value

        # Apply dilation if specified
        for layer, replace_dilation in zip(layers, replace_stride_with_dilation):
            if replace_dilation:
                for block in layer:
                    if block.downsample is not None:
                        block.downsample[0].stride = (1, 1)
                    block.conv2.stride = (1, 1)
                    block.conv2.dilation = (dilation_value, dilation_value)
                    block.conv2.padding = (dilation_value, dilation_value)
                dilation_value *= 2  # Double the dilation for the next layer

        self.global_pool = base_model.global_pool
        if num_classes == 0:
            self.fc = nn.Identity()
        else:
            self.fc = nn.Linear(base_model.fc.in_features, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.return_features:
            return x
        
        x = self.global_pool(x).flatten(1)
        x = self.fc(x)
        return x


class SpecResNet50(nn.Module):
    def __init__(self, num_classes, replace_stride_with_dilation=[False, False, False, False], return_features=False):
        super(SpecResNet50, self).__init__()
        
        self.spectral_adapter = SpectralAdapter()
        
        self.resnet = CustomResNet50(num_classes=num_classes, replace_stride_with_dilation=replace_stride_with_dilation, return_features=return_features)
        self.num_features = self.resnet.num_features

    def forward(self, x):
        x = self.spectral_adapter(x)
        return self.resnet(x)
  
    def get_classifier(self):
        return self.resnet.fc
