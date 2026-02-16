
"""
Contains PyTorch model code to instantiate a MobileNet model.
"""
import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights

# load a pretrained model
weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights=weights)
model.eval()

x = [torch.rand(3, 320, 320), torch.rand(3, 500, 500)]
with torch.no_grad():
    prediction = model(x)

class DepthwiseSeparableConv(nn.Module):
    """
    This is class for creating the Depth Wise Separable Convoltions for Mobile Net Architecture.

    Args:
      in_ch: this is the number of input for Conv2d layer.
      out_ch: this is the number of output for Conv2d layer.
      stride: this is the value for strides which by default is 1.
    """
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()

        self.depthwise = nn.Conv2d(
            in_ch, in_ch,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_ch,
            bias=False
        )
        self.pointwise = nn.Conv2d(
            in_ch, out_ch,
            kernel_size=1,
            bias=False
        )

        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)

# actual class for MobileNet CNN
class SignMobileNet(nn.Module):
    """
    This class creates the Actual MobileNet Architecture from scratch.
    Args:
      num_classes: this is the number of classes the model is going to be trained on.
    """
    def __init__(self, num_classes=35):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            DepthwiseSeparableConv(32, 64),
            DepthwiseSeparableConv(64, 128, stride=2),
            DepthwiseSeparableConv(128, 128),

            DepthwiseSeparableConv(128, 256, stride=2),
            DepthwiseSeparableConv(256, 256),

            DepthwiseSeparableConv(256, 512, stride=2),
            *[DepthwiseSeparableConv(512, 512) for _ in range(4)],

            DepthwiseSeparableConv(512, 1024, stride=2),
            DepthwiseSeparableConv(1024, 1024),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
