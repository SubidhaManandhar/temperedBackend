import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights


class ChannelAttention(nn.Module):
    def __init__(self, in_ch, reduction=16):
        super().__init__()
        hidden = max(in_ch // reduction, 8)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.mx = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, in_ch, 1, bias=False),
        )

    def forward(self, x):
        attn = self.fc(self.avg(x)) + self.fc(self.mx(x))
        return torch.sigmoid(attn) * x


class SpatialAttention(nn.Module):
    def __init__(self, k=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, k, padding=k // 2, bias=False)

    def forward(self, x):
        avg = x.mean(1, keepdim=True)
        mx, _ = x.max(1, keepdim=True)
        attn = self.conv(torch.cat([avg, mx], dim=1))
        return torch.sigmoid(attn) * x


class CBAM(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.ca = ChannelAttention(in_ch)
        self.sa = SpatialAttention()

    def forward(self, x):
        return self.sa(self.ca(x))


def build_resnet50_9ch(num_classes=5, pretrained=False):
    weights = ResNet50_Weights.DEFAULT if pretrained else None
    model = models.resnet50(weights=weights)

    old_conv = model.conv1
    model.conv1 = nn.Conv2d(
        9,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False,
    )

    with torch.no_grad():
        model.conv1.weight.zero_()
        model.conv1.weight[:, :3] = old_conv.weight
        mean_w = old_conv.weight.mean(dim=1, keepdim=True)
        model.conv1.weight[:, 3:6] = mean_w.repeat(1, 3, 1, 1)
        model.conv1.weight[:, 6:9] = mean_w.repeat(1, 3, 1, 1)

    model.layer3 = nn.Sequential(model.layer3, CBAM(1024))
    model.layer4 = nn.Sequential(model.layer4, CBAM(2048))

    # matches fc.1.weight / fc.1.bias
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, num_classes)
    )

    return model