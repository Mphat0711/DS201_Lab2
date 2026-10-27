"""CNN models used in the lab. Constructors receive all architecture choices explicitly."""
from __future__ import annotations

import torch
from torch import nn


class LeNet5(nn.Module):
    def __init__(self, num_classes: int, image_size: int, in_channels: int = 3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 6, 5, padding=2), nn.ReLU(), nn.AvgPool2d(2, stride=2),
            nn.Conv2d(6, 16, 5), nn.ReLU(), nn.AvgPool2d(2),
        )
        with torch.no_grad():
            flat_dim = self.features(torch.zeros(1, in_channels, image_size, image_size)).flatten(1).shape[1]
        self.classifier = nn.Sequential(nn.Linear(flat_dim, 120), nn.ReLU(), nn.Linear(120, 84), nn.ReLU(), nn.Linear(84, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x).flatten(1))


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = nn.Identity() if stride == 1 and in_channels == out_channels else nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride, bias=False), nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.main(x) + self.shortcut(x))


class ResNet18(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 64, 7, 2, 3, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(3, 2, 1))
        self.layer1 = self._stage(64, 64, 1)
        self.layer2 = self._stage(64, 128, 2)
        self.layer3 = self._stage(128, 256, 2)
        self.layer4 = self._stage(256, 512, 2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    @staticmethod
    def _stage(in_channels: int, out_channels: int, stride: int) -> nn.Sequential:
        return nn.Sequential(BasicBlock(in_channels, out_channels, stride), BasicBlock(out_channels, out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for layer in (self.layer1, self.layer2, self.layer3, self.layer4):
            x = layer(x)
        return self.fc(self.pool(x).flatten(1))


class HuggingFaceResNet50(nn.Module):
    """Lab-required `microsoft/resnet-50` backbone with a new classifier."""
    def __init__(self, pretrained_id: str, num_classes: int, freeze_backbone: bool = False) -> None:
        super().__init__()
        from transformers import ResNetForImageClassification

        base = ResNetForImageClassification.from_pretrained(pretrained_id)
        self.backbone = base.resnet
        self.classifier = nn.Linear(2048, num_classes)
        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(pixel_values=images).pooler_output.flatten(1)
        return self.classifier(features)


def build_resnet50_finetune(pretrained_id: str, num_classes: int, freeze_backbone: bool = False) -> nn.Module:
    """Create the exact Hugging Face pretrained variant specified in the lab."""
    return HuggingFaceResNet50(pretrained_id, num_classes, freeze_backbone)


class InceptionBlock(nn.Module):
    """Four-branch GoogLeNet block: 1x1, 3x3, 5x5 and pool-projection."""
    def __init__(self, in_channels: int, f1: int, f3_reduce: int, f3: int, f5_reduce: int, f5: int, pool_proj: int) -> None:
        super().__init__()
        self.branch1 = nn.Sequential(nn.Conv2d(in_channels, f1, 1), nn.ReLU(inplace=True))
        self.branch2 = nn.Sequential(nn.Conv2d(in_channels, f3_reduce, 1), nn.ReLU(inplace=True), nn.Conv2d(f3_reduce, f3, 3, padding=1), nn.ReLU(inplace=True))
        self.branch3 = nn.Sequential(nn.Conv2d(in_channels, f5_reduce, 1), nn.ReLU(inplace=True), nn.Conv2d(f5_reduce, f5, 5, padding=2), nn.ReLU(inplace=True))
        self.branch4 = nn.Sequential(nn.MaxPool2d(3, stride=1, padding=1), nn.Conv2d(in_channels, pool_proj, 1), nn.ReLU(inplace=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat((self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)), dim=1)


class GoogLeNet(nn.Module):
    """GoogLeNet/Inception-v1 classifier built for the lab, without aux heads."""
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3), nn.ReLU(inplace=True), nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(64, 64, 1), nn.ReLU(inplace=True), nn.Conv2d(64, 192, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.inception3 = nn.Sequential(InceptionBlock(192, 64, 96, 128, 16, 32, 32), InceptionBlock(256, 128, 128, 192, 32, 96, 64), nn.MaxPool2d(3, stride=2, padding=1))
        self.inception4 = nn.Sequential(
            InceptionBlock(480, 192, 96, 208, 16, 48, 64), InceptionBlock(512, 160, 112, 224, 24, 64, 64),
            InceptionBlock(512, 128, 128, 256, 24, 64, 64), InceptionBlock(512, 112, 144, 288, 32, 64, 64),
            InceptionBlock(528, 256, 160, 320, 32, 128, 128), nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.inception5 = nn.Sequential(InceptionBlock(832, 256, 160, 320, 32, 128, 128), InceptionBlock(832, 384, 192, 384, 48, 128, 128))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.inception5(self.inception4(self.inception3(self.stem(x))))).flatten(1))


def build_googlenet(num_classes: int) -> nn.Module:
    return GoogLeNet(num_classes)


def build_model(name: str, num_classes: int, image_size: int, freeze_backbone: bool = False, pretrained_id: str | None = None) -> nn.Module:
    builders = {
        "lenet": lambda: LeNet5(num_classes=num_classes, image_size=image_size),
        "googlenet": lambda: build_googlenet(num_classes=num_classes),
        "resnet18": lambda: ResNet18(num_classes=num_classes),
        "resnet50": lambda: build_resnet50_finetune(pretrained_id=pretrained_id or "", num_classes=num_classes, freeze_backbone=freeze_backbone),
    }
    if name not in builders:
        raise ValueError(f"unsupported model: {name}")
    if name == "resnet50" and not pretrained_id:
        raise ValueError("resnet50 requires model.pretrained_id in config")
    return builders[name]()
