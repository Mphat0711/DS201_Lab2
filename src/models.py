"""CNN models used in the lab. Constructors receive all architecture choices explicitly."""
from __future__ import annotations

import torch
from torch import nn


class LeNet5(nn.Module):
    def __init__(self, num_classes: int, image_size: int, in_channels: int = 3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 6, 5), nn.ReLU(), nn.AvgPool2d(2),
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


def build_resnet50_finetune(num_classes: int, freeze_backbone: bool = False) -> nn.Module:
    """Create pretrained torchvision ResNet-50; downloads weights only if absent."""
    from torchvision.models import ResNet50_Weights, resnet50

    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.fc.parameters():
            parameter.requires_grad = True
    return model


def build_googlenet(num_classes: int) -> nn.Module:
    from torchvision.models import googlenet

    model = googlenet(weights=None, aux_logits=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_model(name: str, num_classes: int, image_size: int, freeze_backbone: bool = False) -> nn.Module:
    builders = {
        "lenet": lambda: LeNet5(num_classes=num_classes, image_size=image_size),
        "googlenet": lambda: build_googlenet(num_classes=num_classes),
        "resnet18": lambda: ResNet18(num_classes=num_classes),
        "resnet50": lambda: build_resnet50_finetune(num_classes=num_classes, freeze_backbone=freeze_backbone),
    }
    if name not in builders:
        raise ValueError(f"unsupported model: {name}")
    return builders[name]()
