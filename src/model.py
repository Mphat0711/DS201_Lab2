
from __future__ import annotations

import torch
from torch import nn


class LeNet5(nn.Module):
    """LeNet-5 adapted exactly as specified for 64x64 RGB VinaFood21 images."""
    def __init__(self, num_classes: int, image_size: int, in_channels: int = 3, pool: str = "avg") -> None:
        super().__init__()
        if pool not in {"avg", "max"}:
            raise ValueError("pool must be 'avg' or 'max'")
        Pool = nn.AvgPool2d if pool == "avg" else nn.MaxPool2d
        self.pool_name = pool
        # PDF requirement: valid 5x5 convolutions (padding=0).
        self.c1 = nn.Conv2d(in_channels, 6, kernel_size=5)
        self.s2 = Pool(2, 2)
        self.c3 = nn.Conv2d(6, 16, kernel_size=5)
        self.s4 = Pool(2, 2)
        with torch.no_grad():
            sample = self.s4(self.c3(self.s2(self.c1(torch.zeros(1, in_channels, image_size, image_size)))))
        self.n_flat = int(sample.flatten(1).shape[1])
        self.f5 = nn.Linear(self.n_flat, 120)
        self.f6 = nn.Linear(120, 84)
        self.out = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.s2(torch.relu(self.c1(x)))
        x = self.s4(torch.relu(self.c3(x)))
        x = x.flatten(1)
        x = torch.relu(self.f5(x))
        x = torch.relu(self.f6(x))
        return self.out(x)


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
    """PDF B.4 block: four branches retain H/W then concatenate on channels."""
    def __init__(self, in_channels: int, f1: int, f3_reduce: int, f3: int, f5_reduce: int, f5: int, pool_proj: int) -> None:
        super().__init__()
        self.branch1 = nn.Sequential(nn.Conv2d(in_channels, f1, 1), nn.ReLU())
        self.branch2 = nn.Sequential(nn.Conv2d(in_channels, f3_reduce, 1), nn.ReLU(), nn.Conv2d(f3_reduce, f3, 3, padding=1), nn.ReLU())
        self.branch3 = nn.Sequential(nn.Conv2d(in_channels, f5_reduce, 1), nn.ReLU(), nn.Conv2d(f5_reduce, f5, 5, padding=2), nn.ReLU())
        self.branch4 = nn.Sequential(nn.MaxPool2d(3, stride=1, padding=1), nn.Conv2d(in_channels, pool_proj, 1), nn.ReLU())
        self.out_channels = f1 + f3 + f5 + pool_proj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat((self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)), dim=1)


class GoogLeNet(nn.Module):
    """Reduced GoogLeNet required by the assignment: four Inception blocks."""
    def __init__(self, num_classes: int, dropout: float = 0.4) -> None:
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 1, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2))
        # The first two blocks follow the parameter sets printed in the PDF.
        self.block1 = InceptionBlock(1, 16, 16, 24, 4, 8, 8)       # 56 channels, 4,376 params
        self.block2 = InceptionBlock(56, 32, 32, 48, 8, 16, 16)     # 112 channels
        self.pool1 = nn.MaxPool2d(2, 2)
        self.block3 = InceptionBlock(112, 48, 32, 64, 8, 16, 16)   # 144 channels
        self.block4 = InceptionBlock(144, 64, 48, 80, 12, 24, 24)  # 192 channels
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(192, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.block2(self.block1(x))
        x = self.pool1(x)
        x = self.block4(self.block3(x))
        x = self.pool(x).flatten(1)
        return self.classifier(self.dropout(x))


def build_googlenet(num_classes: int) -> nn.Module:
    return GoogLeNet(num_classes)


def n_params(module: nn.Module, trainable_only: bool = False) -> int:
    parameters = module.parameters()
    if trainable_only:
        parameters = (parameter for parameter in parameters if parameter.requires_grad)
    return sum(parameter.numel() for parameter in parameters)


def layer_summary(model: nn.Module, image_size: int) -> list[dict[str, int | str]]:
    """Small PDF-required table for LeNet layers and Inception output channels."""
    if isinstance(model, LeNet5):
        device = next(model.parameters()).device
        x = torch.zeros(1, 3, image_size, image_size, device=device)
        rows: list[dict[str, int | str]] = []
        with torch.no_grad():
            for name, layer in (("C1", model.c1), ("S2", model.s2), ("C3", model.c3), ("S4", model.s4)):
                x = layer(x)
                rows.append({"layer": name, "output": "x".join(map(str, x.shape[1:])), "parameters": n_params(layer)})
        rows.append({"layer": "flatten", "output": str(x.flatten(1).shape[1]), "parameters": 0})
        for name, layer in (("F5", model.f5), ("F6", model.f6), ("OUT", model.out)):
            rows.append({"layer": name, "output": str(layer.out_features), "parameters": n_params(layer)})
        return rows
    if isinstance(model, GoogLeNet):
        return [{"layer": name, "output": str(block.out_channels), "parameters": n_params(block)} for name, block in (("Inception-1", model.block1), ("Inception-2", model.block2), ("Inception-3", model.block3), ("Inception-4", model.block4))]
    return []


def padding_policy(model: nn.Module) -> str:
    if isinstance(model, LeNet5):
        return "C1 5x5 P=0; C3 5x5 P=0 (valid convolution)"
    if isinstance(model, GoogLeNet):
        return "Inception 3x3 P=1; 5x5 P=2; pool branch P=1"
    if isinstance(model, ResNet18):
        return "stem 7x7 P=3; BasicBlock 3x3 P=1; projection 1x1 P=0"
    if isinstance(model, HuggingFaceResNet50):
        return "padding follows pretrained microsoft/resnet-50 backbone"
    return "not available"


def build_model(name: str, num_classes: int, image_size: int, freeze_backbone: bool = False, pretrained_id: str | None = None, pool: str = "avg") -> nn.Module:
    builders = {
        "lenet": lambda: LeNet5(num_classes=num_classes, image_size=image_size, pool=pool),
        "googlenet": lambda: build_googlenet(num_classes=num_classes),
        "resnet18": lambda: ResNet18(num_classes=num_classes),
        "resnet50": lambda: build_resnet50_finetune(pretrained_id=pretrained_id or "", num_classes=num_classes, freeze_backbone=freeze_backbone),
    }
    if name not in builders:
        raise ValueError(f"unsupported model: {name}")
    if name == "resnet50" and not pretrained_id:
        raise ValueError("resnet50 requires model.pretrained_id in config")
    return builders[name]()
