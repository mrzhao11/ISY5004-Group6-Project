from __future__ import annotations

try:
    import torch
    from torch import nn
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError("PyTorch is required for stage1.models") from exc

from .config import ModelConfig


class FrameCNN(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(128, embedding_dim)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        features = self.net(frames).flatten(1)
        return self.proj(features)


class CNNLSTMBehaviorModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.frame_encoder = FrameCNN(config.hidden_dim)
        self.temporal = nn.LSTM(
            input_size=config.hidden_dim,
            hidden_size=config.hidden_dim,
            batch_first=True,
        )
        self.embedding_head = nn.Linear(config.hidden_dim, config.embedding_dim)
        self.action_classifier = nn.Linear(config.embedding_dim, config.num_action_classes)
        self.look_classifier = nn.Linear(config.embedding_dim, config.num_look_classes)

    def forward(self, clip: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, time_steps, channels, height, width = clip.shape
        frames = clip.view(batch * time_steps, channels, height, width)
        frame_features = self.frame_encoder(frames).view(batch, time_steps, -1)
        _, (hidden, _) = self.temporal(frame_features)
        embedding = self.embedding_head(hidden[-1])
        action_logits = self.action_classifier(embedding)
        look_logits = self.look_classifier(embedding)
        return action_logits, look_logits, embedding


class VideoSwinStyleBehaviorModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.frame_encoder = FrameCNN(config.hidden_dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, config.hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            batch_first=True,
        )
        self.temporal = nn.TransformerEncoder(layer, num_layers=config.num_layers)
        self.embedding_head = nn.Linear(config.hidden_dim, config.embedding_dim)
        self.action_classifier = nn.Linear(config.embedding_dim, config.num_action_classes)
        self.look_classifier = nn.Linear(config.embedding_dim, config.num_look_classes)

    def forward(self, clip: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, time_steps, channels, height, width = clip.shape
        frames = clip.view(batch * time_steps, channels, height, width)
        frame_features = self.frame_encoder(frames).view(batch, time_steps, -1)
        cls = self.cls.expand(batch, -1, -1)
        encoded = self.temporal(torch.cat([cls, frame_features], dim=1))
        pooled = encoded[:, 0]
        embedding = self.embedding_head(pooled)
        action_logits = self.action_classifier(embedding)
        look_logits = self.look_classifier(embedding)
        return action_logits, look_logits, embedding


class FrameTransformerBehaviorModel(nn.Module):
    def __init__(self, config: ModelConfig, backbone_name: str) -> None:
        super().__init__()
        self.frame_encoder = _build_image_encoder(config, backbone_name)
        self.cls = nn.Parameter(torch.zeros(1, 1, config.embedding_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=config.embedding_dim,
            nhead=config.num_heads,
            dim_feedforward=config.embedding_dim * 4,
            dropout=config.dropout,
            batch_first=True,
        )
        self.temporal = nn.TransformerEncoder(layer, num_layers=config.num_layers)
        self.dropout = nn.Dropout(config.dropout)
        self.action_classifier = nn.Linear(config.embedding_dim, config.num_action_classes)
        self.look_classifier = nn.Linear(config.embedding_dim, config.num_look_classes)

    def forward(self, clip: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, time_steps, channels, height, width = clip.shape
        frames = clip.view(batch * time_steps, channels, height, width)
        frame_features = self.frame_encoder(frames).view(batch, time_steps, -1)
        cls = self.cls.expand(batch, -1, -1)
        encoded = self.temporal(torch.cat([cls, frame_features], dim=1))
        embedding = self.dropout(encoded[:, 0])
        action_logits = self.action_classifier(embedding)
        look_logits = self.look_classifier(embedding)
        return action_logits, look_logits, embedding


class TorchvisionVideoSwinBehaviorModel(nn.Module):
    def __init__(self, config: ModelConfig, variant: str = "t") -> None:
        super().__init__()
        try:
            from torchvision.models.video import (
                Swin3D_B_Weights,
                Swin3D_S_Weights,
                Swin3D_T_Weights,
                swin3d_b,
                swin3d_s,
                swin3d_t,
            )
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise ModuleNotFoundError("torchvision is required for torchvision Video Swin models") from exc

        if variant == "t":
            weights = Swin3D_T_Weights.DEFAULT if config.pretrained else None
            backbone = swin3d_t(weights=weights)
        elif variant == "s":
            weights = Swin3D_S_Weights.DEFAULT if config.pretrained else None
            backbone = swin3d_s(weights=weights)
        elif variant == "b":
            weights = Swin3D_B_Weights.DEFAULT if config.pretrained else None
            backbone = swin3d_b(weights=weights)
        else:
            raise ValueError(f"Unsupported Video Swin variant: {variant}")
        feature_dim = backbone.head.in_features
        backbone.head = nn.Identity()
        if config.freeze_backbone:
            for parameter in backbone.parameters():
                parameter.requires_grad = False
        self.backbone = backbone
        self.embedding_head = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(feature_dim, config.embedding_dim),
        )
        self.action_classifier = nn.Linear(config.embedding_dim, config.num_action_classes)
        self.look_classifier = nn.Linear(config.embedding_dim, config.num_look_classes)

    def forward(self, clip: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # torchvision video models expect B,C,T,H,W; Stage1Dataset emits B,T,C,H,W.
        video = clip.permute(0, 2, 1, 3, 4)
        features = self.backbone(video)
        embedding = self.embedding_head(features)
        action_logits = self.action_classifier(embedding)
        look_logits = self.look_classifier(embedding)
        return action_logits, look_logits, embedding


class TorchvisionEncoder(nn.Module):
    def __init__(
        self,
        name: str,
        embedding_dim: int,
        pretrained: bool,
        freeze_backbone: bool,
        freeze_backbone_until: str,
        dropout: float,
    ) -> None:
        super().__init__()
        try:
            from torchvision import models
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise ModuleNotFoundError(
                f"Architecture '{name}' requires torchvision. Install torchvision or use 'paper_simple_cnn'."
            ) from exc

        if name == "alexnet":
            weights = models.AlexNet_Weights.DEFAULT if pretrained else None
            backbone = models.alexnet(weights=weights)
            feature_dim = backbone.classifier[-1].in_features
            backbone.classifier[-1] = nn.Identity()
        elif name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            backbone = models.resnet18(weights=weights)
            feature_dim = backbone.fc.in_features
            backbone.fc = nn.Identity()
        elif name == "resnet34":
            weights = models.ResNet34_Weights.DEFAULT if pretrained else None
            backbone = models.resnet34(weights=weights)
            feature_dim = backbone.fc.in_features
            backbone.fc = nn.Identity()
        elif name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            backbone = models.resnet50(weights=weights)
            feature_dim = backbone.fc.in_features
            backbone.fc = nn.Identity()
        elif name == "mobilenet_v3_small":
            weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            backbone = models.mobilenet_v3_small(weights=weights)
            feature_dim = backbone.classifier[-1].in_features
            backbone.classifier[-1] = nn.Identity()
        elif name == "vit_b_16":
            weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
            backbone = models.vit_b_16(weights=weights)
            feature_dim = backbone.heads.head.in_features
            backbone.heads = nn.Identity()
        elif name == "vit_b_32":
            weights = models.ViT_B_32_Weights.DEFAULT if pretrained else None
            backbone = models.vit_b_32(weights=weights)
            feature_dim = backbone.heads.head.in_features
            backbone.heads = nn.Identity()
        elif name == "swin_t":
            weights = models.Swin_T_Weights.DEFAULT if pretrained else None
            backbone = models.swin_t(weights=weights)
            feature_dim = backbone.head.in_features
            backbone.head = nn.Identity()
        elif name == "swin_s":
            weights = models.Swin_S_Weights.DEFAULT if pretrained else None
            backbone = models.swin_s(weights=weights)
            feature_dim = backbone.head.in_features
            backbone.head = nn.Identity()
        elif name == "swin_b":
            weights = models.Swin_B_Weights.DEFAULT if pretrained else None
            backbone = models.swin_b(weights=weights)
            feature_dim = backbone.head.in_features
            backbone.head = nn.Identity()
        else:
            raise ValueError(f"Unsupported torchvision backbone: {name}")

        if freeze_backbone:
            for parameter in backbone.parameters():
                parameter.requires_grad = False
        elif freeze_backbone_until != "none":
            self._freeze_until(backbone, name, freeze_backbone_until)

        self.backbone = backbone
        self.proj = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, embedding_dim),
        )

    def _freeze_until(self, backbone: nn.Module, name: str, freeze_backbone_until: str) -> None:
        if name in {"resnet18", "resnet34", "resnet50"}:
            trainable_prefixes = {
                "layer4": ("layer4",),
                "layer3": ("layer3", "layer4"),
            }
            if freeze_backbone_until not in trainable_prefixes:
                raise ValueError(f"{name} freeze_backbone_until must be one of: none, layer3, layer4")
            prefixes = trainable_prefixes[freeze_backbone_until]
            for param_name, parameter in backbone.named_parameters():
                parameter.requires_grad = param_name.startswith(prefixes)
            return
        return

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        features = self.backbone(frames)
        return self.proj(features)


def _build_image_encoder(config: ModelConfig, backbone_name: str) -> nn.Module:
    if backbone_name == "simple_cnn":
        return FrameCNN(config.embedding_dim)
    return TorchvisionEncoder(
        name=backbone_name,
        embedding_dim=config.embedding_dim,
        pretrained=config.pretrained,
        freeze_backbone=config.freeze_backbone,
        freeze_backbone_until=config.freeze_backbone_until,
        dropout=config.dropout,
    )


class PaperStyleTwoStreamModel(nn.Module):
    expects_regions = True

    def __init__(self, config: ModelConfig, backbone_name: str) -> None:
        super().__init__()
        self.action_encoder = _build_image_encoder(config, backbone_name)
        self.look_encoder = _build_image_encoder(config, backbone_name)
        self.action_classifier = nn.Linear(config.embedding_dim, config.num_action_classes)
        self.look_classifier = nn.Linear(config.embedding_dim, config.num_look_classes)
        self.embedding_head = nn.Linear(config.embedding_dim * 2, config.embedding_dim)

    def _encode_clip(self, encoder: nn.Module, clip: torch.Tensor) -> torch.Tensor:
        batch, time_steps, channels, height, width = clip.shape
        frames = clip.view(batch * time_steps, channels, height, width)
        frame_features = encoder(frames).view(batch, time_steps, -1)
        return frame_features.mean(dim=1)

    def forward(self, action_clip: torch.Tensor, look_clip: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        action_embedding = self._encode_clip(self.action_encoder, action_clip)
        look_embedding = self._encode_clip(self.look_encoder, look_clip)
        action_logits = self.action_classifier(action_embedding)
        look_logits = self.look_classifier(look_embedding)
        embedding = self.embedding_head(torch.cat([action_embedding, look_embedding], dim=1))
        return action_logits, look_logits, embedding


class SingleFrameClassifier(nn.Module):
    def __init__(self, config: ModelConfig, backbone_name: str, num_classes: int) -> None:
        super().__init__()
        self.encoder = _build_image_encoder(config, backbone_name)
        self.classifier = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.embedding_dim, num_classes),
        )

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.encoder(image)
        logits = self.classifier(embedding)
        return logits, embedding


class TwoViewSingleFrameClassifier(nn.Module):
    def __init__(self, config: ModelConfig, backbone_name: str, num_classes: int) -> None:
        super().__init__()
        self.encoder = _build_image_encoder(config, backbone_name)
        self.classifier = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.embedding_dim * 2, config.embedding_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.embedding_dim, num_classes),
        )

    def forward(self, image: torch.Tensor, image_aux: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.encoder(image)
        aux_embedding = self.encoder(image_aux)
        fused = torch.cat([embedding, aux_embedding], dim=1)
        logits = self.classifier(fused)
        return logits, fused


def build_single_frame_model(config: ModelConfig, task: str) -> nn.Module:
    architecture = config.architecture.lower()
    two_view = False
    if architecture == "paper_simple_cnn":
        backbone = "simple_cnn"
    elif architecture == "paper_alexnet":
        backbone = "alexnet"
    elif architecture == "paper_resnet18":
        backbone = "resnet18"
    elif architecture == "paper_resnet18_twoview":
        backbone = "resnet18"
        two_view = True
    elif architecture == "paper_resnet34":
        backbone = "resnet34"
    elif architecture == "paper_resnet34_twoview":
        backbone = "resnet34"
        two_view = True
    elif architecture == "paper_resnet50":
        backbone = "resnet50"
    elif architecture == "paper_resnet50_twoview":
        backbone = "resnet50"
        two_view = True
    elif architecture == "paper_mobilenet_v3_small":
        backbone = "mobilenet_v3_small"
    elif architecture == "paper_vit_b_16":
        backbone = "vit_b_16"
    elif architecture == "paper_vit_b_32":
        backbone = "vit_b_32"
    elif architecture == "paper_swin_t":
        backbone = "swin_t"
    elif architecture == "paper_swin_s":
        backbone = "swin_s"
    elif architecture == "paper_swin_b":
        backbone = "swin_b"
    else:
        raise ValueError(f"Single-frame paper training does not support architecture: {config.architecture}")

    if task == "action":
        num_classes = config.num_action_classes
    elif task == "look":
        num_classes = config.num_look_classes
    else:
        raise ValueError(f"Unsupported task: {task}")
    if two_view:
        return TwoViewSingleFrameClassifier(config, backbone, num_classes)
    return SingleFrameClassifier(config, backbone, num_classes)


def build_model(config: ModelConfig) -> nn.Module:
    architecture = config.architecture.lower()
    if architecture == "cnn_lstm":
        return CNNLSTMBehaviorModel(config)
    if architecture == "video_swin":
        return VideoSwinStyleBehaviorModel(config)
    if architecture == "paper_resnet18_temporal_transformer":
        return FrameTransformerBehaviorModel(config, "resnet18")
    if architecture == "paper_resnet34_temporal_transformer":
        return FrameTransformerBehaviorModel(config, "resnet34")
    if architecture == "torchvision_swin3d_t":
        return TorchvisionVideoSwinBehaviorModel(config, "t")
    if architecture == "torchvision_swin3d_s":
        return TorchvisionVideoSwinBehaviorModel(config, "s")
    if architecture == "torchvision_swin3d_b":
        return TorchvisionVideoSwinBehaviorModel(config, "b")
    if architecture == "paper_simple_cnn":
        return PaperStyleTwoStreamModel(config, "simple_cnn")
    if architecture == "paper_alexnet":
        return PaperStyleTwoStreamModel(config, "alexnet")
    if architecture == "paper_resnet18":
        return PaperStyleTwoStreamModel(config, "resnet18")
    if architecture == "paper_resnet34":
        return PaperStyleTwoStreamModel(config, "resnet34")
    if architecture == "paper_resnet50":
        return PaperStyleTwoStreamModel(config, "resnet50")
    if architecture == "paper_mobilenet_v3_small":
        return PaperStyleTwoStreamModel(config, "mobilenet_v3_small")
    raise ValueError(f"Unsupported Stage 1 architecture: {config.architecture}")
