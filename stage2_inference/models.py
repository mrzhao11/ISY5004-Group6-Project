from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    from torch import nn
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError("PyTorch is required for stage2.models") from exc

from .config import DataConfig, ModelConfig


class TemporalMLPEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.proj(x)
        return encoded.mean(dim=1)


class GRUEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            dropout=dropout if hidden_dim > 1 else 0.0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(x)
        return hidden[-1]


class TemporalTransformerEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, heads: int, layers: int, dropout: float) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.cls = nn.Parameter(torch.zeros(1, 1, hidden_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.size(0)
        tokens = self.input_proj(x)
        cls = self.cls.expand(batch, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        encoded = self.encoder(tokens)
        return encoded[:, 0]


def build_temporal_encoder(kind: str, input_dim: int, model_cfg: ModelConfig) -> nn.Module:
    kind = kind.lower()
    if kind == "mlp":
        return TemporalMLPEncoder(input_dim, model_cfg.hidden_dim, model_cfg.dropout)
    if kind == "gru":
        return GRUEncoder(input_dim, model_cfg.hidden_dim, model_cfg.dropout)
    if kind == "transformer":
        return TemporalTransformerEncoder(
            input_dim=input_dim,
            hidden_dim=model_cfg.hidden_dim,
            heads=model_cfg.transformer_heads,
            layers=model_cfg.transformer_layers,
            dropout=model_cfg.dropout,
        )
    raise ValueError(f"Unsupported temporal encoder: {kind}")


class GTBehaviorBranch(nn.Module):
    def __init__(self, num_classes: int, hidden_dim: int, encoder_kind: str, model_cfg: ModelConfig) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_classes, hidden_dim)
        self.encoder = build_temporal_encoder(encoder_kind, hidden_dim, model_cfg)

    def forward(self, behavior_indices: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(behavior_indices)
        embedded = embedded * valid_mask.unsqueeze(-1)
        return self.encoder(embedded)


class Stage1BehaviorBranch(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class MLPBranch(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TCLFusionHead(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.event_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, tokens: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
        gated = tokens * self.event_gate(tokens)
        if weights is None:
            fused = gated.mean(dim=1)
        else:
            weights = weights.view(1, -1, 1)
            fused = (gated * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1e-6)
        return self.classifier(fused).squeeze(-1)


class PITFusionHead(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.intent_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        intent = self.intent_token.expand(tokens.size(0), -1, -1)
        encoded = self.encoder(torch.cat([intent, tokens], dim=1))
        return self.classifier(encoded[:, 0]).squeeze(-1)


class TAMFormerFusionHead(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.classifier = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(tokens)
        pooled = encoded.mean(dim=1)
        return self.classifier(pooled).squeeze(-1)


class IntentFormerFusionHead(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.intent_query = nn.Parameter(torch.zeros(1, hidden_dim))
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=dropout, batch_first=True)
        self.classifier = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        query = self.intent_query.unsqueeze(0).expand(tokens.size(0), 1, -1)
        attended, _ = self.attn(query, tokens, tokens)
        return self.classifier(attended[:, 0]).squeeze(-1)


def build_fusion_head(architecture: str, hidden_dim: int, dropout: float) -> nn.Module:
    architecture = architecture.lower()
    if architecture == "tcl":
        return TCLFusionHead(hidden_dim, dropout)
    if architecture == "pit":
        return PITFusionHead(hidden_dim, dropout)
    if architecture == "tamformer":
        return TAMFormerFusionHead(hidden_dim, dropout)
    if architecture == "intentformer":
        return IntentFormerFusionHead(hidden_dim, dropout)
    raise ValueError(f"Unsupported architecture: {architecture}")


class Stage2CrossingModel(nn.Module):
    def __init__(
        self,
        data_cfg: DataConfig,
        model_cfg: ModelConfig,
        trajectory_static_dim: int,
        behavior_feature_dim: int,
        context_feature_dim: int,
        vehicle_feature_dim: int,
    ) -> None:
        super().__init__()
        self.data_cfg = data_cfg
        self.model_cfg = model_cfg

        self.trajectory_seq_encoder = (
            build_temporal_encoder(model_cfg.trajectory_encoder, 8, model_cfg)
            if data_cfg.use_trajectory_sequence
            else None
        )
        self.trajectory_static_branch = (
            MLPBranch(trajectory_static_dim, model_cfg.hidden_dim, model_cfg.dropout)
            if trajectory_static_dim > 0
            else None
        )

        if data_cfg.behavior_mode == "gt_behavior":
            self.behavior_branch = GTBehaviorBranch(
                num_classes=model_cfg.behavior_num_classes,
                hidden_dim=model_cfg.hidden_dim,
                encoder_kind=model_cfg.behavior_encoder,
                model_cfg=model_cfg,
            )
        elif data_cfg.behavior_mode == "stage1_features":
            if behavior_feature_dim <= 0:
                raise ValueError("behavior_mode='stage1_features' requires behavior feature columns in the manifest")
            self.behavior_branch = Stage1BehaviorBranch(behavior_feature_dim, model_cfg.hidden_dim)
        else:
            self.behavior_branch = None

        self.context_branch = (
            MLPBranch(context_feature_dim, model_cfg.hidden_dim, model_cfg.dropout)
            if context_feature_dim > 0
            else None
        )
        self.vehicle_branch = (
            MLPBranch(vehicle_feature_dim, model_cfg.hidden_dim, model_cfg.dropout)
            if vehicle_feature_dim > 0
            else None
        )
        self.fusion = build_fusion_head(model_cfg.architecture, model_cfg.hidden_dim, model_cfg.dropout)
        self.behavior_logit_head = None
        self.behavior_scale_logit = None
        if self.behavior_branch is not None and model_cfg.behavior_fusion in {"scalar_weighted", "late_logit", "raw_logit"}:
            if model_cfg.architecture != "tcl":
                raise ValueError("Alternative behavior fusion modes currently require architecture='tcl'")
            self.behavior_scale_logit = nn.Parameter(torch.tensor(-3.0))
        if self.behavior_branch is not None and model_cfg.behavior_fusion == "late_logit":
            self.behavior_logit_head = nn.Sequential(
                nn.Linear(model_cfg.hidden_dim, model_cfg.hidden_dim),
                nn.ReLU(),
                nn.Dropout(model_cfg.dropout),
                nn.Linear(model_cfg.hidden_dim, 1),
            )
        if self.behavior_branch is not None and model_cfg.behavior_fusion == "raw_logit":
            self.behavior_logit_head = nn.Linear(behavior_feature_dim, 1)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        tokens = []
        behavior_token = None
        if self.trajectory_seq_encoder is not None:
            tokens.append(self.trajectory_seq_encoder(batch["trajectory_seq"]))

        if self.trajectory_static_branch is not None:
            tokens.append(self.trajectory_static_branch(batch["trajectory_static"]))

        if self.behavior_branch is not None:
            if self.data_cfg.behavior_mode == "gt_behavior":
                behavior_token = self.behavior_branch(batch["behavior_input"], batch["behavior_valid"])
            else:
                behavior_token = self.behavior_branch(batch["behavior_input"])
            if self.model_cfg.behavior_fusion == "token":
                tokens.append(behavior_token)

        if self.context_branch is not None:
            tokens.append(self.context_branch(batch["context_vector"]))

        if self.vehicle_branch is not None:
            tokens.append(self.vehicle_branch(batch["vehicle_vector"]))

        if not tokens:
            raise ValueError("Stage2CrossingModel received no enabled input branches")
        stacked = torch.stack(tokens, dim=1)
        if behavior_token is not None and self.model_cfg.behavior_fusion == "scalar_weighted":
            stacked = torch.cat([stacked, behavior_token.unsqueeze(1)], dim=1)
            behavior_weight = torch.sigmoid(self.behavior_scale_logit)
            weights = torch.cat(
                [
                    torch.ones(stacked.size(1) - 1, device=stacked.device),
                    behavior_weight.unsqueeze(0),
                ]
            )
            return self.fusion(stacked, weights=weights)
        if behavior_token is not None and self.model_cfg.behavior_fusion == "late_logit":
            base_logit = self.fusion(stacked)
            behavior_logit = self.behavior_logit_head(behavior_token).squeeze(-1)
            behavior_weight = torch.sigmoid(self.behavior_scale_logit)
            return base_logit + behavior_weight * behavior_logit
        if behavior_token is not None and self.model_cfg.behavior_fusion == "raw_logit":
            base_logit = self.fusion(stacked)
            behavior_logit = self.behavior_logit_head(batch["behavior_input"]).squeeze(-1)
            behavior_weight = torch.sigmoid(self.behavior_scale_logit)
            return base_logit + behavior_weight * behavior_logit
        return self.fusion(stacked)
