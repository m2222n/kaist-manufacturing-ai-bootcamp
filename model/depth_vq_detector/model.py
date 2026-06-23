from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int) -> None:
        super().__init__()
        assert num_layers >= 1
        layers = []
        for i in range(num_layers):
            in_d = input_dim if i == 0 else hidden_dim
            out_d = output_dim if i == num_layers - 1 else hidden_dim
            layers.append(nn.Linear(in_d, out_d))
            if i < num_layers - 1:
                layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int | None = None) -> None:
        super().__init__()
        if p is None:
            p = k // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.conv1 = ConvBNAct(ch, ch, 3, 1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ch),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.conv2(self.conv1(x)))


class DepthCNNBackbone(nn.Module):
    """Small depth-first CNN backbone.

    It is intentionally dependency-free. Later, this module can be replaced by a
    pretrained ConvNeXt/ResNet/Mask2Former backbone while preserving the detector API.
    """

    def __init__(self, in_channels: int, base_dim: int = 64) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            ConvBNAct(in_channels, base_dim // 2, 3, 2),  # H/2
            ConvBNAct(base_dim // 2, base_dim, 3, 2),     # H/4
            ResidualBlock(base_dim),
        )
        self.stage2 = nn.Sequential(ConvBNAct(base_dim, base_dim * 2, 3, 2), ResidualBlock(base_dim * 2))  # H/8
        self.stage3 = nn.Sequential(ConvBNAct(base_dim * 2, base_dim * 4, 3, 2), ResidualBlock(base_dim * 4))  # H/16
        self.stage4 = nn.Sequential(ConvBNAct(base_dim * 4, base_dim * 8, 3, 2), ResidualBlock(base_dim * 8))  # H/32
        self.out_channels = [base_dim, base_dim * 2, base_dim * 4, base_dim * 8]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        c1 = self.stem(x)
        c2 = self.stage2(c1)
        c3 = self.stage3(c2)
        c4 = self.stage4(c3)
        return [c1, c2, c3, c4]


class FPNPixelDecoder(nn.Module):
    """FPN-style pixel decoder.

    Returns:
      mask_features: [B, hidden_dim, H/4, W/4]
      memory_feature: [B, hidden_dim, H/16, W/16] for transformer cross-attention
    """

    def __init__(self, in_channels: list[int], hidden_dim: int = 256) -> None:
        super().__init__()
        self.lateral = nn.ModuleList([nn.Conv2d(c, hidden_dim, 1) for c in in_channels])
        self.output = nn.ModuleList([ConvBNAct(hidden_dim, hidden_dim, 3, 1) for _ in in_channels])
        self.mask_proj = nn.Conv2d(hidden_dim, hidden_dim, 1)
        self.memory_proj = nn.Conv2d(hidden_dim, hidden_dim, 1)

    def forward(self, feats: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        # feats: [C1 H/4, C2 H/8, C3 H/16, C4 H/32]
        laterals = [lat(f) for lat, f in zip(self.lateral, feats)]
        x = laterals[-1]
        outs = [None] * len(laterals)
        outs[-1] = self.output[-1](x)
        for i in range(len(laterals) - 2, -1, -1):
            x = F.interpolate(x, size=laterals[i].shape[-2:], mode="nearest") + laterals[i]
            outs[i] = self.output[i](x)
        mask_features = self.mask_proj(outs[0])
        memory_feature = self.memory_proj(outs[2])  # H/16 by default
        return mask_features, memory_feature


class PositionEmbeddingSine(nn.Module):
    """2D sine-cosine positional encoding, DETR-style."""

    def __init__(self, num_pos_feats: int = 128, temperature: int = 10000, normalize: bool = True) -> None:
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        device = x.device
        y_embed = torch.arange(h, device=device, dtype=torch.float32).unsqueeze(1).repeat(1, w)
        x_embed = torch.arange(w, device=device, dtype=torch.float32).unsqueeze(0).repeat(h, 1)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (h - 1 + eps) * 2 * math.pi
            x_embed = x_embed / (w - 1 + eps) * 2 * math.pi
        dim_t = torch.arange(self.num_pos_feats, device=device, dtype=torch.float32)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / self.num_pos_feats)
        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
        pos = torch.cat((pos_y, pos_x), dim=2).permute(2, 0, 1).unsqueeze(0).repeat(b, 1, 1, 1)
        return pos


class TransformerObjectDecoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 256,
        num_queries: int = 100,
        num_layers: int = 6,
        nheads: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=False,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.pos_embed = PositionEmbeddingSine(hidden_dim // 2)
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim

    def forward(self, memory_feature: torch.Tensor) -> torch.Tensor:
        b, c, h, w = memory_feature.shape
        pos = self.pos_embed(memory_feature)
        memory = (memory_feature + pos).flatten(2).permute(2, 0, 1)  # [S,B,C]
        query = self.query_embed.weight.unsqueeze(1).repeat(1, b, 1)  # [Q,B,C]
        tgt = torch.zeros_like(query)
        hs = self.decoder(tgt + query, memory)
        return hs.permute(1, 0, 2).contiguous()  # [B,Q,C]


def load_cad_codebook(path: str | Path) -> tuple[torch.Tensor, list[str]]:
    """Load CAD memory bank produced by build_cad_memory_bank.py.

    Expected keys:
      embeddings: [M,D]
      cad_ids: [M]
    """
    data = np.load(path, allow_pickle=True)
    if "embeddings" not in data.files:
        raise KeyError(f"{path} does not contain key 'embeddings'. Available: {data.files}")
    embeddings = data["embeddings"].astype(np.float32)
    cad_ids = data["cad_ids"].tolist() if "cad_ids" in data.files else [str(i) for i in range(len(embeddings))]
    cad_ids = [str(x) for x in cad_ids]
    return torch.from_numpy(embeddings), cad_ids


class DepthVQDetector(nn.Module):
    """Depth-only query detector with frozen CAD VQ codebook.

    Forward input:
      x: [B,C,H,W] depth-only channels.

    Outputs:
      pred_logits: [B,Q,num_classes+1] object class logits, last class is no-object.
      pred_boxes: [B,Q,4] normalized cx,cy,w,h.
      pred_masks: [B,Q,H/4,W/4] mask logits.
      pred_obj_emb: [B,Q,D] normalized 2D object embeddings.
      pred_cad_logits: [B,Q,M] CAD codebook similarity logits, if codebook exists.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        cad_codebook: torch.Tensor | None = None,
        num_queries: int = 100,
        hidden_dim: int = 256,
        backbone_dim: int = 64,
        decoder_layers: int = 6,
        nheads: int = 8,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.num_queries = int(num_queries)
        self.hidden_dim = int(hidden_dim)

        self.backbone = DepthCNNBackbone(in_channels, base_dim=backbone_dim)
        self.pixel_decoder = FPNPixelDecoder(self.backbone.out_channels, hidden_dim=hidden_dim)
        self.query_decoder = TransformerObjectDecoder(
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            num_layers=decoder_layers,
            nheads=nheads,
        )

        self.class_head = nn.Linear(hidden_dim, num_classes + 1)  # + no-object
        self.box_head = MLP(hidden_dim, hidden_dim, 4, 3)
        self.mask_embed_head = MLP(hidden_dim, hidden_dim, hidden_dim, 3)

        if cad_codebook is not None:
            if cad_codebook.ndim != 2:
                raise ValueError("cad_codebook must be [num_cads, embed_dim]")
            cad_codebook = F.normalize(cad_codebook.float(), dim=-1)
            embed_dim = cad_codebook.shape[1]
            self.register_buffer("cad_codebook", cad_codebook)
            self.obj_embed_head = MLP(hidden_dim, hidden_dim, embed_dim, 3)
            self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0), dtype=torch.float32))
        else:
            self.register_buffer("cad_codebook", torch.empty(0, hidden_dim))
            self.obj_embed_head = MLP(hidden_dim, hidden_dim, hidden_dim, 3)
            self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0), dtype=torch.float32))

        self._init_parameters()

    @property
    def has_cad_codebook(self) -> bool:
        return self.cad_codebook.numel() > 0

    def _init_parameters(self) -> None:
        nn.init.constant_(self.class_head.bias, 0.0)
        # Bias no-object class slightly high at startup.
        with torch.no_grad():
            self.class_head.bias[-1] = 1.0

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feats = self.backbone(x)
        mask_features, memory_feature = self.pixel_decoder(feats)
        h = self.query_decoder(memory_feature)  # [B,Q,C]

        pred_logits = self.class_head(h)
        pred_boxes = self.box_head(h).sigmoid()
        mask_embed = self.mask_embed_head(h)
        pred_masks = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        pred_obj_emb = F.normalize(self.obj_embed_head(h), dim=-1)
        out: dict[str, torch.Tensor] = {
            "pred_logits": pred_logits,
            "pred_boxes": pred_boxes,
            "pred_masks": pred_masks,
            "pred_obj_emb": pred_obj_emb,
        }
        if self.has_cad_codebook:
            codebook = F.normalize(self.cad_codebook, dim=-1)
            scale = self.logit_scale.exp().clamp(max=100.0)
            pred_cad_logits = scale * pred_obj_emb @ codebook.t()
            out["pred_cad_logits"] = pred_cad_logits
        return out

    def get_config(self) -> dict[str, Any]:
        return {
            "num_classes": self.num_classes,
            "num_queries": self.num_queries,
            "hidden_dim": self.hidden_dim,
            "cad_embed_dim": int(self.cad_codebook.shape[1]) if self.has_cad_codebook else None,
            "num_cads": int(self.cad_codebook.shape[0]) if self.has_cad_codebook else 0,
        }
