"""
Pure-PyTorch PointNet++ encoder for CAD point-cloud embeddings.

Input convention:
    x: FloatTensor [B, N, C]
       C=3  -> xyz only
       C=6  -> xyz + normal

Output:
    global_embedding: [B, embedding_dim], L2-normalized, for CAD retrieval/alignment
    global_feature  : [B, global_dim], unnormalized intermediate feature
    local_xyz       : [B, n_local, 3], sampled canonical anchor points
    local_tokens    : [B, n_local, local_dim], L2-normalized local point tokens

The implementation avoids custom CUDA ops so it is easy to run anywhere.
For large N and large batch size, GPU is strongly recommended.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PointNet2Config:
    input_dim: int = 6
    embedding_dim: int = 256
    local_dim: int = 256
    sa1_npoint: int = 1024
    sa1_radius: float = 0.08
    sa1_nsample: int = 32
    sa2_npoint: int = 256
    sa2_radius: float = 0.16
    sa2_nsample: int = 32
    dropout: float = 0.3


def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """Calculate squared Euclidean distance.

    Args:
        src: [B, N, C]
        dst: [B, M, C]
    Returns:
        dist: [B, N, M]
    """
    return torch.sum((src[:, :, None, :] - dst[:, None, :, :]) ** 2, dim=-1)


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather points using batched indices.

    Args:
        points: [B, N, C]
        idx: [B, S] or [B, S, K]
    Returns:
        gathered: [B, S, C] or [B, S, K, C]
    """
    device = points.device
    B = points.shape[0]
    idx_shape = idx.shape
    idx_flat = idx.reshape(B, -1)
    batch_indices = torch.arange(B, dtype=torch.long, device=device).view(B, 1)
    gathered = points[batch_indices, idx_flat]
    return gathered.reshape(*idx_shape, points.shape[-1])


def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """Farthest point sampling.

    Args:
        xyz: [B, N, 3]
        npoint: number of sampled centroids
    Returns:
        centroids: [B, npoint]
    """
    device = xyz.device
    B, N, _ = xyz.shape
    npoint = min(npoint, N)
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.full((B, N), 1e10, dtype=xyz.dtype, device=device)
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, dim=-1)[1]
    return centroids


def query_ball_point(radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
    """Group local neighborhoods around centroids.

    Args:
        radius: ball radius in normalized canonical coordinates
        nsample: maximum number of neighbors per centroid
        xyz: [B, N, 3]
        new_xyz: [B, S, 3]
    Returns:
        group_idx: [B, S, nsample]
    """
    device = xyz.device
    B, N, _ = xyz.shape
    S = new_xyz.shape[1]
    nsample = min(nsample, N)

    sqrdists = square_distance(new_xyz, xyz)  # [B, S, N]
    group_idx = torch.arange(N, dtype=torch.long, device=device).view(1, 1, N).repeat(B, S, 1)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]

    # If a centroid has fewer than nsample neighbors, repeat the first valid neighbor.
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat(1, 1, nsample)
    group_idx[group_idx == N] = group_first[group_idx == N]
    return group_idx


def sample_and_group(
    npoint: int,
    radius: float,
    nsample: int,
    xyz: torch.Tensor,
    points: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """FPS + ball query grouping."""
    fps_idx = farthest_point_sample(xyz, npoint)
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx)
    grouped_xyz_norm = grouped_xyz - new_xyz[:, :, None, :]

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz_norm
    return new_xyz, new_points


def sample_and_group_all(xyz: torch.Tensor, points: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Group all points into a single global set."""
    B, N, _ = xyz.shape
    new_xyz = torch.zeros(B, 1, 3, dtype=xyz.dtype, device=xyz.device)
    grouped_xyz = xyz.view(B, 1, N, 3)
    grouped_xyz_norm = grouped_xyz - new_xyz[:, :, None, :]
    if points is not None:
        grouped_points = points.view(B, 1, N, -1)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz_norm
    return new_xyz, new_points


class SharedMLP2d(nn.Module):
    def __init__(self, channels: List[int]):
        super().__init__()
        layers = []
        for i in range(len(channels) - 1):
            layers.append(nn.Conv2d(channels[i], channels[i + 1], kernel_size=1, bias=False))
            layers.append(nn.BatchNorm2d(channels[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PointNetSetAbstraction(nn.Module):
    """Single-scale PointNet++ set abstraction layer."""

    def __init__(
        self,
        npoint: Optional[int],
        radius: Optional[float],
        nsample: Optional[int],
        in_channel: int,
        mlp: List[int],
        group_all: bool = False,
    ):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all
        self.mlp = SharedMLP2d([in_channel + 3] + mlp)

    def forward(self, xyz: torch.Tensor, points: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            xyz: [B, N, 3]
            points: [B, N, D] or None
        Returns:
            new_xyz: [B, S, 3]
            new_points: [B, S, D_out]
        """
        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            assert self.npoint is not None and self.radius is not None and self.nsample is not None
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)

        # [B, S, K, D] -> [B, D, K, S]
        new_points = new_points.permute(0, 3, 2, 1).contiguous()
        new_points = self.mlp(new_points)
        # max pool over local neighborhood K -> [B, D_out, S]
        new_points = torch.max(new_points, dim=2)[0]
        # [B, D_out, S] -> [B, S, D_out]
        new_points = new_points.permute(0, 2, 1).contiguous()
        return new_xyz, new_points


class PointNet2Encoder(nn.Module):
    """PointNet++ CAD encoder with global and local outputs."""

    def __init__(self, config: PointNet2Config = PointNet2Config()):
        super().__init__()
        self.config = config
        if config.input_dim < 3:
            raise ValueError("input_dim must be at least 3 for xyz")
        extra_dim = config.input_dim - 3

        self.sa1 = PointNetSetAbstraction(
            npoint=config.sa1_npoint,
            radius=config.sa1_radius,
            nsample=config.sa1_nsample,
            in_channel=extra_dim,
            mlp=[64, 64, 128],
            group_all=False,
        )
        self.sa2 = PointNetSetAbstraction(
            npoint=config.sa2_npoint,
            radius=config.sa2_radius,
            nsample=config.sa2_nsample,
            in_channel=128,
            mlp=[128, 128, 256],
            group_all=False,
        )
        self.sa3 = PointNetSetAbstraction(
            npoint=None,
            radius=None,
            nsample=None,
            in_channel=256,
            mlp=[256, 512, 1024],
            group_all=True,
        )

        self.global_dim = 1024
        self.global_projection = nn.Sequential(
            nn.Linear(self.global_dim, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(512, config.embedding_dim),
        )
        self.local_projection = nn.Sequential(
            nn.Linear(256, config.local_dim, bias=False),
            nn.LayerNorm(config.local_dim),
            nn.ReLU(inplace=True),
            nn.Linear(config.local_dim, config.local_dim),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(f"Expected input [B, N, C], got {tuple(x.shape)}")
        if x.shape[-1] != self.config.input_dim:
            raise ValueError(f"Expected input_dim={self.config.input_dim}, got {x.shape[-1]}")

        xyz = x[:, :, :3].contiguous()
        extra = x[:, :, 3:].contiguous() if self.config.input_dim > 3 else None

        l1_xyz, l1_points = self.sa1(xyz, extra)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        _, l3_points = self.sa3(l2_xyz, l2_points)

        global_feature = l3_points.squeeze(1)
        global_embedding = self.global_projection(global_feature)
        global_embedding = F.normalize(global_embedding, dim=-1)

        local_tokens = self.local_projection(l2_points)
        local_tokens = F.normalize(local_tokens, dim=-1)

        return {
            "global_embedding": global_embedding,
            "global_feature": global_feature,
            "local_xyz": l2_xyz,
            "local_tokens": local_tokens,
        }


class CADPointNet2Model(nn.Module):
    """PointNet++ encoder with optional classification heads."""

    def __init__(
        self,
        config: PointNet2Config,
        num_classes: Optional[int] = None,
        num_cads: Optional[int] = None,
    ):
        super().__init__()
        self.encoder = PointNet2Encoder(config)
        self.num_classes = num_classes
        self.num_cads = num_cads

        if num_classes is not None and num_classes > 0:
            self.class_head = nn.Sequential(
                nn.Linear(self.encoder.global_dim, 512, bias=False),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(config.dropout),
                nn.Linear(512, num_classes),
            )
        else:
            self.class_head = None

        if num_cads is not None and num_cads > 0:
            self.cad_head = nn.Sequential(
                nn.Linear(self.encoder.global_dim, 512, bias=False),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(config.dropout),
                nn.Linear(512, num_cads),
            )
        else:
            self.cad_head = None

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = self.encoder(x)
        global_feature = out["global_feature"]
        if self.class_head is not None:
            out["class_logits"] = self.class_head(global_feature)
        if self.cad_head is not None:
            out["cad_logits"] = self.cad_head(global_feature)
        return out
