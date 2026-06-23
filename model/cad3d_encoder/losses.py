from __future__ import annotations

import torch
import torch.nn.functional as F


def supervised_contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Supervised contrastive loss over a batch of L2-normalized embeddings.

    Args:
        embeddings: [M, D], preferably L2-normalized
        labels: [M], positives are samples sharing the same label
        temperature: softmax temperature
    """
    if embeddings.ndim != 2:
        raise ValueError(f"Expected embeddings [M, D], got {tuple(embeddings.shape)}")
    if labels.ndim != 1:
        labels = labels.view(-1)
    if embeddings.shape[0] != labels.shape[0]:
        raise ValueError("embeddings and labels must have the same batch dimension")

    device = embeddings.device
    labels = labels.to(device)
    embeddings = F.normalize(embeddings, dim=-1)

    logits = torch.matmul(embeddings, embeddings.T) / temperature
    logits = logits - logits.max(dim=1, keepdim=True)[0].detach()

    M = embeddings.shape[0]
    self_mask = torch.eye(M, dtype=torch.bool, device=device)
    positive_mask = labels[:, None].eq(labels[None, :]) & (~self_mask)
    logits_mask = ~self_mask

    exp_logits = torch.exp(logits) * logits_mask.float()
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))

    positive_count = positive_mask.sum(dim=1)
    valid = positive_count > 0
    if not valid.any():
        return embeddings.new_tensor(0.0)

    mean_log_prob_pos = (positive_mask.float() * log_prob).sum(dim=1) / positive_count.clamp_min(1)
    loss = -mean_log_prob_pos[valid].mean()
    return loss
