"""Supervised Contrastive Loss."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """Supervised contrastive loss on L2-normalized projections."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, projections: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = projections.device
        projections = F.normalize(projections, dim=1)
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        logits = torch.div(projections @ projections.T, self.temperature)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        logits_mask = torch.ones_like(mask) - torch.eye(mask.size(0), device=device)
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True).clamp(min=1e-8))

        mean_log_prob = (mask * log_prob).sum(1) / mask.sum(1).clamp(min=1.0)
        return (-mean_log_prob).mean()
