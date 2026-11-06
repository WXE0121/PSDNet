"""Training objective of PSDNet.

    L = EDL(alpha_local) + EDL(alpha_final) + 0.5 * EDL(alpha_sys2_aux)
        + budget_loss_weight * relu(gate_rate - rho)^2   (one-sided compute budget)
        + 0.2 * BCE(boundary_logit, b_t)                 (transition auxiliary task, pos-weighted)
        + 0.05 * KL(mean phase usage || uniform)         (phase-primitive diversity stabiliser)

EDL is the evidential Dirichlet loss of Sensoy et al. (digamma expected cross-entropy plus an
annealed KL(Dir(alpha~) || Dir(1)) term on the non-target evidence).
"""
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PSDNetLoss(nn.Module):
    def __init__(self, num_classes: int, num_primitives: int, max_epochs: int, budget_target: float = 0.25,
                 budget_loss_weight: float = 5.0, kl_scale: float = 0.09):
        super().__init__()
        self.num_classes = num_classes
        self.num_primitives = num_primitives
        self.max_epochs = max_epochs
        self.budget_target = float(budget_target)
        self.budget_loss_weight = float(budget_loss_weight)
        self.kl_scale = float(kl_scale)

    def _kl_dirichlet_to_uniform(self, alpha: torch.Tensor) -> torch.Tensor:
        s = torch.sum(alpha, dim=-1, keepdim=True)
        kl = torch.lgamma(s) - torch.sum(torch.lgamma(alpha), dim=-1, keepdim=True)
        kl = kl - torch.lgamma(torch.tensor(float(self.num_classes), device=alpha.device, dtype=alpha.dtype))
        kl = kl + torch.sum((alpha - 1.0) * (torch.digamma(alpha) - torch.digamma(s)), dim=-1, keepdim=True)
        return kl.squeeze(-1)

    def edl_loss(self, alpha: torch.Tensor, target: torch.Tensor, anneal_coef: float) -> torch.Tensor:
        y_onehot = F.one_hot(target, num_classes=self.num_classes).float()
        s = torch.sum(alpha, dim=-1, keepdim=True)
        loss_ce = torch.sum(y_onehot * (torch.digamma(s) - torch.digamma(alpha)), dim=-1)
        alpha_tilde = y_onehot + (1.0 - y_onehot) * alpha
        loss_kl = self._kl_dirichlet_to_uniform(alpha_tilde)
        return (loss_ce + float(anneal_coef) * self.kl_scale * loss_kl).mean()

    def forward(self, out: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], current_epoch: int) -> Tuple[torch.Tensor, Dict[str, float]]:
        y = batch['y']
        boundary_target = batch['boundary'].float()
        # KL weight is annealed linearly over the first 40% of training
        anneal_coef = min(1.0, float(current_epoch) / max(self.max_epochs * 0.4, 1e-8))

        loss_local = self.edl_loss(out['alpha_local'], y, anneal_coef)
        loss_final = self.edl_loss(out['alpha_final'], y, anneal_coef)
        loss_sys2 = self.edl_loss(out['alpha_sys2_aux'], y, anneal_coef)

        # transitions are rare: re-weight positives so the boundary head cannot collapse to "never"
        num_neg = (boundary_target == 0).sum()
        num_pos = (boundary_target == 1).sum().clamp_min(1.0)
        boundary_loss = F.binary_cross_entropy_with_logits(out['boundary_logit'], boundary_target, pos_weight=(num_neg / num_pos))

        phase_prob = F.softmax(out['cur_phase_logits'], dim=-1)
        mean_phase = phase_prob.mean(dim=0)
        uniform = torch.full_like(mean_phase, 1.0 / self.num_primitives)
        primitive_div = (mean_phase * (mean_phase.clamp_min(1e-8).log() - uniform.log())).sum()

        gate_rate = out['gate_rate_valid']
        loss_budget = F.relu(gate_rate - self.budget_target) ** 2

        total = (loss_local + loss_final + 0.5 * loss_sys2
                 + self.budget_loss_weight * loss_budget
                 + 0.2 * boundary_loss
                 + 0.05 * primitive_div)

        stats = {
            'loss_local': float(loss_local.detach().item()),
            'loss_final': float(loss_final.detach().item()),
            'loss_sys2_aux': float(loss_sys2.detach().item()),
            'loss_budget': float(loss_budget.detach().item()),
            'boundary_loss': float(boundary_loss.detach().item()),
            'primitive_div': float(primitive_div.detach().item()),
            'gate_rate': float(gate_rate.detach().item()),
            'entropy_mean': float(out['entropy'].detach().mean().item()),
        }
        return total, stats
