"""PSDNet: Phase-Script Deliberation Network for sliding-window human activity recognition.

System 1 (always on)
    window encoder -> phase mixture q_t over M latent primitives -> phase feature z_t
    GRU summary c_t of the phase features of the L previous windows
    class-script alignment score g_t for every class
    local Dirichlet evidence alpha^(1) = softplus(logits) + 1 and its expected entropy u_t

System 2 (triggered only when u_t exceeds a learned threshold)
    targeted retrieval over the encoded history, a dense residual correction over all classes and a
    candidate-specific refinement over the top-Kc classes of System 1.  The correction is added to
    the System-1 logits, scaled by a soft gate gamma_t = sigmoid((u_t - threshold) * temperature).

The gate uses the stop-gradient of u_t, and all System-2 inputs are detached, so deliberation can
never lower System-1 uncertainty to avoid being triggered nor corrupt the window encoder.
"""
import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import build_backbone
from .modules import CandidateDeliberation, HistoryRetriever, MaskedGRU, PhaseScriptBank


def expected_dirichlet_entropy(alpha: torch.Tensor) -> torch.Tensor:
    """Expected entropy of a categorical drawn from Dirichlet(alpha)."""
    s = torch.sum(alpha, dim=-1, keepdim=True)
    term = torch.digamma(s + 1.0) - torch.digamma(alpha + 1.0)
    return torch.sum((alpha / s.clamp_min(1e-8)) * term, dim=-1)


class PSDNet(nn.Module):
    def __init__(self, num_classes: int, context_len: int = 10, num_primitives: int = 8, script_len: int = 4,
                 feat_dim: int = 256, candidate_topk: int = 3, gate_temp: float = 10.0, bump_init: float = -1.0,
                 dataset_name: str = 'hapt'):
        super().__init__()
        self.num_classes = num_classes
        self.context_len = context_len
        self.candidate_topk = candidate_topk
        self.gate_temp = gate_temp

        self.window_encoder = build_backbone(feat_dim, dataset_name)

        # ----- System 1 -----
        self.phase_head = nn.Linear(feat_dim, num_primitives)
        self.primitive_bank = PhaseScriptBank(num_classes, num_primitives, feat_dim, script_len)
        self.ctx_encoder = MaskedGRU(feat_dim, feat_dim)
        self.sys1_logits = nn.Sequential(
            nn.Linear(feat_dim * 3, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(feat_dim, num_classes),
        )
        # auxiliary transition (boundary) predictor
        self.boundary_head = nn.Sequential(
            nn.Linear(feat_dim * 2 + num_primitives + 3, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )
        # learnable uncertainty threshold ("speed bump") that triggers System 2
        if bump_init >= 0:
            self.bump_threshold = nn.Parameter(torch.tensor([float(bump_init)]))
        else:
            self.bump_threshold = nn.Parameter(torch.tensor([float(math.log(max(num_classes, 2)) * 0.5)]))

        # ----- System 2 -----
        self.sys2_adaptor = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.ReLU(inplace=True),
        )
        self.epistemic_retriever = HistoryRetriever(feat_dim)
        self.deliberation = CandidateDeliberation(num_classes, feat_dim)
        self.delta_logits_head = nn.Sequential(
            nn.Linear(feat_dim * 3, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(feat_dim, num_classes),
        )

    # ------------------------------------------------------------------ System 2
    def _system2_delta(self, cur_feat: torch.Tensor, ctx_feat_seq: torch.Tensor, cur_phase_feat: torch.Tensor,
                       ctx_mask: torch.Tensor, topk_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Logit correction (B', K) and retrieval attention (B', L) for the selected windows.

        Inputs are detached: System 2 refines on top of System-1 features but never trains them.
        """
        cur_feat = cur_feat.detach()
        ctx_feat_seq = ctx_feat_seq.detach()
        cur_phase_feat = cur_phase_feat.detach()

        ctx_sys2 = self.sys2_adaptor(ctx_feat_seq)
        cur_sys2 = self.sys2_adaptor(cur_feat)

        lengths = ctx_mask.sum(dim=1).clamp(min=1).long()
        key_padding_mask = (ctx_mask == 0.0).bool()
        all_masked = key_padding_mask.all(dim=1)
        if all_masked.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_masked, 0] = False

        retrieved_ctx, attn_w = self.epistemic_retriever(cur_sys2, ctx_sys2, lengths, key_padding_mask)

        # dense residual correction over all classes
        delta_dense = self.delta_logits_head(torch.cat([cur_sys2, retrieved_ctx, cur_phase_feat], dim=-1))
        # candidate-specific refinement, scattered back onto the top-Kc classes (unconstrained sign)
        subset_delta = self.deliberation(cur_feat, cur_phase_feat, retrieved_ctx, topk_idx)
        delib = torch.zeros(cur_feat.size(0), self.num_classes, device=cur_feat.device)
        delib.scatter_add_(1, topk_idx, subset_delta)
        return delta_dense + 0.5 * delib, attn_w

    # ------------------------------------------------------------------ forward
    def forward(self, batch: Dict[str, torch.Tensor], is_inference: bool = False) -> Dict[str, torch.Tensor]:
        x_cur, ctx_x, ctx_mask = batch['x_cur'], batch['ctx_x'], batch['ctx_mask']
        bsz, l_ctx = x_cur.size(0), ctx_x.size(1)
        has_ctx = ctx_mask.sum(dim=1) > 0

        # ----------------- System 1 -----------------
        cur_feat = self.window_encoder(x_cur)
        cur_phase_logits = self.phase_head(cur_feat)
        cur_phase_prob = F.softmax(cur_phase_logits, dim=-1)
        cur_phase_feat = self.primitive_bank.primitive_feature(cur_phase_prob)

        # History windows are encoded without gradient so that the encoder is trained on the
        # current window only.
        with torch.no_grad():
            ctx_feat_flat = self.window_encoder(ctx_x.view(bsz * l_ctx, 1, ctx_x.size(3), ctx_x.size(4)))
            ctx_phase_logits_flat = self.phase_head(ctx_feat_flat)
        ctx_phase_prob_flat = F.softmax(ctx_phase_logits_flat, dim=-1)
        ctx_phase_feat = self.primitive_bank.primitive_feature(ctx_phase_prob_flat).view(bsz, l_ctx, -1)
        ctx_summary = self.ctx_encoder(ctx_phase_feat, ctx_mask)

        script_score, _ = self.primitive_bank.all_class_script_scores(cur_phase_feat, ctx_summary)
        logit_local = self.sys1_logits(torch.cat([cur_feat, cur_phase_feat, ctx_summary], dim=-1)) + script_score

        alpha_local = F.softplus(logit_local) + 1.0
        entropy = expected_dirichlet_entropy(alpha_local)  # u_t

        # routing / boundary features
        phase_entropy = -(cur_phase_prob * cur_phase_prob.clamp_min(1e-8).log()).sum(dim=-1)
        ctx_weight = ctx_mask / ctx_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        ctx_phase_mean = (ctx_phase_prob_flat.view(bsz, l_ctx, -1) * ctx_weight.unsqueeze(-1)).sum(dim=1)
        phase_shift = (cur_phase_prob - ctx_phase_mean).abs().mean(dim=-1)

        e_local_prob = alpha_local / alpha_local.sum(dim=-1, keepdim=True)
        topk_val, topk_idx = e_local_prob.topk(k=min(self.candidate_topk, self.num_classes), dim=-1)
        margin = topk_val[:, 0] - topk_val[:, 1] if topk_val.size(1) > 1 else topk_val[:, 0]

        route_feat = torch.cat([cur_phase_feat, ctx_summary, cur_phase_prob,
                                phase_shift.unsqueeze(-1), phase_entropy.unsqueeze(-1), margin.unsqueeze(-1)], dim=-1)
        boundary_logit = self.boundary_head(route_feat).squeeze(-1)

        ctx_feat_seq_all = ctx_feat_flat.view(bsz, l_ctx, -1)
        threshold = self.bump_threshold.squeeze(0)

        # ----------------- System 2 -----------------
        if not is_inference:
            # Soft gate on the stop-gradient uncertainty; System 2 runs on every window that has
            # history so that it receives a learning signal regardless of the gate.
            gate_soft = torch.sigmoid((entropy.detach() - threshold) * self.gate_temp)
            gate_train = (gate_soft * has_ctx.float()).unsqueeze(-1)

            logit_delta = torch.zeros_like(logit_local)
            if bool(has_ctx.any()):
                delta, _ = self._system2_delta(
                    cur_feat[has_ctx], ctx_feat_seq_all[has_ctx], cur_phase_feat[has_ctx], ctx_mask[has_ctx], topk_idx[has_ctx])
                logit_delta[has_ctx] = delta

            alpha_final = F.softplus(logit_local + gate_train * logit_delta) + 1.0
            # auxiliary branch: System-2 correction supervised on all windows, independent of the gate
            alpha_sys2_aux = F.softplus(logit_local.detach() + logit_delta) + 1.0

            return {
                'alpha_local': alpha_local,
                'alpha_final': alpha_final,
                'alpha_sys2_aux': alpha_sys2_aux,
                'entropy': entropy,
                'boundary_logit': boundary_logit,
                'cur_phase_logits': cur_phase_logits,
                'gate_rate_valid': gate_soft[has_ctx].mean() if bool(has_ctx.any()) else torch.tensor(0.0, device=logit_local.device),
                'active_mask': (gate_soft > 0.5) & has_ctx,
                'has_ctx': has_ctx,
            }

        # ----------------- inference: hard trigger -----------------
        logit_final = logit_local.clone()
        hit_mask = (entropy > threshold) & has_ctx
        trigger_count = int(hit_mask.sum().item())
        attn_weights_all = torch.zeros(bsz, l_ctx, device=x_cur.device)
        if trigger_count > 0:
            delta, attn_w = self._system2_delta(
                cur_feat[hit_mask], ctx_feat_seq_all[hit_mask], cur_phase_feat[hit_mask], ctx_mask[hit_mask], topk_idx[hit_mask])
            attn_weights_all[hit_mask, :attn_w.size(1)] = attn_w
            gate_infer = torch.sigmoid((entropy[hit_mask] - threshold) * self.gate_temp)
            logit_final[hit_mask] += gate_infer.unsqueeze(-1) * delta

        alpha_final = F.softplus(logit_final) + 1.0
        prob_final = alpha_final / alpha_final.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return {
            'prob_final': prob_final,
            'entropy': entropy,
            'trigger_rate': float(trigger_count / bsz) if bsz > 0 else 0.0,
            'attn_weights': attn_weights_all,
            'boundary_prob': torch.sigmoid(boundary_logit),
        }
