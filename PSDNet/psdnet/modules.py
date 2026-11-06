"""Building blocks of PSDNet: history summary, phase/script bank and the System-2 components."""
import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedGRU(nn.Module):
    """GRU over a left-padded history; returns the final hidden state (zeros when there is no history)."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        lengths = mask.sum(dim=1).long()
        out = torch.zeros((x.size(0), self.hidden_dim), dtype=x.dtype, device=x.device)
        valid = lengths > 0
        if valid.any():
            packed = nn.utils.rnn.pack_padded_sequence(x[valid], lengths[valid].cpu(), batch_first=True, enforce_sorted=False)
            _, h = self.gru(packed)
            out[valid] = h[-1]
        return out


class PhaseScriptBank(nn.Module):
    """Latent phase primitives P (M x d) and class-specific scripts S_k (K x R x d).

    A window is described by a soft mixture over the M phase primitives; its phase feature is the
    mixture-weighted sum of primitive embeddings.  Every class owns a script of R ordered phase
    slots.  ``all_class_script_scores`` soft-aligns the current phase feature (and the history
    summary) to each class script and returns one compatibility score per class.
    """

    def __init__(self, num_classes: int, num_primitives: int, primitive_dim: int, script_len: int):
        super().__init__()
        self.primitive_embed = nn.Parameter(torch.randn(num_primitives, primitive_dim) * 0.02)
        self.class_script = nn.Parameter(torch.randn(num_classes, script_len, primitive_dim) * 0.02)
        self.class_transition = nn.Parameter(torch.randn(num_classes, script_len, script_len) * 0.02)
        self.scale = 1.0 / math.sqrt(primitive_dim)

    def primitive_feature(self, phase_prob: torch.Tensor) -> torch.Tensor:
        return phase_prob @ self.primitive_embed

    def all_class_script_scores(self, local_phase_feat: torch.Tensor, ctx_summary: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scores_local = torch.einsum('bd,kmd->bkm', local_phase_feat, self.class_script) * self.scale
        scores_ctx = torch.einsum('bd,kmd->bkm', ctx_summary, self.class_script) * self.scale

        transition_prior = self.class_transition.mean(dim=1)
        transition_prior = F.softmax(transition_prior, dim=-1).unsqueeze(0)

        attn_logits = scores_local + 0.5 * scores_ctx + 0.2 * transition_prior.log().clamp(min=-20)
        attn = F.softmax(attn_logits, dim=-1)  # (B, K, R) alignment of the window to each script slot

        attn_summary = torch.einsum('bkm,kmd->bkd', attn, self.class_script)
        script_score = (attn_summary * local_phase_feat.unsqueeze(1)).sum(dim=-1) * self.scale
        script_score = script_score + 0.35 * (attn_summary * ctx_summary.unsqueeze(1)).sum(dim=-1) * self.scale
        return script_score, attn_summary


class CandidateDeliberation(nn.Module):
    """Candidate-specific refinement: scores each of the top-Kc candidate classes given the evidence."""

    def __init__(self, num_classes: int, feat_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.class_embed = nn.Embedding(num_classes, feat_dim)
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim * 4, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, fused_feat: torch.Tensor, local_phase_feat: torch.Tensor, ctx_summary: torch.Tensor, candidate_ids: torch.Tensor) -> torch.Tensor:
        bsz, k_cand = candidate_ids.shape
        cand_emb = self.class_embed(candidate_ids)
        fused = fused_feat.unsqueeze(1).expand(-1, k_cand, -1)
        local = local_phase_feat.unsqueeze(1).expand(-1, k_cand, -1)
        ctx = ctx_summary.unsqueeze(1).expand(-1, k_cand, -1)
        inp = torch.cat([fused, local, ctx, cand_emb], dim=-1)
        return self.mlp(inp).squeeze(-1)  # (B, Kc) unconstrained logit corrections


class HistoryRetriever(nn.Module):
    """Targeted retrieval over the history: the current window queries GRU-encoded past windows."""

    def __init__(self, feat_dim: int):
        super().__init__()
        self.hist_encoder = nn.GRU(feat_dim, feat_dim, batch_first=True)
        self.attn = nn.MultiheadAttention(embed_dim=feat_dim, num_heads=4, batch_first=True, dropout=0.1)
        self.norm = nn.LayerNorm(feat_dim)

    def forward(self, cur_feat: torch.Tensor, ctx_seq: torch.Tensor, lengths: torch.Tensor, key_padding_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        packed = nn.utils.rnn.pack_padded_sequence(ctx_seq, lengths.cpu(), batch_first=True, enforce_sorted=False)
        gru_out, _ = self.hist_encoder(packed)
        hist_states, _ = nn.utils.rnn.pad_packed_sequence(gru_out, batch_first=True, total_length=ctx_seq.size(1))
        q = cur_feat.unsqueeze(1)
        attn_out, attn_weights = self.attn(query=q, key=hist_states, value=hist_states, key_padding_mask=key_padding_mask)
        retrieved_ctx = self.norm(cur_feat + attn_out.squeeze(1))
        return retrieved_ctx, attn_weights.squeeze(1)
