"""Window encoder that maps a single sensor window (B, 1, T, C) to a feature vector (B, feat_dim)."""
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

# CNN configuration per dataset (looked up case-insensitively, unknown datasets use the first entry).
# UniMiB-SHAR is single-sensor 3-channel data and uses a wider network with a 2-wide channel kernel.
_SMALL = {'channels': (64, 128, 256), 'kernel': 6, 'kernel_widths': None}
_LARGE = {'channels': (128, 256, 384), 'kernel': 7, 'kernel_widths': (1, 1, 2)}
BACKBONE_CONFIGS: Dict[str, Dict[str, Any]] = {
    'uci': _SMALL,
    'usc': _SMALL,
    'pamap2': _SMALL,
    'dsads': _SMALL,
    'hapt': _SMALL,
    'unimib': _LARGE,
    'motionsense': _SMALL,
    'mhealth': _SMALL,
}


def get_backbone_cfg(dataset_name: str) -> Dict[str, Any]:
    return BACKBONE_CONFIGS.get(str(dataset_name).lower(), _SMALL)


class Conv2dBackbone(nn.Module):
    """Three-layer 2-D CNN over (time, channel), projected to ``feat_dim`` and mean-pooled over time.

    ``nn.LazyLinear`` infers its input size on the first forward pass, so run one dummy batch
    through the model before creating the optimizer / loading a checkpoint.
    """

    def __init__(self, feat_dim: int, channels: Tuple[int, int, int], kernel: int, kernel_widths: Optional[Tuple[int, int, int]]):
        super().__init__()
        c1, c2, c3 = channels
        kw = kernel_widths if kernel_widths is not None else (1, 1, 1)
        p_h = (kernel - 1) // 2
        p_w = [max((k - 1) // 2, 0) for k in kw]
        self.conv = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=(kernel, kw[0]), stride=(2, 1), padding=(p_h, p_w[0])),
            nn.BatchNorm2d(c1), nn.ReLU(inplace=True),
            nn.Conv2d(c1, c2, kernel_size=(kernel, kw[1]), stride=(2, 1), padding=(p_h, p_w[1])),
            nn.BatchNorm2d(c2), nn.ReLU(inplace=True),
            nn.Conv2d(c2, c3, kernel_size=(kernel, kw[2]), stride=(2, 1), padding=(p_h, p_w[2])),
            nn.BatchNorm2d(c3), nn.ReLU(inplace=True),
        )
        self.proj = nn.LazyLinear(feat_dim)
        self.out_dim = feat_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        B, ch, Tp, Cp = h.shape
        tokens = self.proj(h.permute(0, 2, 1, 3).reshape(B, Tp, ch * Cp))
        return tokens.mean(dim=1)


def build_backbone(feat_dim: int, dataset_name: str) -> Conv2dBackbone:
    cfg = get_backbone_cfg(dataset_name)
    return Conv2dBackbone(feat_dim, cfg['channels'], cfg['kernel'], cfg['kernel_widths'])
