"""Sliding-window HAR dataset with causal history (previous windows of the same recording)."""
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .utils import flatten_labels


class HARPhaseDataset(Dataset):
    """Windowed HAR dataset that attaches the ``context_len`` preceding windows to every sample.

    Expected inputs:
        x_path   : ``.npy`` array of shape (N, T, C) -- N windows, T time steps, C sensor channels.
        y_path   : ``.npy`` array of shape (N,) with integer labels (one-hot (N, K) is also accepted).
        seq_path : optional ``.npy`` array of shape (N,) with a recording / sequence id per window.
                   History is only collected from windows that belong to the same sequence id and
                   appear earlier in the file, so no future information is leaked.
                   If omitted, ``seq_fallback`` decides how windows are grouped:
                   ``'contiguous'`` treats the whole file as one ordered recording, ``'unique'``
                   gives every window its own id (i.e. no history at all).

    Each item additionally carries a binary ``boundary`` target that is 1 when the label of the
    window differs from the label of its immediate predecessor (an activity transition).
    """

    def __init__(self, x_path: str, y_path: str, seq_path: Optional[str], context_len: int, seq_fallback: str = 'contiguous'):
        x = np.load(x_path).astype(np.float32)
        y = flatten_labels(np.load(y_path)).astype(np.int64)
        if seq_path is None:
            if seq_fallback == 'unique':
                seq = np.arange(x.shape[0], dtype=np.int64)
            else:
                seq = np.zeros((x.shape[0],), dtype=np.int64)
        else:
            seq = np.load(seq_path).astype(np.int64)
            if len(seq) != len(x):
                raise ValueError(f'sequence id length {len(seq)} does not match number of windows {len(x)}')
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)
        self.seq = torch.from_numpy(seq)
        self.context_len = context_len
        self.n, self.t, self.c = self.x.shape

        # prev_same[i] = index of the previous window of the same sequence (-1 if none)
        self.prev_same = torch.full((self.n,), -1, dtype=torch.long)
        last_seen: Dict[int, int] = {}
        for i in range(self.n):
            s = int(self.seq[i].item())
            if s in last_seen:
                self.prev_same[i] = last_seen[s]
            last_seen[s] = i

        self.boundary_target = torch.zeros((self.n,), dtype=torch.float32)
        for i in range(self.n):
            p = int(self.prev_same[i].item())
            if p >= 0 and int(self.y[p].item()) != int(self.y[i].item()):
                self.boundary_target[i] = 1.0

    def __len__(self) -> int:
        return self.n

    def _collect_context(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        xs, ms, ys = [], [], []
        cur = idx
        hist = []
        while len(hist) < self.context_len:
            prev_idx = int(self.prev_same[cur].item())
            if prev_idx < 0:
                break
            hist.append(prev_idx)
            cur = prev_idx
        hist = hist[::-1]  # oldest -> most recent
        pad = self.context_len - len(hist)
        for h in hist:
            xs.append(self.x[h].unsqueeze(0))
            ms.append(1.0)
            ys.append(int(self.y[h].item()))
        for _ in range(pad):
            xs.append(torch.zeros((1, self.t, self.c), dtype=self.x.dtype))
            ms.append(0.0)
            ys.append(-1)
        return torch.stack(xs, 0), torch.tensor(ms, dtype=torch.float32), torch.tensor(ys, dtype=torch.long)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ctx_x, ctx_mask, ctx_y = self._collect_context(idx)
        return {
            'x_cur': self.x[idx].unsqueeze(0),   # (1, T, C)
            'ctx_x': ctx_x,                      # (L, 1, T, C), zero padded on the left
            'ctx_mask': ctx_mask,                # (L,) 1 for real history, 0 for padding
            'ctx_y': ctx_y,                      # (L,) labels of history windows (-1 for padding)
            'y': self.y[idx],
            'seq': self.seq[idx],
            'idx': torch.tensor(idx, dtype=torch.long),
            'boundary': self.boundary_target[idx],
        }


def _find_npy(ds_dir: Path, stem: str) -> Optional[str]:
    """Return ``<ds_dir>/<stem>.npy``, also accepting an upper-cased suffix (``train_X.npy``)."""
    prefix, suffix = stem.rsplit('_', 1)
    for name in (f'{stem}.npy', f'{prefix}_{suffix.upper()}.npy'):
        p = ds_dir / name
        if p.exists():
            return str(p)
    return None


def resolve_dataset_paths(args) -> None:
    """Fill in any of ``train_x/train_y/test_x/test_y/train_seq/test_seq`` that were not given explicitly.

    Files are looked up in ``<dataset_root>/<dataset_name>/`` (falling back to the lower-cased
    dataset name).  ``train_seq.npy`` / ``test_seq.npy`` are optional and only used when present.
    """
    needed = [k for k in ('train_x', 'train_y', 'test_x', 'test_y') if getattr(args, k) is None]
    if not needed and args.train_seq is not None and args.test_seq is not None:
        return
    ds_dir = Path(args.dataset_root) / str(args.dataset_name)
    if not ds_dir.exists():
        ds_dir = Path(args.dataset_root) / str(args.dataset_name).lower()
    for key in needed:
        found = _find_npy(ds_dir, key)
        if found is None:
            raise FileNotFoundError(
                f'Could not find {key}.npy under {ds_dir}. Pass --{key} explicitly or '
                f'place the processed dataset under {Path(args.dataset_root) / str(args.dataset_name)}'
            )
        setattr(args, key, found)
    for key in ('train_seq', 'test_seq'):
        if getattr(args, key) is None:
            setattr(args, key, _find_npy(ds_dir, key))


def make_loaders(args) -> Tuple[DataLoader, DataLoader, int]:
    resolve_dataset_paths(args)
    train_ds = HARPhaseDataset(args.train_x, args.train_y, args.train_seq, args.context_len, seq_fallback=args.seq_fallback)
    test_ds = HARPhaseDataset(args.test_x, args.test_y, args.test_seq, args.context_len, seq_fallback=args.seq_fallback)
    num_classes = int(train_ds.y.max().item() + 1)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=args.pin_memory)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory)
    return train_loader, test_loader, num_classes
