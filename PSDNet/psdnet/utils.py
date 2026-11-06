import json
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def flatten_labels(y: np.ndarray) -> np.ndarray:
    """Accept either integer labels of shape (N,) or one-hot labels of shape (N, C)."""
    if y.ndim > 1:
        return np.argmax(y, axis=1).astype(np.int64)
    return y.astype(np.int64)


def parse_int_list(text: str, name: str = 'value') -> List[int]:
    vals = [int(s) for s in str(text).split(',') if s.strip()]
    if len(vals) == 0:
        raise ValueError(f'{name} cannot be empty')
    return vals


def append_history(path: Path, payload: Dict[str, Any]) -> None:
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(payload, ensure_ascii=False) + '\n')


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
