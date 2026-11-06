# PSDNet: Phase-Script Deliberation Network

PyTorch implementation of **When a Window Is Not an Action: Selective Phase-Script Deliberation
for Sliding-Window Human Activity Recognition**.

PSDNet treats a sliding window as a *phase* of an activity rather than the activity itself.
A fast **System 1** describes each window as a mixture of latent phase primitives, aligns it with
class-specific scripts and a short causal history, and outputs Dirichlet evidence. When that
evidence is uncertain, a **System 2** step retrieves targeted evidence from the history and refines
the top candidate classes, so the extra computation is spent only on ambiguous windows.

```
PSDNet/
├── main.py              train / test entry point
└── psdnet/
    ├── data.py          windows + causal history + transition labels
    ├── backbones.py     2-D CNN window encoder
    ├── modules.py       phase/script bank, history GRU, retrieval, candidate deliberation
    ├── model.py         PSDNet (System 1 + gated System 2)
    ├── losses.py        EDL objective with budget / boundary / diversity terms
    └── engine.py        training, evaluation, multi-seed protocol
```

## Setup

```bash
pip install -r requirements.txt   # torch>=2.0, numpy, scikit-learn
```

## Data format

Place pre-windowed data under `data/<dataset_name>/`:

| File | Shape | Description |
|------|-------|-------------|
| `train_x.npy`, `test_x.npy` | `(N, T, C)` float32 | N windows of T time steps and C channels |
| `train_y.npy`, `test_y.npy` | `(N,)` int | activity label of each window |
| `train_seq.npy`, `test_seq.npy` (optional) | `(N,)` int | recording id; history never crosses recordings |

Windows must be stored in temporal order within each recording: the history of window *t* is the
`L` windows that precede it in the file. Without `*_seq.npy` the whole file is treated as one
ordered recording. Experiments in the paper use UCI-HAR, USC-HAD, PAMAP2, DSADS, HAPT, UniMiB-SHAR,
MotionSense and mHealth.

## Usage

```bash
# train (10 seeds by default; reports mean +/- std of the best test Acc / weighted-F1)
python main.py --dataset_name HAPT --dataset_root ./data

# single run
python main.py --dataset_name HAPT --seeds 1

# evaluate a checkpoint
python main.py --mode test --dataset_name HAPT --ckpt_path runs/<run>/seed1/best.pt
```

Logs (`history.jsonl`), checkpoints (`best.pt`) and the multi-seed `summary.json` are written to
`runs/<dataset>_psdnet_<timestamp>/`.

## Main hyper-parameters

| Flag | Meaning | Default |
|------|---------|---------|
| `--context_len` | L, number of history windows | 10 |
| `--num_primitives` | M, number of latent phase primitives | 8 |
| `--script_len` | R, phase slots per class script | 4 |
| `--candidate_topk` | Kc, candidates refined by System 2 | 3 |
| `--budget_target` | rho, trigger rate allowed without penalty | 0.25 |
| `--kl_scale` | weight of the EDL KL regulariser | 0.09 |
| `--gate_temp` | temperature of the sigmoid gate | 10 |

Training uses AdamW (lr 1e-3, weight decay 1e-4) with a cosine schedule, batch size 64 and 120
epochs.

## Citation

```bibtex
@inproceedings{psdnet,
  title     = {When a Window Is Not an Action: Selective Phase-Script Deliberation for Sliding-Window Human Activity Recognition},
  author    = {},
  booktitle = {},
  year      = {2026}
}
```

## License

MIT (see `LICENSE`).
