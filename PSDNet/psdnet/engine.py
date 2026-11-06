"""Training / evaluation loops and the multi-seed protocol."""
import copy
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, recall_score
from torch.utils.data import DataLoader

from .data import make_loaders
from .losses import PSDNetLoss
from .model import PSDNet
from .utils import append_history, parse_int_list, set_seed, write_json


@dataclass
class EvalResult:
    acc: float
    f1_weighted: float
    f1_macro: float
    recall_weighted: float
    loss: float
    trigger_rate: float
    boundary_auc_proxy: float


@dataclass
class RunPaths:
    run_dir: Path
    save_path: Path
    history_path: Path


def boundary_auc_proxy(score: np.ndarray, boundary_target: np.ndarray) -> float:
    """Rank-based AUC of ``score`` (System-1 uncertainty) for detecting transition windows."""
    pos = np.sort(score[boundary_target > 0.5])
    neg = np.sort(score[boundary_target <= 0.5])
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    below = np.searchsorted(neg, pos, side='left')
    ties = np.searchsorted(neg, pos, side='right') - below
    return float((below.sum() + 0.5 * ties.sum()) / (len(pos) * len(neg)))


def to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


@torch.no_grad()
def evaluate(model: PSDNet, loader: DataLoader, device: torch.device) -> EvalResult:
    model.eval()
    ys, preds, scores, bts = [], [], [], []
    total_loss, total_n, total_route = 0.0, 0, 0.0
    for batch in loader:
        batch = to_device(batch, device)
        n = int(batch['y'].size(0))
        out = model(batch, is_inference=True)
        prob = out['prob_final']
        total_loss += float(F.nll_loss((prob + 1e-8).log(), batch['y'], reduction='sum').item())
        total_n += n
        total_route += float(out['trigger_rate']) * n
        ys.append(batch['y'].cpu().numpy())
        preds.append(prob.argmax(dim=-1).cpu().numpy())
        scores.append(out['entropy'].cpu().numpy())
        bts.append(batch['boundary'].cpu().numpy())
    y, pred = np.concatenate(ys), np.concatenate(preds)
    return EvalResult(
        acc=float(accuracy_score(y, pred)),
        f1_weighted=float(f1_score(y, pred, average='weighted')),
        f1_macro=float(f1_score(y, pred, average='macro')),
        recall_weighted=float(recall_score(y, pred, average='weighted')),
        loss=float(total_loss / max(total_n, 1)),
        trigger_rate=float(total_route / max(total_n, 1)),
        boundary_auc_proxy=boundary_auc_proxy(np.concatenate(scores), np.concatenate(bts)),
    )


def prepare_run_paths(args) -> RunPaths:
    now_tag = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = args.run_name if args.run_name else f'{args.dataset_name}_psdnet_{now_tag}'
    run_dir = Path(args.output_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    save_path = Path(args.save_path) if args.save_path else run_dir / 'best.pt'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    return RunPaths(run_dir=run_dir, save_path=save_path, history_path=run_dir / 'history.jsonl')


def build_model_and_criterion(args, num_classes: int, device: torch.device) -> Tuple[PSDNet, PSDNetLoss]:
    model = PSDNet(
        num_classes=num_classes, context_len=args.context_len,
        num_primitives=args.num_primitives, script_len=args.script_len,
        feat_dim=args.feat_dim, candidate_topk=args.candidate_topk,
        gate_temp=args.gate_temp, bump_init=args.bump_init,
        dataset_name=args.dataset_name,
    ).to(device)
    criterion = PSDNetLoss(
        num_classes=num_classes, num_primitives=args.num_primitives,
        max_epochs=args.epochs, budget_target=args.budget_target,
        budget_loss_weight=args.budget_loss_weight, kl_scale=args.kl_scale,
    )
    return model, criterion


@torch.no_grad()
def materialize_lazy_layers(model: PSDNet, loader: DataLoader, device: torch.device, is_inference: bool) -> None:
    """Run one batch so that ``nn.LazyLinear`` shapes are known before building the optimizer / loading weights."""
    model.eval()
    model(to_device(next(iter(loader)), device), is_inference=is_inference)


def train_one_epoch(model, criterion, loader, optimizer, device, args, epoch: int):
    model.train()
    epoch_loss, epoch_n = 0.0, 0
    track: Dict[str, float] = {}
    all_pred: List[np.ndarray] = []
    all_y: List[np.ndarray] = []
    for batch in loader:
        batch = to_device(batch, device)
        out = model(batch, is_inference=False)
        loss, stats = criterion(out, batch, current_epoch=epoch - 1)
        stats['threshold'] = float(model.bump_threshold.detach().squeeze(0).item())
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        bs = int(batch['y'].size(0))
        epoch_loss += float(loss.item()) * bs
        epoch_n += bs
        all_pred.append(out['alpha_final'].argmax(dim=-1).detach().cpu().numpy())
        all_y.append(batch['y'].detach().cpu().numpy())
        for k, v in stats.items():
            track[k] = track.get(k, 0.0) + float(v) * bs
    mean_loss = float(epoch_loss / max(epoch_n, 1))
    mean_stats = {k: v / max(epoch_n, 1) for k, v in track.items()}
    y, pred = np.concatenate(all_y), np.concatenate(all_pred)
    return mean_loss, mean_stats, float(accuracy_score(y, pred)), float(f1_score(y, pred, average='weighted'))


def train(args) -> Tuple[RunPaths, float, float]:
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    run_paths = prepare_run_paths(args)
    train_loader, test_loader, num_classes = make_loaders(args)
    model, criterion = build_model_and_criterion(args, num_classes, device)
    materialize_lazy_layers(model, train_loader, device, is_inference=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    write_json(run_paths.run_dir / 'args.json', vars(args))

    best_metric, best_acc = -1.0, -1.0
    print(f'Device: {device} | Dataset: {args.dataset_name} | Classes: {num_classes} | Seed: {args.seed}')
    print(f'RunDir: {run_paths.run_dir}')
    print(f'Params: total={sum(p.numel() for p in model.parameters()):,}')
    for epoch in range(1, args.epochs + 1):
        train_loss, train_stats, train_acc, train_f1 = train_one_epoch(model, criterion, train_loader, optimizer, device, args, epoch)
        scheduler.step()
        res = evaluate(model, test_loader, device)
        append_history(run_paths.history_path, {
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'train_f1_weighted': train_f1,
            'train_stats': train_stats,
            'eval': asdict(res),
            'lr': float(optimizer.param_groups[0]['lr']),
        })
        if res.f1_weighted > best_metric:
            best_metric, best_acc = res.f1_weighted, res.acc
            torch.save({'model_state': model.state_dict(), 'args': vars(args),
                        'num_classes': num_classes, 'best_f1_weighted': best_metric}, str(run_paths.save_path))
        print(f'Epoch {epoch:03d} | loss={train_loss:.5f} | train_acc={train_acc:.5f} '
              f'| test_acc={res.acc:.5f} | test_f1={res.f1_weighted:.5f} '
              f"| gate={train_stats['gate_rate']:.4f} | thresh={train_stats['threshold']:.4f} "
              f"| entropy={train_stats['entropy_mean']:.4f} | trigger={res.trigger_rate:.4f}")
    print(f'Best weighted F1: {best_metric:.5f} | Best Acc: {best_acc:.5f}')
    print(f'Best checkpoint saved to: {run_paths.save_path}')
    return run_paths, best_acc, best_metric


def test(args) -> EvalResult:
    if args.ckpt_path is None:
        raise ValueError('--ckpt_path is required in test mode')
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    _, test_loader, num_classes = make_loaders(args)
    model, _ = build_model_and_criterion(args, num_classes, device)
    materialize_lazy_layers(model, test_loader, device, is_inference=True)
    ckpt = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    res = evaluate(model, test_loader, device)
    print(f'Test | acc={res.acc:.5f} | f1w={res.f1_weighted:.5f} | f1m={res.f1_macro:.5f} '
          f'| recall={res.recall_weighted:.5f} | loss={res.loss:.5f} | trigger={res.trigger_rate:.5f} '
          f'| boundary_auc={res.boundary_auc_proxy:.5f}')
    return res


def run_multi_seed(args) -> Path:
    """Train one run per seed in ``args.seeds`` and report mean +/- std of the best test Acc / weighted F1."""
    seeds = parse_int_list(args.seeds, 'seeds')
    now_tag = datetime.now().strftime('%Y%m%d_%H%M%S')
    shared_name = f'{args.dataset_name}_psdnet_{now_tag}'
    shared_run_dir = Path(args.output_root) / shared_name
    shared_run_dir.mkdir(parents=True, exist_ok=True)
    shared_history = shared_run_dir / 'history.jsonl'

    all_acc, all_f1 = [], []
    for s in seeds:
        print(f'\n{"=" * 50}\n  SEED {s} ({len(all_acc) + 1}/{len(seeds)})\n{"=" * 50}')
        run_args = copy.deepcopy(args)
        run_args.seed = int(s)
        run_args.run_name = f'{shared_name}/seed{s}'
        run_args.save_path = None
        run_paths, best_acc, best_f1 = train(run_args)
        with run_paths.history_path.open('r', encoding='utf-8') as fin, shared_history.open('a', encoding='utf-8') as fout:
            for line in fin:
                if line.strip():
                    rec = json.loads(line)
                    rec['seed'] = s
                    fout.write(json.dumps(rec, ensure_ascii=False) + '\n')
        all_acc.append(best_acc)
        all_f1.append(best_f1)
        print(f'[Seed {s}] Best Acc={best_acc:.5f} | Best F1={best_f1:.5f}')

    acc_mean, acc_std = float(np.mean(all_acc)), float(np.std(all_acc))
    f1_mean, f1_std = float(np.mean(all_f1)), float(np.std(all_f1))
    write_json(shared_run_dir / 'summary.json', {
        'dataset': args.dataset_name, 'seeds': seeds,
        'acc_mean': acc_mean, 'acc_std': acc_std, 'f1_mean': f1_mean, 'f1_std': f1_std,
        'per_seed': [{'seed': s, 'acc': all_acc[i], 'f1': all_f1[i]} for i, s in enumerate(seeds)],
        'args': vars(args),
    })
    print(f'\n{"=" * 50}\n  FINAL RESULTS ({args.dataset_name}, {len(seeds)} seeds)\n{"=" * 50}')
    for i, s in enumerate(seeds):
        print(f'  Seed {s:2d}: Acc={all_acc[i]:.5f} | F1={all_f1[i]:.5f}')
    print(f'{"=" * 50}')
    print(f'  Acc = {acc_mean * 100:.2f}% +/- {acc_std * 100:.2f}%')
    print(f'  F1  = {f1_mean * 100:.2f}% +/- {f1_std * 100:.2f}%')
    print(f'{"=" * 50}')
    print(f'All logs & checkpoints saved to: {shared_run_dir}')
    return shared_run_dir
