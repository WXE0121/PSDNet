"""Command-line entry point for PSDNet.

Train (10 seeds by default, mean +/- std of the best test Acc / weighted-F1 is reported):
    python main.py --dataset_name HAPT --dataset_root ./data

Single quick run:
    python main.py --dataset_name HAPT --seeds 1

Evaluate a checkpoint:
    python main.py --mode test --dataset_name HAPT --ckpt_path runs/<run>/seed1/best.pt
"""
import argparse

from psdnet.engine import run_multi_seed, test


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='PSDNet: Phase-Script Deliberation Network for sliding-window HAR',
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--mode', type=str, default='train', choices=['train', 'test'])

    # data
    p.add_argument('--dataset_root', type=str, default='./data', help='folder containing <dataset_name>/{train,test}_{x,y}.npy')
    p.add_argument('--dataset_name', type=str, default='HAPT')
    p.add_argument('--train_x', type=str, default=None, help='override: path to train windows (N, T, C)')
    p.add_argument('--train_y', type=str, default=None, help='override: path to train labels (N,)')
    p.add_argument('--test_x', type=str, default=None)
    p.add_argument('--test_y', type=str, default=None)
    p.add_argument('--train_seq', type=str, default=None, help='optional per-window recording ids (N,); auto-detected as train_seq.npy')
    p.add_argument('--test_seq', type=str, default=None)
    p.add_argument('--seq_fallback', type=str, default='contiguous', choices=['contiguous', 'unique'],
                   help='how windows are grouped into recordings when no sequence ids are available')
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--pin_memory', action='store_true')

    # optimisation
    p.add_argument('--epochs', type=int, default=120)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--seeds', type=str, default='1,2,3,4,5,6,7,8,9,10', help='comma-separated seeds to train')
    p.add_argument('--cpu', action='store_true')

    # model
    p.add_argument('--feat_dim', type=int, default=256, help='window feature dimension d')
    p.add_argument('--context_len', type=int, default=10, help='L: number of previous windows used as history')
    p.add_argument('--num_primitives', type=int, default=8, help='M: number of latent phase primitives')
    p.add_argument('--script_len', type=int, default=4, help='R: number of phase slots per class script')
    p.add_argument('--candidate_topk', type=int, default=3, help='Kc: candidate classes refined by System 2')
    p.add_argument('--gate_temp', type=float, default=10.0, help='temperature of the sigmoid gate')
    p.add_argument('--bump_init', type=float, default=-1.0, help='initial uncertainty threshold; <0 uses 0.5*ln(num_classes)')

    # loss
    p.add_argument('--budget_target', type=float, default=0.25, help='rho: trigger rate allowed without penalty')
    p.add_argument('--budget_loss_weight', type=float, default=5.0)
    p.add_argument('--kl_scale', type=float, default=0.09, help='weight of the EDL KL regulariser')

    # output
    p.add_argument('--output_root', type=str, default='./runs')
    p.add_argument('--run_name', type=str, default=None)
    p.add_argument('--save_path', type=str, default=None)
    p.add_argument('--ckpt_path', type=str, default=None, help='checkpoint to evaluate in test mode')
    return p


if __name__ == '__main__':
    args = build_parser().parse_args()
    if args.mode == 'train':
        run_multi_seed(args)
    else:
        test(args)
