"""
train_control_fixed.py — Train alpha on ONE control system with varying initial states.

The system dynamics (A, B), costs (Q, R, QN), and bounds (xmin, xmax, umin, umax) are
fixed by a single ``--dynamics_seed``.  The dataset consists of ``n_train + n_val``
instances that differ only in the initial state x0.

Usage:
    python -m learned_osqp.train_control_fixed \
        --nx 80 --dynamics_seed 0 --epochs 200 --batch 16 \
        --alpha_mode scalar --model_type gru --loss scaled_residual \
        --normalize_features --device cuda
"""

from __future__ import annotations

import warnings
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore", category=UserWarning, module="cvxpy")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from learned_osqp.config import Config
from learned_osqp.model import PerRowAlphaNet, PerRowGRUNet, ScalarAlphaNet, ScalarGRUNet
from learned_osqp.data import (
    _build_instance_from_qp, QPDataset, collate_fn, dataset_path,
)
from learned_osqp.train import (
    seed_everything, compute_feature_stats, train_epoch, val_epoch,
    _baseline_stats_from_loader, _setup_logger,
)
from problem_classes.control import ControlExample


# --------------------------------------------------------------------------- #
# Dataset generation: one system, many x0
# --------------------------------------------------------------------------- #

def generate_control_fixed_dataset(
    nx: int,
    dynamics_seed: int,
    n_instances: int,
    cfg: Config,
    verbose: bool = True,
) -> QPDataset:
    """Generate *n_instances* control QPs that share one LTI system.

    1. Build a single ``ControlExample(nx, seed=dynamics_seed)`` to fix
       (A, B, Q, R, QN, bounds).
    2. For each instance *i*, sample a new x0 with ``np.random.default_rng(i)``
       and call ``update_x0``.
    3. Convert to a torch instance via ``_build_instance_from_qp``.
    """
    if verbose:
        print(f"Building base control system  nx={nx}  dynamics_seed={dynamics_seed}")
    base = ControlExample(nx, seed=dynamics_seed)

    instances: list[dict] = []
    n_failed = 0

    for i in range(n_instances):
        if verbose and i % 50 == 0:
            print(f"  Generating instance {i}/{n_instances} ...", flush=True)

        # Sample a new x0 within [0.5*xmin, 0.5*xmax]
        rng = np.random.default_rng(i)
        raw = rng.random(base.nx)
        min_x0 = 0.5 * base.xmin
        max_x0 = 0.5 * base.xmax
        x0_new = min_x0 + raw * (max_x0 - min_x0)

        base.update_x0(x0_new)
        inst = _build_instance_from_qp(base, cfg)
        if inst is None:
            n_failed += 1
        else:
            instances.append(inst)

    if verbose:
        print(f"  Generated {len(instances)}/{n_instances} instances ({n_failed} failed)")
    return QPDataset(instances)


# --------------------------------------------------------------------------- #
# Cache path helper
# --------------------------------------------------------------------------- #

def _cache_path(nx: int, dynamics_seed: int, cfg: Config) -> Path:
    return (
        Path(cfg.data_dir)
        / f"control_fixed_nx{nx}_dseed{dynamics_seed}"
          f"_p{cfg.precision}_d{cfg.dtype}_adaptive_rho={cfg.adaptive_rho}.pt"
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    import argparse
    from torch.utils.data import random_split

    parser = argparse.ArgumentParser(
        description='Train alpha on ONE control system with varying x0')
    parser.add_argument('--nx', type=int, default=80,
                        help='State dimension of the control system')
    parser.add_argument('--dynamics_seed', type=int, default=0,
                        help='Seed that fixes A, B, Q, R, QN, bounds')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--T', type=int, default=10, help='steps per stage')
    parser.add_argument('--stages', type=int, default=2000, help='max stages')
    parser.add_argument('--n_train', type=int, default=160)
    parser.add_argument('--n_val', type=int, default=80)
    parser.add_argument('--regen', action='store_true',
                        help='regenerate dataset')
    parser.add_argument('--loss', type=str, default='log_convergence',
                        choices=['log_convergence', 'spectral_radius',
                                 'scaled_residual'],
                        help='loss function')
    parser.add_argument('--ckpt', type=str, default=None,
                        help='checkpoint path (.pt)')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'])
    parser.add_argument('--dtype', type=str, default='float64',
                        choices=['float64', 'float32'])
    parser.add_argument('--precision', type=str, default='low',
                        choices=['low', 'high'])
    parser.add_argument('--adaptive_rho',
                        type=lambda x: x.lower() != 'false',
                        default=True, metavar='BOOL')
    parser.add_argument('--alpha_mode', type=str, default='vector',
                        choices=['vector', 'scalar'])
    parser.add_argument('--model_type', type=str, default='mlp',
                        choices=['mlp', 'gru'])
    parser.add_argument('--normalize_features', action='store_true')
    args = parser.parse_args()

    seed_everything(42)

    cfg = Config(
        n_fixed=args.nx,
        n_epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        T=args.T,
        max_stages=args.stages,
        n_train=args.n_train,
        n_val=args.n_val,
        device=args.device,
        dtype=args.dtype,
        precision=args.precision,
        adaptive_rho=args.adaptive_rho,
        normalize_features=args.normalize_features,
        alpha_mode=args.alpha_mode,
        model_type=args.model_type,
        store_spectral_matrices=(args.loss == 'spectral_radius'),
        qp_types=['control'],
        qp_type_sizes={'control': args.nx},
        # data_dir='learned_osqp/data',
        data_dir='/data/engs-goulart/sedm7756/0319_feat_pri_dua_res_scaled_alpha_1.25_1.95',
    )

    device = torch.device(args.device)
    loss_type = args.loss

    # ---- checkpoint path ------------------------------------------------- #
    if args.ckpt is None:
        ckpt_name = (
            f"best_model_control_fixed_nx{args.nx}_dseed{args.dynamics_seed}"
            f"_precision={args.precision}"
            f"_adaptive_rho={args.adaptive_rho}"
            f"_alpha_mode={args.alpha_mode}"
            f"_model_type={args.model_type}"
            f"_loss={args.loss}"
        )
        # ckpt_path = Path(f"learned_osqp/checkpoints/{ckpt_name}.pt")
        ckpt_path = Path(f"learned_osqp/checkpoints/0319_feat_pri_dua_res_scaled_alpha_1.25_1.95/{ckpt_name}.pt")
    else:
        ckpt_path = Path(args.ckpt)

    log = _setup_logger(ckpt_path)

    alpha_mode = args.alpha_mode
    model_type = args.model_type
    if alpha_mode == 'scalar' and model_type == 'gru':
        model_name = 'ScalarGRUNet'
    elif alpha_mode == 'scalar':
        model_name = 'ScalarAlphaNet'
    elif model_type == 'gru':
        model_name = 'PerRowGRUNet'
    else:
        model_name = 'PerRowAlphaNet'
    log.info(f"Training {model_name}  [loss={loss_type}, alpha_mode={alpha_mode}, model_type={model_type}]")
    log.info(f"  control_fixed  nx={args.nx}  dynamics_seed={args.dynamics_seed}")
    log.info(f"  n_train={args.n_train}  n_val={args.n_val}  batch_size={cfg.batch_size}")
    log.info(f"  T={cfg.T} steps/stage, max_stages={cfg.max_stages}")
    log.info(f"  alpha range: [{cfg.alpha_min}, {cfg.alpha_max}]")
    log.info(f"  device={cfg.device}, dtype={cfg.dtype}, precision={cfg.precision}")
    log.info(f"  checkpoint: {ckpt_path}")

    # ---- dataset --------------------------------------------------------- #
    cache = _cache_path(args.nx, args.dynamics_seed, cfg)
    if args.regen and cache.exists():
        cache.unlink()
        print(f"Removed cached dataset: {cache}")

    if cache.exists():
        print(f"Loading dataset from {cache} ...")
        raw = torch.load(str(cache), weights_only=False)
        dataset = QPDataset(raw)
    else:
        cache.parent.mkdir(parents=True, exist_ok=True)
        dataset = generate_control_fixed_dataset(
            args.nx, args.dynamics_seed, args.n_train + args.n_val, cfg,
        )
        torch.save(dataset.instances, str(cache))
        print(f"Saved dataset to {cache}")

    n_val = min(args.n_val, len(dataset))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        collate_fn=collate_fn, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        collate_fn=collate_fn, drop_last=False,
    )
    log.info(f"  Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # ---- model ----------------------------------------------------------- #
    if alpha_mode == 'scalar':
        if model_type == 'gru':
            model = ScalarGRUNet(cfg).to(dtype=cfg.torch_dtype, device=device)
        else:
            model = ScalarAlphaNet(cfg).to(dtype=cfg.torch_dtype, device=device)
    else:
        if model_type == 'gru':
            model = PerRowGRUNet(cfg).to(dtype=cfg.torch_dtype, device=device)
        else:
            model = PerRowAlphaNet(cfg).to(dtype=cfg.torch_dtype, device=device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"  Model parameters: {n_params:,}")

    if cfg.normalize_features:
        log.info("  Computing feature normalization statistics ...")
        feat_mean, feat_std = compute_feature_stats(
            [train_loader], cfg, device, cfg.torch_dtype)
        model.set_feature_norm(feat_mean, feat_std)
        log.info(f"  Feature norm  mean=[{feat_mean.min():.3f}, {feat_mean.max():.3f}]  "
                 f"std=[{feat_std.min():.3f}, {feat_std.max():.3f}]")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.n_epochs, eta_min=cfg.lr * 0.01)

    # ---- baseline stats -------------------------------------------------- #
    bl_tr_im, bl_tr_is, bl_tr_rm, bl_tr_rs = _baseline_stats_from_loader(train_loader)
    bl_val_im, bl_val_is, bl_val_rm, bl_val_rs = _baseline_stats_from_loader(val_loader)
    log.info(
        f"Baseline (alpha=1.6)  "
        f"train_iters={bl_tr_im:.1f}±{bl_tr_is:.1f}  "
        f"val_iters={bl_val_im:.1f}±{bl_val_is:.1f}  "
        f"train_rho_updates={bl_tr_rm:.2f}±{bl_tr_rs:.2f}  "
        f"val_rho_updates={bl_val_rm:.2f}±{bl_val_rs:.2f}"
    )

    # ---- training loop --------------------------------------------------- #
    best_val_loss = float('inf')
    best_val_iters_m = float('inf')
    best_val_rho_m = float('inf')
    best_val_rho_iters_m = float('inf')
    ckpt_rho_path = ckpt_path.with_name(ckpt_path.stem + '_best_rho' + ckpt_path.suffix)
    t0 = time.time()

    for epoch in range(1, cfg.n_epochs + 1):
        t_ep = time.time()
        train_loss, tr_im, tr_is, tr_rm, tr_rs, tr_am, tr_as = train_epoch(
            model, train_loader, optimizer, cfg, device, loss_type)
        val_loss, val_im, val_is, val_rm, val_rs, val_am, val_as = val_epoch(
            model, val_loader, cfg, device, loss_type)
        scheduler.step()

        elapsed = time.time() - t_ep
        total_elapsed = time.time() - t0
        lr_current = scheduler.get_last_lr()[0]

        log.info(
            f"Epoch {epoch:3d}/{cfg.n_epochs}  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"train_iters={tr_im:.1f}±{tr_is:.1f}  "
            f"val_iters={val_im:.1f}±{val_is:.1f}  "
            f"train_alpha={tr_am:.3f}±{tr_as:.3f}  "
            f"val_alpha={val_am:.3f}±{val_as:.3f}  "
            f"train_rho_updates={tr_rm:.2f}±{tr_rs:.2f}  "
            f"val_rho_updates={val_rm:.2f}±{val_rs:.2f}  "
            f"lr={lr_current:.2e}  "
            f"ep={elapsed:.1f}s  total={total_elapsed / 60:.1f}min"
        )

        if val_im < best_val_iters_m:
            best_val_loss = val_loss
            best_val_iters_m = val_im
            torch.save(
                {'epoch': epoch, 'model_state': model.state_dict(),
                 'val_loss': best_val_loss, 'cfg': cfg,
                 'val_iters_m': val_im, 'val_iters_s': val_is,
                 'feat_norm_active': model.feat_norm_active},
                str(ckpt_path),
            )
            log.info(f"  -> saved checkpoint (val_iters_m={best_val_iters_m:.4f})")

        if (val_rm < best_val_rho_m
                or (val_rm == best_val_rho_m and val_im < best_val_rho_iters_m)):
            best_val_rho_m = val_rm
            best_val_rho_iters_m = val_im
            torch.save(
                {'epoch': epoch, 'model_state': model.state_dict(),
                 'val_loss': val_loss, 'cfg': cfg,
                 'val_iters_m': val_im, 'val_iters_s': val_is,
                 'val_rho_m': val_rm, 'val_rho_s': val_rs,
                 'feat_norm_active': model.feat_norm_active},
                str(ckpt_rho_path),
            )
            log.info(f"  -> saved rho checkpoint (val_rho_m={best_val_rho_m:.4f}, val_iters_m={best_val_rho_iters_m:.4f})")

    log.info(f"Training complete. Best val_iters_m={best_val_iters_m:.4f}")


if __name__ == '__main__':
    main()
