"""
train.py — Multi-stage training loop for learned per-row OSQP alpha.

Training algorithm:
  For each batch of QP instances:
    1. Cold-start OSQP state (x=z=y=0).
    2. For each stage (up to max_stages):
       a. Compute per-row features from current state.
       b. Forward pass through PerRowAlphaNet → alpha_z (B, m).
       c. Run T=10 differentiable OSQP iterations with fixed alpha_z.
       d. Compute log-convergence-ratio loss (detach denominator).
       e. Mask out already-converged instances.
       f. Accumulate loss.
       g. DETACH state before next stage (critical — prevents full history backprop).
       h. Run adaptive rho update (no_grad) and refactorize KKT if needed.
    3. Backprop through accumulated loss, clip gradients, step optimizer.

Key design choices:
  - State is detached between stages → O(T) memory per backward pass.
  - KKT factors are always computed under no_grad → gradients flow through
    the solve RHS only (via x, z, y → alpha_z chain).
  - AdamW + cosine annealing LR schedule.
  - Gradient clipping at max_norm=1.0.
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
import torch.nn as nn

# Suppress cvxpy deprecation warning about matrix multiplication
warnings.filterwarnings("ignore", category=UserWarning, module="cvxpy")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from learned_osqp.config import Config
from learned_osqp.model import PerRowAlphaNet
from learned_osqp.data import make_dataloaders
from learned_osqp.features import compute_per_row_features
from learned_osqp.osqp_torch import (
    factorize_kkt,
    rollout_T_steps,
    maybe_update_rho,
)
from learned_osqp.loss import log_convergence_loss, convergence_mask, spectral_radius_loss, scaled_residual_loss


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _setup_logger(log_path: Path) -> logging.Logger:
    """Logger writing to both stdout and a .log file next to the checkpoint."""
    logger = logging.getLogger('train_' + log_path.stem)
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger
    fmt = logging.Formatter('%(asctime)s  %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(str(log_path.with_suffix('.log')))
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def _baseline_stats_from_loader(loader) -> tuple[float, float, float, float]:
    """
    Collect precomputed baseline_iters / baseline_rho_updates from the dataset.
    Returns (iters_mean, iters_std, rho_mean, rho_std).
    """
    all_iters: list[float] = []
    all_rho:   list[float] = []
    for batch in loader:
        if 'baseline_iters' in batch:
            all_iters.extend(batch['baseline_iters'].tolist())
            all_rho.extend(batch['baseline_rho_updates'].tolist())
    if not all_iters:
        return float('nan'), float('nan'), float('nan'), float('nan')
    ia = np.array(all_iters, dtype=float)
    ra = np.array(all_rho,   dtype=float)
    return float(ia.mean()), float(ia.std()), float(ra.mean()), float(ra.std())


def _osqp_converged_batched(
    P: torch.Tensor,      # (B, n, n)  SCALED
    A: torch.Tensor,      # (B, m, n)  SCALED
    q: torch.Tensor,      # (B, n)     SCALED
    x: torch.Tensor,      # (B, n)     scaled iterate
    z: torch.Tensor,      # (B, m)     scaled iterate
    y: torch.Tensor,      # (B, m)     scaled iterate
    d_inv: torch.Tensor,  # (B, n)     D^{-1} diagonal
    e_inv: torch.Tensor,  # (B, m)     E^{-1} diagonal
    c_inv: torch.Tensor,  # (B,)       1 / c_scale
    cfg: 'Config',
) -> torch.Tensor:
    """
    OSQP SOLVED criterion with unscaling (batched, no_grad).

    All inputs are in the SCALED problem space. Residuals are unscaled before
    comparison to match _osqp.py compute_pri_res / compute_dua_res with
    scaled_termination=False.

    Unscaled residuals:
        pri_res = E^{-1} * (A_sc @ x_sc - z_sc)
        dua_res = c^{-1} * D^{-1} * (P_sc @ x_sc + q_sc + A_sc^T @ y_sc)

    Tolerances (in unscaled space):
        eps_pri = eps_abs + eps_rel * max(||A*x_orig||_inf, ||z_orig||_inf)
        eps_dua = eps_abs + eps_rel * max(||A^T*y_orig||_inf, ||P*x_orig||_inf, ||q||_inf)
    """
    Ax  = torch.bmm(A, x.unsqueeze(-1)).squeeze(-1)                     # (B, m)
    ATy = torch.bmm(A.transpose(1, 2), y.unsqueeze(-1)).squeeze(-1)     # (B, n)
    Px  = torch.bmm(P, x.unsqueeze(-1)).squeeze(-1)                     # (B, n)

    c_inv_u = c_inv.unsqueeze(1)                                         # (B, 1)

    # Unscaled residuals
    pri_res_unc = e_inv * (Ax - z)                                       # (B, m)
    dua_res_unc = c_inv_u * d_inv * (Px + q + ATy)                      # (B, n)

    # Tolerances in original (unscaled) space
    eps_pri = cfg.eps_abs + cfg.eps_rel * torch.maximum(
        (e_inv * Ax).abs().amax(dim=1),
        (e_inv * z).abs().amax(dim=1),
    )                                                                     # (B,)
    eps_dua = cfg.eps_abs + cfg.eps_rel * torch.stack([
        (c_inv_u * d_inv * ATy).abs().amax(dim=1),
        (c_inv_u * d_inv * Px).abs().amax(dim=1),
        (c_inv_u * d_inv * q).abs().amax(dim=1),
    ], dim=1).amax(dim=1)                                                 # (B,)

    return (pri_res_unc.abs().amax(dim=1) < eps_pri) & \
           (dua_res_unc.abs().amax(dim=1) < eps_dua)                     # (B,)


# --------------------------------------------------------------------------- #
# Training epoch
# --------------------------------------------------------------------------- #

def train_epoch(
    model: PerRowAlphaNet,
    loader,
    optimizer: torch.optim.Optimizer,
    cfg: Config,
    device: torch.device,
    loss_type: str = "log_convergence",
) -> tuple[float, float, float, float, float]:
    """Run one training epoch.

    Returns:
        (mean_loss, iters_mean, iters_std, rho_updates_mean, rho_updates_std)
        iters: OSQP steps to convergence per instance (capped at max_stages*T).
        rho_updates: number of rho updates per instance across all stages.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0
    all_iters: list[float] = []
    all_rho: list[float] = []

    dtype = cfg.torch_dtype

    for batch in loader:
        batch = {
            k: (v.to(device=device, dtype=dtype) if v.is_floating_point()
                else v.to(device=device))
            if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        P = batch['P']
        A = batch['A']
        q = batch['q']
        l = batch['l']
        u = batch['u']
        x_star = batch['x_star']
        y_star = batch['y_star']
        z_star = batch['z_star']
        constr_type = batch['constr_type']
        R    = batch['R']
        AR   = batch['AR']
        ARAt = batch['ARAt']
        d_inv = batch['d_inv']   # (B, n)
        e_inv = batch['e_inv']   # (B, m)
        c_inv = batch['c_inv']   # (B,)

        B = P.shape[0]
        n = P.shape[1]
        m = A.shape[1]

        x = torch.zeros(B, n, dtype=dtype, device=device)
        z = torch.zeros(B, m, dtype=dtype, device=device)
        y = torch.zeros(B, m, dtype=dtype, device=device)

        rho_scalar = torch.full((B,), cfg.rho, dtype=dtype, device=device)
        rho_vec = batch['rho_vec'].clone()
        rho_inv = batch['rho_inv'].clone()
        factors = factorize_kkt(P, A, cfg.sigma, rho_inv)

        optimizer.zero_grad()

        stage_loss = torch.tensor(0.0, dtype=dtype, device=device)
        n_active_stages = 0

        # Per-instance tracking
        osqp_iters  = torch.full((B,), float(cfg.max_stages * cfg.T))
        osqp_done   = torch.zeros(B, dtype=torch.bool)
        rho_updates = torch.zeros(B)

        # Previous state (T steps ago); zeros at stage 0
        x_prev = torch.zeros(B, n, dtype=dtype, device=device)
        z_prev = torch.zeros(B, m, dtype=dtype, device=device)
        y_prev = torch.zeros(B, m, dtype=dtype, device=device)

        for stage in range(cfg.max_stages):
            with torch.no_grad():
                active = convergence_mask(P, A, q, x, z, y, d_inv, e_inv, c_inv, cfg)
                if not active.any():
                    break

            features = compute_per_row_features(P, q, A, l, u, x, z, y, rho_vec, x_star,
                                                x_prev, z_prev, y_prev)
            alpha_z = model(features)

            _batch_data = {'P': P, 'A': A, 'q': q, 'l': l, 'u': u,
                           'rho_vec': rho_vec, 'rho_inv': rho_inv,
                           'R': R, 'AR': AR, 'ARAt': ARAt}

            if loss_type == "spectral_radius":
                loss_i = spectral_radius_loss(z, alpha_z, _batch_data, cfg)
                with torch.no_grad():
                    x_new, z_new, y_new = rollout_T_steps(
                        x, z, y, alpha_z.detach(), factors, _batch_data, cfg,
                    )
            else:
                x_new, z_new, y_new = rollout_T_steps(
                    x, z, y, alpha_z, factors, _batch_data, cfg,
                )
                if loss_type == "scaled_residual":
                    loss_i = scaled_residual_loss(
                        x_new, z_new, y_new, x, z, y, _batch_data, mask=active,
                        eps_abs=cfg.eps_abs, eps_rel=cfg.eps_rel)
                else:
                    loss_i = log_convergence_loss(
                        x_new=x_new, x_prev=x, x_star=x_star,
                        y_new=y_new, y_prev=y, y_star=y_star,
                        z_new=z_new, z_prev=z, z_star=z_star,
                        cfg=cfg, mask=active,
                    )

            if not (torch.isnan(loss_i) or torch.isinf(loss_i)):
                stage_loss = stage_loss + loss_i
                n_active_stages += 1

            x_prev = x.detach()
            z_prev = z.detach()
            y_prev = y.detach()
            x = x_new.detach()
            z = z_new.detach()
            y = y_new.detach()

            with torch.no_grad():
                newly = _osqp_converged_batched(
                    P, A, q, x, z, y, d_inv, e_inv, c_inv, cfg) & ~osqp_done
                osqp_iters[newly] = float((stage + 1) * cfg.T)
                osqp_done |= newly

            if cfg.adaptive_rho:
                _batch_rho = {'P': P, 'A': A, 'q': q,
                              'rho_vec': rho_vec, 'rho_inv': rho_inv,
                              'constr_type': constr_type}
                rho_scalar, rho_vec, rho_inv, rho_updated = maybe_update_rho(
                    _batch_rho, x, z, y, rho_scalar, cfg
                )
                rho_updates += (rho_updated.cpu() & ~osqp_done).float()
                if rho_updated.any():
                    factors = factorize_kkt(P, A, cfg.sigma, rho_inv)

        if n_active_stages > 0:
            avg_loss = stage_loss / n_active_stages
            avg_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)
            optimizer.step()
            total_loss += avg_loss.item()
            n_batches += 1

        all_iters.extend(osqp_iters.tolist())
        all_rho.extend(rho_updates.tolist())

    iters_arr = np.array(all_iters) if all_iters else np.array([float('nan')])
    rho_arr   = np.array(all_rho)   if all_rho   else np.array([float('nan')])
    return (total_loss / max(n_batches, 1),
            float(iters_arr.mean()), float(iters_arr.std()),
            float(rho_arr.mean()),   float(rho_arr.std()))


# --------------------------------------------------------------------------- #
# Validation epoch
# --------------------------------------------------------------------------- #

@torch.no_grad()
def val_epoch(
    model: PerRowAlphaNet,
    loader,
    cfg: Config,
    device: torch.device,
    loss_type: str = "log_convergence",
) -> tuple[float, float, float, float, float]:
    """Run one validation epoch.

    Returns:
        (mean_loss, iters_mean, iters_std, rho_updates_mean, rho_updates_std)
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_iters: list[float] = []
    all_rho:   list[float] = []

    dtype = cfg.torch_dtype

    for batch in loader:
        batch = {
            k: (v.to(device=device, dtype=dtype) if v.is_floating_point()
                else v.to(device=device))
            if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        P = batch['P']
        A = batch['A']
        q = batch['q']
        l = batch['l']
        u = batch['u']
        x_star = batch['x_star']
        y_star = batch['y_star']
        z_star = batch['z_star']
        constr_type = batch['constr_type']
        R    = batch['R']
        AR   = batch['AR']
        ARAt = batch['ARAt']
        d_inv = batch['d_inv']   # (B, n)
        e_inv = batch['e_inv']   # (B, m)
        c_inv = batch['c_inv']   # (B,)

        B, n = P.shape[0], P.shape[1]
        m = A.shape[1]

        x = torch.zeros(B, n, dtype=dtype, device=device)
        z = torch.zeros(B, m, dtype=dtype, device=device)
        y = torch.zeros(B, m, dtype=dtype, device=device)

        rho_scalar = torch.full((B,), cfg.rho, dtype=dtype, device=device)
        rho_vec = batch['rho_vec'].clone()
        rho_inv = batch['rho_inv'].clone()
        factors = factorize_kkt(P, A, cfg.sigma, rho_inv)

        stage_loss = 0.0
        n_active = 0
        osqp_iters  = torch.full((B,), float(cfg.max_stages * cfg.T))
        osqp_done   = torch.zeros(B, dtype=torch.bool)
        rho_updates = torch.zeros(B)

        # Previous state (T steps ago); zeros at stage 0
        x_prev = torch.zeros(B, n, dtype=dtype, device=device)
        z_prev = torch.zeros(B, m, dtype=dtype, device=device)
        y_prev = torch.zeros(B, m, dtype=dtype, device=device)

        for stage in range(cfg.max_stages):
            active = convergence_mask(P, A, q, x, z, y, d_inv, e_inv, c_inv, cfg)
            if not active.any():
                break

            features = compute_per_row_features(P, q, A, l, u, x, z, y, rho_vec, x_star,
                                                x_prev, z_prev, y_prev)
            alpha_z = model(features)

            _batch_data = {'P': P, 'A': A, 'q': q, 'l': l, 'u': u,
                           'rho_vec': rho_vec, 'rho_inv': rho_inv,
                           'R': R, 'AR': AR, 'ARAt': ARAt}

            x_new, z_new, y_new = rollout_T_steps(
                x, z, y, alpha_z, factors, _batch_data, cfg,
            )

            if loss_type == "spectral_radius":
                loss_i = spectral_radius_loss(z, alpha_z, _batch_data, cfg)
            elif loss_type == "scaled_residual":
                loss_i = scaled_residual_loss(
                    x_new, z_new, y_new, x, z, y, _batch_data, mask=active,
                    eps_abs=cfg.eps_abs, eps_rel=cfg.eps_rel)
            else:
                loss_i = log_convergence_loss(
                    x_new, x, x_star, y_new, y, y_star, z_new, z, z_star,
                    cfg, mask=active)

            if not (torch.isnan(loss_i) or torch.isinf(loss_i)):
                stage_loss += loss_i.item()
                n_active += 1

            x_prev, z_prev, y_prev = x.detach(), z.detach(), y.detach()
            x, z, y = x_new, z_new, y_new

            newly = _osqp_converged_batched(
                P, A, q, x, z, y, d_inv, e_inv, c_inv, cfg) & ~osqp_done
            osqp_iters[newly] = float((stage + 1) * cfg.T)
            osqp_done |= newly

            if cfg.adaptive_rho:
                _batch_rho = {'P': P, 'A': A, 'q': q,
                              'rho_vec': rho_vec, 'rho_inv': rho_inv,
                              'constr_type': constr_type}
                rho_scalar, rho_vec, rho_inv, rho_updated = maybe_update_rho(
                    _batch_rho, x, z, y, rho_scalar, cfg
                )
                rho_updates += (rho_updated.cpu() & ~osqp_done).float()
                if rho_updated.any():
                    factors = factorize_kkt(P, A, cfg.sigma, rho_inv)

        if n_active > 0:
            total_loss += stage_loss / n_active
            n_batches += 1

        all_iters.extend(osqp_iters.tolist())
        all_rho.extend(rho_updates.tolist())

    iters_arr = np.array(all_iters) if all_iters else np.array([float('nan')])
    rho_arr   = np.array(all_rho)   if all_rho   else np.array([float('nan')])
    return (total_loss / max(n_batches, 1),
            float(iters_arr.mean()), float(iters_arr.std()),
            float(rho_arr.mean()),   float(rho_arr.std()))


# --------------------------------------------------------------------------- #
# Main training loop
# --------------------------------------------------------------------------- #

def train(
    cfg: Config | None = None,
    n_fixed: int | None = None,
    loss_type: str = "log_convergence",
    checkpoint_path: str = 'learned_osqp/checkpoints/best_model.pt',
) -> PerRowAlphaNet:
    """
    Full training run.

    Args:
        cfg             : Config instance (uses defaults if None)
        n_fixed         : QP problem size (overrides cfg.n_fixed if given)
        loss_type       : "log_convergence" (default) or "spectral_radius"
        checkpoint_path : path to save the best model (.pt); a .log file is
                          written alongside it for the full training log.

    Returns:
        Trained PerRowAlphaNet model.
    """
    if cfg is None:
        cfg = Config()

    ckpt_path = Path(checkpoint_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    log = _setup_logger(ckpt_path)

    device = cfg.torch_device

    log.info(f"Training PerRowAlphaNet  [loss_type={loss_type}]")
    log.info(f"  n={cfg.n_fixed}, m={cfg.m_fixed}, batch_size={cfg.batch_size}")
    log.info(f"  T={cfg.T} steps/stage, max_stages={cfg.max_stages}")
    log.info(f"  alpha range: [{cfg.alpha_min}, {cfg.alpha_max}]")
    log.info(f"  device={cfg.device}, dtype={cfg.dtype}")
    log.info(f"  checkpoint: {ckpt_path}")

    train_loader, val_loader = make_dataloaders(cfg, n_fixed, verbose=False)
    log.info(f"  Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    model = PerRowAlphaNet(cfg).to(dtype=cfg.torch_dtype, device=device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"  Model parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.n_epochs, eta_min=cfg.lr * 0.01
    )

    # Log precomputed baseline (alpha=1.6) stats once for reference
    bl_tr_im, bl_tr_is, bl_tr_rm, bl_tr_rs = _baseline_stats_from_loader(train_loader)
    bl_val_im, bl_val_is, bl_val_rm, bl_val_rs = _baseline_stats_from_loader(val_loader)
    log.info(
        f"Baseline (alpha=1.6)  "
        f"train_iters={bl_tr_im:.1f}±{bl_tr_is:.1f}  "
        f"val_iters={bl_val_im:.1f}±{bl_val_is:.1f}  "
        f"train_rho_updates={bl_tr_rm:.2f}±{bl_tr_rs:.2f}  "
        f"val_rho_updates={bl_val_rm:.2f}±{bl_val_rs:.2f}"
    )

    best_val_loss = float('inf')
    t0 = time.time()

    for epoch in range(1, cfg.n_epochs + 1):
        t_ep = time.time()
        train_loss, tr_iters_m, tr_iters_s, tr_rho_m, tr_rho_s = train_epoch(
            model, train_loader, optimizer, cfg, device, loss_type)
        val_loss, val_iters_m, val_iters_s, val_rho_m, val_rho_s = val_epoch(
            model, val_loader, cfg, device, loss_type)
        scheduler.step()

        elapsed = time.time() - t_ep
        total_elapsed = time.time() - t0
        lr_current = scheduler.get_last_lr()[0]

        log.info(
            f"Epoch {epoch:3d}/{cfg.n_epochs}  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"train_iters={tr_iters_m:.1f}±{tr_iters_s:.1f}  "
            f"val_iters={val_iters_m:.1f}±{val_iters_s:.1f}  "
            f"train_rho_updates={tr_rho_m:.2f}±{tr_rho_s:.2f}  "
            f"val_rho_updates={val_rho_m:.2f}±{val_rho_s:.2f}  "
            f"lr={lr_current:.2e}  "
            f"ep={elapsed:.1f}s  total={total_elapsed/60:.1f}min"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {'epoch': epoch, 'model_state': model.state_dict(),
                 'val_loss': best_val_loss, 'cfg': cfg},
                str(ckpt_path),
            )
            log.info(f"  -> saved checkpoint (val_loss={best_val_loss:.4f})")

    log.info(f"Training complete. Best val_loss={best_val_loss:.4f}")
    return model


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train PerRowAlphaNet for OSQP')
    parser.add_argument('--n', type=int, default=20, help='QP problem size n')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--T', type=int, default=10, help='steps per stage')
    parser.add_argument('--stages', type=int, default=300, help='max stages')
    parser.add_argument('--n_train', type=int, default=50)
    parser.add_argument('--regen', action='store_true', help='regenerate dataset')
    parser.add_argument('--loss', type=str, default='log_convergence',
                        choices=['log_convergence', 'spectral_radius', 'scaled_residual'],
                        help='loss function to use')
    parser.add_argument('--ckpt', type=str,
                        default='learned_osqp/checkpoints/best_model.pt',
                        help='path to save best checkpoint (.pt); .log written alongside')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                        help='compute device')
    parser.add_argument('--dtype', type=str, default='float64',
                        choices=['float64', 'float32'],
                        help='floating-point dtype (float64 recommended for KKT stability)')
    parser.add_argument('--precision', type=str, default='low', choices=['low', 'high'],
                        help='convergence tolerance: low → eps=1e-3, high → eps=1e-5')
    args = parser.parse_args()

    cfg = Config(
        n_fixed=args.n,
        n_epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        T=args.T,
        max_stages=args.stages,
        n_train=args.n_train,
        device=args.device,
        dtype=args.dtype,
        precision=args.precision,
        data_path=f"learned_osqp/data/qp_dataset_n={args.n}_precision={args.precision}_dtype={args.dtype}.pt",
    )

    if args.regen:
        p = Path(cfg.data_path)
        if p.exists():
            p.unlink()
            print(f"Removed existing dataset at {p}")

    train(cfg, loss_type=args.loss, checkpoint_path=args.ckpt)
