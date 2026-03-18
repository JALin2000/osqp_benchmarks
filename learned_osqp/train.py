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
from learned_osqp.model import PerRowAlphaNet, ScalarAlphaNet, ScalarGRUNet
import itertools
from learned_osqp.data import make_dataloaders_multi, dataset_path
from learned_osqp.features import compute_per_row_features, compute_global_features


from learned_osqp.osqp_torch import (
    factorize_kkt,
    selective_factorize_kkt,
    rollout_T_steps,
    maybe_update_rho,
)
from learned_osqp.loss import log_convergence_loss, convergence_mask, spectral_radius_loss, scaled_residual_loss


# --------------------------------------------------------------------------- #
# Feature normalization statistics
# --------------------------------------------------------------------------- #

def compute_feature_stats(
    train_loaders, cfg: Config, device: torch.device, dtype: torch.dtype,
    n_stages: int = 20,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run a short ADMM rollout on the training set (fixed alpha=1.6, no grad) and
    collect per-row features at every stage to cover the full convergence trajectory.

    Args:
        n_stages : number of T-step ADMM stages to run per batch (default 20,
                   i.e. 20*T total steps — enough to span early→mid→late convergence)

    Returns:
        mean : (feature_dim,) tensor
        std  : (feature_dim,) tensor — entries with std < 1e-6 are set to 1.0
    """
    scalar_mode = getattr(cfg, 'alpha_mode', 'vector') == 'scalar'
    feat_dim  = cfg.scalar_feature_dim if scalar_mode else cfg.feature_dim
    feat_sum = torch.zeros(feat_dim, dtype=dtype, device=device)
    feat_sq  = torch.zeros(feat_dim, dtype=dtype, device=device)
    count    = 0

    iterable = itertools.chain(*train_loaders) if isinstance(train_loaders, list) else train_loaders
    with torch.no_grad():
        for batch in iterable:
            batch = {
                k: (v.to(device=device, dtype=dtype) if isinstance(v, torch.Tensor) and v.is_floating_point()
                    else v.to(device=device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            P, A, q = batch['P'], batch['A'], batch['q']
            l, u    = batch['l'], batch['u']
            rho_vec = batch['rho_vec']
            x_star  = batch.get('x_star', None)
            B, m, n = A.shape

            x = torch.zeros(B, n, dtype=dtype, device=device)
            z = torch.zeros(B, m, dtype=dtype, device=device)
            y = torch.zeros(B, m, dtype=dtype, device=device)
            alpha_z = torch.full((B, m), 1.6, dtype=dtype, device=device)
            rho_scalar_vec = torch.full((B,), cfg.rho, dtype=dtype, device=device)

            factors = factorize_kkt(P, A, cfg.sigma, rho_vec)
            alpha_stats = torch.full((B,), 1.6, dtype=dtype, device=device)

            for _ in range(n_stages):
                x_prev, z_prev, y_prev = x, z, y

                if scalar_mode:
                    feat = compute_global_features(
                        P, q, A, x, z, y, rho_scalar_vec,
                        x_prev, z_prev, y_prev, alpha_stats,
                    )  # (B, scalar_feature_dim)
                    flat = feat  # (B, scalar_feature_dim)
                else:
                    feat = compute_per_row_features(
                        P, q, A, l, u, x, z, y, rho_vec, x_star,
                        x_prev, z_prev, y_prev,
                    )  # (B, m, feature_dim)
                    flat = feat.reshape(-1, cfg.feature_dim)
                feat_sum += flat.sum(0)
                feat_sq  += (flat ** 2).sum(0)
                count    += flat.shape[0]

                x, z, y = rollout_T_steps(
                    x, z, y, alpha_z, factors, batch, cfg,
                )

    mean = feat_sum / count
    std  = (feat_sq / count - mean ** 2).clamp(min=0.0).sqrt()
    std  = torch.where(std < 1e-6, torch.ones_like(std), std)
    return mean, std


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


def _baseline_stats_from_loader(loader_or_list) -> tuple[float, float, float, float]:
    """
    Collect precomputed baseline_iters / baseline_rho_updates from the dataset.
    Accepts a single DataLoader or a list of DataLoaders.
    Returns (iters_mean, iters_std, rho_mean, rho_std).
    """
    loaders = loader_or_list if isinstance(loader_or_list, list) else [loader_or_list]
    all_iters: list[float] = []
    all_rho:   list[float] = []
    for loader in loaders:
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
    AT: torch.Tensor | None = None,  # (B, n, m) precomputed A^T contiguous
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
    if AT is None:
        AT = A.transpose(1, 2)
    Ax  = torch.bmm(A, x.unsqueeze(-1)).squeeze(-1)                     # (B, m)
    ATy = torch.bmm(AT, y.unsqueeze(-1)).squeeze(-1)                    # (B, n)
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
    model: nn.Module,
    loaders,
    optimizer: torch.optim.Optimizer,
    cfg: Config,
    device: torch.device,
    loss_type: str = "log_convergence",
) -> tuple[float, float, float, float, float]:
    """Run one training epoch.

    ``loaders`` may be a single DataLoader or a list of DataLoaders (one per QP
    type).  When a list is given batches are iterated in round-robin order so
    every type contributes equally within the epoch.

    Returns:
        (mean_loss, iters_mean, iters_std, rho_updates_mean, rho_updates_std,
         alpha_mean, alpha_std)
    """
    model.train()
    total_loss = 0.0
    n_batches = 0
    all_iters: list[float] = []
    all_rho: list[float] = []
    alpha_sum = alpha_sum_sq = 0.0
    alpha_count = 0

    dtype = cfg.torch_dtype

    # Build a single iterable: round-robin across all loaders each epoch
    if isinstance(loaders, list):
        loader_iters = [iter(l) for l in loaders]
        sentinel = object()

        def _round_robin():
            active = list(range(len(loader_iters)))
            while active:
                remaining = []
                for i in active:
                    item = next(loader_iters[i], sentinel)
                    if item is not sentinel:
                        remaining.append(i)
                        yield item
                active = remaining

        iterable = _round_robin()
    else:
        iterable = loaders

    for batch in iterable:
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
        d_inv = batch['d_inv']   # (B, n)
        e_inv = batch['e_inv']   # (B, m)
        c_inv = batch['c_inv']   # (B,)

        B = P.shape[0]
        n = P.shape[1]
        m = A.shape[1]
        AT = A.transpose(1, 2).contiguous()  # (B, n, m) — precompute once per batch

        x = torch.zeros(B, n, dtype=dtype, device=device)
        z = torch.zeros(B, m, dtype=dtype, device=device)
        y = torch.zeros(B, m, dtype=dtype, device=device)

        # GRU hidden state — reset to None (zeros) at the start of each problem
        h_state: torch.Tensor | None = None

        rho_scalar = torch.full((B,), cfg.rho, dtype=dtype, device=device)
        rho_vec = batch['rho_vec'].clone()
        rho_inv = batch['rho_inv'].clone()
        factors = factorize_kkt(P, A, cfg.sigma, rho_vec, AT=AT)

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
        alpha_prev = torch.full((B,), cfg.alpha_x, dtype=dtype, device=device)

        for stage in range(cfg.max_stages):
            # Skip convergence check at stage 0: x=z=y=0 can give spurious
            # convergence (e.g. when q=0 as in control QPs).
            if stage > 0:
                with torch.no_grad():
                    active = convergence_mask(P, A, q, x, z, y, d_inv, e_inv, c_inv, cfg, AT=AT)
                    if not active.any():
                        break
            else:
                active = torch.ones(B, dtype=torch.bool, device=device)

            _batch_data = {'P': P, 'A': A, 'q': q, 'l': l, 'u': u,
                           'rho_vec': rho_vec, 'rho_inv': rho_inv}
            if loss_type == "spectral_radius":
                _batch_data['R']    = batch['R']
                _batch_data['AR']   = batch['AR']
                _batch_data['ARAt'] = batch['ARAt']

            if getattr(cfg, 'alpha_mode', 'vector') == 'scalar':
                # ScalarAlphaNet / ScalarGRUNet: (B,) → unsqueeze to (B, 1) for broadcasting
                global_feat  = compute_global_features(
                    P, q, A, x, z, y, rho_scalar, x_prev, z_prev, y_prev, alpha_prev,
                    AT=AT,
                )  # (B, scalar_feature_dim)
                if isinstance(model, ScalarGRUNet):
                    alpha_scalar, h_state = model(global_feat, h_state)  # (B,), (B, hidden)
                else:
                    alpha_scalar = model(global_feat)          # (B,)
                alpha_prev   = alpha_scalar.detach()
                alpha_z      = alpha_scalar.unsqueeze(-1)  # (B, 1) — broadcasts vs (B, m/n)
                alpha_x_override = alpha_z                 # same (B, 1)
            else:
                features = compute_per_row_features(P, q, A, l, u, x, z, y, rho_vec, x_star,
                                                    x_prev, z_prev, y_prev, AT=AT)
                alpha_z = model(features)   # (B, m)
                alpha_x_override = None
                alpha_scalar = None

            with torch.no_grad():
                _az = (alpha_scalar if alpha_scalar is not None else alpha_z).detach()
                alpha_sum    += _az.sum().item()
                alpha_sum_sq += (_az ** 2).sum().item()
                alpha_count  += _az.numel()

            if loss_type == "spectral_radius":
                loss_i = spectral_radius_loss(z, alpha_z, _batch_data, cfg)
                with torch.no_grad():
                    _az_d = alpha_z.detach()
                    x_new, z_new, y_new = rollout_T_steps(
                        x, z, y, _az_d, factors, _batch_data, cfg,
                        alpha_x_override=_az_d if alpha_x_override is not None else None,
                    )
            else:
                x_new, z_new, y_new = rollout_T_steps(
                    x, z, y, alpha_z, factors, _batch_data, cfg,
                    alpha_x_override=alpha_x_override,
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
            # Detach GRU hidden state between stages (same convention as x/z/y)
            if h_state is not None:
                h_state = h_state.detach()

            with torch.no_grad():
                newly = _osqp_converged_batched(
                    P, A, q, x, z, y, d_inv, e_inv, c_inv, cfg, AT=AT) & ~osqp_done
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
                    factors = selective_factorize_kkt(
                        factors, P, A, cfg.sigma, rho_vec, AT, rho_updated)

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
    if alpha_count > 0:
        alpha_mean = alpha_sum / alpha_count
        alpha_std  = (max(alpha_sum_sq / alpha_count - alpha_mean ** 2, 0.0)) ** 0.5
    else:
        alpha_mean = alpha_std = float('nan')
    return (total_loss / max(n_batches, 1),
            float(iters_arr.mean()), float(iters_arr.std()),
            float(rho_arr.mean()),   float(rho_arr.std()),
            alpha_mean, alpha_std)


# --------------------------------------------------------------------------- #
# Validation epoch
# --------------------------------------------------------------------------- #

@torch.no_grad()
def val_epoch(
    model: nn.Module,
    loaders,
    cfg: Config,
    device: torch.device,
    loss_type: str = "log_convergence",
) -> tuple[float, float, float, float, float]:
    """Run one validation epoch.

    ``loaders`` may be a single DataLoader or a list of DataLoaders.

    Returns:
        (mean_loss, iters_mean, iters_std, rho_updates_mean, rho_updates_std,
         alpha_mean, alpha_std)
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_iters: list[float] = []
    all_rho:   list[float] = []
    alpha_sum = alpha_sum_sq = 0.0
    alpha_count = 0

    dtype = cfg.torch_dtype

    iterable = itertools.chain(*loaders) if isinstance(loaders, list) else loaders

    for batch in iterable:
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
        d_inv = batch['d_inv']   # (B, n)
        e_inv = batch['e_inv']   # (B, m)
        c_inv = batch['c_inv']   # (B,)

        B, n = P.shape[0], P.shape[1]
        m = A.shape[1]
        AT = A.transpose(1, 2).contiguous()  # (B, n, m) — precompute once per batch

        x = torch.zeros(B, n, dtype=dtype, device=device)
        z = torch.zeros(B, m, dtype=dtype, device=device)
        y = torch.zeros(B, m, dtype=dtype, device=device)

        # GRU hidden state — reset to None (zeros) at the start of each problem
        h_state: torch.Tensor | None = None

        rho_scalar = torch.full((B,), cfg.rho, dtype=dtype, device=device)
        rho_vec = batch['rho_vec'].clone()
        rho_inv = batch['rho_inv'].clone()
        factors = factorize_kkt(P, A, cfg.sigma, rho_vec, AT=AT)

        stage_loss = 0.0
        n_active = 0
        osqp_iters  = torch.full((B,), float(cfg.max_stages * cfg.T))
        osqp_done   = torch.zeros(B, dtype=torch.bool)
        rho_updates = torch.zeros(B)

        # Previous state (T steps ago); zeros at stage 0
        x_prev = torch.zeros(B, n, dtype=dtype, device=device)
        z_prev = torch.zeros(B, m, dtype=dtype, device=device)
        y_prev = torch.zeros(B, m, dtype=dtype, device=device)
        alpha_prev = torch.full((B,), cfg.alpha_x, dtype=dtype, device=device)

        for stage in range(cfg.max_stages):
            # Skip convergence check at stage 0: x=z=y=0 can give spurious
            # convergence (e.g. when q=0 as in control QPs).
            if stage > 0:
                active = convergence_mask(P, A, q, x, z, y, d_inv, e_inv, c_inv, cfg, AT=AT)
                if not active.any():
                    break
            else:
                active = torch.ones(B, dtype=torch.bool, device=device)

            _batch_data = {'P': P, 'A': A, 'q': q, 'l': l, 'u': u,
                           'rho_vec': rho_vec, 'rho_inv': rho_inv}
            if loss_type == "spectral_radius":
                _batch_data['R']    = batch['R']
                _batch_data['AR']   = batch['AR']
                _batch_data['ARAt'] = batch['ARAt']

            if getattr(cfg, 'alpha_mode', 'vector') == 'scalar':
                global_feat  = compute_global_features(
                    P, q, A, x, z, y, rho_scalar, x_prev, z_prev, y_prev, alpha_prev,
                    AT=AT,
                )  # (B, scalar_feature_dim)
                if isinstance(model, ScalarGRUNet):
                    alpha_scalar, h_state = model(global_feat, h_state)  # (B,), (B, hidden)
                else:
                    alpha_scalar = model(global_feat)          # (B,)
                alpha_prev   = alpha_scalar.detach()
                alpha_z      = alpha_scalar.unsqueeze(-1)  # (B, 1)
                alpha_x_override = alpha_z
            else:
                features = compute_per_row_features(P, q, A, l, u, x, z, y, rho_vec, x_star,
                                                    x_prev, z_prev, y_prev, AT=AT)
                alpha_z = model(features)   # (B, m)
                alpha_x_override = None
                alpha_scalar = None

            _az = alpha_scalar if alpha_scalar is not None else alpha_z
            alpha_sum    += _az.sum().item()
            alpha_sum_sq += (_az ** 2).sum().item()
            alpha_count  += _az.numel()

            x_new, z_new, y_new = rollout_T_steps(
                x, z, y, alpha_z, factors, _batch_data, cfg,
                alpha_x_override=alpha_x_override,
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
                P, A, q, x, z, y, d_inv, e_inv, c_inv, cfg, AT=AT) & ~osqp_done
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
                    factors = selective_factorize_kkt(
                        factors, P, A, cfg.sigma, rho_vec, AT, rho_updated)

        if n_active > 0:
            total_loss += stage_loss / n_active
            n_batches += 1

        all_iters.extend(osqp_iters.tolist())
        all_rho.extend(rho_updates.tolist())

    iters_arr = np.array(all_iters) if all_iters else np.array([float('nan')])
    rho_arr   = np.array(all_rho)   if all_rho   else np.array([float('nan')])
    if alpha_count > 0:
        alpha_mean = alpha_sum / alpha_count
        alpha_std  = (max(alpha_sum_sq / alpha_count - alpha_mean ** 2, 0.0)) ** 0.5
    else:
        alpha_mean = alpha_std = float('nan')
    return (total_loss / max(n_batches, 1),
            float(iters_arr.mean()), float(iters_arr.std()),
            float(rho_arr.mean()),   float(rho_arr.std()),
            alpha_mean, alpha_std)


# --------------------------------------------------------------------------- #
# Main training loop
# --------------------------------------------------------------------------- #

def train(
    cfg: Config | None = None,
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

    alpha_mode = getattr(cfg, 'alpha_mode', 'vector')
    model_type = getattr(cfg, 'model_type', 'mlp')
    if alpha_mode != 'scalar':
        model_name = 'PerRowAlphaNet'
    elif model_type == 'gru':
        model_name = 'ScalarGRUNet'
    else:
        model_name = 'ScalarAlphaNet'
    log.info(f"Training {model_name}  [loss_type={loss_type}, alpha_mode={alpha_mode}, model_type={model_type}]")
    log.info(f"  qp_types={cfg.qp_types}, qp_type_sizes={cfg.qp_type_sizes}, batch_size={cfg.batch_size}")
    log.info(f"  T={cfg.T} steps/stage, max_stages={cfg.max_stages}")
    log.info(f"  alpha range: [{cfg.alpha_min}, {cfg.alpha_max}]")
    log.info(f"  device={cfg.device}, dtype={cfg.dtype}, precision={cfg.precision}")
    log.info(f"  checkpoint: {ckpt_path}")

    type_loaders = make_dataloaders_multi(cfg, verbose=True)
    train_loaders = [v[0] for v in type_loaders.values()]
    val_loaders   = [v[1] for v in type_loaders.values()]
    n_tr = sum(len(l) for l in train_loaders)
    n_va = sum(len(l) for l in val_loaders)
    log.info(f"  Total train batches: {n_tr}, Total val batches: {n_va}")

    if alpha_mode == 'scalar':
        if model_type == 'gru':
            model = ScalarGRUNet(cfg).to(dtype=cfg.torch_dtype, device=device)
        else:
            model = ScalarAlphaNet(cfg).to(dtype=cfg.torch_dtype, device=device)
    else:
        model = PerRowAlphaNet(cfg).to(dtype=cfg.torch_dtype, device=device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"  Model parameters: {n_params:,}")

    if cfg.normalize_features:
        log.info("  Computing feature normalization statistics from training data...")
        feat_mean, feat_std = compute_feature_stats(train_loaders, cfg, device, cfg.torch_dtype)
        model.set_feature_norm(feat_mean, feat_std)
        log.info(f"  Feature norm set  mean=[{feat_mean.min():.3f}, {feat_mean.max():.3f}]  "
                 f"std=[{feat_std.min():.3f}, {feat_std.max():.3f}]")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.n_epochs, eta_min=cfg.lr * 0.01
    )

    # Log precomputed baseline (alpha=1.6) stats once for reference
    bl_tr_im, bl_tr_is, bl_tr_rm, bl_tr_rs = _baseline_stats_from_loader(train_loaders)
    bl_val_im, bl_val_is, bl_val_rm, bl_val_rs = _baseline_stats_from_loader(val_loaders)
    log.info(
        f"Baseline (alpha=1.6)  "
        f"train_iters={bl_tr_im:.1f}±{bl_tr_is:.1f}  "
        f"val_iters={bl_val_im:.1f}±{bl_val_is:.1f}  "
        f"train_rho_updates={bl_tr_rm:.2f}±{bl_tr_rs:.2f}  "
        f"val_rho_updates={bl_val_rm:.2f}±{bl_val_rs:.2f}"
    )

    best_val_loss = float('inf')
    best_val_iters_m = float('inf')
    best_val_rho_m = float('inf')
    best_val_rho_iters_m = float('inf')
    ckpt_rho_path = ckpt_path.with_name(ckpt_path.stem + '_best_rho' + ckpt_path.suffix)
    t0 = time.time()

    for epoch in range(1, cfg.n_epochs + 1):
        t_ep = time.time()
        train_loss, tr_iters_m, tr_iters_s, tr_rho_m, tr_rho_s, tr_alpha_m, tr_alpha_s = train_epoch(
            model, train_loaders, optimizer, cfg, device, loss_type)
        val_loss, val_iters_m, val_iters_s, val_rho_m, val_rho_s, val_alpha_m, val_alpha_s = val_epoch(
            model, val_loaders, cfg, device, loss_type)
        scheduler.step()

        elapsed = time.time() - t_ep
        total_elapsed = time.time() - t0
        lr_current = scheduler.get_last_lr()[0]

        log.info(
            f"Epoch {epoch:3d}/{cfg.n_epochs}  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"train_iters={tr_iters_m:.1f}±{tr_iters_s:.1f}  "
            f"val_iters={val_iters_m:.1f}±{val_iters_s:.1f}  "
            f"train_alpha={tr_alpha_m:.3f}±{tr_alpha_s:.3f}  "
            f"val_alpha={val_alpha_m:.3f}±{val_alpha_s:.3f}  "
            f"train_rho_updates={tr_rho_m:.2f}±{tr_rho_s:.2f}  "
            f"val_rho_updates={val_rho_m:.2f}±{val_rho_s:.2f}  "
            f"lr={lr_current:.2e}  "
            f"ep={elapsed:.1f}s  total={total_elapsed/60:.1f}min"
        )

        if val_iters_m < best_val_iters_m:
            best_val_loss = val_loss
            best_val_iters_m = val_iters_m
            torch.save(
                {'epoch': epoch, 'model_state': model.state_dict(),
                 'val_loss': best_val_loss, 'cfg': cfg,
                 'val_iters_m': val_iters_m, 'val_iters_s': val_iters_s,
                 'feat_norm_active': model.feat_norm_active},
                str(ckpt_path),
            )
            log.info(f"  -> saved checkpoint (val_iters_m={best_val_iters_m:.4f})")

        # Second checkpoint: best val_rho_m (fewest rho updates),
        # with val_iters_m as tiebreaker when val_rho_m is equal.
        if (val_rho_m < best_val_rho_m
                or (val_rho_m == best_val_rho_m and val_iters_m < best_val_rho_iters_m)):
            best_val_rho_m = val_rho_m
            best_val_rho_iters_m = val_iters_m
            torch.save(
                {'epoch': epoch, 'model_state': model.state_dict(),
                 'val_loss': val_loss, 'cfg': cfg,
                 'val_iters_m': val_iters_m, 'val_iters_s': val_iters_s,
                 'val_rho_m': val_rho_m, 'val_rho_s': val_rho_s,
                 'feat_norm_active': model.feat_norm_active},
                str(ckpt_rho_path),
            )
            log.info(f"  -> saved rho checkpoint (val_rho_m={best_val_rho_m:.4f}, val_iters_m={best_val_rho_iters_m:.4f})")

    log.info(f"Training complete. Best val_loss={best_val_loss:.4f}")
    return model


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train PerRowAlphaNet for OSQP')
    parser.add_argument('--n', type=int, default=100,
                        help='Size parameter for random_qp (used when --types is not set)')
    parser.add_argument('--types', type=str, default=None,
                        help='Comma-separated QP types, e.g. "random_qp,control,lasso". '
                             'Supported: random_qp, control, eq_qp, huber, lasso, portfolio')
    parser.add_argument('--sizes', type=str, default=None,
                        help='Comma-separated size params matching --types, e.g. "20,10,3". '
                             'If omitted, --n is used for all types.')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch', type=int, default=10)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--T', type=int, default=10, help='steps per stage')
    parser.add_argument('--stages', type=int, default=2000, help='max stages')
    parser.add_argument('--n_train', type=int, default=50)
    parser.add_argument('--regen', action='store_true', help='regenerate dataset(s)')
    parser.add_argument('--loss', type=str, default='log_convergence',
                        choices=['log_convergence', 'spectral_radius', 'scaled_residual'],
                        help='loss function to use')
    parser.add_argument('--ckpt', type=str, default=None,
                        help='path to save best checkpoint (.pt); .log written alongside. If not set, see below for default naming based on --types.')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                        help='compute device')
    parser.add_argument('--dtype', type=str, default='float64',
                        choices=['float64', 'float32'],
                        help='floating-point dtype (float64 recommended for KKT stability)')
    parser.add_argument('--precision', type=str, default='low', choices=['low', 'high'],
                        help='convergence tolerance: low → eps=1e-3, high → eps=1e-5')
    parser.add_argument('--adaptive_rho', type=lambda x: x.lower() != 'false',
                        default=True, metavar='BOOL',
                        help='enable adaptive rho updates (default: True); pass false to disable')
    parser.add_argument('--alpha_mode', type=str, default='vector',
                        choices=['vector', 'scalar'],
                        help='vector: per-row alpha_z via PerRowAlphaNet (default); '
                             'scalar: single alpha for both alpha_x and alpha_z via ScalarAlphaNet/ScalarGRUNet')
    parser.add_argument('--model_type', type=str, default='mlp',
                        choices=['mlp', 'gru'],
                        help='model architecture for scalar alpha mode: '
                             'mlp (default, memoryless MLP) or '
                             'gru (GRU with hidden state across stages)')
    parser.add_argument('--normalize_features', action='store_true',
                        help='normalize input features to zero mean / unit std before training')
    args = parser.parse_args()

    # Parse types and per-type size parameters
    if args.types:
        types_list = [t.strip() for t in args.types.split(',')]
        if args.sizes:
            sizes_list = [int(s.strip()) for s in args.sizes.split(',')]
            if len(sizes_list) != len(types_list):
                parser.error('--sizes must have the same number of entries as --types')
        else:
            sizes_list = [args.n] * len(types_list)
        qp_type_sizes = dict(zip(types_list, sizes_list))
    else:
        types_list = ['random_qp']
        qp_type_sizes = {'random_qp': args.n}

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
        adaptive_rho=args.adaptive_rho,
        normalize_features=args.normalize_features,
        alpha_mode=args.alpha_mode,
        model_type=args.model_type,
        store_spectral_matrices=(args.loss == 'spectral_radius'),
        qp_types=types_list,
        qp_type_sizes=qp_type_sizes,
        # data_dir='learned_osqp/data',
        data_dir='/data/engs-goulart/sedm7756/float64_optimized',
    )

    if args.regen:
        for type_name in cfg.qp_types:
            size_param = cfg.qp_type_sizes[type_name]
            p = dataset_path(type_name, size_param, cfg)
            if p.exists():
                p.unlink()
                print(f"Removed dataset: {p}")

    if args.ckpt is None:
        ckpt_name = f"best_model_{args.types}_precision={args.precision}_adaptive_rho={args.adaptive_rho}_alpha_mode={args.alpha_mode}_model_type={args.model_type}_loss={args.loss}"
        # ckpt_path = f"learned_osqp/checkpoints/{ckpt_name}_1111.pt"
        ckpt_path = f"learned_osqp/checkpoints/0318_feat_pri_dua_res_scaled_alpha_1.3_1.9/{ckpt_name}.pt"
    else:
        ckpt_path = args.ckpt
    train(cfg, loss_type=args.loss, checkpoint_path=ckpt_path)