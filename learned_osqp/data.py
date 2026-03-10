"""
data.py — QP dataset generation and loading.

Uses RandomQPExample from problem_classes/random_qp.py to generate QP instances
and cvxpy to compute ground-truth solutions x_star.

All sparse matrices are converted to dense float64 tensors for batched PyTorch ops.
"""

import sys
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path

# Allow imports from the benchmarks root
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from problem_classes.random_qp import RandomQPExample
from learned_osqp.config import (
    Config, RHO_MIN, RHO_EQ_OVER_RHO_INEQ,
    OSQP_INFTY, MIN_SCALING, RHO_TOL,
)
from learned_osqp.scaling import ruiz_equilibration, apply_scaling

try:
    import cvxpy
    _CVXPY_AVAILABLE = True
except ImportError:
    _CVXPY_AVAILABLE = False


# --------------------------------------------------------------------------- #
# Constraint type constants (mirror _osqp.py)
# --------------------------------------------------------------------------- #
CONSTR_LOOSE = -1    # l = -inf AND u = +inf  → rho_min
CONSTR_EQ = 1        # u - l < RHO_TOL        → rho_eq_factor * rho
CONSTR_INEQ = 0      # otherwise              → rho


def get_constr_type_np(l: np.ndarray, u: np.ndarray) -> np.ndarray:
    """
    Classify constraints as in _osqp.py set_rho_vec (lines 537-562).
    Returns integer array with values CONSTR_LOOSE / CONSTR_EQ / CONSTR_INEQ.
    """
    m = l.shape[0]
    constr_type = np.full(m, CONSTR_INEQ, dtype=np.int32)

    loose_mask = (l < -OSQP_INFTY * MIN_SCALING) & (u > OSQP_INFTY * MIN_SCALING)
    eq_mask = (u - l) < RHO_TOL

    constr_type[loose_mask] = CONSTR_LOOSE
    constr_type[eq_mask] = CONSTR_EQ
    return constr_type


def build_rho_vec(constr_type: np.ndarray, rho: float) -> np.ndarray:
    """
    Build per-constraint rho vector from constraint types.
    Mirrors _osqp.py set_rho_vec lines 558-560.
    """
    rho_vec = np.empty(constr_type.shape[0])
    rho_vec[constr_type == CONSTR_LOOSE] = RHO_MIN
    rho_vec[constr_type == CONSTR_EQ] = RHO_EQ_OVER_RHO_INEQ * rho
    rho_vec[constr_type == CONSTR_INEQ] = rho
    return rho_vec


# --------------------------------------------------------------------------- #
# Baseline stats (run once per instance at generation time)
# --------------------------------------------------------------------------- #

def compute_baseline_stats(instance: dict, cfg: Config, alpha: float = 1.6) -> tuple[int, int]:
    """
    Run baseline OSQP (fixed alpha_z = alpha, default 1.6) on a single instance.
    Uses the same convergence criterion and adaptive-rho logic as the training loop.

    Returns:
        (iters_to_converge, rho_updates)
        iters_to_converge : iterations until OSQP_SOLVED, or cfg.max_stages*cfg.T if never
        rho_updates       : number of rho update events (each event = one stage boundary)

    Uses local imports to avoid circular dependency with osqp_torch.
    """
    # Local imports — osqp_torch imports update_rho_vec_batched from this module,
    # so we defer the import here to avoid a module-level circular dependency.
    from learned_osqp.osqp_torch import factorize_kkt, osqp_step, maybe_update_rho

    max_iters = cfg.max_stages * cfg.T
    dtype = torch.float64

    # Build 1-element batch
    P = instance['P'].unsqueeze(0)            # (1, n, n)
    q = instance['q'].unsqueeze(0)            # (1, n)
    A = instance['A'].unsqueeze(0)            # (1, m, n)
    l = instance['l'].unsqueeze(0)            # (1, m)
    u = instance['u'].unsqueeze(0)            # (1, m)
    constr_type = instance['constr_type'].unsqueeze(0)  # (1, m)

    n_dim = P.shape[1]
    m_dim = A.shape[1]

    x = torch.zeros(1, n_dim, dtype=dtype)
    z = torch.zeros(1, m_dim, dtype=dtype)
    y = torch.zeros(1, m_dim, dtype=dtype)
    alpha_z = torch.full((1, m_dim), alpha, dtype=dtype)
    rho_scalar = torch.tensor([cfg.rho], dtype=dtype)
    rho_vec = instance['rho_vec'].unsqueeze(0).clone()
    rho_inv = instance['rho_inv'].unsqueeze(0).clone()

    iters_to_converge = max_iters
    rho_updates = 0
    eps_abs, eps_rel = 3e-4, 3e-4

    # Scaling params for unscaled convergence check (all in scaled space)
    d_inv = instance['d_inv'].unsqueeze(0)   # (1, n)
    e_inv = instance['e_inv'].unsqueeze(0)   # (1, m)
    c_inv = float(instance['c_inv'].item())  # scalar

    factors = factorize_kkt(P, A, cfg.sigma, rho_inv)

    def _converged(x_, z_, y_):
        Ax_  = torch.bmm(A, x_.unsqueeze(-1)).squeeze(-1)
        ATy_ = torch.bmm(A.transpose(1, 2), y_.unsqueeze(-1)).squeeze(-1)
        Px_  = torch.bmm(P, x_.unsqueeze(-1)).squeeze(-1)
        # Unscale: pri_res = E^{-1}*(A_sc*x_sc - z_sc)
        pri_res = e_inv * (Ax_ - z_)                        # (1, m)
        # Unscale: dua_res = c^{-1}*D^{-1}*(P_sc*x_sc + q_sc + A_sc^T*y_sc)
        dua_res = c_inv * d_inv * (Px_ + q + ATy_)          # (1, n)
        pr = pri_res.abs().amax(dim=1)
        ep = eps_abs + eps_rel * torch.maximum(
            (e_inv * Ax_).abs().amax(dim=1),
            (e_inv * z_).abs().amax(dim=1),
        )
        dr = dua_res.abs().amax(dim=1)
        ed = eps_abs + eps_rel * torch.stack([
            (c_inv * d_inv * Px_).abs().amax(dim=1),
            (c_inv * d_inv * ATy_).abs().amax(dim=1),
            (c_inv * d_inv * q).abs().amax(dim=1),
        ], dim=1).amax(dim=1)
        return bool((pr < ep).all() and (dr < ed).all())

    with torch.no_grad():
        for stage in range(cfg.max_stages):
            # Run T steps (same granularity as training loop)
            for _ in range(cfg.T):
                x, z, y, _, _ = osqp_step(
                    x, z, y, q, l, u, rho_vec, rho_inv,
                    factors, cfg.alpha_x, alpha_z, cfg.sigma,
                )

            # Check convergence at stage boundary (same as training)
            if _converged(x, z, y):
                iters_to_converge = (stage + 1) * cfg.T
                break

            # Adaptive rho at stage boundary
            if cfg.adaptive_rho:
                _b = {'P': P, 'A': A, 'q': q,
                      'rho_vec': rho_vec, 'rho_inv': rho_inv,
                      'constr_type': constr_type}
                rho_scalar, rho_vec, rho_inv, updated = maybe_update_rho(
                    _b, x, z, y, rho_scalar, cfg)
                if updated.any():
                    rho_updates += 1
                    factors = factorize_kkt(P, A, cfg.sigma, rho_inv)
                    alpha_z = torch.full((1, m_dim), alpha, dtype=dtype)

    return iters_to_converge, rho_updates


# --------------------------------------------------------------------------- #
# Instance generation
# --------------------------------------------------------------------------- #

def generate_qp_instance(n: int, seed: int, cfg: Config) -> dict | None:
    """
    Generate one QP instance using RandomQPExample, solve with cvxpy for x_star.

    Returns a dict of float64 torch tensors:
        P          : (n, n)
        q          : (n,)
        A          : (m, n)
        l          : (m,)   may contain -inf
        u          : (m,)
        x_star     : (n,)
        y_star     : (m,)
        z_star     : (m,)   A @ x_star  (slack at optimality)
        rho_vec    : (m,)   initial per-constraint penalties
        rho_inv    : (m,)   1 / rho_vec
        constr_type: (m,)   int32  — CONSTR_LOOSE / EQ / INEQ

        # Precomputed matrices for spectral_radius_loss.
        # These depend only on P, A, and cfg.sigma — never on rho or the iterate,
        # so they are valid for the entire lifetime of the dataset.
        R          : (n, n)   (P + σI)^{-1}
        AR         : (m, n)   A @ R
        ARAt       : (m, m)   A @ R @ A^T

    Returns None if cvxpy fails to find a solution.
    """
    if not _CVXPY_AVAILABLE:
        raise RuntimeError("cvxpy is required for data generation")

    qp = RandomQPExample(n, seed=seed)

    # Solve with cvxpy to get x_star
    try:
        qp.cvxpy_problem.solve(
            solver=cvxpy.OSQP,
            eps_abs=1e-8,
            eps_rel=1e-8,
            max_iter=10000,
            warm_start=False,
            verbose=False,
        )
    except Exception:
        return None

    if qp.cvxpy_problem.status not in ('optimal', 'optimal_inaccurate'):
        return None

    x_star_np, y_star_np, z_star_np = qp.revert_cvxpy_solution()
    if x_star_np is None or np.any(np.isnan(x_star_np)):
        return None
    if y_star_np is None or np.any(np.isnan(y_star_np)):
        return None
    if z_star_np is None or np.any(np.isnan(z_star_np)):
        return None
    m = int(n * 10)
    l_np = qp.l.copy()
    u_np = qp.u.copy()

    # Classify constraints and build initial rho_vec (from original l, u)
    constr_type = get_constr_type_np(l_np, u_np)
    rho_vec = build_rho_vec(constr_type, cfg.rho)
    rho_inv = 1.0 / rho_vec

    P_np = qp.P.toarray()
    A_np = qp.A.toarray()

    # ---- Ruiz equilibration (mirrors _osqp.py scale_data) ----
    d_vec, e_vec, c_scale = ruiz_equilibration(P_np, qp.q, A_np, l_np, u_np)
    d_inv_np = 1.0 / d_vec
    e_inv_np = 1.0 / e_vec
    c_inv_val = 1.0 / c_scale

    # Scaled problem matrices
    P_sc, q_sc, A_sc, l_sc, u_sc = apply_scaling(
        P_np, qp.q, A_np, l_np, u_np, d_vec, e_vec, c_scale
    )

    # Scaled optimal solutions:
    #   x_sc = D^{-1} @ x_orig   (x_orig = D @ x_sc)
    #   y_sc = c * E^{-1} @ y_orig
    #   z_sc = E @ z_orig         (z_orig = E^{-1} @ z_sc)
    x_star_sc = d_inv_np * x_star_np
    y_star_sc = c_scale * e_inv_np * y_star_np
    z_star_sc = e_vec * z_star_np

    # Precompute static matrices for spectral_radius_loss (from scaled P̃, Ã)
    K_sc = P_sc + cfg.sigma * np.eye(n)
    R_sc    = np.linalg.inv(K_sc)       # (n, n)
    AR_sc   = A_sc @ R_sc               # (m, n)
    ARAt_sc = AR_sc @ A_sc.T            # (m, m)

    instance = {
        # Scaled problem matrices (used directly in solver)
        'P': torch.tensor(P_sc, dtype=torch.float64),
        'q': torch.tensor(q_sc, dtype=torch.float64),
        'A': torch.tensor(A_sc, dtype=torch.float64),
        'l': torch.tensor(l_sc, dtype=torch.float64),
        'u': torch.tensor(u_sc, dtype=torch.float64),
        # Scaled optimal solutions (for loss / convergence_mask)
        'x_star': torch.tensor(x_star_sc, dtype=torch.float64),
        'y_star': torch.tensor(y_star_sc, dtype=torch.float64),
        'z_star': torch.tensor(z_star_sc, dtype=torch.float64),
        # ADMM penalty (unaffected by scaling, mirrors OSQP set_rho_vec order)
        'rho_vec': torch.tensor(rho_vec, dtype=torch.float64),
        'rho_inv': torch.tensor(rho_inv, dtype=torch.float64),
        'constr_type': torch.tensor(constr_type, dtype=torch.int32),
        'n': n,
        'm': m,
        # Precomputed spectral-loss matrices (from scaled P̃, Ã)
        'R': torch.tensor(R_sc, dtype=torch.float64),
        'AR': torch.tensor(AR_sc, dtype=torch.float64),
        'ARAt': torch.tensor(ARAt_sc, dtype=torch.float64),
        # Scaling parameters (for unscaling residuals in convergence check)
        'd_scale': torch.tensor(d_vec, dtype=torch.float64),
        'e_scale': torch.tensor(e_vec, dtype=torch.float64),
        'c_scale': torch.tensor(c_scale, dtype=torch.float64),
        'd_inv':   torch.tensor(d_inv_np, dtype=torch.float64),
        'e_inv':   torch.tensor(e_inv_np, dtype=torch.float64),
        'c_inv':   torch.tensor(c_inv_val, dtype=torch.float64),
    }

    bl_iters, bl_rho = compute_baseline_stats(instance, cfg)
    instance['baseline_iters'] = torch.tensor(bl_iters, dtype=torch.int64)
    instance['baseline_rho_updates'] = torch.tensor(bl_rho, dtype=torch.int64)
    return instance


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #

class QPDataset(Dataset):
    """
    Pre-generated dataset of fixed-size QP instances.
    All instances must have the same (n, m) for batching without padding.
    """

    def __init__(self, instances: list):
        self.instances = [x for x in instances if x is not None]

    def __len__(self) -> int:
        return len(self.instances)

    def __getitem__(self, idx: int) -> dict:
        return self.instances[idx]


def collate_fn(batch: list) -> dict:
    """
    Collate a list of instance dicts into a batched dict.
    Stacks tensor fields along a new batch dimension (dim=0).
    Scalar fields (n, m) are taken from the first element (all equal).
    """
    out = {}
    for key in batch[0]:
        if key in ('n', 'm'):
            out[key] = batch[0][key]
        else:
            out[key] = torch.stack([b[key] for b in batch], dim=0)
    return out


# --------------------------------------------------------------------------- #
# Dataset generation and caching
# --------------------------------------------------------------------------- #

def generate_dataset(cfg: Config, n_fixed: int | None = None, verbose: bool = True) -> QPDataset:
    """
    Generate cfg.n_train + cfg.n_val QP instances at fixed n.
    Seeds 0..N-1 are used for reproducibility.
    """
    n = n_fixed if n_fixed is not None else cfg.n_fixed
    N = cfg.n_train + cfg.n_val

    instances = []
    n_failed = 0
    for seed in range(N):
        if verbose and seed % 100 == 0:
            print(f"  Generating instance {seed}/{N}...", flush=True)
        inst = generate_qp_instance(n, seed, cfg)
        if inst is None:
            n_failed += 1
        else:
            instances.append(inst)

    if verbose:
        print(f"  Generated {len(instances)}/{N} instances ({n_failed} failed)")

    return QPDataset(instances)


def load_or_generate(
    cfg: Config,
    n_fixed: int | None = None,
    verbose: bool = True,
) -> tuple[QPDataset, QPDataset]:
    """
    Load dataset from disk if it exists, otherwise generate and save.

    Returns:
        (train_dataset, val_dataset) — both are QPDataset instances.
    """
    path = Path(cfg.data_path)

    if path.exists():
        if verbose:
            print(f"Loading dataset from {path}...", flush=True)
        raw = torch.load(str(path), weights_only=False)
        dataset = QPDataset(raw)
    else:
        if verbose:
            n = n_fixed if n_fixed is not None else cfg.n_fixed
            print(f"Generating dataset (n={n}, N={cfg.n_train + cfg.n_val})...", flush=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        dataset = generate_dataset(cfg, n_fixed, verbose)
        torch.save(dataset.instances, str(path))
        if verbose:
            print(f"Saved dataset to {path}")

    # Split into train/val
    n_val = min(cfg.n_val, len(dataset))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    return train_ds, val_ds


def make_dataloaders(
    cfg: Config,
    n_fixed: int | None = None,
    verbose: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """Convenience wrapper: returns (train_loader, val_loader)."""
    train_ds, val_ds = load_or_generate(cfg, n_fixed, verbose)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
    )
    return train_loader, val_loader


# --------------------------------------------------------------------------- #
# Batched utility: update rho_vec/rho_inv after rho scalar changes
# --------------------------------------------------------------------------- #

def update_rho_vec_batched(
    constr_type: torch.Tensor,   # (B, m) int32
    rho_scalar: torch.Tensor,    # (B,) float64
    rho_min: float = RHO_MIN,
    rho_eq_factor: float = RHO_EQ_OVER_RHO_INEQ,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Recompute rho_vec and rho_inv for a batch after rho scalar update.
    Mirrors _osqp.py update_rho (lines 1704-1719).

    Args:
        constr_type : (B, m) — constraint type flags
        rho_scalar  : (B,)  — new per-instance rho scalar
        rho_min     : minimum rho for loose constraints

    Returns:
        rho_vec : (B, m)
        rho_inv : (B, m)
    """
    B, m = constr_type.shape
    device = constr_type.device
    dtype = rho_scalar.dtype

    # Expand rho_scalar to (B, m) for broadcasting
    rho_b = rho_scalar.unsqueeze(1).expand(B, m)   # (B, m)

    rho_vec = torch.where(
        constr_type == CONSTR_LOOSE,
        torch.full((B, m), rho_min, dtype=dtype, device=device),
        torch.where(
            constr_type == CONSTR_EQ,
            rho_eq_factor * rho_b,
            rho_b,  # CONSTR_INEQ
        ),
    )
    rho_inv = 1.0 / rho_vec
    return rho_vec, rho_inv
