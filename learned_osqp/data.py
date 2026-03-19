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

SUPPORTED_QP_TYPES = [
    'random_qp', 'control', 'eq_qp', 'huber', 'lasso', 'portfolio',
    'svm', 'suitesparse_huber', 'suitesparse_lasso',
]

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
    eps_abs, eps_rel = cfg.eps_abs, cfg.eps_rel

    # Scaling params for unscaled convergence check (all in scaled space)
    d_inv = instance['d_inv'].unsqueeze(0)   # (1, n)
    e_inv = instance['e_inv'].unsqueeze(0)   # (1, m)
    c_inv = float(instance['c_inv'].item())  # scalar

    factors = factorize_kkt(P, A, cfg.sigma, rho_vec)

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
                    factors, cfg.alpha_x, alpha_z, cfg.sigma, A,
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
                    factors = factorize_kkt(P, A, cfg.sigma, rho_vec)
                    alpha_z = torch.full((1, m_dim), alpha, dtype=dtype)

    return iters_to_converge, rho_updates


# --------------------------------------------------------------------------- #
# Instance generation
# --------------------------------------------------------------------------- #

def _suitesparse_files(size_param: str) -> list:
    """
    Return a sorted list of .mat file paths (without extension) for suitesparse types.

    If size_param is a directory, scan for all *.mat files inside.
    If size_param is a file path (with or without .mat extension), return it alone.
    Files within a single dataset should all share the same (n, m) so that
    instances can be batched together.
    """
    p = Path(size_param)
    if p.is_dir():
        return sorted(str(f.with_suffix('')) for f in p.glob('*.mat'))
    # Treat as file path; strip .mat suffix if present
    return [str(p.with_suffix(''))]


def _make_qp_object(type_name: str, size_param, seed: int):
    """Instantiate the right problem class for the given type.

    size_param:
      - int   for random_qp, control, eq_qp, huber, lasso, portfolio, svm
      - str   for suitesparse_huber (directory or file path)
              and suitesparse_lasso (file path; lambda varied by seed)
    """
    if type_name == 'random_qp':
        return RandomQPExample(size_param, seed=seed)
    # Lazy imports so non-random_qp types don't require all dependencies at module load
    if type_name == 'control':
        from problem_classes.control import ControlExample
        return ControlExample(size_param, seed)
    if type_name == 'eq_qp':
        from problem_classes.eq_qp import EqQPExample
        return EqQPExample(size_param, seed)
    if type_name == 'huber':
        from problem_classes.huber import HuberExample
        return HuberExample(size_param, seed=seed)
    if type_name == 'lasso':
        from problem_classes.lasso import LassoExample
        return LassoExample(size_param, seed=seed)
    if type_name == 'portfolio':
        from problem_classes.portfolio import PortfolioExample
        return PortfolioExample(size_param, seed=seed)
    if type_name == 'svm':
        from problem_classes.svm import SVMExample
        return SVMExample(size_param, seed=seed)
    if type_name == 'suitesparse_huber':
        from problem_classes.suitesparse_huber import SuitesparseHuber
        files = _suitesparse_files(str(size_param))
        if not files:
            raise ValueError(f"No .mat files found for suitesparse_huber at: {size_param}")
        # Cycle through available files using seed
        return SuitesparseHuber(files[seed % len(files)])
    if type_name == 'suitesparse_lasso':
        from problem_classes.suitesparse_lasso import SuitesparseLasso
        files = _suitesparse_files(str(size_param))
        if not files:
            raise ValueError(f"No .mat files found for suitesparse_lasso at: {size_param}")
        # Use a single file per dataset (user should provide one file so all instances
        # share the same (n, m)). Vary lambda_param by seed to get different instances.
        qp = SuitesparseLasso(files[seed % len(files)])
        rng = np.random.default_rng(seed)
        lam = float(rng.uniform(0.05, 1.0)) * qp.lambda_max
        qp.update_lambda(lam)
        return qp
    raise ValueError(f"Unknown QP type '{type_name}'. Supported: {SUPPORTED_QP_TYPES}")


def _solve_with_osqppurepy(qp_prob: dict):
    """
    Solve a QP directly using the pure-Python OSQP implementation.

    Bypasses cvxpy entirely — used as a fallback when cvxpy's numerical PSD
    certification fails (e.g. ARPACK non-convergence on large QN matrices).

    Returns (x_star, y_star, z_star) numpy arrays, or (None, None, None).
    The dual y_star is already in OSQP convention (Px + q + A^T y = 0).
    """
    from solvers.osqppurepy import OSQP as _OSQP
    P = qp_prob['P']   # sparse
    q = np.asarray(qp_prob['q']).ravel()
    A = qp_prob['A']   # sparse
    l = np.asarray(qp_prob['l']).ravel()
    u = np.asarray(qp_prob['u']).ravel()
    try:
        solver = _OSQP()
        solver.setup(P, q, A, l, u,
                     eps_abs=1e-6, eps_rel=1e-6,
                     max_iter=100000, verbose=False, polish=True, scaling=10)
        result = solver.solve()
    except Exception:
        return None, None, None
    if result.status not in ('optimal', 'optimal inaccurate'):
        return None, None, None
    x = np.asarray(result.x, dtype=float)
    y = np.asarray(result.y, dtype=float)
    z = A.dot(x) if hasattr(A, 'dot') else A @ x
    if np.any(np.isnan(x)) or np.any(np.isnan(y)):
        return None, None, None
    return x, y, z


def _build_instance_from_qp(qp, cfg: Config) -> dict | None:
    """
    Shared logic: solve `qp`, apply Ruiz scaling, compute baseline stats.

    `qp` must expose:
        .qp_problem  — dict with keys P (sparse), q, A (sparse), l, u, n, m
        .cvxpy_problem
        .revert_cvxpy_solution() -> (x_star_np, y_star_np, z_star_np)

    Returns a dict of float64 torch tensors (same schema as generate_qp_instance),
    or None on failure.

    Strategy:
      1. Try CLARABEL (avoids ARPACK-based PSD check that fails for large QN).
      2. Try OSQP via cvxpy (works for smaller problems).
      3. Fall back to osqppurepy directly on the sparse matrices — completely
         bypasses cvxpy's PSD certification, always works.

    z_star=None from revert_cvxpy_solution (ControlExample TODO) is filled in
    as A @ x_star.
    """
    qp_prob = qp.qp_problem
    n = int(qp_prob['n'])
    m = int(qp_prob['m'])
    P_np = qp_prob['P'].toarray() if hasattr(qp_prob['P'], 'toarray') else np.asarray(qp_prob['P'])
    A_np = qp_prob['A'].toarray() if hasattr(qp_prob['A'], 'toarray') else np.asarray(qp_prob['A'])
    q_np = np.asarray(qp_prob['q']).ravel().copy()
    l_np = np.asarray(qp_prob['l']).ravel().copy()
    u_np = np.asarray(qp_prob['u']).ravel().copy()

    # Step 1-2: try cvxpy
    x_star_np = y_star_np = z_star_np = None
    _solver_configs = [
        dict(solver=cvxpy.OSQP, eps_abs=1e-6, eps_rel=1e-6,
             max_iter=100000, warm_start=False, verbose=False),
    ]
    for kwargs in _solver_configs:
        try:
            qp.cvxpy_problem.solve(**kwargs)
            if qp.cvxpy_problem.status in ('optimal', 'optimal_inaccurate'):
                rev = qp.revert_cvxpy_solution()
                # Some classes return (x, y) and others return (x, y, z)
                if len(rev) == 3:
                    x_star_np, y_star_np, z_star_np = rev
                else:
                    x_star_np, y_star_np = rev
                    z_star_np = None
                break
        except Exception:
            continue

    # Step 3: fall back to osqppurepy if cvxpy failed or returned bad values
    if (x_star_np is None or np.any(np.isnan(x_star_np))
            or y_star_np is None or np.any(np.isnan(y_star_np))):
        x_star_np, y_star_np, z_star_np = _solve_with_osqppurepy(qp_prob)

    if x_star_np is None or np.any(np.isnan(x_star_np)):
        return None
    if y_star_np is None or np.any(np.isnan(y_star_np)):
        return None

    # z_star=None (ControlExample has TODO): compute from A @ x_star
    if z_star_np is None:
        z_star_np = A_np @ x_star_np
    if np.any(np.isnan(z_star_np)):
        return None

    constr_type = get_constr_type_np(l_np, u_np)
    rho_vec = build_rho_vec(constr_type, cfg.rho)
    rho_inv = 1.0 / rho_vec

    d_vec, e_vec, c_scale = ruiz_equilibration(P_np, q_np, A_np, l_np, u_np)
    d_inv_np = 1.0 / d_vec
    e_inv_np = 1.0 / e_vec
    c_inv_val = 1.0 / c_scale

    P_sc, q_sc, A_sc, l_sc, u_sc = apply_scaling(
        P_np, q_np, A_np, l_np, u_np, d_vec, e_vec, c_scale
    )

    x_star_sc = d_inv_np * x_star_np
    y_star_sc = c_scale * e_inv_np * y_star_np
    z_star_sc = e_vec * z_star_np

    instance = {
        'P': torch.tensor(P_sc, dtype=torch.float64),
        'q': torch.tensor(q_sc, dtype=torch.float64),
        'A': torch.tensor(A_sc, dtype=torch.float64),
        'l': torch.tensor(l_sc, dtype=torch.float64),
        'u': torch.tensor(u_sc, dtype=torch.float64),
        'x_star': torch.tensor(x_star_sc, dtype=torch.float64),
        'y_star': torch.tensor(y_star_sc, dtype=torch.float64),
        'z_star': torch.tensor(z_star_sc, dtype=torch.float64),
        'rho_vec': torch.tensor(rho_vec, dtype=torch.float64),
        'rho_inv': torch.tensor(rho_inv, dtype=torch.float64),
        'constr_type': torch.tensor(constr_type, dtype=torch.int32),
        'n': n,
        'm': m,
        'd_scale': torch.tensor(d_vec,     dtype=torch.float64),
        'e_scale': torch.tensor(e_vec,     dtype=torch.float64),
        'c_scale': torch.tensor(c_scale,   dtype=torch.float64),
        'd_inv':   torch.tensor(d_inv_np,  dtype=torch.float64),
        'e_inv':   torch.tensor(e_inv_np,  dtype=torch.float64),
        'c_inv':   torch.tensor(c_inv_val, dtype=torch.float64),
    }

    # R, AR, ARAt are only needed for spectral_radius_loss.
    # Skip them to save memory (e.g. SVM: 127MB/instance × 60 = 7.6GB).
    if getattr(cfg, 'store_spectral_matrices', True):
        K_sc    = P_sc + cfg.sigma * np.eye(n)
        R_sc    = np.linalg.inv(K_sc)
        AR_sc   = A_sc @ R_sc
        ARAt_sc = AR_sc @ A_sc.T
        instance['R']    = torch.tensor(R_sc,    dtype=torch.float64)
        instance['AR']   = torch.tensor(AR_sc,   dtype=torch.float64)
        instance['ARAt'] = torch.tensor(ARAt_sc, dtype=torch.float64)

    bl_iters, bl_rho = compute_baseline_stats(instance, cfg)
    instance['baseline_iters'] = torch.tensor(bl_iters, dtype=torch.int64)
    instance['baseline_rho_updates'] = torch.tensor(bl_rho, dtype=torch.int64)
    return instance


def generate_qp_instance(n: int, seed: int, cfg: Config) -> dict | None:
    """Generate one random_qp instance (backward-compatible wrapper)."""
    if not _CVXPY_AVAILABLE:
        raise RuntimeError("cvxpy is required for data generation")
    return _build_instance_from_qp(RandomQPExample(n, seed=seed), cfg)


def generate_qp_instance_for_type(
    type_name: str, size_param, seed: int, cfg: Config
) -> dict | None:
    """Generate one QP instance of the given type and size parameter."""
    if not _CVXPY_AVAILABLE:
        raise RuntimeError("cvxpy is required for data generation")
    return _build_instance_from_qp(_make_qp_object(type_name, size_param, seed), cfg)


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
    Generate cfg.n_train + cfg.n_val random_qp instances at fixed n.
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


def generate_dataset_for_type(
    type_name: str, size_param, cfg: Config, verbose: bool = True
) -> QPDataset:
    """
    Generate cfg.n_train + cfg.n_val instances of the given QP type.
    All instances share the same (n, m) determined by type_name and size_param.
    """
    N = cfg.n_train + cfg.n_val
    instances = []
    n_failed = 0
    for seed in range(N):
        if verbose and seed % 50 == 0:
            print(f"  [{type_name}] Generating instance {seed}/{N}...", flush=True)
        inst = generate_qp_instance_for_type(type_name, size_param, seed, cfg)
        if inst is None:
            n_failed += 1
        else:
            instances.append(inst)
    if verbose:
        print(f"  [{type_name}] Generated {len(instances)}/{N} instances ({n_failed} failed)")
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


def dataset_path(type_name: str, size_param, cfg: Config) -> Path:
    """Return the cache path for a given type/size/precision/dtype combination.

    For suitesparse types size_param is a file/dir path string; it is sanitised
    to produce a valid filename component.
    """
    sp_str = str(size_param).replace('/', '_').replace('\\', '_').replace(' ', '_')
    return (
        Path(cfg.data_dir)
        / f"{type_name}_s{sp_str}_p{cfg.precision}_d{cfg.dtype}_adaptive_rho={cfg.adaptive_rho}.pt"
    )


def load_or_generate_type(
    type_name: str, cfg: Config, verbose: bool = True
) -> tuple[Dataset, Dataset]:
    """
    Load (or generate and cache) a dataset for one QP type.

    The size parameter is taken from cfg.qp_type_sizes[type_name], falling back
    to cfg.n_fixed for 'random_qp' and 1 for other int-based types.
    For suitesparse_huber/lasso there is no default — the user must provide a path
    in cfg.qp_type_sizes.

    Returns (train_dataset, val_dataset).
    """
    size_param = cfg.qp_type_sizes.get(
        type_name, cfg.n_fixed if type_name == 'random_qp' else 1
    )
    path = dataset_path(type_name, size_param, cfg)

    if path.exists():
        if verbose:
            print(f"Loading [{type_name}] dataset from {path}...", flush=True)
        raw = torch.load(str(path), weights_only=False)
        ds = QPDataset(raw)
    else:
        if verbose:
            print(
                f"Generating [{type_name}] dataset "
                f"(size_param={size_param}, N={cfg.n_train + cfg.n_val})...",
                flush=True,
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        ds = generate_dataset_for_type(type_name, size_param, cfg, verbose)
        torch.save(ds.instances, str(path))
        if verbose:
            print(f"Saved [{type_name}] dataset to {path}")

    n_val = min(cfg.n_val, len(ds))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(
        ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    return train_ds, val_ds


def make_dataloaders_multi(
    cfg: Config, verbose: bool = True
) -> dict[str, tuple[DataLoader, DataLoader]]:
    """
    Build one (train_loader, val_loader) pair per QP type listed in cfg.qp_types.

    Instances within each type share the same (n, m), so type-homogeneous
    batching requires no padding.  The caller is responsible for iterating
    across types (e.g. round-robin or sequential) during training.

    Returns:
        dict mapping type_name → (train_loader, val_loader)
    """
    result: dict[str, tuple[DataLoader, DataLoader]] = {}
    for type_name in cfg.qp_types:
        train_ds, val_ds = load_or_generate_type(type_name, cfg, verbose)
        train_loader = DataLoader(
            train_ds, batch_size=cfg.batch_size, shuffle=True,
            collate_fn=collate_fn, drop_last=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=cfg.batch_size, shuffle=False,
            collate_fn=collate_fn, drop_last=False,
        )
        result[type_name] = (train_loader, val_loader)
    return result


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
