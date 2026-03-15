"""
osqp_torch.py — Differentiable OSQP ADMM iterations in PyTorch.

Ports the iteration logic from solvers/osqppurepy/_osqp.py into batched,
differentiable PyTorch operations. The implementation is a faithful translation
of the three ADMM steps (update_xz_tilde, update_x, update_z, update_y).

Key design decisions:
  - Uses the Schur complement to reduce the (n+m)×(n+m) KKT system to an n×n
    system: M = P + σI + Aᵀ diag(ρ) A. This gives O(n³) factorization + O(n²)
    per-step solve instead of O((n+m)³) + O((n+m)²) — a large speedup when m >> n.
  - The Schur complement factorization is computed once per stage under no_grad
    and stored as detached LU/pivot tensors. Gradients flow only through the
    RHS of each solve, not through the factored matrix.
  - Per-row alpha_z (B, m) replaces the scalar alpha_z in the reference code.
  - Adaptive rho is applied between stages, never within a 10-step rollout.
  - All tensors must be float64 for numerical stability.

Reference equations (_osqp.py lines 682-741), rewritten via Schur complement:

  Step 1 (update_xz_tilde via Schur complement):
    M = P + σI + Aᵀ diag(ρ) A                        (factored once per stage)
    rhs_reduced = σ * x_prev - q + Aᵀ(ρ·z_prev - y)  (built each iteration)
    x_tilde = M^{-1} rhs_reduced                      (n×n lu_solve)
    z_tilde = A @ x_tilde                             (matvec)

  Step 2a (update_x):
    x = alpha_x * x_tilde + (1 - alpha_x) * x_prev

  Step 2b (update_z):
    z_unclip = alpha_z * z_tilde + (1 - alpha_z) * z_prev + rho_inv * y
    z = clamp(z_unclip, l, u)

  Step 3 (update_y):
    delta_y = rho * (alpha_z * z_tilde + (1 - alpha_z) * z_prev - z)
    y = y + delta_y
"""

from __future__ import annotations

from typing import NamedTuple, TYPE_CHECKING
import torch

if TYPE_CHECKING:
    from learned_osqp.config import Config


# --------------------------------------------------------------------------- #
# Schur Complement Factorization
# --------------------------------------------------------------------------- #

class KKTFactors(NamedTuple):
    """Pre-computed LU factorization of the (B, n, n) Schur complement matrix
    M = P + σI + Aᵀ diag(ρ) A, plus cached Aᵀ for per-step RHS assembly."""
    LU: torch.Tensor      # (B, n, n) — packed LU factors of M
    pivots: torch.Tensor  # (B, n)    — pivot indices
    AT: torch.Tensor      # (B, n, m) — precomputed A^T contiguous


def factorize_kkt(
    P: torch.Tensor,        # (B, n, n)
    A: torch.Tensor,        # (B, m, n)
    sigma: float,
    rho_vec: torch.Tensor,  # (B, m)
    AT: torch.Tensor | None = None,  # (B, n, m) precomputed A^T contiguous
) -> KKTFactors:
    """
    Factor the n×n Schur complement: M = P + σI + Aᵀ diag(ρ) A.

    Eliminates the dual variable ν from the full (n+m)×(n+m) KKT system,
    reducing to an n×n system.  This gives O(n³) factorization + O(n²)
    per-step solve instead of O((n+m)³) + O((n+m)²).

    ALWAYS called under torch.no_grad() — LU factors and pivots are detached
    constants. Gradients only flow through the RHS of each solve.

    Args:
        P       : (B, n, n) positive semidefinite cost Hessian
        A       : (B, m, n) constraint matrix
        sigma   : scalar regularization (e.g. 1e-6)
        rho_vec : (B, m) per-constraint penalty parameter
        AT      : (B, n, m) optional precomputed A^T (contiguous)

    Returns:
        KKTFactors(LU, pivots, AT)
    """
    B, m, n = A.shape
    dtype, device = P.dtype, P.device

    with torch.no_grad():
        if AT is None:
            AT = A.transpose(1, 2).contiguous()  # (B, n, m)

        # M = P + σI + Aᵀ diag(ρ) A
        AT_rho = AT * rho_vec.unsqueeze(1)       # (B, n, m) — scale cols by ρ
        AtRA = torch.bmm(AT_rho, A)              # (B, n, n)

        I_n = sigma * torch.eye(n, dtype=dtype, device=device).unsqueeze(0).expand(B, -1, -1)
        M = P + I_n + AtRA                       # (B, n, n)

        LU, pivots = torch.linalg.lu_factor(M)

    return KKTFactors(LU=LU.detach(), pivots=pivots.detach(), AT=AT.detach())


# --------------------------------------------------------------------------- #
# Single ADMM Step
# --------------------------------------------------------------------------- #

def osqp_step(
    x: torch.Tensor,        # (B, n)
    z: torch.Tensor,        # (B, m)
    y: torch.Tensor,        # (B, m)
    q: torch.Tensor,        # (B, n)
    l: torch.Tensor,        # (B, m) — may contain -inf
    u: torch.Tensor,        # (B, m)
    rho_vec: torch.Tensor,  # (B, m)
    rho_inv: torch.Tensor,  # (B, m)
    factors: KKTFactors,    # pre-computed Schur complement factorization
    alpha_x: float,         # scalar, fixed (e.g. 1.6)
    alpha_z: torch.Tensor,  # (B, m) per-row relaxation parameter
    sigma: float,
    A: torch.Tensor,        # (B, m, n) constraint matrix
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Execute one OSQP ADMM step via Schur complement. Differentiable w.r.t. alpha_z.

    Uses the n×n Schur complement solve instead of the full (n+m)×(n+m) KKT:
        rhs_reduced = σx - q + Aᵀ(ρ·z - y)
        x_tilde = M⁻¹ rhs_reduced         (n×n lu_solve)
        z_tilde = A @ x_tilde              (matvec)

    Returns:
        x_new   : (B, n)
        z_new   : (B, m)
        y_new   : (B, m)
        x_tilde : (B, n)  — intermediate (useful for features)
        z_tilde : (B, m)  — intermediate (useful for features)
    """
    AT = factors.AT                                    # (B, n, m)

    # ---- Step 1: Schur complement solve ----------------------------------- #
    # rhs_reduced = σx - q + Aᵀ(ρ·z - y)
    rhs_reduced = sigma * x - q + torch.bmm(
        AT, (rho_vec * z - y).unsqueeze(-1)
    ).squeeze(-1)                                      # (B, n)

    # x_tilde = M⁻¹ rhs_reduced
    x_tilde = torch.linalg.lu_solve(
        factors.LU, factors.pivots, rhs_reduced.unsqueeze(-1)
    ).squeeze(-1)                                      # (B, n)

    # z_tilde = A @ x_tilde
    z_tilde = torch.bmm(A, x_tilde.unsqueeze(-1)).squeeze(-1)  # (B, m)

    # ---- Step 2a: x update ------------------------------------------------ #
    x_new = alpha_x * x_tilde + (1.0 - alpha_x) * x   # (B, n)

    # ---- Step 2b: z update ------------------------------------------------ #
    z_unclip = alpha_z * z_tilde + (1.0 - alpha_z) * z + rho_inv * y   # (B, m)
    z_new = torch.clamp(z_unclip, min=l, max=u)        # (B, m)

    # ---- Step 3: y update ------------------------------------------------- #
    z_bar = alpha_z * z_tilde + (1.0 - alpha_z) * z    # (B, m)
    delta_y = rho_vec * (z_bar - z_new)                # (B, m)
    y_new = y + delta_y                                 # (B, m)

    return x_new, z_new, y_new, x_tilde, z_tilde


# --------------------------------------------------------------------------- #
# T-step rollout
# --------------------------------------------------------------------------- #

def rollout_T_steps(
    x: torch.Tensor,        # (B, n) — initial state (from previous stage, detached)
    z: torch.Tensor,        # (B, m)
    y: torch.Tensor,        # (B, m)
    alpha_z: torch.Tensor,  # (B, m) — fixed for all T steps in this stage
    factors: KKTFactors,    # pre-computed LU factorization, from factorize_kkt
    batch: dict,            # contains q, l, u, rho_vec, rho_inv
    cfg: 'Config',
    alpha_x_override=None,  # (B, 1) tensor or None; overrides cfg.alpha_x when given
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Run cfg.T OSQP iterations with a fixed alpha_z.

    Features are computed ONCE at the stage start (before this call).
    alpha_z is held constant across all T iterations.

    Args:
        x, z, y          : current OSQP iterate (B, n/m/m)
        alpha_z          : per-row relaxation (B, m) — carries gradient
        factors          : KKTFactors — pre-computed LU factorization from factorize_kkt
        batch            : QP data dict
        cfg              : Config
        alpha_x_override : (B, 1) tensor or scalar float; if provided, overrides
                           cfg.alpha_x for the x-update.  Used in scalar alpha mode
                           where the same learned value is used for both alpha_x and
                           alpha_z.

    Returns:
        (x_new, z_new, y_new) after T steps
    """
    A = batch['A']
    q = batch['q']
    l = batch['l']
    u = batch['u']
    rho_vec = batch['rho_vec']
    rho_inv = batch['rho_inv']

    alpha_x = alpha_x_override if alpha_x_override is not None else cfg.alpha_x

    for _ in range(cfg.T):
        x, z, y, _, _ = osqp_step(
            x, z, y, q, l, u, rho_vec, rho_inv,
            factors, alpha_x, alpha_z, cfg.sigma, A,
        )

    return x, z, y


# --------------------------------------------------------------------------- #
# Adaptive rho
# --------------------------------------------------------------------------- #

def compute_rho_estimate_batched(
    P: torch.Tensor,        # (B, n, n)
    A: torch.Tensor,        # (B, m, n)
    q: torch.Tensor,        # (B, n)
    x: torch.Tensor,        # (B, n)
    z: torch.Tensor,        # (B, m)
    y: torch.Tensor,        # (B, m)
    rho_scalar: torch.Tensor,  # (B,) current rho per instance
    rho_min: float,
    rho_max: float,
) -> torch.Tensor:
    """
    Compute new rho estimates for each batch item.
    Ports _osqp.py compute_rho_estimate (lines 945-973).

    new_rho_i = rho_i * sqrt(pri_res_norm_i / dua_res_norm_i)
    clamped to [rho_min, rho_max].

    Called under torch.no_grad() — rho is not a learnable parameter.

    Returns:
        rho_new : (B,) new rho scalars
    """
    # Primal residual: ||Ax - z||_inf  (B,)
    Ax = torch.bmm(A, x.unsqueeze(-1)).squeeze(-1)      # (B, m)
    pri_res = torch.norm(Ax - z, p=float('inf'), dim=1)  # (B,)
    pri_norm = (
        torch.maximum(
            torch.norm(Ax, p=float('inf'), dim=1),
            torch.norm(z, p=float('inf'), dim=1),
        ) + 1e-10
    )   # (B,)
    pri_res_norm = pri_res / pri_norm   # (B,)

    # Dual residual: ||Px + q + A^T y||_inf  (B,)
    Px = torch.bmm(P, x.unsqueeze(-1)).squeeze(-1)              # (B, n)
    ATy = torch.bmm(A.transpose(1, 2), y.unsqueeze(-1)).squeeze(-1)  # (B, n)
    dua_res = torch.norm(Px + q + ATy, p=float('inf'), dim=1)   # (B,)
    dua_norm = (
        torch.maximum(
            torch.norm(ATy, p=float('inf'), dim=1),
            torch.maximum(
                torch.norm(Px, p=float('inf'), dim=1),
                torch.norm(q, p=float('inf'), dim=1),
            ),
        ) + 1e-10
    )   # (B,)
    dua_res_norm = dua_res / dua_norm   # (B,)

    rho_new = rho_scalar * torch.sqrt(pri_res_norm / (dua_res_norm + 1e-10))
    return rho_new.clamp(rho_min, rho_max)   # (B,)


def maybe_update_rho(
    batch: dict,
    x: torch.Tensor,        # (B, n) — detached state after stage
    z: torch.Tensor,        # (B, m)
    y: torch.Tensor,        # (B, m)
    rho_scalar: torch.Tensor,   # (B,) current rho per instance
    cfg: 'Config',
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute adaptive rho update between stages (always under no_grad).

    Mirrors _osqp.py adapt_rho logic (lines 975-995):
      - Compute new rho estimate
      - Only update if |rho_new/rho| > tol or |rho/rho_new| > tol
      - Update rho_vec and rho_inv for changed instances

    Args:
        batch       : QP data dict (rho_vec, rho_inv, constr_type updated in-place)
        x, z, y    : current iterate (detached)
        rho_scalar  : (B,) current per-instance rho
        cfg         : Config

    Returns:
        rho_scalar_new : (B,) updated rho scalars
        rho_vec_new    : (B, m) updated per-constraint rho
        rho_inv_new    : (B, m) updated per-constraint rho_inv
        updated        : (B,) bool — True for each instance whose rho was updated
    """
    from learned_osqp.data import update_rho_vec_batched

    P = batch['P']
    A = batch['A']
    q = batch['q']
    constr_type = batch['constr_type']   # (B, m) int32

    rho_new = compute_rho_estimate_batched(
        P, A, q, x, z, y, rho_scalar,
        cfg.rho_min, float(1e6),
    )   # (B,)

    tol = cfg.adaptive_rho_tolerance
    # Trigger update if rho_new > tol*rho or rho_new < rho/tol
    should_update = (rho_new > tol * rho_scalar) | (rho_new < rho_scalar / tol)   # (B,) bool

    if not should_update.any():
        return rho_scalar, batch['rho_vec'], batch['rho_inv'], should_update

    # Update rho_scalar where needed
    rho_scalar_new = torch.where(should_update, rho_new, rho_scalar)

    # Rebuild rho_vec / rho_inv for all instances with new scalar
    # (cheap; only re-assigned for changed instances below for safety)
    rho_vec_new, rho_inv_new = update_rho_vec_batched(
        constr_type, rho_scalar_new,
        rho_min=cfg.rho_min, rho_eq_factor=cfg.rho_eq_factor,
    )

    # Keep old values for instances that didn't change
    rho_vec_new = torch.where(
        should_update.unsqueeze(1), rho_vec_new, batch['rho_vec']
    )
    rho_inv_new = torch.where(
        should_update.unsqueeze(1), rho_inv_new, batch['rho_inv']
    )

    return rho_scalar_new, rho_vec_new, rho_inv_new, should_update


# --------------------------------------------------------------------------- #
# Baseline rollout (fixed scalar alpha — for evaluation)
# --------------------------------------------------------------------------- #

def baseline_rollout(
    batch: dict,
    cfg: 'Config',
    T_total: int = 100,
    alpha_scalar: float = 1.6,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Run T_total steps with fixed scalar alpha_z = alpha_scalar.
    Used as the baseline in eval.py.

    Returns:
        state_history : list of (x_t, z_t, y_t) after each step
    """
    P = batch['P']
    q = batch['q']
    A = batch['A']
    l = batch['l']
    u = batch['u']

    B, n = P.shape[0], P.shape[1]
    m = A.shape[1]
    device = P.device
    dtype = P.dtype

    rho_vec = batch['rho_vec'].clone()
    rho_inv = batch['rho_inv'].clone()
    rho_scalar = torch.full((B,), cfg.rho, dtype=dtype, device=device)
    constr_type = batch['constr_type']

    x = torch.zeros(B, n, dtype=dtype, device=device)
    z = torch.zeros(B, m, dtype=dtype, device=device)
    y = torch.zeros(B, m, dtype=dtype, device=device)

    alpha_z = torch.full((B, m), alpha_scalar, dtype=dtype, device=device)

    state_history = []

    # Build initial Schur complement factors
    factors = factorize_kkt(P, A, cfg.sigma, rho_vec)

    with torch.no_grad():
        for t in range(T_total):
            x, z, y, _, _ = osqp_step(
                x, z, y, q, l, u, rho_vec, rho_inv,
                factors, cfg.alpha_x, alpha_z, cfg.sigma, A,
            )
            state_history.append((x.clone(), z.clone(), y.clone()))

            # Adaptive rho every cfg.T steps (matching training frequency)
            if cfg.adaptive_rho and ((t + 1) % cfg.T == 0):
                _batch = {
                    'P': P, 'A': A, 'q': q,
                    'rho_vec': rho_vec, 'rho_inv': rho_inv,
                    'constr_type': constr_type,
                }
                rho_scalar, rho_vec, rho_inv, updated = maybe_update_rho(
                    _batch, x, z, y, rho_scalar, cfg
                )
                if updated.any():
                    factors = factorize_kkt(P, A, cfg.sigma, rho_vec)
                    alpha_z = torch.full((B, m), alpha_scalar, dtype=dtype, device=device)

    return state_history
