"""
loss.py — Loss functions for the learned OSQP pipeline.

Three loss functions are available (selected by loss_type in train.py):

1. log_convergence_loss (default):
    loss = log( (||x_{k+T} - x*||^2 + ||y_{k+T} - y*||^2 + eps)
              / (||x_k   - x*||^2 + ||y_k    - y*||^2 + eps) )
   Minimise this to maximise the geometric reduction per stage.

2. spectral_radius_loss:
    Minimises (spectral_radius(T))^2 where T is the (n+2m)×(n+2m)
    ADMM transition operator at the current iterate (x, z, y).
    T depends on alpha_z through the sub-blocks Sx, Sz, Sy.
    Gradients flow through alpha_z → Sx/Sz/Sy → T → max|eigval(T)|.

3. scaled_residual_loss:
    Uses the normalised primal and dual residuals (inf-norm variant of the
    OSQP termination criterion):
        scaled_prim = ||Ax - z||_inf / max(||Ax||_inf, ||z||_inf)
        scaled_dual = ||Px + q + A^T y||_inf / max(||Px||_inf, ||A^T y||_inf, ||q||_inf)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from learned_osqp.config import Config


def log_convergence_loss(
    x_new: torch.Tensor,    # (B, n) — iterate after T steps (has gradient)
    x_prev: torch.Tensor,   # (B, n) — iterate before T steps (detached)
    x_star: torch.Tensor,   # (B, n) — optimal solution (detached)
    y_new: torch.Tensor,    # (B, m) — dual iterate after T steps (has gradient)
    y_prev: torch.Tensor,   # (B, m) — dual iterate before T steps (detached)
    y_star: torch.Tensor,   # (B, m) — optimal dual solution (detached)
    z_new: torch.Tensor,    # (B, m) — slack iterate after T steps (has gradient)
    z_prev: torch.Tensor,   # (B, m) — slack iterate before T steps (detached)
    z_star: torch.Tensor,   # (B, m) — optimal slack solution (detached)
    cfg: 'Config',
    mask: torch.Tensor | None = None,  # (B,) bool — True = include in loss
) -> torch.Tensor:
    """
    Per-instance log convergence ratio loss.

    loss_i = log( (||x_new_i - x_star_i||^2 + eps)
                / (||x_prev_i - x_star_i||^2 + eps) )

    x_prev and x_star should both be detached (no gradient).
    x_new carries gradients through the T-step OSQP rollout.

    Args:
        x_new  : (B, n) output of rollout_T_steps (grad-enabled)
        x_prev : (B, n) state before the rollout (detached)
        x_star : (B, n) ground-truth optimal (precomputed, detached)
        cfg    : Config containing loss_eps and convergence_tol
        mask   : (B,) bool tensor — if given, only masked=True instances
                 contribute to the mean. If None, all instances are used.

    Returns:
        scalar loss (mean over active instances)
    """
    eps = cfg.loss_eps

    ### 1. log_convergence_mul ###
    # # Numerator: depends on x_new (gradients flow here)
    # # err_new = torch.norm(x_new - x_star.detach(), dim=1) ** 2   # (B,)
    # err_new = torch.norm(x_new - x_star.detach(), dim=1) ** 2 * torch.norm(y_new - y_star.detach(), dim=1) ** 2  # (B,)
    # num = err_new + eps

    # # Denominator: fully detached
    # # err_prev = torch.norm(x_prev.detach() - x_star.detach(), dim=1) ** 2   # (B,)
    # err_prev = torch.norm(x_prev.detach() - x_star.detach(), dim=1) ** 2 * torch.norm(y_prev.detach() - y_star.detach(), dim=1) ** 2  # (B,)
    # denom = (err_prev + eps).detach()

    # ratio_log = torch.log(num / denom)   # (B,)  negative = improvement



    ### 2. log_convergence_add ###
    # # Numerator: depends on x_new (gradients flow here)
    # err_new = torch.norm(x_new - x_star.detach(), dim=1) ** 2 + torch.norm(y_new - y_star.detach(), dim=1) ** 2  # (B,)
    # num = err_new + eps

    # # Denominator: fully detached
    # err_prev = torch.norm(x_prev.detach() - x_star.detach(), dim=1) ** 2 + torch.norm(y_prev.detach() - y_star.detach(), dim=1) ** 2  # (B,)
    # denom = (err_prev + eps).detach()

    # ratio_log = torch.log(num / denom)   # (B,)  negative = improvement



    ### 3. log_convergence_add_sqrt ###
    # Numerator: depends on x_new (gradients flow here)
    err_new = torch.norm(x_new - x_star.detach(), dim=1) ** 2 + torch.norm(y_new - y_star.detach(), dim=1) ** 2  # (B,)
    num = err_new + eps

    # Denominator: fully detached
    err_prev = torch.norm(x_prev.detach() - x_star.detach(), dim=1) ** 2 + torch.norm(y_prev.detach() - y_star.detach(), dim=1) ** 2  # (B,)
    denom = (err_prev + eps).detach()

    ratio_log = torch.log(torch.sqrt(num / denom))   # (B,)  negative = improvement



    ### 4. log_convergence_add_sqrt_z ###
    # Numerator: depends on z_new (gradients flow here)
    # err_new = torch.norm(x_new - x_star.detach(), dim=1) ** 2 + torch.norm(y_new - y_star.detach(), dim=1) ** 2 + torch.norm(z_new - z_star.detach(), dim=1) ** 2  # (B,)
    # num = err_new + eps

    # # Denominator: fully detached
    # err_prev = torch.norm(x_prev.detach() - x_star.detach(), dim=1) ** 2 + torch.norm(y_prev.detach() - y_star.detach(), dim=1) ** 2 + torch.norm(z_prev.detach() - z_star.detach(), dim=1) ** 2  # (B,)
    # denom = (err_prev + eps).detach()

    # ratio_log = torch.log(torch.sqrt(num / denom))   # (B,)  negative = improvement

    if mask is not None:
        mask_f = mask.float()   # (B,)
        n_active = mask_f.sum().clamp(min=1.0)
        return (ratio_log * mask_f).sum() / n_active
    else:
        return ratio_log.mean()


def convergence_mask(
    P: torch.Tensor,       # (B, n, n)  SCALED
    A: torch.Tensor,       # (B, m, n)  SCALED
    q: torch.Tensor,       # (B, n)     SCALED
    x: torch.Tensor,       # (B, n)     scaled iterate
    z: torch.Tensor,       # (B, m)     scaled iterate
    y: torch.Tensor,       # (B, m)     scaled iterate
    d_inv: torch.Tensor,   # (B, n)     D^{-1} diagonal
    e_inv: torch.Tensor,   # (B, m)     E^{-1} diagonal
    c_inv: torch.Tensor,   # (B,)       1 / c_scale
    cfg: 'Config',
    AT: torch.Tensor | None = None,  # (B, n, m) precomputed A^T contiguous
) -> torch.Tensor:         # (B,) bool — True = NOT yet converged (include in loss)
    """
    Return a (B,) bool mask: True = instance has NOT yet converged.

    Mirrors the OSQP termination criterion in _osqp.py with scaled_termination=False:

        unscaled pri_res = E^{-1} * (A_sc @ x_sc - z_sc)
        unscaled dua_res = c^{-1} * D^{-1} * (P_sc @ x_sc + q_sc + A_sc^T @ y_sc)

        eps_pri = eps_abs + eps_rel * max(||E^{-1} A x||_inf, ||E^{-1} z||_inf)
        eps_dua = eps_abs + eps_rel * max(||c^{-1} D^{-1} A^T y||_inf,
                                          ||c^{-1} D^{-1} P x||_inf,
                                          ||c^{-1} D^{-1} q||_inf)

    eps_abs and eps_rel are taken from cfg.eps_abs / cfg.eps_rel (set by cfg.precision).

    Returns:
        active : (B,) bool  — True means "not yet converged, include in loss"
    """
    with torch.no_grad():
        if AT is None:
            AT = A.transpose(1, 2)
        Ax  = torch.bmm(A, x.unsqueeze(-1)).squeeze(-1)                    # (B, m)
        ATy = torch.bmm(AT, y.unsqueeze(-1)).squeeze(-1)                   # (B, n)
        Px  = torch.bmm(P, x.unsqueeze(-1)).squeeze(-1)                    # (B, n)

        c_inv_u = c_inv.unsqueeze(1)                                        # (B, 1)

        # Unscaled residuals
        pri_res_unc = e_inv * (Ax - z)                                      # (B, m)
        dua_res_unc = c_inv_u * d_inv * (Px + q + ATy)                     # (B, n)

        # Tolerances in original (unscaled) space
        eps_pri = cfg.eps_abs + cfg.eps_rel * torch.maximum(
            (e_inv * Ax).abs().amax(dim=1),
            (e_inv * z).abs().amax(dim=1),
        )                                                                    # (B,)
        eps_dua = cfg.eps_abs + cfg.eps_rel * torch.stack([
            (c_inv_u * d_inv * ATy).abs().amax(dim=1),
            (c_inv_u * d_inv * Px).abs().amax(dim=1),
            (c_inv_u * d_inv * q).abs().amax(dim=1),
        ], dim=1).amax(dim=1)                                               # (B,)

        converged = (
            pri_res_unc.abs().amax(dim=1) < eps_pri
        ) & (
            dua_res_unc.abs().amax(dim=1) < eps_dua
        )
        return ~converged   # True = still active (not yet converged)


def primal_residual(
    A: torch.Tensor,   # (B, m, n)
    x: torch.Tensor,   # (B, n)
    z: torch.Tensor,   # (B, m)
) -> torch.Tensor:
    """
    Compute ||Ax - z||_2 per batch instance. Returns (B,).
    """
    Ax = torch.bmm(A, x.unsqueeze(-1)).squeeze(-1)   # (B, m)
    return torch.norm(Ax - z, dim=1)                 # (B,)


def dual_residual(
    P: torch.Tensor,   # (B, n, n)
    A: torch.Tensor,   # (B, m, n)
    q: torch.Tensor,   # (B, n)
    x: torch.Tensor,   # (B, n)
    y: torch.Tensor,   # (B, m)
) -> torch.Tensor:
    """
    Compute ||Px + q + A^T y||_2 per batch instance. Returns (B,).
    """
    Px = torch.bmm(P, x.unsqueeze(-1)).squeeze(-1)             # (B, n)
    ATy = torch.bmm(A.transpose(1, 2), y.unsqueeze(-1)).squeeze(-1)  # (B, n)
    return torch.norm(Px + q + ATy, dim=1)                     # (B,)


# --------------------------------------------------------------------------- #
# Spectral radius loss
# --------------------------------------------------------------------------- #

def spectral_radius_loss(
    z: torch.Tensor,         # (B, m) current slack — used for active-set detection
    alpha_z: torch.Tensor,   # (B, m) per-row relaxation — CARRIES GRADIENT
    batch: dict,
    cfg: 'Config',
    active_tol: float = 1e-4,
) -> torch.Tensor:
    """
    Spectral radius of the ADMM transition operator T, squared.

    T is the (n+2m)×(n+2m) block matrix governing the linear convergence of
    OSQP at the current iterate.  Its spectral radius ρ(T) < 1 is necessary
    and sufficient for local convergence; minimising ρ(T) maximises contraction.

    Block structure (generalised from scalar alpha to per-row alpha_z):

        T = [ T_xx         | T_xz          | T_xy         ]  (n rows)
            [ D_free Sx    | D_free Sz      | D_free Sy    ]  (m rows — inactive)
            [ D_ρ D_act Sx | D_ρ D_act Sz  | D_ρ D_act Sy ]  (m rows — active)

    where:
        K  = P + σI  →  R  = K⁻¹
        S  = A R Aᵀ + diag(ρ_inv)  →  W = S⁻¹
        a  = alpha_z * rho_inv  (per-row, carries gradient)
        Sx = σ · diag(a) @ W @ A @ R           (m, n)
        Sz = I − diag(a) @ W                   (m, m)
        Sy = diag(a) @ W @ diag(ρ_inv) + diag((1−α_z)·ρ_inv)   (m, m)

        T_xx = (1−α_x) I + α_x σ (R − R Aᵀ W A R)   (uses fixed scalar alpha_x)
        T_xz = α_x R Aᵀ W
        T_xy = −α_x R Aᵀ W diag(ρ_inv)

        D_free[i,i] = 1 if z[i] is NOT at a bound (inactive constraint)
        D_act [i,i] = 1 if z[i] IS  at a bound (active  constraint)

    R and W are computed under no_grad (they do not depend on alpha_z).
    Gradients flow: alpha_z → a → Sx/Sz/Sy → T → eigvals → ρ(T)² → loss.

    Args:
        z         : (B, m) current slack — only used to determine active set
        alpha_z   : per-row over-relaxation (B, m), output of PerRowAlphaNet
        batch     : QP data dict (P, A, l, u, rho_vec, rho_inv)
        cfg       : Config (uses cfg.sigma, cfg.alpha_x)
        active_tol: tolerance for deciding a constraint is active (z at bound)

    Returns:
        scalar — mean(spectral_radius²) over the batch
    """
    A = batch['A']          # (B, m, n)
    l = batch['l']          # (B, m)
    u = batch['u']          # (B, m)
    rho_vec = batch['rho_vec']  # (B, m)
    rho_inv = batch['rho_inv']  # (B, m)

    # Static precomputed matrices (stored in dataset, never depend on rho)
    R = batch['R']          # (B, n, n)  (P + σI)^{-1}
    AR = batch['AR']        # (B, m, n)  A @ R
    ARAt = batch['ARAt']    # (B, m, m)  A @ R @ A^T

    B, n = R.shape[:2]
    m = A.shape[1]
    device = A.device
    dtype = A.dtype

    # ---- Compute W from current rho_inv (no grad — doesn't depend on alpha_z) ---- #
    with torch.no_grad():
        S = ARAt + torch.diag_embed(rho_inv)    # (B, m, m)
        W = torch.linalg.inv(S)                 # (B, m, m)

    W = W.detach()

    # ---- Active constraint detection ---- #
    with torch.no_grad():
        # A constraint is "active" if z is at (or beyond) one of its bounds.
        # For loose constraints (l=-inf, u=+inf), these comparisons are always
        # False, so loose rows are never marked active. ✓
        at_lower = z <= l + active_tol          # (B, m)
        at_upper = z >= u - active_tol          # (B, m)
        active = at_lower | at_upper            # (B, m) bool
        inactive = ~active                      # (B, m) bool

    # Diagonal mask matrices  (B, m, m)
    D_free = torch.diag_embed(inactive.to(dtype))   # 1 for inactive (free) rows
    D_act  = torch.diag_embed(active.to(dtype))      # 1 for active rows
    D_rho  = torch.diag_embed(rho_vec)               # diag(rho_vec)

    # ---- Sub-blocks Sx, Sz, Sy — depend on alpha_z (carry gradient) ---- #
    # a = alpha_z * rho_inv   (B, m)  — per-row diag(alpha/rho)
    a = alpha_z * rho_inv   # (B, m), grad flows here

    WAR = torch.bmm(W, AR)      # (B, m, n) = W @ A @ R  (W detached)

    # Sx = sigma * diag(a) @ W @ A @ R  →  row-scale WAR by a
    Sx = cfg.sigma * torch.einsum('bi,bij->bij', a, WAR)      # (B, m, n)

    # Sz = I − diag(a) @ W  →  row-scale W by a, subtract from I
    I_m = torch.eye(m, dtype=dtype, device=device).unsqueeze(0).expand(B, -1, -1)
    Sz = I_m - torch.einsum('bi,bij->bij', a, W)              # (B, m, m)

    # Sy = diag(a) @ W @ diag(rho_inv) + diag((1-alpha_z)*rho_inv)
    #    = col-scale(W, rho_inv) row-scaled by a,  + diag-part
    W_rhoinv = torch.einsum('bij,bj->bij', W, rho_inv)         # (B, m, m) = W @ diag(rho_inv)
    Sy = (torch.einsum('bi,bij->bij', a, W_rhoinv)             # diag(a) @ W @ diag(rho_inv)
          + torch.diag_embed((1.0 - alpha_z) * rho_inv))       # (B, m, m)

    # ---- X-block rows of T (use fixed scalar alpha_x) ---- #
    # R_At_W = R @ Aᵀ @ W  (B, n, m)
    R_At = R.bmm(A.transpose(1, 2))                             # (B, n, m)
    R_At_W = torch.bmm(R_At, W)                                 # (B, n, m)

    I_n = torch.eye(n, dtype=dtype, device=device).unsqueeze(0).expand(B, -1, -1)

    # T_xx = (1-alpha_x) I + alpha_x * sigma * (R - R Aᵀ W A R)
    R_At_W_AR = torch.bmm(R_At_W, AR)                           # (B, n, n) = R Aᵀ W A R
    T_xx = ((1.0 - cfg.alpha_x) * I_n
            + cfg.alpha_x * cfg.sigma * (R - R_At_W_AR))        # (B, n, n)

    # T_xz = alpha_x * R Aᵀ W                                   (B, n, m)
    T_xz = cfg.alpha_x * R_At_W

    # T_xy = -alpha_x * R Aᵀ W diag(rho_inv)  →  col-scale T_xz by rho_inv
    T_xy = -cfg.alpha_x * torch.einsum('bij,bj->bij', R_At_W, rho_inv)   # (B, n, m)

    # ---- Z-block rows (inactive, D_free @ [Sx, Sz, Sy]) ---- #
    T_zx = torch.bmm(D_free, Sx)    # (B, m, n)
    T_zz = torch.bmm(D_free, Sz)    # (B, m, m)
    T_zy = torch.bmm(D_free, Sy)    # (B, m, m)

    # ---- Y-block rows (active,  D_rho @ D_act @ [Sx, Sz, Sy]) ---- #
    D_rho_act = torch.bmm(D_rho, D_act)  # (B, m, m) — zero for inactive rows
    T_yx = torch.bmm(D_rho_act, Sx)      # (B, m, n)
    T_yz = torch.bmm(D_rho_act, Sz)      # (B, m, m)
    T_yy = torch.bmm(D_rho_act, Sy)      # (B, m, m)

    # ---- Assemble T: (B, n+2m, n+2m) ---- #
    top = torch.cat([T_xx, T_xz, T_xy], dim=2)    # (B, n, n+2m)
    mid = torch.cat([T_zx, T_zz, T_zy], dim=2)    # (B, m, n+2m)
    bot = torch.cat([T_yx, T_yz, T_yy], dim=2)    # (B, m, n+2m)
    T_mat = torch.cat([top, mid, bot], dim=1)      # (B, n+2m, n+2m)

    # ---- Spectral radius = max |eigenvalue| ---- #
    # eigvals returns complex (B, n+2m); take abs then max over eigenvalue dim
    eigvals_abs = torch.linalg.eigvals(T_mat).abs()          # (B, n+2m) real
    spectral_radius = eigvals_abs.max(dim=1).values   # (B,) real
    eigvals_loss = eigvals_abs.pow(2).mean()

    # MSE loss (target spectral radius = 0 → fastest convergence)
    # return spectral_radius.pow(2).mean()
    return eigvals_loss


# --------------------------------------------------------------------------- #
# Scaled residual loss
# --------------------------------------------------------------------------- #

def _progress_score(
    x: torch.Tensor,   # (B, n)
    z: torch.Tensor,   # (B, m)
    y: torch.Tensor,   # (B, m)
    P: torch.Tensor,   # (B, n, n)
    A: torch.Tensor,   # (B, m, n)
    q: torch.Tensor,   # (B, n)
    eps_abs: float,
    eps_rel: float,
) -> torch.Tensor:     # (B,)
    """
    s(x, z, y) = smoothmax( log(r_prim/ε_prim), log(r_dual/ε_dual) )
               = log( r_prim/ε_prim + r_dual/ε_dual )

    where smoothmax(a, b) = log(eᵃ + eᵇ)  (log-sum-exp).

    ε_prim = ε_abs + ε_rel * max(||Ax||_inf, ||z||_inf)
    ε_dual = ε_abs + ε_rel * max(||Px||_inf, ||A^T y||_inf, ||q||_inf)

    s ≤ 0  ⟺  iterate is within OSQP tolerance on both residuals.
    """
    Ax  = torch.bmm(A, x.unsqueeze(-1)).squeeze(-1)                   # (B, m)
    ATy = torch.bmm(A.transpose(1, 2), y.unsqueeze(-1)).squeeze(-1)   # (B, n)
    Px  = torch.bmm(P, x.unsqueeze(-1)).squeeze(-1)                   # (B, n)

    r_prim   = (Ax - z).abs().amax(dim=1)                             # (B,)
    eps_prim = eps_abs + eps_rel * torch.maximum(
        Ax.abs().amax(dim=1), z.abs().amax(dim=1))                    # (B,)

    r_dual   = (Px + q + ATy).abs().amax(dim=1)                       # (B,)
    eps_dual = eps_abs + eps_rel * torch.stack([
        Px.abs().amax(dim=1),
        ATy.abs().amax(dim=1),
        q.abs().amax(dim=1).expand(x.shape[0]),
    ], dim=1).amax(dim=1)                                              # (B,)

    # smoothmax(log(r/ε_p), log(r/ε_d)) = log(r_prim/ε_prim + r_dual/ε_dual)
    return torch.log(r_prim / eps_prim + r_dual / eps_dual + 1e-30)   # (B,)


def scaled_residual_loss(
    x_new: torch.Tensor,   # (B, n) — primal after T steps  (carries gradient)
    z_new: torch.Tensor,   # (B, m) — slack  after T steps  (carries gradient)
    y_new: torch.Tensor,   # (B, m) — dual   after T steps  (carries gradient)
    x_prev: torch.Tensor,  # (B, n) — primal before T steps (detached)
    z_prev: torch.Tensor,  # (B, m) — slack  before T steps (detached)
    y_prev: torch.Tensor,  # (B, m) — dual   before T steps (detached)
    batch: dict,           # must contain P, A, q
    mask: torch.Tensor | None = None,  # (B,) bool — True = include in loss
    eps_abs: float = 1e-3,
    eps_rel: float = 1e-3,
) -> torch.Tensor:
    """
    Progress-score loss:  L_ratio = s_new - s_prev

    s(x, z, y) = smoothmax( log(r_prim/ε_prim), log(r_dual/ε_dual) )
               = log( r_prim/ε_prim + r_dual/ε_dual )

    Minimising L_ratio encourages s to decrease as fast as possible.
    s ≤ 0 means both residuals are within OSQP tolerance.
    Gradients flow through x_new, z_new, y_new; s_prev is fully detached.

    Args:
        x_new/z_new/y_new  : iterates after  T steps (grad-enabled)
        x_prev/z_prev/y_prev: iterates before T steps (detached)
        batch    : QP data dict containing P (B,n,n), A (B,m,n), q (B,n)
        mask     : (B,) bool — if given, only True instances enter the mean
        eps_abs, eps_rel : OSQP tolerance parameters (default 1e-3)

    Returns:
        scalar loss (mean over active instances)
    """
    P = batch['P']   # (B, n, n)
    A = batch['A']   # (B, m, n)
    q = batch['q']   # (B, n)

    s_new = _progress_score(x_new, z_new, y_new, P, A, q, eps_abs, eps_rel)  # (B,)
    with torch.no_grad():
        s_prev = _progress_score(
            x_prev, z_prev, y_prev, P, A, q, eps_abs, eps_rel)               # (B,)

    loss_per_instance = s_new - s_prev   # (B,)  negative = improvement

    if mask is not None:
        mask_f   = mask.float()
        n_active = mask_f.sum().clamp(min=1.0)
        return (loss_per_instance * mask_f).sum() / n_active
    return loss_per_instance.mean()
