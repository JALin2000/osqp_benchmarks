"""
features.py — Per-row feature computation for PerRowAlphaNet.

Computes a (B, m, feature_dim) tensor of per-constraint-row features
from the current OSQP iterate state (x, z, y) and problem data.

Feature vector per row i (feature_dim = 13):
    f[0]  = log10(clamp(z_i - l_i))                      log distance to lower bound
    f[1]  = log10(clamp(u_i - z_i))                      log distance to upper bound
    f[2]  = log10(clamp(|pri_res_i| / pri_scale))         log scaled absolute primal residual
    f[3]  = sign(pri_res_i)                               sign of primal residual
    f[4]  = log10(clamp(|y_i|))                           log absolute dual variable
    f[5]  = log10(clamp(||pri_res||_inf / pri_scale))     log scaled primal residual inf norm (broadcast)
    f[6]  = log10(clamp(||dua_res||_inf / dua_scale))     log scaled dual residual inf norm (broadcast)
    f[7]  = log10(clamp(rho_i))                           log penalty parameter
    f[8]  = ||A[i,:]||_inf                                row inf norm of A
    f[9]  = log10(clamp(scaled_|pri_res_i| / scaled_|pri_res_i_prev|))  scaled per-row primal ratio
    f[10] = log10(clamp(scaled_||pri||_inf / scaled_||pri_prev||_inf))  scaled primal inf norm ratio (broadcast)
    f[11] = log10(clamp(scaled_||dua||_inf / scaled_||dua_prev||_inf))  scaled dual inf norm ratio (broadcast)
    f[12] = log10(clamp(scaled_||pri||_inf / scaled_||dua||_inf))       scaled primal/dual imbalance (broadcast)

Notes:
  - l and u are clamped to ±1e6 only for the feature vector. The raw tensors
    (which may contain ±inf) are used for the actual OSQP clamping in osqp_torch.py.
  - All operations are batched over the B dimension.
  - This function is called under the autograd graph — Ax and pri_res carry
    gradients w.r.t. x (which depend on alpha_z from the previous stage).
    However, at the start of each stage x is detached, so no cross-stage grad.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from learned_osqp.config import Config

# Clamp bounds in features to avoid ±inf propagating into the network
_LOG_UPPER_BOUND_CLAMP = 1e6
_LOG_LOWER_BOUND_CLAMP = 1e-6
_BOUND_CLAMP = 1e6
_EPS = 1e-8


def compute_per_row_features(
    P: torch.Tensor,          # (B, n, n)
    q: torch.Tensor,          # (B, n)
    A: torch.Tensor,          # (B, m, n)
    l: torch.Tensor,          # (B, m)  — raw, may be -inf
    u: torch.Tensor,          # (B, m)  — raw, may be +inf
    x: torch.Tensor,          # (B, n)  current primal
    z: torch.Tensor,          # (B, m)  current slack
    y: torch.Tensor,          # (B, m)  current dual
    rho_vec: torch.Tensor,    # (B, m)
    x_star: torch.Tensor,     # (B, n)  only for debugging/analysis, not used in features
    x_prev: torch.Tensor,     # (B, n)  primal T steps ago  (zeros at stage 0)
    z_prev: torch.Tensor,     # (B, m)  slack  T steps ago
    y_prev: torch.Tensor,     # (B, m)  dual   T steps ago
    AT: torch.Tensor | None = None,  # (B, n, m) precomputed A^T contiguous
) -> torch.Tensor:            # (B, m, feature_dim)
    """
    Compute per-row feature matrix for the alpha network.

    Args:
        A      : (B, m, n)  constraint matrix (dense)
        l      : (B, m)     lower bounds (may contain -inf)
        u      : (B, m)     upper bounds (may contain +inf)
        x      : (B, n)     current primal iterate
        z      : (B, m)     current slack variable
        y      : (B, m)     current dual variable
        x_star : (B, n)     optimal primal solution (precomputed)

    Returns:
        features : (B, m, 9) float64 tensor
    """
    # ---- Row norms of A: ||A[i,:]||_2  (B, m) ----
    # row_norms = torch.norm(A, dim=2)    # (B, m)
    # A_norm = row_norms + _EPS           # (B, m) — safe denominator

    if AT is None:
        AT = A.transpose(1, 2)

    # ---- Constraint activations ----
    # Ax = A @ x  →  (B, m)
    Ax = torch.bmm(A, x.unsqueeze(-1)).squeeze(-1)          # (B, m)
    Ax_prev = torch.bmm(A, x_prev.unsqueeze(-1)).squeeze(-1)  # (B, m)

    m = A.shape[1]
    lower_dist = z - l
    upper_dist = u - z
    pri_res_vec = z - Ax
    abs_pri_res_vec = torch.abs(pri_res_vec)
    sign_pri_res_vec = torch.sign(pri_res_vec)
    pri_res_inf_norm = torch.norm(pri_res_vec, p=float('inf'), dim=1)  # (B,)
    abs_y = torch.abs(y) # no need for sign_y since already indicated by lower_dist and upper_dist

    Px = torch.bmm(P, x.unsqueeze(-1)).squeeze(-1)             # (B, n)
    ATy = torch.bmm(AT, y.unsqueeze(-1)).squeeze(-1)           # (B, n)
    dua_res_inf_norm = torch.norm(Px + q + ATy, p=float('inf'), dim=1)  # (B,)

    A_inf_norm = torch.norm(A, p=float('inf'), dim=2)  # (B, m)  # for potential future use in normalization
    pri_res_vec_prev = z_prev - Ax_prev
    abs_pri_res_vec_prev = torch.abs(pri_res_vec_prev)
    pri_res_inf_norm_prev = torch.norm(pri_res_vec_prev, p=float('inf'), dim=1)  # (B,)
    Px_prev = torch.bmm(P, x_prev.unsqueeze(-1)).squeeze(-1)             # (B, n)
    ATy_prev = torch.bmm(AT, y_prev.unsqueeze(-1)).squeeze(-1)           # (B, n)
    dua_res_inf_norm_prev = torch.norm(Px_prev + q + ATy_prev, p=float('inf'), dim=1)  # (B,)

    # ---- Scaled residuals (OSQP-style normalization) ----
    q_inf = torch.norm(q, p=float('inf'), dim=1)                         # (B,)
    # pri_scale      = torch.maximum(torch.norm(Ax,      p=float('inf'), dim=1),
    #                                torch.norm(z,       p=float('inf'), dim=1)) + _EPS  # (B,)
    # dua_scale      = torch.maximum(torch.norm(Px,      p=float('inf'), dim=1),
    #                  torch.maximum(torch.norm(ATy,     p=float('inf'), dim=1),
    #                                q_inf)) + _EPS                         # (B,)
    # pri_scale_prev = torch.maximum(torch.norm(Ax_prev, p=float('inf'), dim=1),
    #                                torch.norm(z_prev,  p=float('inf'), dim=1)) + _EPS  # (B,)
    # dua_scale_prev = torch.maximum(torch.norm(Px_prev, p=float('inf'), dim=1),
    #                  torch.maximum(torch.norm(ATy_prev,p=float('inf'), dim=1),
    #                                q_inf)) + _EPS                         # (B,)

    # Scaled versions — pri_scale is (B,), broadcast to (B, m) for per-row features
    # abs_pri_res_vec_scaled      = abs_pri_res_vec      / pri_scale.unsqueeze(1)       # (B, m)
    # abs_pri_res_vec_prev_scaled = abs_pri_res_vec_prev / pri_scale_prev.unsqueeze(1)  # (B, m)
    # pri_res_inf_norm_scaled      = pri_res_inf_norm      / pri_scale       # (B,)
    # dua_res_inf_norm_scaled      = dua_res_inf_norm      / dua_scale       # (B,)
    # pri_res_inf_norm_prev_scaled = pri_res_inf_norm_prev / pri_scale_prev  # (B,)
    # dua_res_inf_norm_prev_scaled = dua_res_inf_norm_prev / dua_scale_prev  # (B,)


    # ---- Stack all features along the last dimension ----
    # features = torch.stack(
    #     [
    #         Ax,                   # f[0]  raw activation
    #         z,                    # f[1]  slack
    #         y,                    # f[2]  dual
    #         l_feat,               # f[3]  lower bound (clamped)
    #         u_feat,               # f[4]  upper bound (clamped)
    #         Ax_star,              # f[5]  optimal activation
    #         pri_res,              # f[6]  primal residual
    #         Ax / A_norm,          # f[7]  normalized activation
    #         pri_res / A_norm,     # f[8]  normalized residual
    #     ],
    #     dim=-1,
    # )   # (B, m, 9)
    features = torch.stack([
        torch.log10(torch.clamp(lower_dist, _LOG_LOWER_BOUND_CLAMP, _LOG_UPPER_BOUND_CLAMP)),  # f[0] log distance to lower bound
        torch.log10(torch.clamp(upper_dist, _LOG_LOWER_BOUND_CLAMP, _LOG_UPPER_BOUND_CLAMP)),  # f[1] log distance to upper bound
        torch.log10(torch.clamp(abs_pri_res_vec, _LOG_LOWER_BOUND_CLAMP, _LOG_UPPER_BOUND_CLAMP)),  # f[2] log absolute primal residual (unscaled)
        # torch.log10(torch.clamp(abs_pri_res_vec_scaled, _LOG_LOWER_BOUND_CLAMP, _LOG_UPPER_BOUND_CLAMP)),  # f[2] log scaled absolute primal residual
        sign_pri_res_vec,  # f[3] sign of primal residual
        torch.log10(torch.clamp(abs_y, _LOG_LOWER_BOUND_CLAMP, _LOG_UPPER_BOUND_CLAMP)),  # f[4] log absolute dual variable
        torch.log10(torch.clamp(pri_res_inf_norm.unsqueeze(1).expand(-1, m), _LOG_LOWER_BOUND_CLAMP, _LOG_UPPER_BOUND_CLAMP)),  # f[5] log infinity norm of primal residual (unscaled)
        torch.log10(torch.clamp(dua_res_inf_norm.unsqueeze(1).expand(-1, m), _LOG_LOWER_BOUND_CLAMP, _LOG_UPPER_BOUND_CLAMP)),  # f[6] log infinity norm of dual residual (unscaled)
        # torch.log10(torch.clamp(pri_res_inf_norm_scaled.unsqueeze(1).expand(-1, m), _LOG_LOWER_BOUND_CLAMP, _LOG_UPPER_BOUND_CLAMP)),  # f[5] log scaled infinity norm of primal residual
        # torch.log10(torch.clamp(dua_res_inf_norm_scaled.unsqueeze(1).expand(-1, m), _LOG_LOWER_BOUND_CLAMP, _LOG_UPPER_BOUND_CLAMP)),  # f[6] log scaled infinity norm of dual residual
        torch.log10(torch.clamp(rho_vec, _LOG_LOWER_BOUND_CLAMP, _LOG_UPPER_BOUND_CLAMP)),  # f[7] log of penalty parameter
        A_inf_norm,  # f[8]
        torch.log10(torch.clamp(abs_pri_res_vec / (abs_pri_res_vec_prev + _EPS), _LOG_LOWER_BOUND_CLAMP, _LOG_UPPER_BOUND_CLAMP)),  # f[9] ratio of current to previous primal residual (elementwise, unscaled)
        # torch.log10(torch.clamp(abs_pri_res_vec_scaled / (abs_pri_res_vec_prev_scaled + _EPS), _LOG_LOWER_BOUND_CLAMP, _LOG_UPPER_BOUND_CLAMP)),  # f[9] scaled ratio of current to previous primal residual (elementwise)
        torch.log10(torch.clamp(pri_res_inf_norm / (pri_res_inf_norm_prev + _EPS), _LOG_LOWER_BOUND_CLAMP, _LOG_UPPER_BOUND_CLAMP)).unsqueeze(1).expand(-1, m),  # f[10] ratio of current to previous primal residual (inf norm, unscaled)
        torch.log10(torch.clamp(dua_res_inf_norm / (dua_res_inf_norm_prev + _EPS), _LOG_LOWER_BOUND_CLAMP, _LOG_UPPER_BOUND_CLAMP)).unsqueeze(1).expand(-1, m),  # f[11] ratio of current to previous dual residual (inf norm, unscaled)
        # torch.log10(torch.clamp(pri_res_inf_norm_scaled / (pri_res_inf_norm_prev_scaled + _EPS), _LOG_LOWER_BOUND_CLAMP, _LOG_UPPER_BOUND_CLAMP)).unsqueeze(1).expand(-1, m),  # f[10] scaled ratio of current to previous primal residual (inf norm)
        # torch.log10(torch.clamp(dua_res_inf_norm_scaled / (dua_res_inf_norm_prev_scaled + _EPS), _LOG_LOWER_BOUND_CLAMP, _LOG_UPPER_BOUND_CLAMP)).unsqueeze(1).expand(-1, m),  # f[11] scaled ratio of current to previous dual residual (inf norm)
        torch.log10(torch.clamp(pri_res_inf_norm / (dua_res_inf_norm + _EPS), _LOG_LOWER_BOUND_CLAMP, _LOG_UPPER_BOUND_CLAMP)).unsqueeze(1).expand(-1, m),  # f[12] log primal/dual imbalance (unscaled)
        # torch.log10(torch.clamp(pri_res_inf_norm_scaled / (dua_res_inf_norm_scaled + _EPS), _LOG_LOWER_BOUND_CLAMP, _LOG_UPPER_BOUND_CLAMP)).unsqueeze(1).expand(-1, m),  # f[12] log scaled primal/dual imbalance
    ], dim=-1)  # (B, m, 13)

    return features


def compute_global_features(
    P: torch.Tensor,          # (B, n, n)
    q: torch.Tensor,          # (B, n)
    A: torch.Tensor,          # (B, m, n)
    x: torch.Tensor,          # (B, n)
    z: torch.Tensor,          # (B, m)
    y: torch.Tensor,          # (B, m)
    rho_scalar: torch.Tensor, # (B,)   base rho (not per-row)
    x_prev: torch.Tensor,     # (B, n)  state T steps ago  (zeros at stage 0)
    z_prev: torch.Tensor,     # (B, m)
    y_prev: torch.Tensor,     # (B, m)
    alpha: torch.Tensor | None = None,  # (B,)   alpha used at previous stage (unused, kept for caller compat)
    AT: torch.Tensor | None = None,  # (B, n, m) precomputed A^T contiguous
) -> torch.Tensor:            # (B, 6)
    """
    Compute 6-dim global feature vector for ScalarAlphaNet.

    Features:
        f[0] = log10(clamped pri_res_inf_norm)
        f[1] = log10(clamped dua_res_inf_norm)
        f[2] = log10(clamped rho_scalar)
        f[3] = log10(clamped pri_res_inf_norm / pri_res_inf_norm_prev)
        f[4] = log10(clamped dua_res_inf_norm / dua_res_inf_norm_prev)
        f[5] = log10(clamped pri_res_inf_norm / dua_res_inf_norm)   (primal/dual imbalance)

    All residuals are computed in the (scaled) problem space that the training
    loop operates in.  Ratio features are robust to absolute magnitude and
    encode convergence speed.

    At stage 0 (x_prev=z_prev=y_prev=0) the prev norms are 0, so the ratio
    features are clamped to log10(1e6) = 6.  This is a valid warm-start signal.
    """
    def _log10c(v: torch.Tensor) -> torch.Tensor:
        return torch.log10(torch.clamp(v, _LOG_LOWER_BOUND_CLAMP, _LOG_UPPER_BOUND_CLAMP))

    if AT is None:
        AT = A.transpose(1, 2)

    # Current residuals
    Ax  = torch.bmm(A, x.unsqueeze(-1)).squeeze(-1)           # (B, m)
    pri_res_inf = torch.norm(z - Ax, p=float('inf'), dim=1)   # (B,)

    Px  = torch.bmm(P, x.unsqueeze(-1)).squeeze(-1)           # (B, n)
    ATy = torch.bmm(AT, y.unsqueeze(-1)).squeeze(-1)          # (B, n)
    dua_res_inf = torch.norm(Px + q + ATy, p=float('inf'), dim=1)    # (B,)

    # Previous residuals (T steps ago)
    Ax_prev  = torch.bmm(A, x_prev.unsqueeze(-1)).squeeze(-1)          # (B, m)
    pri_res_inf_prev = torch.norm(z_prev - Ax_prev, p=float('inf'), dim=1)  # (B,)

    Px_prev  = torch.bmm(P, x_prev.unsqueeze(-1)).squeeze(-1)          # (B, n)
    ATy_prev = torch.bmm(AT, y_prev.unsqueeze(-1)).squeeze(-1)         # (B, n)
    dua_res_inf_prev = torch.norm(Px_prev + q + ATy_prev, p=float('inf'), dim=1)  # (B,)

    # ---- Scaled residuals (OSQP-style normalization) ----
    # Scaling denominators reuse already-computed tensors — negligible overhead.
    # q_inf = torch.norm(q, p=float('inf'), dim=1)                       # (B,)
    # pri_scale      = torch.maximum(torch.norm(Ax,      p=float('inf'), dim=1),
    #                                torch.norm(z,       p=float('inf'), dim=1)) + _EPS  # (B,)
    # dua_scale      = torch.maximum(torch.norm(Px,      p=float('inf'), dim=1),
    #                  torch.maximum(torch.norm(ATy,     p=float('inf'), dim=1),
    #                                q_inf)) + _EPS                       # (B,)
    # pri_scale_prev = torch.maximum(torch.norm(Ax_prev, p=float('inf'), dim=1),
    #                                torch.norm(z_prev,  p=float('inf'), dim=1)) + _EPS  # (B,)
    # dua_scale_prev = torch.maximum(torch.norm(Px_prev, p=float('inf'), dim=1),
    #                  torch.maximum(torch.norm(ATy_prev,p=float('inf'), dim=1),
    #                                q_inf)) + _EPS                       # (B,)

    # pri_res_inf_scaled      = pri_res_inf      / pri_scale       # (B,)
    # dua_res_inf_scaled      = dua_res_inf      / dua_scale       # (B,)
    # pri_res_inf_prev_scaled = pri_res_inf_prev / pri_scale_prev  # (B,)
    # dua_res_inf_prev_scaled = dua_res_inf_prev / dua_scale_prev  # (B,)

    return torch.stack([
        _log10c(pri_res_inf),                                        # f[0] (unscaled)
        _log10c(dua_res_inf),                                        # f[1] (unscaled)
        # _log10c(pri_res_inf_scaled),                                   # f[0] scaled primal residual
        # _log10c(dua_res_inf_scaled),                                   # f[1] scaled dual residual
        _log10c(rho_scalar),                                           # f[2]
        _log10c(pri_res_inf / (pri_res_inf_prev + _EPS)),            # f[3] (unscaled ratio)
        _log10c(dua_res_inf / (dua_res_inf_prev + _EPS)),            # f[4] (unscaled ratio)
        # _log10c(pri_res_inf_scaled / (pri_res_inf_prev_scaled + _EPS)),  # f[3] scaled primal ratio
        # _log10c(dua_res_inf_scaled / (dua_res_inf_prev_scaled + _EPS)),  # f[4] scaled dual ratio
        _log10c(pri_res_inf / (dua_res_inf + _EPS)),                 # f[5] (unscaled imbalance)
        # _log10c(pri_res_inf_scaled / (dua_res_inf_scaled + _EPS)),     # f[5] scaled primal/dual imbalance
        # alpha,                                                        # (commented out) previous-stage alpha
    ], dim=-1)  # (B, 6)
