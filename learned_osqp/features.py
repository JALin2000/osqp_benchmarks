"""
features.py — Per-row feature computation for PerRowAlphaNet.

Computes a (B, m, feature_dim) tensor of per-constraint-row features
from the current OSQP iterate state (x, z, y) and problem data.

Feature vector per row i (feature_dim = 9):
    f[0] = A[i,:] @ x_k                          constraint activation (raw)
    f[1] = z_k[i]                                current slack
    f[2] = y_k[i]                                current dual variable
    f[3] = l[i]  (clamped to [-1e6, 1e6])        lower bound
    f[4] = u[i]  (clamped to [-1e6, 1e6])        upper bound
    f[5] = A[i,:] @ x_star                       optimal constraint activation
    f[6] = A[i,:] @ x_k - z_k[i]                primal residual for this row
    f[7] = (A[i,:] @ x_k) / (||A[i,:]|| + eps)  normalized activation
    f[8] = primal_residual_i / (||A[i,:]|| + eps) normalized residual

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
    x_star: torch.Tensor,     # (B, n)
    x_prev: torch.Tensor,     # (B, n)  primal T steps ago  (zeros at stage 0)
    z_prev: torch.Tensor,     # (B, m)  slack  T steps ago
    y_prev: torch.Tensor,     # (B, m)  dual   T steps ago
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

    # ---- Constraint activations ----
    # Ax = A @ x  →  (B, m)
    Ax = torch.bmm(A, x.unsqueeze(-1)).squeeze(-1)          # (B, m)
    Ax_prev = torch.bmm(A, x_prev.unsqueeze(-1)).squeeze(-1)  # (B, m)

    # A @ x_star  →  (B, m)
    # Ax_star = torch.bmm(A, x_star.unsqueeze(-1)).squeeze(-1)   # (B, m)

    # ---- Primal residual per row ----
    # pri_res = Ax - z   # (B, m)

    # ---- Bounded versions of l, u for feature embedding ----
    # l_feat = torch.clamp(l, min=-_BOUND_CLAMP, max=_BOUND_CLAMP)   # (B, m)
    # u_feat = torch.clamp(u, min=-_BOUND_CLAMP, max=_BOUND_CLAMP)   # (B, m)

    m = A.shape[1]
    lower_dist = z - l
    upper_dist = u - z
    pri_res_vec = z - Ax
    abs_pri_res_vec = torch.abs(pri_res_vec)
    sign_pri_res_vec = torch.sign(pri_res_vec)
    pri_res_inf_norm = torch.norm(pri_res_vec, p=float('inf'), dim=1)  # (B,)
    abs_y = torch.abs(y) # no need for sign_y since already indicated by lower_dist and upper_dist
    
    Px = torch.bmm(P, x.unsqueeze(-1)).squeeze(-1)             # (B, n)
    ATy = torch.bmm(A.transpose(1, 2), y.unsqueeze(-1)).squeeze(-1)  # (B, n)
    dua_res_inf_norm = torch.norm(Px + q + ATy, p=float('inf'), dim=1)  # (B,)

    A_inf_norm = torch.norm(A, p=float('inf'), dim=2)  # (B, m)  # for potential future use in normalization
    pri_res_vec_prev = z_prev - Ax_prev
    abs_pri_res_vec_prev = torch.abs(pri_res_vec_prev)
    pri_res_inf_norm_prev = torch.norm(pri_res_vec_prev, p=float('inf'), dim=1)  # (B,)
    Px_prev = torch.bmm(P, x_prev.unsqueeze(-1)).squeeze(-1)             # (B, n)
    ATy_prev = torch.bmm(A.transpose(1, 2), y_prev.unsqueeze(-1)).squeeze(-1)  # (B, n)
    dua_res_inf_norm_prev = torch.norm(Px_prev + q + ATy_prev, p=float('inf'), dim=1)  # (B,)



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
        torch.log10(torch.clamp(abs_pri_res_vec, _LOG_LOWER_BOUND_CLAMP, _LOG_UPPER_BOUND_CLAMP)),  # f[2] log absolute primal residual
        sign_pri_res_vec,  # f[3] sign of primal residual
        torch.log10(torch.clamp(abs_y, _LOG_LOWER_BOUND_CLAMP, _LOG_UPPER_BOUND_CLAMP)),  # f[4] log absolute dual variable
        torch.log10(torch.clamp(pri_res_inf_norm.unsqueeze(1).repeat(1, m), _LOG_LOWER_BOUND_CLAMP, _LOG_UPPER_BOUND_CLAMP)),  # f[5] log infinity norm of primal residual
        torch.log10(torch.clamp(dua_res_inf_norm.unsqueeze(1).repeat(1, m), _LOG_LOWER_BOUND_CLAMP, _LOG_UPPER_BOUND_CLAMP)),  # f[6] log infinity norm of dual residual
        torch.log10(torch.clamp(rho_vec, _LOG_LOWER_BOUND_CLAMP, _LOG_UPPER_BOUND_CLAMP)),  # f[7] log of penalty parameter
        A_inf_norm,
        torch.log10(torch.clamp(abs_pri_res_vec / (abs_pri_res_vec_prev + _EPS), _LOG_LOWER_BOUND_CLAMP, _LOG_UPPER_BOUND_CLAMP)),  # f[9] ratio of current to previous primal residual (elementwise)
        torch.log10(torch.clamp(pri_res_inf_norm / (pri_res_inf_norm_prev + _EPS), _LOG_LOWER_BOUND_CLAMP, _LOG_UPPER_BOUND_CLAMP)).unsqueeze(1).repeat(1, m),  # f[10] ratio of current to previous primal residual (inf norm)
        torch.log10(torch.clamp(dua_res_inf_norm / (dua_res_inf_norm_prev + _EPS), _LOG_LOWER_BOUND_CLAMP, _LOG_UPPER_BOUND_CLAMP)).unsqueeze(1).repeat(1, m),  # f[11] ratio of current to previous dual residual (inf norm)
    ], dim=-1)  # (B, m, 12)

    return features
