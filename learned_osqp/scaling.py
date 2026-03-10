"""
scaling.py — Ruiz equilibration for QP problem scaling.

Mirrors scale_data() from solvers/osqppurepy/_osqp.py.

The scaling transforms the QP:
    min  (1/2) x^T P x + q^T x
    s.t. l <= Ax <= u

to the scaled version:
    min  (1/2) x_sc^T P_sc x_sc + q_sc^T x_sc
    s.t. l_sc <= A_sc x_sc <= u_sc

where:
    P_sc = c * D @ P @ D     (D = diag(d_vec))
    q_sc = c * D @ q
    A_sc = E @ A @ D         (E = diag(e_vec))
    l_sc = E @ l
    u_sc = E @ u

Recovering the original solution from scaled iterates (x_sc, y_sc):
    x_orig = D  * x_sc            (elementwise, D = diag(d_vec))
    y_orig = (c^{-1} * E) * y_sc  (elementwise)

Unscaling residuals for convergence checking (mirrors compute_pri/dua_res):
    pri_res_orig = E^{-1} * (A_sc @ x_sc - z_sc)
    dua_res_orig = c^{-1} * D^{-1} * (P_sc @ x_sc + q_sc + A_sc^T @ y_sc)

Both reduce to the standard OSQP residuals (A@x - z, P@x + q + A^T@y) in
the original variable space.
"""

import numpy as np

_MIN_SCALING = 1e-4   # matches MIN_SCALING in _osqp.py
_MAX_SCALING = 1e4    # matches MAX_SCALING in _osqp.py


def _limit_scaling_vec(vals: np.ndarray) -> np.ndarray:
    """
    Apply _limit_scaling to an array, matching _osqp.py logic:
      - val < MIN_SCALING  →  1.0   (not MIN_SCALING; leave that column unscaled)
      - val > MAX_SCALING  →  MAX_SCALING
      - else               →  val
    """
    out = vals.copy()
    out[vals < _MIN_SCALING] = 1.0
    out[vals > _MAX_SCALING] = _MAX_SCALING
    return out


def _limit_scaling_scalar(val: float) -> float:
    """Scalar version of _limit_scaling."""
    if val < _MIN_SCALING:
        return 1.0
    elif val > _MAX_SCALING:
        return _MAX_SCALING
    return val


def ruiz_equilibration(
    P_np: np.ndarray,   # (n, n) dense symmetric PSD
    q_np: np.ndarray,   # (n,)
    A_np: np.ndarray,   # (m, n) dense
    l_np: np.ndarray,   # (m,)  may contain -inf (not used in computation)
    u_np: np.ndarray,   # (m,)  may contain +inf (not used in computation)
    n_iter: int = 10,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Ruiz equilibration with cost normalisation, mirroring _osqp.py scale_data().

    Performs n_iter rounds of:
      1. Ruiz column-equilibration of the KKT matrix [[P, A^T], [A, 0]]
         via symmetric scaling (D_temp, E_temp) = diag(1/sqrt(limited col-norms)).
      2. Cost normalisation: c_temp = 1 / limit(max(limit(||q||_inf), mean_col||P||_inf)).

    Args:
        P_np, q_np, A_np, l_np, u_np : problem data (l, u not used directly)
        n_iter : Ruiz iterations (default 10, matches OSQP default)

    Returns:
        d_vec : (n,)  diagonal of D
        e_vec : (m,)  diagonal of E
        c     : float cost scaling

    Usage::
        d, e, c = ruiz_equilibration(P, q, A, l, u)
        P_sc, q_sc, A_sc, l_sc, u_sc = apply_scaling(P, q, A, l, u, d, e, c)
    """
    n = P_np.shape[0]
    m = A_np.shape[0]

    d = np.ones(n)
    e = np.ones(m) if m > 0 else np.ones(0)
    c = 1.0

    # Work on copies so the caller's arrays are not modified
    P = P_np.astype(np.float64, copy=True)
    A = A_np.astype(np.float64, copy=True)
    q = q_np.astype(np.float64, copy=True)

    for _ in range(n_iter):
        # ---- Step 1: Ruiz column equilibration ----
        # Column infinity norms of the KKT matrix [[P, A^T], [A, 0]]:
        #   first n cols : max(|P[:,j]|_inf, |A[:,j]|_inf)   j = 0..n-1
        #   last  m cols : |A[i,:]|_inf                       i = 0..m-1
        norm_P_cols = np.abs(P).max(axis=0)              # (n,)  col inf-norm of P
        if m > 0:
            norm_A_cols  = np.abs(A).max(axis=0)         # (n,)  col inf-norm of A
            norm_first   = np.maximum(norm_P_cols, norm_A_cols)
            norm_second  = np.abs(A).max(axis=1)         # (m,)  row inf-norm of A
        else:
            norm_first  = norm_P_cols
            norm_second = np.zeros(0)

        norm_cols = _limit_scaling_vec(
            np.concatenate([norm_first, norm_second])
        )  # (n+m,)  — values < MIN become 1.0

        s_temp = 1.0 / np.sqrt(norm_cols)               # (n+m,)
        d_temp = s_temp[:n]                              # (n,)
        e_temp = s_temp[n:] if m > 0 else np.ones(0)    # (m,)

        # Apply: P <- D_temp @ P @ D_temp
        P = (d_temp[:, None] * P) * d_temp[None, :]
        # Apply: A <- E_temp @ A @ D_temp
        if m > 0:
            A = (e_temp[:, None] * A) * d_temp[None, :]
        # Apply: q <- D_temp @ q
        q = d_temp * q

        # Accumulate overall scaling
        d = d_temp * d
        if m > 0:
            e = e_temp * e

        # ---- Step 2: Cost normalisation ----
        # Mirrors _osqp.py lines 481-506
        norm_P_mean = float(np.abs(P).max(axis=0).mean()) if n > 0 else 0.0
        inf_norm_q  = _limit_scaling_scalar(
            float(np.abs(q).max()) if n > 0 else 0.0
        )
        scale_cost  = _limit_scaling_scalar(max(inf_norm_q, norm_P_mean))
        c_temp      = 1.0 / scale_cost

        P = c_temp * P
        q = c_temp * q
        c = c_temp * c

    return d, e, c


def apply_scaling(
    P_np: np.ndarray,
    q_np: np.ndarray,
    A_np: np.ndarray,
    l_np: np.ndarray,
    u_np: np.ndarray,
    d_vec: np.ndarray,
    e_vec: np.ndarray,
    c: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply pre-computed Ruiz scaling (d_vec, e_vec, c) to problem matrices.

    Returns (P_sc, q_sc, A_sc, l_sc, u_sc).
    Safe for l/u containing ±inf (inf * positive_scale = ±inf).
    """
    m = A_np.shape[0]
    P_sc = c * (d_vec[:, None] * P_np * d_vec[None, :])
    q_sc = c * d_vec * q_np
    if m > 0:
        A_sc = e_vec[:, None] * A_np * d_vec[None, :]
        l_sc = e_vec * l_np
        u_sc = e_vec * u_np
    else:
        A_sc = A_np.copy()
        l_sc = l_np.copy()
        u_sc = u_np.copy()
    return P_sc, q_sc, A_sc, l_sc, u_sc
