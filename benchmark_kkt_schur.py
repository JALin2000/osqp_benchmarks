"""
Benchmark: Full KKT solve vs Schur complement (n×n reduced system).

Compares correctness and timing of two approaches for the OSQP KKT solve:

  Current (full KKT):
    Factor (n+m, n+m) system, then lu_solve per step.

  Schur complement (reduced):
    Eliminate ν to get n×n system:
      M = P + σI + Aᵀ diag(ρ) A        → factor once (n×n)
      M x_tilde = σx - q + Aᵀ(ρ·z - y) → solve per step (n×n)
      z_tilde = A @ x_tilde              → matvec per step

Both produce identical x_tilde and z_tilde (up to floating point).
"""

import time
import torch
import numpy as np


def make_problem(B, n, m, dtype=torch.float64, device='cpu', seed=42):
    """Generate a random but well-conditioned KKT-compatible problem."""
    rng = np.random.default_rng(seed)

    # P: symmetric positive definite (n, n)
    L = rng.standard_normal((n, n)) * 0.1
    P_np = L.T @ L + 0.01 * np.eye(n)
    P_np = 0.5 * (P_np + P_np.T)

    # A: random (m, n)
    A_np = rng.standard_normal((m, n)) * 0.5

    # Other vectors
    q_np = rng.standard_normal(n)
    x_np = rng.standard_normal(n) * 0.1
    z_np = rng.standard_normal(m) * 0.1
    y_np = rng.standard_normal(m) * 0.1

    # rho: per-constraint penalty (positive)
    rho_vec_np = np.full(m, 0.1)
    rho_vec_np[:m // 10] = 1e-6  # some loose constraints
    rho_inv_np = 1.0 / rho_vec_np

    sigma = 1e-6

    # Expand to batch
    P = torch.tensor(P_np, dtype=dtype, device=device).unsqueeze(0).expand(B, -1, -1).contiguous()
    A = torch.tensor(A_np, dtype=dtype, device=device).unsqueeze(0).expand(B, -1, -1).contiguous()
    q = torch.tensor(q_np, dtype=dtype, device=device).unsqueeze(0).expand(B, -1).contiguous()
    x = torch.tensor(x_np, dtype=dtype, device=device).unsqueeze(0).expand(B, -1).contiguous()
    z = torch.tensor(z_np, dtype=dtype, device=device).unsqueeze(0).expand(B, -1).contiguous()
    y = torch.tensor(y_np, dtype=dtype, device=device).unsqueeze(0).expand(B, -1).contiguous()
    rho_vec = torch.tensor(rho_vec_np, dtype=dtype, device=device).unsqueeze(0).expand(B, -1).contiguous()
    rho_inv = torch.tensor(rho_inv_np, dtype=dtype, device=device).unsqueeze(0).expand(B, -1).contiguous()

    return P, A, q, x, z, y, rho_vec, rho_inv, sigma


# ──────────────────────────────────────────────────────────────────────
# Method 1: Current full KKT approach  (from osqp_torch.py)
# ──────────────────────────────────────────────────────────────────────

def full_kkt_factor(P, A, sigma, rho_inv):
    """Factor the (n+m, n+m) KKT matrix."""
    B, m, n = A.shape
    dtype, device = P.dtype, P.device

    I_n = sigma * torch.eye(n, dtype=dtype, device=device).unsqueeze(0).expand(B, -1, -1)
    top_left = P + I_n
    top_right = A.transpose(1, 2)
    bottom_right = -torch.diag_embed(rho_inv)

    top = torch.cat([top_left, top_right], dim=2)
    bottom = torch.cat([A, bottom_right], dim=2)
    KKT = torch.cat([top, bottom], dim=1)

    LU, pivots = torch.linalg.lu_factor(KKT)
    return LU, pivots


def full_kkt_solve(LU, pivots, rhs):
    """Solve KKT @ sol = rhs using pre-factored LU."""
    return torch.linalg.lu_solve(LU, pivots, rhs.unsqueeze(-1)).squeeze(-1)


def full_kkt_step(x, z, y, q, rho_vec, rho_inv, LU, pivots, sigma, alpha_x=1.6):
    """One OSQP step using the full KKT approach. Returns (x_tilde, z_tilde)."""
    B, n = x.shape

    rhs_top = sigma * x - q
    rhs_bot = z - rho_inv * y
    rhs = torch.cat([rhs_top, rhs_bot], dim=1)

    sol = full_kkt_solve(LU, pivots, rhs)

    x_tilde = sol[:, :n]
    z_tilde = z + rho_inv * (sol[:, n:] - y)

    return x_tilde, z_tilde


# ──────────────────────────────────────────────────────────────────────
# Method 2: Schur complement (n×n reduced system)
# ──────────────────────────────────────────────────────────────────────

def schur_factor(P, A, sigma, rho_vec):
    """
    Factor the n×n reduced system:  M = P + σI + Aᵀ diag(ρ) A

    Also precompute AT_contiguous for reuse in solves.
    """
    B, m, n = A.shape
    dtype, device = P.dtype, P.device

    AT = A.transpose(1, 2).contiguous()  # (B, n, m)

    # Aᵀ diag(ρ) A  =  (Aᵀ * ρ) @ A  =  ATρ @ A
    # ATρ[b, i, j] = AT[b, i, j] * rho_vec[b, j]
    AT_rho = AT * rho_vec.unsqueeze(1)  # (B, n, m)  — row-scale AT by ρ
    AtRA = torch.bmm(AT_rho, A)  # (B, n, n)

    I_n = sigma * torch.eye(n, dtype=dtype, device=device).unsqueeze(0).expand(B, -1, -1)
    M = P + I_n + AtRA  # (B, n, n)

    LU_M, pivots_M = torch.linalg.lu_factor(M)
    return LU_M, pivots_M, AT


def schur_solve(LU_M, pivots_M, AT, A, rhs_reduced):
    """Solve M @ x_tilde = rhs_reduced, then z_tilde = A @ x_tilde."""
    x_tilde = torch.linalg.lu_solve(LU_M, pivots_M, rhs_reduced.unsqueeze(-1)).squeeze(-1)
    z_tilde = torch.bmm(A, x_tilde.unsqueeze(-1)).squeeze(-1)
    return x_tilde, z_tilde


def schur_step(x, z, y, q, rho_vec, AT, A, LU_M, pivots_M, sigma, alpha_x=1.6):
    """One OSQP step using the Schur complement approach. Returns (x_tilde, z_tilde)."""
    # rhs_reduced = σx - q + Aᵀ(ρ·z - y)
    rhs_reduced = sigma * x - q + torch.bmm(AT, (rho_vec * z - y).unsqueeze(-1)).squeeze(-1)

    x_tilde, z_tilde = schur_solve(LU_M, pivots_M, AT, A, rhs_reduced)
    return x_tilde, z_tilde


# ──────────────────────────────────────────────────────────────────────
# Benchmark
# ──────────────────────────────────────────────────────────────────────

def benchmark_one(B, n, m, T=10, n_warmup=3, n_trials=20):
    """Compare both methods for a given problem size."""
    print(f"\n{'='*70}")
    print(f"  B={B}, n={n}, m={m}  (KKT size={n+m}, ratio m/n={m/n:.1f})")
    print(f"{'='*70}")

    P, A, q, x, z, y, rho_vec, rho_inv, sigma = make_problem(B, n, m)

    # ── Correctness check ─────────────────────────────────────────────
    with torch.no_grad():
        LU_full, piv_full = full_kkt_factor(P, A, sigma, rho_inv)
        xt_full, zt_full = full_kkt_step(x, z, y, q, rho_vec, rho_inv,
                                          LU_full, piv_full, sigma)

        LU_sch, piv_sch, AT = schur_factor(P, A, sigma, rho_vec)
        xt_sch, zt_sch = schur_step(x, z, y, q, rho_vec, AT, A,
                                     LU_sch, piv_sch, sigma)

    err_x = (xt_full - xt_sch).abs().max().item()
    err_z = (zt_full - zt_sch).abs().max().item()
    print(f"\n  Correctness:  max|x_tilde diff| = {err_x:.2e}")
    print(f"                max|z_tilde diff| = {err_z:.2e}")
    assert err_x < 1e-8, f"x_tilde mismatch too large: {err_x}"
    assert err_z < 1e-8, f"z_tilde mismatch too large: {err_z}"
    print(f"                ✓ Both methods agree")

    # Also verify correctness after T steps of chained solves
    with torch.no_grad():
        xf, zf, yf = x.clone(), z.clone(), y.clone()
        xs, zs, ys = x.clone(), z.clone(), y.clone()
        alpha_x = 1.6
        l = torch.full_like(z, -1e30)
        u = torch.full_like(z, 1e30)

        for _ in range(T):
            xt_f, zt_f = full_kkt_step(xf, zf, yf, q, rho_vec, rho_inv,
                                        LU_full, piv_full, sigma)
            # x update
            xf_new = alpha_x * xt_f + (1 - alpha_x) * xf
            # z update
            alpha_z = torch.full_like(zf, alpha_x)
            z_unclip = alpha_z * zt_f + (1 - alpha_z) * zf + rho_inv * yf
            zf_new = torch.clamp(z_unclip, min=l, max=u)
            # y update
            z_bar = alpha_z * zt_f + (1 - alpha_z) * zf
            yf_new = yf + rho_vec * (z_bar - zf_new)
            xf, zf, yf = xf_new, zf_new, yf_new

            xt_s, zt_s = schur_step(xs, zs, ys, q, rho_vec, AT, A,
                                     LU_sch, piv_sch, sigma)
            xs_new = alpha_x * xt_s + (1 - alpha_x) * xs
            z_unclip_s = alpha_z * zt_s + (1 - alpha_z) * zs + rho_inv * ys
            zs_new = torch.clamp(z_unclip_s, min=l, max=u)
            z_bar_s = alpha_z * zt_s + (1 - alpha_z) * zs
            ys_new = ys + rho_vec * (z_bar_s - zs_new)
            xs, zs, ys = xs_new, zs_new, ys_new

    err_x_T = (xf - xs).abs().max().item()
    err_z_T = (zf - zs).abs().max().item()
    err_y_T = (yf - ys).abs().max().item()
    print(f"  After {T} steps: max|x diff|={err_x_T:.2e}, "
          f"max|z diff|={err_z_T:.2e}, max|y diff|={err_y_T:.2e}")
    assert err_x_T < 1e-6 and err_z_T < 1e-6 and err_y_T < 1e-6, "Diverged after T steps"
    print(f"                ✓ {T}-step rollout matches")

    # ── Timing: factorization ─────────────────────────────────────────
    for _ in range(n_warmup):
        full_kkt_factor(P, A, sigma, rho_inv)
        schur_factor(P, A, sigma, rho_vec)

    times_factor_full = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        LU_full, piv_full = full_kkt_factor(P, A, sigma, rho_inv)
        times_factor_full.append(time.perf_counter() - t0)

    times_factor_schur = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        LU_sch, piv_sch, AT = schur_factor(P, A, sigma, rho_vec)
        times_factor_schur.append(time.perf_counter() - t0)

    tf_full = np.median(times_factor_full) * 1e3
    tf_schur = np.median(times_factor_schur) * 1e3

    # ── Timing: T-step solve (the hot path) ───────────────────────────
    for _ in range(n_warmup):
        for _ in range(T):
            full_kkt_step(x, z, y, q, rho_vec, rho_inv, LU_full, piv_full, sigma)
        for _ in range(T):
            schur_step(x, z, y, q, rho_vec, AT, A, LU_sch, piv_sch, sigma)

    times_solve_full = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        for _ in range(T):
            full_kkt_step(x, z, y, q, rho_vec, rho_inv, LU_full, piv_full, sigma)
        times_solve_full.append(time.perf_counter() - t0)

    times_solve_schur = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        for _ in range(T):
            schur_step(x, z, y, q, rho_vec, AT, A, LU_sch, piv_sch, sigma)
        times_solve_schur.append(time.perf_counter() - t0)

    ts_full = np.median(times_solve_full) * 1e3
    ts_schur = np.median(times_solve_schur) * 1e3

    # ── Timing: total (1 factorization + T solves) — one stage ────────
    total_full = tf_full + ts_full
    total_schur = tf_schur + ts_schur

    # ── Memory: matrix sizes ──────────────────────────────────────────
    kkt_bytes = B * (n + m) ** 2 * 8
    schur_bytes = B * n * n * 8 + B * n * m * 8  # M + AT
    at_rho_a_bytes = B * n * n * 8  # intermediate Aᵀ diag(ρ) A

    print(f"\n  Factorization (median of {n_trials}):")
    print(f"    Full KKT ({n+m}×{n+m}): {tf_full:8.2f} ms")
    print(f"    Schur    ({n}×{n}):     {tf_schur:8.2f} ms")
    print(f"    Speedup:                 {tf_full / tf_schur:8.1f}×")

    print(f"\n  {T}-step solve (median of {n_trials}):")
    print(f"    Full KKT:                {ts_full:8.2f} ms")
    print(f"    Schur:                   {ts_schur:8.2f} ms")
    print(f"    Speedup:                 {ts_full / ts_schur:8.1f}×")

    print(f"\n  Total per stage (factor + {T} solves):")
    print(f"    Full KKT:                {total_full:8.2f} ms")
    print(f"    Schur:                   {total_schur:8.2f} ms")
    print(f"    Speedup:                 {total_full / total_schur:8.1f}×")

    print(f"\n  LU factor memory:")
    print(f"    Full KKT: {kkt_bytes / 1e6:.1f} MB")
    print(f"    Schur (M + AT): {schur_bytes / 1e6:.1f} MB")

    return {
        'n': n, 'm': m,
        'factor_full_ms': tf_full, 'factor_schur_ms': tf_schur,
        'solve_full_ms': ts_full, 'solve_schur_ms': ts_schur,
        'total_full_ms': total_full, 'total_schur_ms': total_schur,
    }


if __name__ == '__main__':
    B = 10  # batch size matching training
    T = 10  # steps per stage

    configs = [
        # (n, m, description)
        (20, 200, "random_qp s=20 (current default)"),
        (150, 1500, "random_qp s=150 (your training)"),
        (1515, 3000, "SVM s=15"),
        (1280, 2160, "control s=80"),
    ]

    results = []
    for n, m, desc in configs:
        print(f"\n>>> {desc}")
        r = benchmark_one(B, n, m, T=T)
        r['desc'] = desc
        results.append(r)

    # ── Summary table ─────────────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print(f"  SUMMARY  (B={B}, T={T})")
    print(f"{'='*70}")
    print(f"{'Problem':<30} {'Factor':>10} {'Solve':>10} {'Total':>10}")
    print(f"{'':30} {'speedup':>10} {'speedup':>10} {'speedup':>10}")
    print(f"{'-'*70}")
    for r in results:
        fs = r['factor_full_ms'] / r['factor_schur_ms']
        ss = r['solve_full_ms'] / r['solve_schur_ms']
        ts = r['total_full_ms'] / r['total_schur_ms']
        print(f"{r['desc']:<30} {fs:>9.1f}× {ss:>9.1f}× {ts:>9.1f}×")
