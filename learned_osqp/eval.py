"""
eval.py — Evaluation: compare learned per-row alpha vs. baseline alpha=1.6.

Compares:
  1. Baseline : OSQP with scalar alpha_z = 1.6 for all rows (no network).
  2. Learned  : OSQP with per-row alpha_z from trained PerRowAlphaNet.

Metrics reported at each checkpoint (iterations 10, 20, 50, 100):
  - Mean primal residual ||Ax - z||_2 over the val set
  - Mean dual   residual ||Px + q + A^T y||_2
  - Mean ||x - x*||_2

Also saves convergence curves (residuals vs. iteration) for a sample of
instances to learned_osqp/results/.

Usage:
    python -m learned_osqp.eval [--checkpoint path] [--T 100] [--n 20]
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from learned_osqp.config import Config
from learned_osqp.model import PerRowAlphaNet, ScalarAlphaNet, ScalarGRUNet
from learned_osqp.data import make_dataloaders_multi
from learned_osqp.features import compute_per_row_features, compute_global_features
from learned_osqp.osqp_torch import (
    factorize_kkt,
    osqp_step,
    maybe_update_rho,
    baseline_rollout,
)
from learned_osqp.loss import primal_residual, dual_residual


# --------------------------------------------------------------------------- #
# Learned rollout (with network alpha)
# --------------------------------------------------------------------------- #

@torch.no_grad()
def learned_rollout(
    model: PerRowAlphaNet | ScalarAlphaNet | ScalarGRUNet,
    batch: dict,
    cfg: Config,
    T_total: int = 100,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Run T_total steps using the trained PerRowAlphaNet to set alpha_z.
    Alpha is recomputed every cfg.T steps (one stage = one network call).

    Returns:
        state_history : list of (x_t, z_t, y_t) after each iteration
    """
    model.eval()

    P = batch['P']
    A = batch['A']
    q = batch['q']
    l = batch['l']
    u = batch['u']
    x_star = batch['x_star']
    constr_type = batch['constr_type']

    B, n = P.shape[0], P.shape[1]
    m = A.shape[1]
    device = P.device
    dtype = P.dtype

    x = torch.zeros(B, n, dtype=dtype, device=device)
    z = torch.zeros(B, m, dtype=dtype, device=device)
    y = torch.zeros(B, m, dtype=dtype, device=device)

    rho_scalar = torch.full((B,), cfg.rho, dtype=dtype, device=device)
    rho_vec = batch['rho_vec'].clone()
    rho_inv = batch['rho_inv'].clone()

    scalar_mode = getattr(cfg, 'alpha_mode', 'vector') == 'scalar'

    factors = factorize_kkt(P, A, cfg.sigma, rho_vec)
    state_history = []
    step_in_stage = 0
    # will be updated at stage boundaries; init to baseline
    alpha_z       = torch.full((B, m), 1.6, dtype=dtype, device=device)
    alpha_x_cur   = cfg.alpha_x  # may become (B, 1) in scalar mode

    # GRU hidden state — reset to None (zeros) at solve start
    h_state: torch.Tensor | None = None

    # Previous state (T steps ago); zeros at stage 0
    x_prev = torch.zeros(B, n, dtype=dtype, device=device)
    z_prev = torch.zeros(B, m, dtype=dtype, device=device)
    y_prev = torch.zeros(B, m, dtype=dtype, device=device)

    for t in range(T_total):
        # Recompute alpha at the start of each stage
        if step_in_stage == 0:
            if scalar_mode:
                global_feat   = compute_global_features(
                    P, q, A, x, z, y, rho_scalar, x_prev, z_prev, y_prev,
                )
                if isinstance(model, ScalarGRUNet):
                    alpha_scalar, h_state = model(global_feat, h_state)  # (B,), (B, hidden)
                else:
                    alpha_scalar  = model(global_feat)          # (B,)
                alpha_z       = alpha_scalar.unsqueeze(-1)  # (B, 1)
                alpha_x_cur   = alpha_z
            else:
                features = compute_per_row_features(P, q, A, l, u, x, z, y, rho_vec, x_star,
                                                    x_prev, z_prev, y_prev)
                alpha_z = model(features)   # (B, m)
                alpha_x_cur = cfg.alpha_x

            # Save state at start of this stage; will become x_prev for the next stage
            x_stage_start = x.clone()
            z_stage_start = z.clone()
            y_stage_start = y.clone()

        x, z, y, _, _ = osqp_step(
            x, z, y, q, l, u, rho_vec, rho_inv,
            factors, alpha_x_cur, alpha_z, cfg.sigma, A,
        )
        state_history.append((x.clone(), z.clone(), y.clone()))

        step_in_stage = (step_in_stage + 1) % cfg.T

        # Update prev state at stage boundaries (matches training: prev = start of this stage)
        if step_in_stage == 0:
            x_prev, z_prev, y_prev = x_stage_start, z_stage_start, y_stage_start

        # Adaptive rho at stage boundaries
        if cfg.adaptive_rho and step_in_stage == 0:
            _batch_rho = {
                'P': P, 'A': A, 'q': q,
                'rho_vec': rho_vec, 'rho_inv': rho_inv,
                'constr_type': constr_type,
            }
            rho_scalar, rho_vec, rho_inv, changed = maybe_update_rho(
                _batch_rho, x, z, y, rho_scalar, cfg
            )
            if changed.any():
                factors = factorize_kkt(P, A, cfg.sigma, rho_vec)

    return state_history


# --------------------------------------------------------------------------- #
# Aggregate metrics
# --------------------------------------------------------------------------- #

def compute_metrics_at_steps(
    state_history: list[tuple],
    batch: dict,
    steps: list[int],
) -> dict:
    """
    Extract primal/dual residuals and ||x - x*|| at specified iteration counts.

    Args:
        state_history : list of (x_t, z_t, y_t) at each iteration
        batch         : contains P, A, q, x_star
        steps         : 0-indexed iteration numbers to report

    Returns:
        dict mapping step -> {'pri': float, 'dua': float, 'err': float}
    """
    P = batch['P']
    A = batch['A']
    q = batch['q']
    x_star = batch['x_star']
    results = {}

    for s in steps:
        if s >= len(state_history):
            continue
        x_t, z_t, y_t = state_history[s]
        pri = primal_residual(A, x_t, z_t).mean().item()
        dua = dual_residual(P, A, q, x_t, y_t).mean().item()
        err = torch.norm(x_t - x_star, dim=1).mean().item()
        results[s + 1] = {'pri': pri, 'dua': dua, 'err': err}

    return results


# --------------------------------------------------------------------------- #
# Main evaluation
# --------------------------------------------------------------------------- #

def evaluate(
    cfg: Config | None = None,
    checkpoint_path: str | None = None,
    T_total: int = 100,
    n_batches_eval: int = 10,
    save_plots: bool = True,
) -> None:
    """
    Load trained model and compare learned vs. baseline OSQP.

    Args:
        cfg             : Config (uses defaults if None)
        checkpoint_path : path to .pt checkpoint (uses cfg.checkpoint_path if None)
        T_total         : total OSQP iterations to run for comparison
        n_batches_eval  : number of val batches to average metrics over
        save_plots      : whether to save matplotlib convergence curves
    """
    if cfg is None:
        cfg = Config()

    ckpt_path = checkpoint_path or 'learned_osqp/checkpoints/best_model.pt'

    # ---- Load model ---- #
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    # Restore cfg from checkpoint if available, then apply any runtime overrides
    if 'cfg' in ckpt:
        cfg_ckpt = ckpt['cfg']
        # Preserve device/dtype/precision from the caller's cfg (runtime choice)
        cfg_ckpt.device    = cfg.device
        cfg_ckpt.dtype     = cfg.dtype
        cfg_ckpt.precision = cfg.precision
        # Back-compat: old checkpoints may not have alpha_mode / scalar_feature_dim / model_type
        if not hasattr(cfg_ckpt, 'alpha_mode'):
            cfg_ckpt.alpha_mode = 'vector'
        if not hasattr(cfg_ckpt, 'scalar_feature_dim'):
            cfg_ckpt.scalar_feature_dim = 5
        if not hasattr(cfg_ckpt, 'model_type'):
            cfg_ckpt.model_type = 'mlp'
        cfg = cfg_ckpt

    device = cfg.torch_device
    dtype  = cfg.torch_dtype

    alpha_mode = getattr(cfg, 'alpha_mode', 'vector')
    model_type = getattr(cfg, 'model_type', 'mlp')
    if alpha_mode == 'scalar':
        if model_type == 'gru':
            model = ScalarGRUNet(cfg).to(dtype=dtype, device=device)
        else:
            model = ScalarAlphaNet(cfg).to(dtype=dtype, device=device)
    else:
        model = PerRowAlphaNet(cfg).to(dtype=dtype, device=device)
    model.load_state_dict(ckpt['model_state'])
    model.feat_norm_active = ckpt.get('feat_norm_active', False)
    model.eval()
    print(f"  Loaded epoch {ckpt.get('epoch', '?')}, val_loss={ckpt.get('val_loss', '?'):.4f}")
    print(f"  device={cfg.device}, dtype={cfg.dtype}")

    # ---- Data ---- #
    import itertools
    type_loaders = make_dataloaders_multi(cfg, verbose=False)
    val_loader = itertools.chain(*[v[1] for v in type_loaders.values()])

    # ---- Evaluation ---- #
    report_steps = [s - 1 for s in [10, 20, 50, T_total] if s <= T_total]

    learned_metrics: dict[int, list] = {s + 1: [] for s in report_steps}
    baseline_metrics: dict[int, list] = {s + 1: [] for s in report_steps}

    # For convergence curves
    sample_learned_hist = None
    sample_baseline_hist = None
    sample_batch = None

    for i, batch in enumerate(val_loader):
        if i >= n_batches_eval:
            break

        batch = {
            k: (v.to(device=device, dtype=dtype) if v.is_floating_point()
                else v.to(device=device))
            if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        lrn_hist = learned_rollout(model, batch, cfg, T_total)
        bas_hist = baseline_rollout(batch, cfg, T_total)

        lrn_m = compute_metrics_at_steps(lrn_hist, batch, report_steps)
        bas_m = compute_metrics_at_steps(bas_hist, batch, report_steps)

        for step, vals in lrn_m.items():
            learned_metrics[step].append(vals)
        for step, vals in bas_m.items():
            baseline_metrics[step].append(vals)

        if sample_learned_hist is None:
            sample_learned_hist = lrn_hist
            sample_baseline_hist = bas_hist
            sample_batch = batch

    # ---- Print table ---- #
    print(f"\n{'Iter':>6}  {'Learned pri':>12} {'Baseline pri':>13}  "
          f"{'Learned dua':>12} {'Baseline dua':>13}  "
          f"{'Learned err':>12} {'Baseline err':>13}")
    print("-" * 90)

    for step in sorted(learned_metrics.keys()):
        lv = learned_metrics[step]
        bv = baseline_metrics[step]
        if not lv or not bv:
            continue

        l_pri = sum(d['pri'] for d in lv) / len(lv)
        b_pri = sum(d['pri'] for d in bv) / len(bv)
        l_dua = sum(d['dua'] for d in lv) / len(lv)
        b_dua = sum(d['dua'] for d in bv) / len(bv)
        l_err = sum(d['err'] for d in lv) / len(lv)
        b_err = sum(d['err'] for d in bv) / len(bv)

        print(f"{step:>6}  {l_pri:>12.3e} {b_pri:>13.3e}  "
              f"{l_dua:>12.3e} {b_dua:>13.3e}  "
              f"{l_err:>12.3e} {b_err:>13.3e}")

    # ---- Convergence curves ---- #
    if save_plots and sample_learned_hist is not None:
        _save_convergence_plots(
            sample_learned_hist, sample_baseline_hist, sample_batch, cfg, T_total
        )


def _save_convergence_plots(
    learned_hist, baseline_hist, batch, cfg: Config, T_total: int
) -> None:
    """Save primal and dual residual convergence curves for a few example instances.

    Layout: 2 rows × n_show columns.
      Row 0: primal residual  ||Ax - z||
      Row 1: dual   residual  ||Px + q + A^T y||
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plots")
        return

    results_dir = Path(cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    P = batch['P']
    A = batch['A']
    q = batch['q']

    n_show = min(3, P.shape[0])
    iters = list(range(1, T_total + 1))

    fig, axes = plt.subplots(2, n_show, figsize=(5 * n_show, 8), sharey='row')
    # axes shape: (2, n_show) — guarantee 2D even when n_show == 1
    if n_show == 1:
        axes = axes.reshape(2, 1)

    for b_idx in range(n_show):
        ax_pri = axes[0, b_idx]
        ax_dua = axes[1, b_idx]

        # ---- per-instance residual histories ---- #
        A_i = A[b_idx:b_idx+1]    # (1, m, n)
        P_i = P[b_idx:b_idx+1]    # (1, n, n)
        q_i = q[b_idx:b_idx+1]    # (1, n)

        def _pri(hist):
            return [
                torch.norm(
                    torch.bmm(A_i, x_t[b_idx:b_idx+1].unsqueeze(-1)).squeeze(-1)
                    - z_t[b_idx:b_idx+1],
                    dim=1,
                ).item()
                for x_t, z_t, _ in hist
            ]

        def _dua(hist):
            return [
                torch.norm(
                    torch.bmm(P_i, x_t[b_idx:b_idx+1].unsqueeze(-1)).squeeze(-1)
                    + q_i
                    + torch.bmm(A_i.transpose(1, 2), y_t[b_idx:b_idx+1].unsqueeze(-1)).squeeze(-1),
                    dim=1,
                ).item()
                for x_t, _, y_t in hist
            ]

        learned_pri  = _pri(learned_hist)
        baseline_pri = _pri(baseline_hist)
        learned_dua  = _dua(learned_hist)
        baseline_dua = _dua(baseline_hist)

        # ---- primal row ---- #
        ax_pri.semilogy(iters, learned_pri,  label='Learned',        color='tab:blue')
        ax_pri.semilogy(iters, baseline_pri, label='Baseline (α=1.6)', color='tab:orange', linestyle='--')
        ax_pri.set_ylabel('||Ax - z||')
        ax_pri.set_title(f'Instance {b_idx + 1}')
        ax_pri.legend(fontsize=8)
        ax_pri.grid(True, which='both', alpha=0.3)

        # ---- dual row ---- #
        ax_dua.semilogy(iters, learned_dua,  color='tab:blue')
        ax_dua.semilogy(iters, baseline_dua, color='tab:orange', linestyle='--')
        ax_dua.set_xlabel('Iteration')
        ax_dua.set_ylabel('||Px + q + Aᵀy||')
        ax_dua.grid(True, which='both', alpha=0.3)

    axes[0, 0].annotate('Primal residual', xy=(0, 0.5), xycoords='axes fraction',
                         fontsize=10, fontweight='bold', ha='right', va='center',
                         rotation=90, xytext=(-40, 0), textcoords='offset points')
    axes[1, 0].annotate('Dual residual', xy=(0, 0.5), xycoords='axes fraction',
                         fontsize=10, fontweight='bold', ha='right', va='center',
                         rotation=90, xytext=(-40, 0), textcoords='offset points')

    plt.suptitle('Convergence: Learned α vs. Fixed α=1.6', fontsize=12)
    plt.tight_layout()
    plot_path = results_dir / 'convergence_curves.png'
    plt.savefig(str(plot_path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved convergence plot to {plot_path}")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate learned OSQP alpha')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--T', type=int, default=1500, help='total iterations')
    parser.add_argument('--n', type=int, default=20, help='QP size n')
    parser.add_argument('--batches', type=int, default=10)
    parser.add_argument('--no-plots', action='store_true')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                        help='compute device')
    parser.add_argument('--dtype', type=str, default='float64',
                        choices=['float64', 'float32'],
                        help='floating-point dtype')
    parser.add_argument('--precision', type=str, default='low', choices=['low', 'high'],
                        help='convergence tolerance: low → eps=1e-3, high → eps=1e-5')
    parser.add_argument('--normalize_features', action='store_true',
                        help='(informational only — normalization state is restored from checkpoint)')
    parser.add_argument('--alpha_mode', type=str, default='vector',
                        choices=['vector', 'scalar'],
                        help='(informational only — alpha_mode is restored from checkpoint)')
    args = parser.parse_args()

    cfg = Config(n_fixed=args.n, device=args.device, dtype=args.dtype,
                 precision=args.precision, alpha_mode=args.alpha_mode)
    evaluate(
        cfg=cfg,
        checkpoint_path=args.checkpoint,
        T_total=args.T,
        n_batches_eval=args.batches,
        save_plots=not args.no_plots,
    )
