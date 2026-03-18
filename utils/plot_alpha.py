"""
Per-instance alpha history plot for NeuralScalarAlpha solvers.

Each plot shows, vs ADMM stage boundary (every T iterations):
  Left  y-axis (log scale): scaled_prim / scaled_dual ratio
  Right y-axis (linear):    alpha chosen by the neural net

scaled_prim = ||Ax - z||_inf / max(||Ax||_inf, ||z||_inf)
scaled_dual = ||Px + q + A^T y||_inf / max(||Px||_inf, ||A^T y||_inf, ||q||_inf)
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot_alpha_history(history: dict, title: str, output_path: str) -> None:
    """
    Generate and save a dual-axis alpha-history plot.

    Args:
        history:     dict with keys 'iter', 'alpha', 'scaled_prim', 'scaled_dual'
                     (all lists of the same length, recorded at stage boundaries)
        title:       plot title string (problem name + solver)
        output_path: full file path to save (e.g. .../alpha_plots/problem.png)
    """
    iters   = np.asarray(history['iter'],        dtype=float)
    alpha   = np.asarray(history['alpha'],       dtype=float)
    sc_prim = np.asarray(history['scaled_prim'], dtype=float)
    sc_dual = np.asarray(history['scaled_dual'], dtype=float)

    if len(iters) == 0:
        return

    _EPS  = 1e-14
    ratio = sc_prim / np.clip(sc_dual, _EPS, None)

    fig, ax_left = plt.subplots(figsize=(8, 4))
    ax_right = ax_left.twinx()

    # ---- Left axis: scaled_prim / scaled_dual ratio ----
    l1, = ax_left.semilogy(iters, np.clip(ratio, _EPS, None),
                           color='royalblue', linewidth=1.4,
                           label='scaled_prim / scaled_dual')
    ax_left.axhline(1.0, color='royalblue', linewidth=0.8, linestyle=':',
                    alpha=0.6)
    ax_left.set_xlabel('ADMM iteration (stage boundary)')
    ax_left.set_ylabel('scaled_prim / scaled_dual (log)', color='royalblue')
    ax_left.tick_params(axis='y', labelcolor='royalblue')

    # ---- Right axis: alpha (step function) ----
    l2, = ax_right.step(iters, alpha, where='post', color='seagreen',
                        linewidth=1.6, linestyle='--', label='alpha')
    ax_right.set_ylabel('alpha', color='seagreen')
    ax_right.tick_params(axis='y', labelcolor='seagreen')

    # ---- Legend ----
    lines  = [l1, l2]
    labels = [l.get_label() for l in lines]
    ax_left.legend(lines, labels, loc='upper right', fontsize=8)

    ax_left.set_title(title, fontsize=9)
    ax_left.grid(True, which='both', alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
