"""
Per-instance alpha history plots for Neural OSQP solvers.

plot_alpha_history:  dual-axis (scaled_prim/dual ratio + alpha value) — scalar only.
plot_alpha_change:   alpha change magnitude per stage boundary — both scalar and vector.
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


def plot_alpha_change(history: dict, title: str, output_path: str) -> None:
    """
    Plot alpha change magnitude per stage boundary on a log-scale y-axis.

    Works for both scalar and vector alpha modes.  The history dict must
    contain keys 'iter' and 'alpha_change' (both lists of the same length).
    For vector mode, alpha_change[i] = max(|alpha_z[i] - alpha_z[i-1]|).
    For scalar mode, alpha_change[i] = |alpha[i] - alpha[i-1]|.

    Args:
        history:     dict with keys 'iter', 'alpha_change'
        title:       plot title string
        output_path: full file path to save (.png)
    """
    iters  = np.asarray(history['iter'],         dtype=float)
    change = np.asarray(history['alpha_change'], dtype=float)

    if len(iters) < 2:
        return

    # Skip the first entry (change = 0 by definition)
    iters  = iters[1:]
    change = change[1:]

    _EPS = 1e-14

    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    ax.semilogy(iters, np.clip(change, _EPS, None),
                color='crimson', linewidth=1.0)
    ax.set_xlabel('OSQP iteration', fontsize=8)
    ax.set_ylabel(r'$\alpha$ change magnitude', fontsize=8)
    ax.set_title(title, fontsize=7)
    ax.tick_params(labelsize=7)
    ax.grid(True, which='both', alpha=0.3)

    fig.tight_layout(pad=0.3)
    fig.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close(fig)
