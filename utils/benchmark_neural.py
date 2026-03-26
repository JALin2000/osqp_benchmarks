"""
Benchmark utilities for neural OSQP comparison.

Splits performance profiles into arho vs no_arho groups
(4 curves each) and uses fresh figures to avoid accumulation.
"""

import os
import pandas as pd
import numpy as np
import solvers.statuses as statuses
from solvers.solvers import time_limit

import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt

MAX_TIMING = time_limit


def get_cumulative_data(solvers, problems, output_folder):
    for solver in solvers:
        path = os.path.join('.', 'results', output_folder, solver)
        results = []
        for problem in problems:
            file_name = os.path.join(path, problem, 'full.csv')
            results.append(pd.read_csv(file_name))
        df = pd.concat(results)
        solver_file_name = os.path.join(path, 'results.csv')
        df.to_csv(solver_file_name, index=False)


MAX_ITER = int(1e09)  # fallback for failed solves (iter metric)


def _compute_perf_profile_data(solvers, output_folder, metric='run_time'):
    """Compute performance profile raw data for given solvers.

    Args:
        metric: 'run_time' or 'iter'
    """
    t = {}
    status = {}
    fallback = MAX_TIMING if metric == 'run_time' else MAX_ITER

    for solver in solvers:
        path = os.path.join('.', 'results', output_folder, solver, 'results.csv')
        df = pd.read_csv(path)
        n_problems = len(df)
        t[solver] = df[metric].values.astype(float)
        status[solver] = df['status'].values
        for idx in range(n_problems):
            if status[solver][idx] not in statuses.SOLUTION_PRESENT:
                t[solver][idx] = fallback

    # Compute relative values
    r = {s_: np.zeros(n_problems) for s_ in solvers}
    for p in range(n_problems):
        min_val = np.min([t[s_][p] for s_ in solvers])
        for s_ in solvers:
            r[s_][p] = t[s_][p] / min_val

    # Compute curves
    n_tau = 1000
    tau_vec = np.linspace(1, 2, n_tau)
    rho = {'tau': tau_vec}
    for s_ in solvers:
        rho[s_] = np.zeros(n_tau)
        for tau_idx in range(n_tau):
            rho[s_][tau_idx] = np.sum(r[s_] <= tau_vec[tau_idx]) / n_problems

    return rho


def _plot_profile(rho, solvers, title, output_path):
    """Plot performance profile as (1,2) subplots: MLP+baselines | GRU+baselines.

    Performance ratios in ``rho`` are computed over ALL solvers jointly.
    The two subplots simply filter which curves to show.
    """
    baselines = [s_ for s_ in solvers if '_neural_' not in s_]
    mlp_solvers = [s_ for s_ in solvers if '_neural_mlp_' in s_]
    gru_solvers = [s_ for s_ in solvers if '_neural_gru_' in s_]

    # If we can split into mlp/gru groups, draw side-by-side subplots
    if mlp_solvers and gru_solvers:
        fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(18, 6))
        for solver in baselines + mlp_solvers:
            ax_l.plot(rho['tau'], rho[solver], label=solver)
        ax_l.set_xlim(1., 2.)
        ax_l.set_ylim(0., 1.)
        ax_l.set_xlabel(r'Performance ratio $\tau$')
        ax_l.set_ylabel('Ratio of problems solved')
        ax_l.set_title(f'{title} — MLP')
        ax_l.legend(fontsize=7)
        ax_l.grid(True)

        for solver in baselines + gru_solvers:
            ax_r.plot(rho['tau'], rho[solver], label=solver)
        ax_r.set_xlim(1., 2.)
        ax_r.set_ylim(0., 1.)
        ax_r.set_xlabel(r'Performance ratio $\tau$')
        ax_r.set_ylabel('Ratio of problems solved')
        ax_r.set_title(f'{title} — GRU')
        ax_r.legend(fontsize=7)
        ax_r.grid(True)
    else:
        # Fallback: single plot (no mlp/gru split possible)
        fig, ax = plt.subplots(figsize=(10, 6))
        for solver in solvers:
            ax.plot(rho['tau'], rho[solver], label=solver)
        ax.set_xlim(1., 2.)
        ax.set_ylim(0., 1.)
        ax.set_xlabel(r'Performance ratio $\tau$')
        ax.set_ylabel('Ratio of problems solved')
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True)

    fig.tight_layout()
    print("Saving plot to %s" % output_path)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def compute_failure_rates(solvers, output_folder):
    failure_rates = {}
    for solver in solvers:
        results_file = os.path.join('.', 'results', output_folder, solver, 'results.csv')
        df = pd.read_csv(results_file)
        n_problems = len(df)
        failed_statuses = np.logical_and(*[df['status'].values != s_
                                           for s_ in statuses.SOLUTION_PRESENT])
        n_failed_problems = np.sum(failed_statuses)
        failure_rates[solver] = 100 * (n_failed_problems / n_problems)

    failure_rates_file = os.path.join('.', 'results', output_folder, 'failure_rates.csv')
    pd.Series(failure_rates).to_frame().transpose().to_csv(failure_rates_file, index=False)


def geom_mean(t, shift=10.):
    return np.exp(np.sum(np.log(np.maximum(1, t + shift)) / len(t))) - shift


def compute_shifted_geometric_means(solvers, output_folder):
    t = {}
    status = {}
    g_mean = {}

    for solver in solvers:
        path = os.path.join('.', 'results', output_folder, solver, 'results.csv')
        df = pd.read_csv(path)
        n_problems = len(df)
        t[solver] = df['run_time'].values
        status[solver] = df['status'].values
        for idx in range(n_problems):
            if status[solver][idx] not in statuses.SOLUTION_PRESENT:
                t[solver][idx] = MAX_TIMING
        g_mean[solver] = geom_mean(t[solver])

    best_g_mean = np.min([g_mean[s_] for s_ in solvers])
    for s_ in solvers:
        g_mean[s_] /= best_g_mean

    g_mean_file = os.path.join('.', 'results', output_folder, 'geom_mean.csv')
    pd.Series(g_mean).to_frame().transpose().to_csv(g_mean_file, index=False)


def compute_stats_info_split(all_solvers, output_folder, problems=None,
                             high_accuracy=False):
    """
    Like compute_stats_info but splits performance profiles into
    arho and no_arho groups (4 curves each).
    """
    if problems is not None:
        get_cumulative_data(all_solvers, problems, output_folder)

    # Failure rates & geom means over all solvers
    compute_failure_rates(all_solvers, output_folder)
    compute_shifted_geometric_means(all_solvers, output_folder)

    # Split solvers into arho / no_arho groups based on OSQP *runtime* arho setting.
    # Cross-test names embed the osqp setting as '_noarho_osqp_' or '_arho_osqp_'.
    # Existing/baseline names use the plain 'no_arho' / 'arho' suffix convention.
    def _osqp_uses_arho(name):
        if '_osqp_' in name:
            return '_noarho_osqp_' not in name   # cross-test: check osqp part
        return 'no_arho' not in name              # existing names

    arho_solvers = [s_ for s_ in all_solvers if _osqp_uses_arho(s_)]
    no_arho_solvers = [s_ for s_ in all_solvers if not _osqp_uses_arho(s_)]

    results_dir = os.path.join('.', 'results', output_folder)

    for group_name, group_solvers in [('arho', arho_solvers), ('no_arho', no_arho_solvers)]:
        if len(group_solvers) < 2:
            continue
        label = 'adaptive_rho' if group_name == 'arho' else 'no adaptive_rho'

        for metric, metric_label in [('run_time', 'time'), ('iter', 'iterations')]:
            rho = _compute_perf_profile_data(group_solvers, output_folder, metric=metric)
            df = pd.DataFrame(rho)
            df.to_csv(os.path.join(results_dir,
                                   f'performance_profiles_{group_name}_{metric_label}.csv'),
                      index=False)
            _plot_profile(rho, group_solvers,
                          f'{output_folder} — {label} ({metric_label})',
                          os.path.join(results_dir,
                                       f'{output_folder}_{group_name}_{metric_label}.png'))
