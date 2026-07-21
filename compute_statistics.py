"""Compute average solve_time, iter and their std from results/0324* folders.

Produces a pivoted CSV matching the paper table layout:
  Rows    = problem types (Random QP, Portfolio, Lasso, SVM, Control)
  Columns = 8 solver configs grouped under "no rho adapt" and "rho adapt"
  Cells   = "iter_mean (solve_time_mean)"
"""

import os
import csv
import glob
import numpy as np

RESULTS_ROOT = os.path.join(os.path.dirname(__file__), 'results')
OUTPUT_CSV = os.path.join(RESULTS_ROOT, 'summary_statistics_alpha_freeze.csv')

# ---- Problem folder → display name ----
PROBLEM_MAP = {
    '0324_neural_comparison_random_qp_solver_alpha_freeze': 'Random QP',
    '0324_neural_comparison_portfolio_solver_alpha_freeze': 'Portfolio',
    '0324_neural_comparison_lasso_solver_alpha_freeze': 'Lasso',
    '0324_neural_comparison_svm_solver_alpha_freeze': 'SVM',
    '0324_neural_comparison_control_solver_alpha_freeze': 'Control',
}
PROBLEM_ORDER = ['Random QP', 'Portfolio', 'Lasso', 'SVM', 'Control']

# ---- Column definitions: (column_label, solver_folder_name) ----
COLUMNS = [
    ('osqp',             'OSQP_python_no_arho'),
    ('scalar',           'OSQP_python_neural_mlp_scalar_no_arho_best_iter'),
    ('vector',           'OSQP_python_neural_mlp_vector_no_arho_best_iter'),
    ('osqp',             'OSQP_python_arho'),
    ('scalar best iter', 'OSQP_python_neural_mlp_scalar_arho_best_iter'),
    ('scalar best rho',  'OSQP_python_neural_mlp_scalar_arho_best_rho'),
    ('vector best iter', 'OSQP_python_neural_mlp_vector_arho_best_iter'),
    ('vector best rho',  'OSQP_python_neural_mlp_vector_arho_best_rho'),
]


def _read_stats(csv_path: str) -> tuple[float, float] | None:
    """Read results.csv → (iter_mean, solve_time_mean) or None."""
    iters, times = [], []
    with open(csv_path, 'r') as f:
        for row in csv.DictReader(f):
            try:
                iters.append(float(row['iter']))
                times.append(float(row['solve_time']))
            except (ValueError, KeyError):
                continue
    if not iters:
        return None
    return float(np.mean(iters)), float(np.mean(times))


def _fmt(stats: tuple[float, float] | None) -> str:
    """Format as 'iter_mean (solve_time_mean)'."""
    if stats is None:
        return ''
    it, st = stats
    return f'{it:.2f} ({st:.3f})'


# ---- Collect data keyed by (problem_display, solver_folder) ----
data: dict[tuple[str, str], tuple[float, float]] = {}

for problem_dir in sorted(glob.glob(os.path.join(RESULTS_ROOT, '0324*_alpha_freeze'))):
    if not os.path.isdir(problem_dir):
        continue
    folder = os.path.basename(problem_dir)
    display = PROBLEM_MAP.get(folder)
    if display is None:
        continue

    for _, solver_name in COLUMNS:
        csv_path = os.path.join(problem_dir, solver_name, 'results.csv')
        if os.path.isfile(csv_path):
            stats = _read_stats(csv_path)
            if stats is not None:
                data[(display, solver_name)] = stats

# ---- Write pivoted CSV ----
with open(OUTPUT_CSV, 'w', newline='') as f:
    w = csv.writer(f)
    # Header row 1: group labels
    w.writerow(['', 'no \\rho adapt', '', '', '\\rho adapt', '', '', '', ''])
    # Header row 2: column labels
    w.writerow([''] + [label for label, _ in COLUMNS])
    # Data rows
    for prob in PROBLEM_ORDER:
        cells = [_fmt(data.get((prob, solver))) for _, solver in COLUMNS]
        w.writerow([prob] + cells)

print(f'Wrote {OUTPUT_CSV}')
