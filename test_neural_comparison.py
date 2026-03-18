'''
Test neural OSQP vs baseline on all QP types.

For each QP type, compares 8 solver configs:
  1. OSQP_python with adaptive_rho
  2. OSQP_python without adaptive_rho
  3. Neural scalar + arho (best_iter checkpoint)
  4. Neural scalar + arho (best_rho checkpoint)
  5. Neural vector + arho (best_iter checkpoint)
  6. Neural vector + arho (best_rho checkpoint)
  7. Neural scalar + no_arho (best_iter only — best_rho same when no rho updates)
  8. Neural vector + no_arho (best_iter only)
'''

import os
from functools import partial
from benchmark_problems.example import Example
from suitesparse_problems.suitesparse_problem import SuitesparseRunner
from maros_meszaros_problems.maros_meszaros_problem import MarosMeszarosRunner
import solvers.solvers as s
from solvers.osqppurepy import OSQP as OSQPPythonSolver
from learned_osqp.neural_osqp_solver import NeuralOSQPSolver
from utils.general import gen_int_log_space
from utils.benchmark_neural import compute_stats_info_split
import argparse


parser = argparse.ArgumentParser(description='Neural OSQP Comparison')
parser.add_argument('--high_accuracy', help='Test with high accuracy', default=False,
                    action='store_true')
parser.add_argument('--verbose', help='Verbose solvers', default=False,
                    action='store_true')
parser.add_argument('--parallel', help='Parallel solution', default=False,
                    action='store_true')
parser.add_argument('--small', help='Use small test (fast)', default=False,
                    action='store_true')
args = parser.parse_args()
high_accuracy = args.high_accuracy
verbose = args.verbose
parallel = args.parallel
small_test = args.small

print('high_accuracy:', high_accuracy)
print('verbose:', verbose)
print('parallel:', parallel)
print('small test:', small_test)

# Number of instances and dimensions
if small_test:
    n_instances = 3
    n_dim = 5
else:
    n_instances = 10
    n_dim = 10

# --------------------------------------------------------------------------- #
# QP types and dimensions
# --------------------------------------------------------------------------- #
problems = [
    'Random QP',
    'Portfolio',
    'Lasso',
    'SVM',
    'Control',
]

# problem_dimensions = {
#     'Random QP': gen_int_log_space(100, 250, n_dim),
#     'Portfolio': gen_int_log_space(5, 50, n_dim),
#     'Lasso': gen_int_log_space(5, 50, n_dim),
#     'SVM': gen_int_log_space(5, 25, n_dim),
#     'Control': gen_int_log_space(40, 160, n_dim),
# }

# problem_dimensions = {
#     'Random QP': gen_int_log_space(500, 500, n_dim),
#     'Portfolio': gen_int_log_space(50, 100, n_dim),
#     'Lasso': gen_int_log_space(50, 100, n_dim),
#     'SVM': gen_int_log_space(50, 100, n_dim),
#     'Control': gen_int_log_space(200, 200, n_dim),
# }

# for plotting
problem_dimensions = {
    'Random QP': gen_int_log_space(500, 10, 1),
    'Portfolio': gen_int_log_space(100, 10, 1),
    'Lasso': gen_int_log_space(100, 10, 1),
    'SVM': gen_int_log_space(100, 10, 1),
    'Control': gen_int_log_space(300, 10, 1),
}

problem_parallel = {p: parallel for p in problems}

# Mapping from Example problem name to checkpoint QP type string
QP_TYPE_MAP = {
    'Random QP': 'random_qp',
    'Portfolio': 'portfolio',
    'Lasso': 'lasso',
    'SVM': 'svm',
    'Control': 'control',
}

CHECKPOINT_DIR = os.path.join('learned_osqp', 'checkpoints_arc', 'float64_optimized')

# --------------------------------------------------------------------------- #
# Shared OSQP settings
# --------------------------------------------------------------------------- #
eps_low = 1e-03
time_limit = 1000.

_base_settings = {
    'max_iter': int(1e09),
    'eps_abs': eps_low,
    'eps_rel': eps_low,
    'polish': False,
    'verbose': False,
    'eps_prim_inf': 1e-15,
    'eps_dual_inf': 1e-15,
    'time_limit': time_limit,
}


def _make_settings(adaptive_rho: bool) -> dict:
    """Create a settings dict with adaptive_rho on or off."""
    d = dict(_base_settings)
    if not adaptive_rho:
        d['adaptive_rho'] = False
    return d

precision = 'high' if high_accuracy else 'low'


def _register_cross_neural_solvers(qp_key: str) -> list:
    """
    Register cross-test neural solvers where ckpt_arho != osqp_arho.

    Naming: OSQP_python_neural_{alpha_mode}_{ckpt_arho_s}_ckpt_{osqp_arho_s}_osqp_{ckpt_type}
      e.g.  OSQP_python_neural_scalar_arho_ckpt_noarho_osqp_best_iter
            (checkpoint trained with arho, running OSQP without arho)

    Returns list of added solver names.
    """
    names = []
    for alpha_mode in ['scalar', 'vector']:
        for ckpt_arho in [True, False]:
            ckpt_arho_s = 'arho' if ckpt_arho else 'noarho'
            osqp_arho = not ckpt_arho  # cross: opposite of ckpt
            osqp_arho_s = 'arho' if osqp_arho else 'noarho'
            ckpt_types = ['best_iter', 'best_rho'] if ckpt_arho else ['best_iter']
            for ckpt_type in ckpt_types:
                ckpt_base = (f'best_model_{qp_key}_precision={precision}'
                             f'_adaptive_rho={ckpt_arho}'
                             f'_alpha_mode={alpha_mode}')
                ckpt_file = ckpt_base + ('_best_rho.pt' if ckpt_type == 'best_rho' else '.pt')
                ckpt_path = os.path.join(CHECKPOINT_DIR, ckpt_file)

                name = (f'OSQP_python_neural_{alpha_mode}'
                        f'_{ckpt_arho_s}_ckpt_{osqp_arho_s}_osqp'
                        f'_{ckpt_type}')
                s.SOLVER_MAP[name] = partial(
                    NeuralOSQPSolver,
                    checkpoint_path=ckpt_path,
                    alpha_mode=alpha_mode,
                    record_history=(alpha_mode == 'scalar'),
                )
                s.settings[name] = _make_settings(adaptive_rho=osqp_arho)
                names.append(name)
    return names


# --------------------------------------------------------------------------- #
# Register solvers and run per QP type
# --------------------------------------------------------------------------- #
for problem in problems:
    qp_key = QP_TYPE_MAP[problem]
    solver_names = []

    # ---- 1. OSQP_python with adaptive_rho ----
    name = 'OSQP_python_arho'
    s.SOLVER_MAP[name] = OSQPPythonSolver
    s.settings[name] = _make_settings(adaptive_rho=True)
    solver_names.append(name)

    # ---- 2. OSQP_python without adaptive_rho ----
    name = 'OSQP_python_no_arho'
    s.SOLVER_MAP[name] = OSQPPythonSolver
    s.settings[name] = _make_settings(adaptive_rho=False)
    solver_names.append(name)

    # ---- 3-8. Neural variants ----
    for arho in [True, False]:
        arho_str = 'arho' if arho else 'no_arho'
        # With adaptive_rho: test both best_iter and best_rho checkpoints
        # Without adaptive_rho: best_rho is same as best_iter (0 rho updates), skip it
        ckpt_types = ['best_iter', 'best_rho'] if arho else ['best_iter']

        for alpha_mode in ['scalar', 'vector']:
            for ckpt_type in ckpt_types:
                # Build checkpoint path
                ckpt_base = (f'best_model_{qp_key}_precision={precision}'
                             f'_adaptive_rho={arho}'
                             f'_alpha_mode={alpha_mode}')
                if ckpt_type == 'best_rho':
                    ckpt_file = ckpt_base + '_best_rho.pt'
                else:
                    ckpt_file = ckpt_base + '.pt'
                ckpt_path = os.path.join(CHECKPOINT_DIR, ckpt_file)

                # Solver name must start with 'OSQP_python' for example.py branch
                name = f'OSQP_python_neural_{alpha_mode}_{arho_str}_{ckpt_type}'
                s.SOLVER_MAP[name] = partial(
                    NeuralOSQPSolver,
                    checkpoint_path=ckpt_path,
                    alpha_mode=alpha_mode,
                    record_history=(alpha_mode == 'scalar'),
                )
                s.settings[name] = _make_settings(adaptive_rho=arho)
                solver_names.append(name)

    # ---- Cross-test: ckpt_arho != osqp_arho ----
    solver_names += _register_cross_neural_solvers(qp_key)

    if verbose:
        for name in solver_names:
            s.settings[name]['verbose'] = True

    OUTPUT_FOLDER = f'0318_neural_comparison_{qp_key}_solver_alpha_plot'

    print("\n" + "=" * 80)
    print(f"Testing {problem} — {len(solver_names)} solver configs")
    print("Solvers:", solver_names)
    print("=" * 80 + "\n")

    example = Example(
        problem,
        problem_dimensions[problem],
        solver_names,
        s.settings,
        OUTPUT_FOLDER,
        n_instances,
    )
    example.solve(parallel=problem_parallel[problem])

    # Compute results statistics for this QP type
    print("\n" + "=" * 80)
    print(f"Computing Statistics for {problem}")
    print("=" * 80 + "\n")

    try:
        compute_stats_info_split(
            solver_names,
            OUTPUT_FOLDER,
            problems=[problem],
            high_accuracy=high_accuracy,
        )
    except FileNotFoundError as e:
        print("Note: compute_stats_info skipped (missing files): %s" % e)

print("\n" + "=" * 80)
print("All QP types completed!")
print("=" * 80)

# --------------------------------------------------------------------------- #
# SuitesparseLasso with lasso-trained neural solvers
# --------------------------------------------------------------------------- #
print("\n" + "=" * 80)
print("Testing SuitesparseLasso with lasso neural solvers")
print("=" * 80 + "\n")

SS_OUTPUT_FOLDER = f'0318_neural_comparison_suitesparse_lasso_solver_alpha_plot'
ss_solver_names = []

# ---- 1. OSQP_python with adaptive_rho ----
name = 'OSQP_python_arho'
s.SOLVER_MAP[name] = OSQPPythonSolver
s.settings[name] = _make_settings(adaptive_rho=True)
ss_solver_names.append(name)

# ---- 2. OSQP_python without adaptive_rho ----
name = 'OSQP_python_no_arho'
s.SOLVER_MAP[name] = OSQPPythonSolver
s.settings[name] = _make_settings(adaptive_rho=False)
ss_solver_names.append(name)

# ---- 3-8. Neural variants (lasso checkpoints) ----
for arho in [True, False]:
    arho_str = 'arho' if arho else 'no_arho'
    ckpt_types = ['best_iter', 'best_rho'] if arho else ['best_iter']

    for alpha_mode in ['scalar', 'vector']:
        for ckpt_type in ckpt_types:
            ckpt_base = (f'best_model_lasso_precision={precision}'
                         f'_adaptive_rho={arho}'
                         f'_alpha_mode={alpha_mode}')
            if ckpt_type == 'best_rho':
                ckpt_file = ckpt_base + '_best_rho.pt'
            else:
                ckpt_file = ckpt_base + '.pt'
            ckpt_path = os.path.join(CHECKPOINT_DIR, ckpt_file)

            name = f'OSQP_python_neural_{alpha_mode}_{arho_str}_{ckpt_type}'
            s.SOLVER_MAP[name] = partial(
                NeuralOSQPSolver,
                checkpoint_path=ckpt_path,
                alpha_mode=alpha_mode,
            )
            s.settings[name] = _make_settings(adaptive_rho=arho)
            ss_solver_names.append(name)

# ---- Cross-test: ckpt_arho != osqp_arho ----
ss_solver_names += _register_cross_neural_solvers('lasso')

if verbose:
    for name in ss_solver_names:
        s.settings[name]['verbose'] = True

print("Solvers:", ss_solver_names)

ss_runner = SuitesparseRunner(
    'Lasso',
    ss_solver_names,
    s.settings,
    SS_OUTPUT_FOLDER,
)
ss_runner.solve(parallel=parallel)

print("\n" + "=" * 80)
print("Computing Statistics for SuitesparseLasso")
print("=" * 80 + "\n")

try:
    compute_stats_info_split(
        ss_solver_names,
        SS_OUTPUT_FOLDER,
        problems=['Lasso'],
        high_accuracy=high_accuracy,
    )
except FileNotFoundError as e:
    print("Note: compute_stats_info skipped (missing files): %s" % e)

print("\n" + "=" * 80)
print("SuitesparseLasso completed!")
print("=" * 80)

# --------------------------------------------------------------------------- #
# Maros-Meszaros problems with control-trained neural solvers
# --------------------------------------------------------------------------- #
print("\n" + "=" * 80)
print("Testing Maros-Meszaros with control neural solvers")
print("=" * 80 + "\n")

MM_OUTPUT_FOLDER = f'0318_neural_comparison_maros_meszaros_solver_alpha_plot'
mm_solver_names = []

# ---- 1. OSQP_python with adaptive_rho ----
name = 'OSQP_python_arho'
s.SOLVER_MAP[name] = OSQPPythonSolver
s.settings[name] = _make_settings(adaptive_rho=True)
mm_solver_names.append(name)

# ---- 2. OSQP_python without adaptive_rho ----
name = 'OSQP_python_no_arho'
s.SOLVER_MAP[name] = OSQPPythonSolver
s.settings[name] = _make_settings(adaptive_rho=False)
mm_solver_names.append(name)

# ---- 3-8. Neural variants (control checkpoints) ----
for arho in [True, False]:
    arho_str = 'arho' if arho else 'no_arho'
    ckpt_types = ['best_iter', 'best_rho'] if arho else ['best_iter']

    for alpha_mode in ['scalar', 'vector']:
        for ckpt_type in ckpt_types:
            ckpt_base = (f'best_model_control_precision={precision}'
                         f'_adaptive_rho={arho}'
                         f'_alpha_mode={alpha_mode}')
            if ckpt_type == 'best_rho':
                ckpt_file = ckpt_base + '_best_rho.pt'
            else:
                ckpt_file = ckpt_base + '.pt'
            ckpt_path = os.path.join(CHECKPOINT_DIR, ckpt_file)

            name = f'OSQP_python_neural_{alpha_mode}_{arho_str}_{ckpt_type}'
            s.SOLVER_MAP[name] = partial(
                NeuralOSQPSolver,
                checkpoint_path=ckpt_path,
                alpha_mode=alpha_mode,
            )
            s.settings[name] = _make_settings(adaptive_rho=arho)
            mm_solver_names.append(name)

# ---- Cross-test: ckpt_arho != osqp_arho ----
mm_solver_names += _register_cross_neural_solvers('control')

if verbose:
    for name in mm_solver_names:
        s.settings[name]['verbose'] = True

print("Solvers:", mm_solver_names)

mm_runner = MarosMeszarosRunner(
    mm_solver_names,
    s.settings,
    MM_OUTPUT_FOLDER,
)
mm_runner.solve(parallel=parallel)

print("\n" + "=" * 80)
print("Computing Statistics for Maros-Meszaros")
print("=" * 80 + "\n")

try:
    # MarosMeszarosRunner writes results.csv directly (no per-class subdir),
    # so pass problems=None to skip get_cumulative_data
    compute_stats_info_split(
        mm_solver_names,
        MM_OUTPUT_FOLDER,
        problems=None,
        high_accuracy=high_accuracy,
    )
except FileNotFoundError as e:
    print("Note: compute_stats_info skipped (missing files): %s" % e)

print("\n" + "=" * 80)
print("Maros-Meszaros completed!")
print("=" * 80)
