'''
Test neural OSQP vs baseline on all QP types.

For each QP type, compares 14 solver configs:
  1.  OSQP_python with adaptive_rho
  2.  OSQP_python without adaptive_rho
  --- MLP ---
  3.  Neural mlp scalar + arho (best_iter checkpoint)
  4.  Neural mlp scalar + arho (best_rho checkpoint)
  5.  Neural mlp vector + arho (best_iter checkpoint)
  6.  Neural mlp vector + arho (best_rho checkpoint)
  7.  Neural mlp scalar + no_arho (best_iter only)
  8.  Neural mlp vector + no_arho (best_iter only)
  --- GRU ---
  9.  Neural gru scalar + arho (best_iter checkpoint)
  10. Neural gru scalar + arho (best_rho checkpoint)
  11. Neural gru vector + arho (best_iter checkpoint)
  12. Neural gru vector + arho (best_rho checkpoint)
  13. Neural gru scalar + no_arho (best_iter only)
  14. Neural gru vector + no_arho (best_iter only)
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

problem_dimensions = {
    'Random QP': gen_int_log_space(500, 500, n_dim),
    'Portfolio': gen_int_log_space(50, 100, n_dim),
    'Lasso': gen_int_log_space(50, 100, n_dim),
    'SVM': gen_int_log_space(50, 100, n_dim),
    'Control': gen_int_log_space(200, 200, n_dim),
}

# for plotting
# problem_dimensions = {
#     'Random QP': gen_int_log_space(500, 10, 1),
#     'Portfolio': gen_int_log_space(100, 10, 1),
#     'Lasso': gen_int_log_space(100, 10, 1),
#     'SVM': gen_int_log_space(100, 10, 1),
#     'Control': gen_int_log_space(300, 10, 1),
# }

problem_parallel = {p: parallel for p in problems}

# Mapping from Example problem name to checkpoint QP type string
QP_TYPE_MAP = {
    'Random QP': 'random_qp',
    'Portfolio': 'portfolio',
    'Lasso': 'lasso',
    'SVM': 'svm',
    'Control': 'control',
}

CHECKPOINT_DIR = os.path.join('learned_osqp', 'checkpoints_arc', '0319_feat_pri_dua_res_scaled_alpha_1.25_1.95')

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


def _ckpt_path(qp_key: str, arho: bool, alpha_mode: str, model_type: str,
               ckpt_type: str) -> str:
    """Build checkpoint path with backward-compatible fallback.

    New naming: ..._model_type={model_type}_loss=log_convergence[_best_rho].pt
    Old naming (mlp only): ...alpha_mode={alpha_mode}[_best_rho].pt
    """
    suffix = '_best_rho.pt' if ckpt_type == 'best_rho' else '.pt'
    # New naming (includes model_type and loss)
    new_base = (f'best_model_{qp_key}_precision={precision}'
                f'_adaptive_rho={arho}'
                f'_alpha_mode={alpha_mode}'
                f'_model_type={model_type}'
                f'_loss=log_convergence')
    new_path = os.path.join(CHECKPOINT_DIR, new_base + suffix)
    if os.path.exists(new_path):
        return new_path
    # Old naming fallback (no model_type / loss fields, mlp only)
    old_base = (f'best_model_{qp_key}_precision={precision}'
                f'_adaptive_rho={arho}'
                f'_alpha_mode={alpha_mode}')
    old_path = os.path.join(CHECKPOINT_DIR, old_base + suffix)
    return old_path


def _register_cross_neural_solvers(qp_key: str) -> list:
    """
    Register cross-test neural solvers where ckpt_arho != osqp_arho.

    Naming: OSQP_python_neural_{model_type}_{alpha_mode}_{ckpt_arho_s}_ckpt_{osqp_arho_s}_osqp_{ckpt_type}
      e.g.  OSQP_python_neural_mlp_scalar_arho_ckpt_noarho_osqp_best_iter
            (checkpoint trained with arho, running OSQP without arho)

    Returns list of added solver names.
    """
    names = []
    for model_type in ['mlp', 'gru']:
        for alpha_mode in ['scalar', 'vector']:
            for ckpt_arho in [True, False]:
                ckpt_arho_s = 'arho' if ckpt_arho else 'noarho'
                osqp_arho = not ckpt_arho  # cross: opposite of ckpt
                osqp_arho_s = 'arho' if osqp_arho else 'noarho'
                ckpt_types = ['best_iter', 'best_rho'] if ckpt_arho else ['best_iter']
                for ckpt_type in ckpt_types:
                    ckpt_path = _ckpt_path(qp_key, ckpt_arho, alpha_mode,
                                           model_type, ckpt_type)

                    name = (f'OSQP_python_neural_{model_type}_{alpha_mode}'
                            f'_{ckpt_arho_s}_ckpt_{osqp_arho_s}_osqp'
                            f'_{ckpt_type}')
                    s.SOLVER_MAP[name] = partial(
                        NeuralOSQPSolver,
                        checkpoint_path=ckpt_path,
                        alpha_mode=alpha_mode,
                        model_type=model_type,
                        record_history=False,
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

    # ---- 3-14. Neural variants (mlp + gru) ----
    for model_type in ['mlp', 'gru']:
        for arho in [True, False]:
            arho_str = 'arho' if arho else 'no_arho'
            # With adaptive_rho: test both best_iter and best_rho checkpoints
            # Without adaptive_rho: best_rho is same as best_iter (0 rho updates), skip it
            ckpt_types = ['best_iter', 'best_rho'] if arho else ['best_iter']

            for alpha_mode in ['scalar', 'vector']:
                for ckpt_type in ckpt_types:
                    ckpt_path = _ckpt_path(qp_key, arho, alpha_mode,
                                           model_type, ckpt_type)

                    # Solver name must start with 'OSQP_python' for example.py branch
                    name = f'OSQP_python_neural_{model_type}_{alpha_mode}_{arho_str}_{ckpt_type}'
                    s.SOLVER_MAP[name] = partial(
                        NeuralOSQPSolver,
                        checkpoint_path=ckpt_path,
                        alpha_mode=alpha_mode,
                        model_type=model_type,
                        record_history=False,
                    )
                    s.settings[name] = _make_settings(adaptive_rho=arho)
                    solver_names.append(name)

    # ---- Cross-test: ckpt_arho != osqp_arho ----
    solver_names += _register_cross_neural_solvers(qp_key)

    if verbose:
        for name in solver_names:
            s.settings[name]['verbose'] = True

    OUTPUT_FOLDER = f'0319_neural_comparison_{qp_key}_solver'

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

SS_OUTPUT_FOLDER = f'0319_neural_comparison_suitesparse_lasso_solver'
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

# ---- 3-14. Neural variants (lasso checkpoints, mlp + gru) ----
for model_type in ['mlp', 'gru']:
    for arho in [True, False]:
        arho_str = 'arho' if arho else 'no_arho'
        ckpt_types = ['best_iter', 'best_rho'] if arho else ['best_iter']

        for alpha_mode in ['scalar', 'vector']:
            for ckpt_type in ckpt_types:
                ckpt_path = _ckpt_path('lasso', arho, alpha_mode,
                                       model_type, ckpt_type)

                name = f'OSQP_python_neural_{model_type}_{alpha_mode}_{arho_str}_{ckpt_type}'
                s.SOLVER_MAP[name] = partial(
                    NeuralOSQPSolver,
                    checkpoint_path=ckpt_path,
                    alpha_mode=alpha_mode,
                    model_type=model_type,
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

MM_OUTPUT_FOLDER = f'0319_neural_comparison_maros_meszaros_solver'
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

# ---- 3-14. Neural variants (control checkpoints, mlp + gru) ----
for model_type in ['mlp', 'gru']:
    for arho in [True, False]:
        arho_str = 'arho' if arho else 'no_arho'
        ckpt_types = ['best_iter', 'best_rho'] if arho else ['best_iter']

        for alpha_mode in ['scalar', 'vector']:
            for ckpt_type in ckpt_types:
                ckpt_path = _ckpt_path('control', arho, alpha_mode,
                                       model_type, ckpt_type)

                name = f'OSQP_python_neural_{model_type}_{alpha_mode}_{arho_str}_{ckpt_type}'
                s.SOLVER_MAP[name] = partial(
                    NeuralOSQPSolver,
                    checkpoint_path=ckpt_path,
                    alpha_mode=alpha_mode,
                    model_type=model_type,
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
