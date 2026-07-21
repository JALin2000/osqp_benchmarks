'''
Test neural OSQP on control-fixed problem with unseen initial states.

Uses the same dynamics (A, B, Q, R, QN, bounds) as training but samples
test x0's from seeds that don't overlap with training (0..n_train-1) or
validation (n_train..n_train+n_val-1).

Usage:
    python test_control_fixed.py --mode benchmark
    python test_control_fixed.py --mode plot
'''

import os
import argparse
import numpy as np
import pandas as pd
from functools import partial

import solvers.solvers as s
from problem_classes.control import ControlExample
from solvers.osqppurepy import OSQP as OSQPPythonSolver
from learned_osqp.neural_osqp_solver import NeuralOSQPSolver
from utils.general import make_sure_path_exists
from utils.plot_alpha import plot_alpha_history, plot_alpha_change

# --------------------------------------------------------------------------- #
# Arguments
# --------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description='Test neural OSQP on control-fixed')
parser.add_argument('--nx', type=int, default=100)
parser.add_argument('--dynamics_seed', type=int, default=0)
parser.add_argument('--n_train', type=int, default=160,
                    help='Number of training instances (seeds 0..n_train-1)')
parser.add_argument('--n_val', type=int, default=80,
                    help='Number of validation instances (seeds n_train..n_train+n_val-1)')
parser.add_argument('--n_test', type=int, default=100,
                    help='Number of test instances (benchmark mode)')
parser.add_argument('--mode', type=str, default='benchmark',
                    choices=['benchmark', 'plot'])
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--precision', type=str, default='low',
                    choices=['low', 'high'])
parser.add_argument('--ckpt_dir', type=str,
                    default=os.path.join('learned_osqp', 'checkpoints_arc',
                                         '0324_control_fixed'))
args = parser.parse_args()

nx = args.nx
dynamics_seed = args.dynamics_seed
n_train = args.n_train
n_val = args.n_val
n_test = args.n_test if args.mode == 'benchmark' else 3
mode = args.mode
precision = args.precision
CHECKPOINT_DIR = args.ckpt_dir

# Test x0 seeds start AFTER train + val
x0_seed_offset = n_train + n_val  # 240 by default

print(f'nx={nx}, dynamics_seed={dynamics_seed}')
print(f'n_train={n_train}, n_val={n_val}, n_test={n_test}')
print(f'x0 test seeds: {x0_seed_offset} .. {x0_seed_offset + n_test - 1}')
print(f'mode={mode}, precision={precision}')
print(f'checkpoint dir: {CHECKPOINT_DIR}')

# --------------------------------------------------------------------------- #
# OSQP settings
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
    d = dict(_base_settings)
    if not adaptive_rho:
        d['adaptive_rho'] = False
    return d


# --------------------------------------------------------------------------- #
# Checkpoint path builder
# --------------------------------------------------------------------------- #
def _ckpt_path(arho: bool, alpha_mode: str, model_type: str,
               ckpt_type: str) -> str:
    suffix = '_best_rho.pt' if ckpt_type == 'best_rho' else '.pt'
    base = (f'best_model_control_fixed_nx{nx}_dseed{dynamics_seed}'
            f'_precision={precision}'
            f'_adaptive_rho={arho}'
            f'_alpha_mode={alpha_mode}'
            f'_model_type={model_type}'
            f'_loss=log_convergence')
    return os.path.join(CHECKPOINT_DIR, base + suffix)


# --------------------------------------------------------------------------- #
# Register solvers
# --------------------------------------------------------------------------- #
solver_names = []

# Baseline: OSQP with adaptive rho
name = 'OSQP_python_arho'
s.SOLVER_MAP[name] = OSQPPythonSolver
s.settings[name] = _make_settings(adaptive_rho=True)
solver_names.append(name)

# Baseline: OSQP without adaptive rho
name = 'OSQP_python_no_arho'
s.SOLVER_MAP[name] = OSQPPythonSolver
s.settings[name] = _make_settings(adaptive_rho=False)
solver_names.append(name)

# Neural variants
record_history = (mode == 'plot')
for model_type in ['mlp']:
    for arho in [True, False]:
        arho_str = 'arho' if arho else 'no_arho'
        ckpt_types = ['best_iter', 'best_rho'] if arho else ['best_iter']
        for alpha_mode in ['scalar', 'vector']:
            for ckpt_type in ckpt_types:
                ckpt = _ckpt_path(arho, alpha_mode, model_type, ckpt_type)
                if not os.path.isfile(ckpt):
                    print(f'  [skip] {ckpt} not found')
                    continue
                name = (f'OSQP_python_neural_{model_type}_{alpha_mode}'
                        f'_{arho_str}_{ckpt_type}')
                s.SOLVER_MAP[name] = partial(
                    NeuralOSQPSolver,
                    checkpoint_path=ckpt,
                    alpha_mode=alpha_mode,
                    model_type=model_type,
                    record_history=record_history,
                )
                s.settings[name] = _make_settings(adaptive_rho=arho)
                solver_names.append(name)

if args.verbose:
    for name in solver_names:
        s.settings[name]['verbose'] = True

print(f"\nSolvers ({len(solver_names)}):", solver_names)

# --------------------------------------------------------------------------- #
# Build base control system (shared dynamics)
# --------------------------------------------------------------------------- #
base = ControlExample(nx, seed=dynamics_seed)

OUTPUT_FOLDER = f'control_fixed_nx{nx}_dseed{dynamics_seed}_alpha_freeze_new'
output_base = os.path.join('.', 'results', OUTPUT_FOLDER)

# --------------------------------------------------------------------------- #
# Test loop
# --------------------------------------------------------------------------- #
for solver_name in solver_names:
    settings = s.settings[solver_name]
    results_list = []

    solver_dir = os.path.join(output_base, solver_name)
    make_sure_path_exists(solver_dir)
    results_file = os.path.join(solver_dir, 'results.csv')

    if os.path.isfile(results_file):
        print(f'\n[skip] {solver_name}: results.csv already exists')
        continue

    print(f'\n--- Solving with {solver_name} ---')

    for i in range(n_test):
        seed = x0_seed_offset + i

        # Sample x0 (same method as training)
        rng = np.random.default_rng(seed)
        raw = rng.random(base.nx)
        min_x0 = 0.5 * base.xmin
        max_x0 = 0.5 * base.xmax
        x0_new = min_x0 + raw * (max_x0 - min_x0)

        base.update_x0(x0_new)
        qp = base.qp_problem

        print(f'  instance {i} (x0 seed={seed})', end=' ... ', flush=True)

        # Create and setup solver
        solver_obj = s.SOLVER_MAP[solver_name]()
        solver_obj.setup(
            P=qp['P'], q=qp['q'], A=qp['A'],
            l=qp['l'].copy(), u=qp['u'].copy(),
            **settings,
        )
        result = solver_obj.solve()

        print(f'status={result.status}, iter={result.niter}, '
              f'time={result.run_time:.4f}s')

        # Record result
        row = {
            'class': 'Control_fixed',
            'solver': solver_name,
            'status': result.status,
            'run_time': result.run_time,
            'iter': result.niter,
            'obj_val': result.obj_val,
            'n': nx,
            'x0_seed': seed,
            'instance': i,
        }
        if solver_name[:4] == 'OSQP':
            row['setup_time'] = result.setup_time
            row['solve_time'] = result.solve_time
            row['update_time'] = result.update_time
            row['rho_updates'] = result.rho_updates

        results_list.append(row)

        # Alpha change plots (plot mode only)
        if hasattr(solver_obj, 'get_alpha_history'):
            hist = solver_obj.get_alpha_history()
            if hist and hist.get('iter'):
                plot_dir = os.path.join(solver_dir, 'alpha_plots')
                make_sure_path_exists(plot_dir)
                title = (f'Control_fixed nx={nx} x0_seed={seed}'
                         f' | {solver_name}')

                if 'alpha' in hist:
                    fname = os.path.join(
                        plot_dir, f'alpha_seed{seed}.png')
                    plot_alpha_history(hist, title, fname)

                if 'alpha_change' in hist:
                    fname = os.path.join(
                        plot_dir, f'alpha_change_seed{seed}.pdf')
                    plot_alpha_change(hist, title, fname)

    # Save results
    df = pd.DataFrame(results_list)
    df.to_csv(results_file, index=False)
    print(f'  Saved: {results_file}')

# --------------------------------------------------------------------------- #
# Summary statistics
# --------------------------------------------------------------------------- #
print('\n' + '=' * 80)
print('Summary Statistics')
print('=' * 80)

for solver_name in solver_names:
    results_file = os.path.join(output_base, solver_name, 'results.csv')
    if not os.path.isfile(results_file):
        continue
    df = pd.read_csv(results_file)
    iter_mean = df['iter'].mean()
    iter_std = df['iter'].std()
    time_col = 'solve_time' if 'solve_time' in df.columns else 'run_time'
    time_mean = df[time_col].mean()
    time_std = df[time_col].std()
    rho_str = ''
    if 'rho_updates' in df.columns:
        rho_str = f'  rho_updates={df["rho_updates"].mean():.1f}'
    print(f'  {solver_name:55s}  iter={iter_mean:8.1f}+/-{iter_std:6.1f}  '
          f'time={time_mean:.4f}+/-{time_std:.4f}{rho_str}')

# --------------------------------------------------------------------------- #
# Verify no x0 seed overlap between train/val and test
# --------------------------------------------------------------------------- #
print('\n' + '=' * 80)
print('x0 Seed Overlap Verification')
print('=' * 80)

train_seeds = set(range(n_train))
val_seeds = set(range(n_train, n_train + n_val))
test_seeds = set(range(x0_seed_offset, x0_seed_offset + n_test))

train_overlap = train_seeds & test_seeds
val_overlap = val_seeds & test_seeds

if train_overlap:
    print(f'  WARNING: test seeds overlap with training: {sorted(train_overlap)}')
else:
    print(f'  OK: No overlap with training seeds (0..{n_train - 1})')

if val_overlap:
    print(f'  WARNING: test seeds overlap with validation: {sorted(val_overlap)}')
else:
    print(f'  OK: No overlap with validation seeds ({n_train}..{n_train + n_val - 1})')

print(f'  Test seeds: {x0_seed_offset}..{x0_seed_offset + n_test - 1}')

print('\n' + '=' * 80)
print('Done!')
print('=' * 80)
