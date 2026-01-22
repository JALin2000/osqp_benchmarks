'''
Test ADMM solver on Random QP problems.

This script tests the ADMM solver against OSQP on random QP problems
following the same structure as run_benchmark_problems.py
'''

from benchmark_problems.example import Example
import solvers.solvers as s
from utils.general import gen_int_log_space
from utils.benchmark import compute_stats_info
import argparse


parser = argparse.ArgumentParser(description='ADMM Test on Random QP')
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

# Configure solvers
if high_accuracy:
    solvers = [# s.ADMM_high, 
               # s.OSQP_high, s.OSQP_polish_high, 
               s.SuperADMM_high, 
               s.Super_ruiz_high,
            # #    s.Super_ruiz_kaczmarz_high,
               s.Super_ruiz_ldlt_high,
               s.Super_ruiz_new_fact_high,
               s.Super_ruiz_cg_high,
               s.Super_ruiz_cg_precond_high,
               s.Super_ldlt_high,
               s.Super_new_fact_high,
               s.Super_cg_high,
               s.Super_cg_precond_high,
               ]
    # OUTPUT_FOLDER = 'ADMM_alpha=1.0_gradual_change4_high_accuracy'
    OUTPUT_FOLDER = 'ablation_study_high_accuracy'
    for key in s.settings:
        s.settings[key]['high_accuracy'] = True
else:
    solvers = [# s.ADMM, 
               s.OSQP, s.OSQP_polish, 
               s.SuperADMM, 
               s.Super_ruiz,
               s.Super_ruiz_ldlt,
               s.Super_ruiz_new_fact,
               s.Super_ruiz_cg,
               s.Super_ruiz_cg_precond,
            #    s.Super_ldlt,
            #    s.Super_new_fact,
            #    s.Super_cg,
            #    s.Super_cg_precond,
               ]
    OUTPUT_FOLDER = 'ablation_study_high_dim'

if verbose:
    for key in s.settings:
        s.settings[key]['verbose'] = True

# Number of instances and dimensions
if small_test:
    n_instances = 3
    n_dim = 5
else:
    n_instances = 10
    n_dim = 20

# Problem dimensions
# problem_dimensions = {
#     'Random QP': gen_int_log_space(10, 200, n_dim) if not small_test else [10, 20, 30]
# }

# problem_parallel = {
#     'Random QP': parallel
# }

# Problem dimensions
problems = [
            'Random QP',
            # 'Eq QP',
            # 'Portfolio',
            #'Lasso',
            #'SVM',
            #'Huber',
            # 'Control'
            ]

problem_dimensions = {# 'Random QP': gen_int_log_space(300, 100, 1),
                      'Random QP': gen_int_log_space(10, 50, n_dim),
                    #   'Random QP': gen_int_log_space(200, 50, 2),
                      'Eq QP': gen_int_log_space(10, 200, n_dim),
                      'Portfolio': gen_int_log_space(5, 10, 5),
                      'Lasso': gen_int_log_space(10, 50, n_dim),
                      'SVM': gen_int_log_space(10, 50, n_dim),
                      'Huber': gen_int_log_space(10, 50, n_dim),
                      'Control': gen_int_log_space(10, 50, n_dim)}

problem_parallel = {'Random QP': parallel,
                    'Eq QP': parallel,
                    'Portfolio': parallel,
                    'Lasso': parallel,
                    'SVM': parallel,
                    'Huber': parallel,
                    'Control': parallel}

# Run Random QP test
# print("\n" + "="*80)
# print("Testing ADMM Solver on Random QP Problems")
# print("="*80 + "\n")

# example = Example(
#     'Random QP',
#     problem_dimensions['Random QP'],
#     solvers,
#     s.settings,
#     OUTPUT_FOLDER,
#     n_instances
# )
# example.solve(parallel=problem_parallel['Random QP'])

# Run all examples
for problem in problems:
    example = Example(problem,
                      problem_dimensions[problem],
                      solvers,
                      s.settings,
                      OUTPUT_FOLDER,
                      n_instances)
    example.solve(parallel=problem_parallel[problem])

# Compute results statistics
print("\n" + "="*80)
print("Computing Statistics")
print("="*80 + "\n")

# compute_stats_info(
#     solvers,
#     OUTPUT_FOLDER,
#     problems=['Random QP'],
#     high_accuracy=high_accuracy
# )

compute_stats_info(solvers, OUTPUT_FOLDER,
                   problems=problems,
                   high_accuracy=high_accuracy)

print("\n" + "="*80)
print("✓ ADMM Test Completed!")
print("="*80)
