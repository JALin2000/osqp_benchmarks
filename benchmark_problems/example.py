import os
from multiprocessing import Pool, cpu_count
from itertools import repeat
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from solvers.solvers import SOLVER_MAP
from problem_classes.random_qp import RandomQPExample
from problem_classes.eq_qp import EqQPExample
from problem_classes.portfolio import PortfolioExample
from problem_classes.lasso import LassoExample
from problem_classes.svm import SVMExample
from problem_classes.huber import HuberExample
from problem_classes.control import ControlExample
from utils.general import make_sure_path_exists
from utils.plot_alpha import plot_alpha_history, plot_alpha_change

examples = [RandomQPExample,
            EqQPExample,
            PortfolioExample,
            LassoExample,
            SVMExample,
            HuberExample,
            ControlExample]

EXAMPLES_MAP = {example.name(): example for example in examples}


def plot_R_and_b_history(R_history, b_history, output_path):
    """Plot R diagonal elements with b and 1/b bounds over iterations.
    
    Args:
        R_history: list of R matrices (diagonal elements are plotted)
        b_history: list of b scalar values
        output_path: path to save the plot
    """
    if R_history is None or b_history is None:
        return
    
    try:
        iterations = np.arange(len(b_history))
        
        # Extract diagonal elements from R matrices
        R_diag_list = [np.diag(R_mat) for R_mat in R_history]
        R_diag_array = np.array(R_diag_list)  # shape: (iterations, m)
        num_constraints = R_diag_array.shape[1]
        
        # Compute b bounds
        b_upper = np.array(b_history)  # b is the upper bound
        b_lower = 1.0 / b_upper         # 1/b is the lower bound
        
        # Create figure with single y-axis
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Plot each R diagonal element
        colors = plt.cm.tab20(np.linspace(0, 1, min(num_constraints, 20)))
        if num_constraints > 20:
            colors = plt.cm.gist_rainbow(np.linspace(0, 1, num_constraints))
        
        for i in range(num_constraints):
            R_i = R_diag_array[:, i]
            ax.semilogy(iterations, R_i, color=colors[i], linewidth=1.0, 
                       label=f'R[{i},{i}]', alpha=0.7)
        
        # Plot bounds: b (upper bound) and 1/b (lower bound)
        ax.semilogy(iterations, b_upper, color='red', linewidth=3, 
                   label='b (upper bound)', linestyle='-', zorder=10)
        ax.semilogy(iterations, b_lower, color='blue', linewidth=3, 
                   label='1/b (lower bound)', linestyle='--', zorder=10)
        
        # Labels and formatting
        ax.set_xlabel('Iteration', fontsize=13, fontweight='bold')
        ax.set_ylabel('Value (log scale)', fontsize=13, fontweight='bold')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, which='both', linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.15, which='minor', linestyle=':', linewidth=0.3)
        
        # Title
        plt.title(f'SuperADMM: R Matrix Diagonal Elements (m={num_constraints}) with Bounds [1/b, b]', 
                 fontsize=14, fontweight='bold')
        
        # Legend handling based on number of constraints
        if num_constraints <= 10:
            ax.legend(loc='best', fontsize=10, framealpha=0.95)
        else:
            # Show legend for b bounds only, add annotation for R elements
            handles, labels = ax.get_legend_handles_labels()
            # Keep only the last two (b and 1/b)
            ax.legend(handles[-2:], labels[-2:], loc='best', fontsize=11, framealpha=0.95)
            # Add annotation about R elements
            ax.text(0.02, 0.02, f'Showing all {num_constraints} R diagonal elements', 
                   transform=ax.transAxes, fontsize=11, verticalalignment='bottom',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8, edgecolor='gray'))
        
        fig.tight_layout()
        
        # Save figure
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Error plotting R and b history: {e}")


class Example(object):
    '''
    Examples runner
    '''
    def __init__(self, name,
                 dims,
                 solvers,
                 settings,
                 output_folder,
                 n_instances=10):
        self.name = name
        self.dims = dims
        self.n_instances = n_instances
        self.solvers = solvers
        self.settings = settings
        self.output_folder = output_folder

    def solve(self, parallel=True):
        '''
        Solve problems of type example

        The results are stored as

            ./results/{self.output_folder}/{solver}/{class}/n{dimension}.csv

        using a pandas table with fields
            - 'class': example class
            - 'solver': solver name
            - 'status': solver status
            - 'run_time': execution time
            - 'iter': number of iterations
            - 'obj_val': objective value
            - 'n': leading dimension
            - 'N': nnz dimension (nnz(P) + nnz(A))
        '''

        print("Solving %s" % self.name)
        print("-----------------")

        if parallel:
            pool = Pool(processes=min(self.n_instances, cpu_count()))

        # Iterate over all solvers
        for solver in self.solvers:
            settings = self.settings[solver]

            # Initialize solver results
            results_solver = []

            # Solution directory
            path = os.path.join('.', 'results', self.output_folder,
                                solver,
                                self.name
                                )

            # Create directory for the results
            make_sure_path_exists(path)

            # Get solver file name
            solver_file_name = os.path.join(path, 'full.csv')

            for n in self.dims:

                # Check if solution already exists
                n_file_name = os.path.join(path, 'n%i.csv' % n)

                if not os.path.isfile(n_file_name):

                    if parallel and solver not in ['ECOS', 'ECOS_high', 'qpOASES']:
                        # NB. ECOS and qpOASES crahs if the problem sizes are too large
                        instances_list = list(range(self.n_instances))
                        n_results = pool.starmap(self.solve_single_example,
                                                 zip(repeat(n),
                                                     instances_list,
                                                     repeat(solver),
                                                     repeat(settings)))
                    else:
                        n_results = []
                        for instance in range(self.n_instances):
                            n_results.append(
                                self.solve_single_example(n,
                                                          instance,
                                                          solver,
                                                          settings)
                                )

                    # Combine n_results
                    df = pd.concat(n_results)

                    # Store n_results
                    df.to_csv(n_file_name, index=False)

                else:
                    # Load from file
                    df = pd.read_csv(n_file_name)

                # Combine list of dataframes
                results_solver.append(df)

            # Create total dataframe for the solver from list
            df_solver = pd.concat(results_solver)

            # Store dataframe
            df_solver.to_csv(solver_file_name, index=False)

        if parallel:
            pool.close()  # Not accepting any more jobs on this pool
            pool.join()   # Wait for all processes to finish

    def solve_single_example(self,
                             dimension, instance_number,
                             solver, settings):
        '''
        Solve 'example' with 'solver'

        Args:
            dimension: problem leading dimension
            instance_number: number of the instance
            solver: solver name
            settings: settings dictionary for the solver

        '''

        # Create example instance
        example_instance = EXAMPLES_MAP[self.name](dimension,
                                                   instance_number)

        print(" - Solving %s with n = %i, instance = %i with solver %s" %
              (self.name, dimension, instance_number, solver))

        # Solve problem
        if solver[:11] == 'OSQP_python':
            s = SOLVER_MAP[solver]()
            s.setup(P=example_instance.qp_problem['P'],
                    q=example_instance.qp_problem['q'],
                    A=example_instance.qp_problem['A'],
                    l=example_instance.qp_problem['l'],
                    u=example_instance.qp_problem['u'],
                    **settings)
            results = s.solve()
        else:
            s = SOLVER_MAP[solver](settings)
            results = s.solve(example_instance)

        # Create solution as pandas table
        P = example_instance.qp_problem['P']
        A = example_instance.qp_problem['A']
        N = P.nnz + A.nnz
        solution_dict = {'class': [self.name],
                         'solver': [solver],
                         'status': [results.status],
                         'run_time': [results.run_time],
                         'iter': [results.niter],
                         'obj_val': [results.obj_val],
                         'n': [dimension],
                         'N': [N]}

        # Add status polish if OSQP
        if solver[:4] == 'OSQP':
            solution_dict['setup_time'] = results.setup_time
            solution_dict['solve_time'] = results.solve_time
            solution_dict['update_time'] = results.update_time
            solution_dict['rho_updates'] = results.rho_updates
            if 'python' not in solver:
                solution_dict['status_polish'] = results.status_polish
        
        if solver[:9] == 'SuperADMM':
            solution_dict['b'] = results.b
            solution_dict['R_bounded_ratio'] = results.R_bounded_ratio

            # Plot R and b history
            if hasattr(results, 'R_history') and hasattr(results, 'b_history') and (dimension == 10 or dimension == 55 or dimension == 209):
                plot_dir = os.path.join('.', 'results', self.output_folder, solver, self.name, 'plots')
                make_sure_path_exists(plot_dir)
                plot_filename = os.path.join(plot_dir, f'R_b_history_n{dimension}_inst{instance_number}.png')
                plot_R_and_b_history(results.R_history, results.b_history, plot_filename)

        # Save per-instance alpha history plots for neural solvers
        if hasattr(s, 'get_alpha_history'):
            hist = s.get_alpha_history()
            if hist and hist.get('iter'):
                plot_dir = os.path.join('.', 'results', self.output_folder,
                                        solver, self.name, 'alpha_plots')
                make_sure_path_exists(plot_dir)
                title = f'{self.name} n={dimension} inst={instance_number} | {solver}'

                # Dual-axis plot (scalar only — needs 'alpha' key)
                if 'alpha' in hist:
                    fname = os.path.join(plot_dir,
                                         f'alpha_n{dimension}_inst{instance_number}.png')
                    plot_alpha_history(hist, title, fname)

                # Alpha change magnitude plot (both scalar and vector)
                if 'alpha_change' in hist:
                    fname = os.path.join(plot_dir,
                                         f'alpha_change_n{dimension}_inst{instance_number}.pdf')
                    plot_alpha_change(hist, title, fname)

        # Return solution
        return pd.DataFrame(solution_dict)