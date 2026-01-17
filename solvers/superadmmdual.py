import numpy as np
import scipy.sparse as spa
import time
from . import statuses as s
from .results import Results
from utils.general import is_qp_solution_optimal


class SuperADMMDualSolver(object):
    """SuperADMMDual solver for QP problems.
    
    Solves the QP problem:
        minimize   0.5 x^T P x + q^T x
        subject to l <= A x <= u
    
    Using ADMM formulation:
        minimize   f(x) + g(y)
        subject to A x + B y = b
    
    Where:
        f(x) = 0.5 x^T P x + q^T x
        g(y) = indicator of box constraints
        B y encodes the box constraints: y is decomposed as [y_l, y_u]
        and we have y_l >= 0, y_u >= 0, y_l - y_u = A x - c (implicit)
    """

    STATUS_MAP = {}  # Will be populated if needed

    def __init__(self, settings={}):
        """Initialize solver object with settings."""
        self._settings = settings

    @property
    def settings(self):
        """Solver settings"""
        return self._settings

    def solve(self, example):
        """Solve QP problem using ADMM.
        
        Args:
            example: problem structure with QP matrices
            
        Returns:
            Results structure
        """
        problem = example.qp_problem
        settings = self._settings.copy()
        high_accuracy = settings.pop('high_accuracy', False)
        verbose = settings.pop('verbose', False)
        time_limit = settings.pop('time_limit', 1000.0)
        
        # Extract QP problem data
        P = problem['P']
        q = problem['q']
        A = problem['A']
        u = problem['u']
        l = problem['l']
        
        # Convert sparse to dense if needed for solver
        if spa.issparse(P):
            P_dense = P.toarray()
        else:
            P_dense = P
            
        if spa.issparse(A):
            A_dense = A.toarray()
        else:
            A_dense = A
        
        start_time = time.time()
        
        try:
            # Solve using SuperADMMDual
            x_sol, y_sol, u_sol, niter = self._superadmmdual_solve(
                P_dense, q, A_dense, l, u,
                rho=settings.get('rho', 1.0),
                alpha=settings.get('alpha', 0.99),
                max_iter=settings.get('max_iter', 1000),
                abstol=settings.get('abstol', 1e-4),
                reltol=settings.get('reltol', 1e-3),
                verbose=verbose,
                time_limit=time_limit
            )
            
            elapsed_time = time.time() - start_time
            
            # Compute objective value
            obj_val = 0.5 * (x_sol @ (P_dense @ x_sol)) + q @ x_sol
            
            # Determine status
            # Check feasibility
            residual = A_dense @ x_sol
            constraint_violation = np.maximum(0, l - residual).max() + np.maximum(0, residual - u).max()
            
            if constraint_violation < 1e-3:
                status = s.OPTIMAL
            else:
                status = s.SOLVER_ERROR
            
            if elapsed_time > time_limit:
                status = s.TIME_LIMIT
            
            # Create return results
            return_results = Results(
                status,
                obj_val,
                x_sol,
                y_sol,
                elapsed_time,
                niter
            )
            
            return return_results
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            return Results(
                s.SOLVER_ERROR,
                np.inf,
                np.zeros(problem['n']),
                np.zeros(problem['m']),
                elapsed_time,
                0
            )

    def _superadmmdual_solve(self, P, q, A, l, u, rho=1.0, alpha=0.99, max_iter=1000, 
                    abstol=1e-4, reltol=1e-3, verbose=False, time_limit=1000.0):
        """SuperADMMDual solver for the box-constrained QP.
        
        Solves: minimize 0.5 x^T P x + q^T x
                subject to l <= A x <= u
        
        Using the formulation:
            minimize f(x) + g(y)
            subject to A x - y = 0, and y in [l, u]
        
        Returns:
            x_sol: primal solution
            y_sol: dual variables (constraint slacks)
            u_sol: scaled dual variables
            niter: number of iterations
        """
        m, n = A.shape
        
        # Initialize variables
        x = np.zeros(n)
        y = np.clip(np.zeros_like(l), l, u)
        u_dual = np.zeros(m)  # scaled dual variable
        
        # Precompute matrices
        P_plus_rho = P + rho * (A.T @ A)
        
        start_time = time.time()
        
        for k in range(1, max_iter + 1):
            if k == 5000:
                a = 1
            # Check time limit
            if time.time() - start_time > time_limit:
                if verbose:
                    print(f"Time limit reached at iteration {k}")
                break
            
            # x-update: solve (P + rho A^T A) x = -q + rho A^T (y - u_dual)
            try:
                rhs = -q + rho * (A.T @ (y - u_dual))
                x = np.linalg.solve(P_plus_rho, rhs)
            except np.linalg.LinAlgError:
                # Use least squares if matrix is singular
                x = np.linalg.lstsq(P_plus_rho, rhs, rcond=None)[0]
            
            # y-update: project A x + u_dual onto [l, u]
            y_unconstrained = A @ x + u_dual
            y_old = y.copy()
            y = np.clip(y_unconstrained, l, u)
            
            # dual update
            residual = A @ x - y
            decrease_mask = (residual < -1e-1)
            # increase_mask = (residual > 1e-2)
            u_dual[decrease_mask] = alpha * u_dual[decrease_mask] + residual[decrease_mask]
            u_dual[~decrease_mask] = u_dual[~decrease_mask] + residual[~decrease_mask]
            # u_dual[increase_mask] =  1 / alpha * u_dual[increase_mask] + residual[increase_mask]
            # u_dual[~(decrease_mask | increase_mask)] = u_dual[~(decrease_mask | increase_mask)] + residual[~(decrease_mask | increase_mask)]
            # u_dual = u_dual + residual
            
            # Compute residuals for stopping criteria
            r_norm = np.linalg.norm(residual, ord=np.inf)
            s = rho * (A.T @ (y - y_old))
            s_norm = np.linalg.norm(s, ord=np.inf)
            
            if verbose and k % 100 == 0:
                print(f"ADMM iter {k:3d}: r_norm={r_norm:.4e}, s_norm={s_norm:.4e}")
            
            # Stopping criteria
            eps_pri = np.sqrt(m) * abstol + reltol * np.maximum(np.linalg.norm(A @ x), np.linalg.norm(y))
            eps_dual = np.sqrt(n) * abstol + reltol * np.linalg.norm(A.T @ u_dual)
            
            if r_norm <= eps_pri and s_norm <= eps_dual:
                if verbose:
                    print(f"ADMM converged in {k} iterations")
                break
        
        return x, y, u_dual, k
