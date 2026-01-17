import numpy as np
import scipy.sparse as spa
import time
from . import statuses as s
from .results import Results
from utils.general import is_qp_solution_optimal


class SuperADMMSolver(object):
    """SuperADMM solver for QP problems.
    
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
        """Solve QP problem using SuperADMM.
        
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
            # Solve using ADMM
            x_sol, y_sol, u_sol, niter, b, R_bounded_ratio, R_history, b_history = self._superadmm_solve(
                P_dense, q, A_dense, l, u,
                rho=settings.get('rho', 1.0),
                alpha=settings.get('alpha', 500.0),
                tau=settings.get('tau', 0.5),
                b0=settings.get('b0', 1e8),
                sigma=settings.get('sigma', 1e-6),
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
            return_results.b = b
            return_results.R_bounded_ratio = R_bounded_ratio
            return_results.R_history = R_history
            return_results.b_history = b_history
            
            return return_results
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            return_results = Results(
                s.SOLVER_ERROR,
                np.inf,
                np.zeros(problem['n']),
                np.zeros(problem['m']),
                elapsed_time,
                0
            )
            return_results.b = None
            return_results.R_bounded_ratio = None
            return return_results

    def _superadmm_solve(self, P, q, A, l, u, rho=0.1, alpha=500.0, tau=0.5, b0=1e8, sigma=1e-6, 
                         max_iter=1000, abstol=1e-4, reltol=1e-3, verbose=False, time_limit=1000.0):
        """SuperADMM solver for the box-constrained QP.
        
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
        u_dual = np.zeros(m)  # dual variable
        R = rho * np.eye(m) # step size matrix
        R_inv = 1 / rho * np.eye(m)
        b = b0

        # Store history of R matrices
        R_history = [R.copy()]
        b_history = [b]
        
        start_time = time.time()
        
        for k in range(1, max_iter + 1):
            # Check time limit
            if time.time() - start_time > time_limit:
                if verbose:
                    print(f"Time limit reached at iteration {k}")
                break
            
            # x-update
            lhs_matrix = P + A.T @ R @ A + sigma * np.eye(n)
            try:
                rhs = -q + A.T @ (R @ y - u_dual) + sigma * x
                x = np.linalg.solve(lhs_matrix, rhs)
            except np.linalg.LinAlgError:
                # Use least squares if matrix is singular
                x = np.linalg.lstsq(lhs_matrix, rhs, rcond=None)[0]
            
            # y-update
            y_unconstrained = R_inv @ u_dual + A @ x
            y_old = y.copy()
            y = np.clip(y_unconstrained, l, u)
            
            # dual update: u_dual = u_dual + (A x - y)
            residual = A @ x - y
            u_dual = u_dual + R @ residual
            
            # Compute residuals for stopping criteria
            r_norm = np.linalg.norm(residual, ord=np.inf)
            s = A.T @ R @ (y - y_old)
            s_norm = np.linalg.norm(s, ord=np.inf)

            # numerical error & b update
            epsilon = np.linalg.norm(rhs - lhs_matrix @ x, ord=np.inf)
            b = tau * b if epsilon >= r_norm else b

            # R matrix update
            update_mask = np.logical_or(np.abs(y - l) < 1e-8, np.abs(y - u) < 1e-8)
            for diag_idx in range(m):
                if update_mask[diag_idx]:
                    R[diag_idx, diag_idx] = min(R[diag_idx, diag_idx] * alpha, b)
                    # R[diag_idx, diag_idx] = b
                else:
                    R[diag_idx, diag_idx] = max(R[diag_idx, diag_idx] / alpha, 1 / b)
                    # R[diag_idx, diag_idx] = 1 / b
            R_inv = 1.0 / np.diag(R) * np.eye(m)

            R_history.append(R.copy())
            b_history.append(b)
            
            if verbose and k % 100 == 0:
                print(f"ADMM iter {k:3d}: r_norm={r_norm:.4e}, s_norm={s_norm:.4e}")
            
            # Stopping criteria
            eps_pri = np.sqrt(m) * abstol + reltol * np.maximum(np.linalg.norm(A @ x), np.linalg.norm(y))
            eps_dual = np.sqrt(n) * abstol + reltol * np.linalg.norm(A.T @ u_dual)
            
            if r_norm <= eps_pri and s_norm <= eps_dual:
                if verbose:
                    print(f"SuperADMM converged in {k} iterations")
                break
        
        R_bounded_ratio = round(np.logical_or(np.diag(R) >= b, np.diag(R) <= 1 / b).sum() / m, 4)
        return x, y, u_dual, k, b, R_bounded_ratio, R_history, b_history
