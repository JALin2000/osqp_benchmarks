import numpy as np
import scipy.sparse as spa
from scipy.sparse.linalg import gmres, cg, LinearOperator
from scipy.linalg import ldl, solve_triangular
import time
from . import statuses as s
from .results import Results
from utils.general import is_qp_solution_optimal


class SuperADMMTricksSolver(object):
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
                time_limit=time_limit,
                use_preconditioning=settings.get('use_preconditioning', True),
                solving_method=settings.get('solving_method', 'direct'),
                max_iter_kacz=settings.get('max_iter_kacz', 5)
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

    def _kkt_col_norms(self, P, A):
        """Compute infinity norms of columns in the KKT matrix [P, A^T; A, 0].
        
        Follows COSMO.jl implementation:
        - For P: compute column norms of symmetric matrix (assumes triu or tril supplied)
        - For A: incrementally add column norms from A, then compute row norms separately
        
        Returns:
            d_norms: infinity norms of columns of P (combined with A columns)
            e_norms: infinity norms of rows of A (= column norms of A^T)
        """
        m, n = A.shape
        
        # Initialize norm vectors
        d_norms = np.zeros(n)
        e_norms = np.zeros(m)
        
        # Handle P: symmetric matrix column norms
        if spa.issparse(P):
            # For sparse symmetric matrix, traverse carefully
            P_coo = P.tocoo()
            for i, j, v in zip(P_coo.row, P_coo.col, P_coo.data):
                d_norms[j] = max(d_norms[j], abs(v))
                # If matrix is lower triangular, also update from the transpose
                if i != j:
                    d_norms[i] = max(d_norms[i], abs(v))
        else:
            # Dense matrix: compute column norms directly
            d_norms = np.max(np.abs(P), axis=0)
        
        # Add A contributions to d_norms (incrementally from P)
        if spa.issparse(A):
            A_coo = A.tocoo()
            for i, j, v in zip(A_coo.row, A_coo.col, A_coo.data):
                d_norms[j] = max(d_norms[j], abs(v))
        else:
            # Dense matrix: column norms of A
            A_col_norms = np.max(np.abs(A), axis=0)
            d_norms = np.maximum(d_norms, A_col_norms)
        
        # Row norms of A (= column norms of A^T)
        if spa.issparse(A):
            A_coo = A.tocoo()
            for i, j, v in zip(A_coo.row, A_coo.col, A_coo.data):
                e_norms[i] = max(e_norms[i], abs(v))
        else:
            # Dense matrix: row norms
            e_norms = np.max(np.abs(A), axis=1)
        
        return d_norms, e_norms

    def _scale_data(self, P, q, A, l, u, d_scale, e_scale):
        """Apply scaling to problem data.
        
        Args:
            P, q, A, l, u: problem matrices
            d_scale: scaling for x variables (diagonal of D)
            e_scale: scaling for y variables (diagonal of E)
            
        Returns:
            P_scaled, q_scaled, A_scaled, l_scaled, u_scaled: scaled matrices
        """
        # P_scaled = D @ P @ D where D = diag(d_scale)
        P_scaled = (d_scale[:, np.newaxis] * P) * d_scale[np.newaxis, :]
        
        # q_scaled = D @ q
        q_scaled = d_scale * q
        
        # A_scaled = E @ A @ D where E = diag(e_scale)
        A_scaled = (e_scale[:, np.newaxis] * A) * d_scale[np.newaxis, :]
        
        # l_scaled = E @ l, u_scaled = E @ u
        l_scaled = e_scale * l
        u_scaled = e_scale * u
        
        return P_scaled, q_scaled, A_scaled, l_scaled, u_scaled

    def _ruiz_equilibration(self, P, q, A, l, u, 
                           max_iter_equil=10, scale_min=1e-4, scale_max=1e4, verbose=False):
        """Apply Ruiz equilibration for preconditioning (following COSMO.jl implementation).
        
        This method implements the Ruiz equilibration procedure:
        1. Compute column norms of KKT matrix [P, A^T; A, 0]
        2. Scale to make columns have similar norms
        3. Apply cost scaling to normalize objective
        4. Repeat until convergence or max iterations
        
        Args:
            P, q, A, l, u: QP problem matrices
            max_iter_equil: maximum iterations for equilibration
            scale_min: minimum allowed scaling factor
            scale_max: maximum allowed scaling factor
            verbose: print equilibration progress
            
        Returns:
            P_scaled, q_scaled, A_scaled, l_scaled, u_scaled: equilibrated matrices
            d, e, c: scaling factors for variable/constraint/cost (for solution recovery)
        """
        m, n = A.shape
        
        # Initialize scaling vectors
        d = np.ones(n)  # scaling for x variables
        e = np.ones(m)  # scaling for y variables
        c = 1.0         # cost scaling factor
        
        # Work copies
        P_work = P.copy()
        q_work = q.copy()
        A_work = A.copy()
        l_work = l.copy()
        u_work = u.copy()
        
        if verbose:
            print("Starting Ruiz equilibration...")
        
        # Ruiz scaling loop
        for iter_equil in range(max_iter_equil):
            # Compute column norms of KKT matrix
            d_norms, e_norms = self._kkt_col_norms(P_work, A_work)
            
            # Create scaling vectors (inverse square root of norms)
            d_work = np.ones(n)
            e_work = np.ones(m)
            
            # Avoid division by zero
            d_work = np.where(d_norms > 0, 1.0 / np.sqrt(d_norms), 1.0)
            e_work = np.where(e_norms > 0, 1.0 / np.sqrt(e_norms), 1.0)
            
            # Bound the cumulative scaling
            d_work = np.clip(d_work, scale_min / d, scale_max / d)
            e_work = np.clip(e_work, scale_min / e, scale_max / e)
            
            # Scale the problem data
            P_work, q_work, A_work, l_work, u_work = self._scale_data(
                P_work, q_work, A_work, l_work, u_work, d_work, e_work)
            
            # Update scaling matrices
            d *= d_work
            e *= e_work
            
            # Cost scaling step
            # Compute mean column norm of P and infinity norm of q
            P_col_norms = np.linalg.norm(P_work, axis=0, ord=np.inf)
            mean_col_norm_P = np.mean(P_col_norms)
            inf_norm_q = np.linalg.norm(q_work, ord=np.inf)
            
            if mean_col_norm_P > 0 and inf_norm_q > 0:
                scale_cost = max(inf_norm_q, mean_col_norm_P)
                c_tmp = 1.0 / scale_cost
                
                # Bound the cost scaling
                c_tmp = np.clip(c_tmp, scale_min / c, scale_max / c)
                
                # Apply cost scaling
                P_work *= c_tmp
                q_work *= c_tmp
                c *= c_tmp
            
            # Check convergence: if scaling factors are close to 1
            delta = np.linalg.norm(1.0 - d_work, ord=np.inf)
            if verbose and iter_equil % 1 == 0:
                print(f"  Iter {iter_equil}: delta = {delta:.2e}")
            
            if delta < 1e-4:
                if verbose:
                    print(f"  Equilibration converged in {iter_equil} iterations")
                break
        
        if verbose:
            print(f"Final scaling: c = {c:.2e}")
        
        return P_work, q_work, A_work, l_work, u_work, d, e, c

    def _kaczmarz_solve(
        self,
        A,
        b,
        x_init,
        max_sweeps=5,
        tol=1e-6,
        omega=1.0,
        check_every_sweeps=2,
        randomized=True,
        seed=None,
        eps_row_norm=1e-14,
    ):
        """
        Efficient warm-started Kaczmarz solver for A x = b.

        Notes on efficiency choices:
        - Uses randomized row-norm-weighted Kaczmarz by default, which is typically superior
        under heavy row scaling (e.g., factors 500 and 1/500).
        - Checks convergence only every `check_every_sweeps` sweeps to avoid doubling cost.
        - Includes a CSR sparse fast path.

        Args:
            A: (m, n) array-like or scipy.sparse.csr_matrix
            b: (m,) array-like
            x_init: (n,) warm start
            max_sweeps: number of passes worth of row-updates (sweeps)
            tol: stopping tolerance on ||A x - b||_inf (checked periodically)
            omega: relaxation parameter in (0, 2); 1.0 is standard
            check_every_sweeps: compute full residual every this many sweeps
            randomized: if True use row-norm^2 weighted sampling; else cyclic
            seed: RNG seed for reproducibility
            eps_row_norm: threshold to skip near-zero rows

        Returns:
            x: (n,) solution estimate
        """
        # Ensure contiguous float arrays for x and b
        b = np.asarray(b, dtype=np.float64)
        x = np.array(x_init, dtype=np.float64, copy=True, order="C")

        rng = np.random.default_rng(seed)

        # ---- Sparse CSR fast path ----
        # if sp is not None and sp.isspmatrix_csr(A):
        #     A_csr = A
        #     m, n = A_csr.shape
        #     if b.shape[0] != m:
        #         raise ValueError(f"b has length {b.shape[0]} but A has {m} rows")

        #     indptr = A_csr.indptr
        #     indices = A_csr.indices
        #     data = A_csr.data

        #     # Row norm^2 for CSR: sum of squares per row
        #     data_sq = data * data
        #     row_norms_sq = np.add.reduceat(data_sq, indptr[:-1])
        #     valid = row_norms_sq > eps_row_norm
        #     valid_rows = np.flatnonzero(valid)
        #     if valid_rows.size == 0:
        #         return x  # Degenerate: all-zero rows

        #     inv_row_norms_sq = np.zeros(m, dtype=np.float64)
        #     inv_row_norms_sq[valid] = 1.0 / row_norms_sq[valid]

        #     # Sampling distribution proportional to row norm^2 (restricted to valid rows)
        #     if randomized:
        #         p = row_norms_sq[valid_rows]
        #         p = p / p.sum()

        #     total_updates = int(max_sweeps) * int(m)

        #     # Convergence checking schedule
        #     check_every_updates = max(1, int(check_every_sweeps) * int(m))

        #     for t in range(total_updates):
        #         if randomized:
        #             i = valid_rows[rng.choice(valid_rows.size, p=p)]
        #         else:
        #             i = (t % m)
        #             if not valid[i]:
        #                 continue

        #         start, end = indptr[i], indptr[i + 1]
        #         cols = indices[start:end]
        #         vals = data[start:end]

        #         # residual_i = b[i] - A[i,:] @ x
        #         r = b[i] - np.dot(vals, x[cols])

        #         # x[cols] += omega * (r / ||a_i||^2) * a_i
        #         scale = omega * (r * inv_row_norms_sq[i])
        #         if scale != 0.0:
        #             x[cols] += scale * vals

        #         # Periodic full residual check
        #         if (t + 1) % check_every_updates == 0:
        #             # For CSR, A @ x is efficient
        #             res_inf = np.max(np.abs(A_csr @ x - b))
        #             if res_inf < tol:
        #                 break

        #     return x

        # ---- Dense path ----
        A = np.asarray(A, dtype=np.float64)
        m, n = A.shape
        if b.shape[0] != m:
            raise ValueError(f"b has length {b.shape[0]} but A has {m} rows")

        # Row norm^2 efficiently
        row_norms_sq = np.einsum("ij,ij->i", A, A)
        valid = row_norms_sq > eps_row_norm
        valid_rows = np.flatnonzero(valid)
        if valid_rows.size == 0:
            return x

        inv_row_norms_sq = np.zeros(m, dtype=np.float64)
        inv_row_norms_sq[valid] = 1.0 / row_norms_sq[valid]

        if randomized:
            p = row_norms_sq[valid_rows]
            p = p / p.sum()

        total_updates = int(max_sweeps) * int(m)
        check_every_updates = max(1, int(check_every_sweeps) * int(m))

        try:
            for t in range(total_updates):
                if randomized:
                    i = valid_rows[rng.choice(valid_rows.size, p=p)]
                else:
                    i = (t % m)
                    if not valid[i]:
                        continue

                ai = A[i]  # view (no copy)
                r = b[i] - ai @ x
                scale = omega * (r * inv_row_norms_sq[i])
                if scale != 0.0:
                    x += scale * ai

                if (t + 1) % check_every_updates == 0:
                    res_inf = np.max(np.abs(A @ x - b))
                    if res_inf < tol:
                        break
        
        except:
            a=1

        return x
    
    def _solve_indirect_cg(self, P, A, R_diag, sigma, x_k, z_k, y_k, q, current_x_guess=None, r_norm=None, precond=False):
        """
        Solves the SuperADMM step using Conjugate Gradient (Matrix-Free).
        
        Args:
            P: (n, n) Cost matrix (sparse or dense)
            A: (m, n) Constraint matrix (sparse)
            R_diag: (m,) Diagonal of the weight matrix R^k
            sigma: Scalar smoothing parameter
            x_k, z_k, y_k: ADMM variables from previous step
            q: Linear cost vector
            current_x_guess: (Optional) Warm start from previous iteration
        
        Returns:
            x_new: The updated primal variable
        """
        n = P.shape[0]
        
        # --- 1. Construct the Right Hand Side (RHS) ---
        # Formula: RHS = sigma*x^k - q + A.T @ (R^k * z^k - y^k)
        # Note: R^k is diagonal, so we element-wise multiply
        
        term_1 = sigma * x_k - q
        
        # efficient diagonal multiplication: R z - y
        # (R_diag * z_k) is element-wise
        temp_vec = (R_diag * z_k) - y_k 
        
        # A.T @ temp_vec
        term_2 = A.T @ temp_vec
        
        rhs = term_1 + term_2

        # --- 2. Define the Linear Operator (The Matrix-Vector Product) ---
        # We want to perform: v -> (P + sigma*I + A.T R A) v
        def matvec(v):
            P_sigma_v = self.P_sigma @ v  # P + sigma*I part
            
            # A.T R A v
            # Compute inside-out to keep vectors small:
            # Av -> R(Av) -> A.T(R(Av))
            Av = A @ v
            RAv = R_diag * Av  # Element-wise multiply for diagonal R
            ATRAv = A.T @ RAv
            
            return P_sigma_v + ATRAv

        # Wrap as a LinearOperator for Scipy
        A_op = LinearOperator((n, n), matvec=matvec, dtype=np.float64)

        # --- 3. Solve using Conjugate Gradient ---
        # warm_start: Use the previous x as the initial guess
        x0 = current_x_guess if current_x_guess is not None else x_k
        
        # tol: You can loosen this. ADMM is robust to inexact inner solves.
        M = np.diag(1.0 / (np.diag(P) + sigma * np.ones(n) + (A**2).T @ R_diag)) if precond else None
        if r_norm is None:
            x_new, info = cg(A_op, rhs, x0=x0, rtol=1e-7, maxiter=100, M=M)
        else:
            x_new, info = cg(A_op, rhs, x0=x0, atol=0.1*r_norm, rtol=0.0, maxiter=100, M=M)
        
        # if info > 0:
        #     print(f"Warning: CG did not converge within {info} iterations")
        
        return x_new

    def _superadmm_solve(self, P, q, A, l, u, rho=1.0, alpha=500.0, tau=0.5, b0=1e8, sigma=1e-6, 
                         max_iter=1000, abstol=1e-4, reltol=1e-3, verbose=False, time_limit=1000.0,
                         use_preconditioning=True, solving_method='direct', max_iter_kacz=5):
        """SuperADMM solver for the box-constrained QP with preconditioning.
        
        Solves: minimize 0.5 x^T P x + q^T x
                subject to l <= A x <= u
        
        Using the formulation:
            minimize f(x) + g(y)
            subject to A x - y = 0, and y in [l, u]
        
        Args:
            solving_method: Method to solve the x-update ('direct', 'kaczmarz', 'LDLT', 'new_fact', 'CG', 'CG_precond')
            max_iter_kacz: Maximum iterations for Kaczmarz algorithm
        
        Returns:
            x_sol: primal solution (in original coordinates)
            y_sol: dual variables (constraint slacks)
            u_sol: scaled dual variables
            niter: number of iterations
        """
        m, n = A.shape
        start_time = time.time()
        
        # Apply preconditioning if requested
        if use_preconditioning:
            P_eq, q_eq, A_eq, l_eq, u_eq, d, e, c = self._ruiz_equilibration(
                P, q, A, l, u, verbose=verbose)
            
            if verbose:
                print(f"Preconditioning applied: d range = [{np.min(d):.2e}, {np.max(d):.2e}], "
                      f"e range = [{np.min(e):.2e}, {np.max(e):.2e}]")
        else:
            P_eq, q_eq, A_eq, l_eq, u_eq = P, q, A, l, u
            d = np.ones(n)
            e = np.ones(m)
            c = 1.0
        
        # Initialize variables (in scaled coordinates)
        x_scaled = np.zeros(n)
        y_scaled = np.clip(np.zeros_like(l_eq), l_eq, u_eq)
        u_dual = np.zeros(m)  # dual variable
        R = rho * np.eye(m) # step size matrix
        R_inv = 1 / rho * np.eye(m)
        b = b0
        r_norm = 1e-2
        output = np.zeros(n+m) # for GMRES
        self.P_sigma = P_eq + sigma * np.eye(n)

        # Store history of R matrices
        R_history = [R.copy()]
        b_history = [b]
        
        for k in range(1, max_iter + 1):
            # Check time limit
            if time.time() - start_time > time_limit:
                if verbose:
                    print(f"Time limit reached at iteration {k}")
                break
            
            # x-update (in scaled coordinates)
            if solving_method == 'kaczmarz':
                # Use Kaczmarz algorithm with warm-start
                # Solve: lhs_matrix @ x = rhs
                lhs_matrix = self.P_sigma + self.P_sigma + A_eq.T @ (np.diag(R)[:, None] * A_eq)
                rhs = -q_eq + A_eq.T @ (np.diag(R) * y_scaled - u_dual) + sigma * x_scaled
                if r_norm >= 1e-2:
                    x_scaled = np.linalg.solve(lhs_matrix, rhs)
                else:
                    x_scaled = self._kaczmarz_solve(lhs_matrix, rhs, x_scaled, max_sweeps=max_iter_kacz, tol=0.1*r_norm)
            
            elif solving_method == 'direct':
                lhs_matrix = self.P_sigma + A_eq.T @ (np.diag(R)[:, None] * A_eq)
                rhs = -q_eq + A_eq.T @ (np.diag(R) * y_scaled - u_dual) + sigma * x_scaled
                x_scaled_old = x_scaled.copy()
                x_scaled = np.linalg.solve(lhs_matrix, rhs)

            elif solving_method == 'LDLT':
                # don't know if it is useful by now
                lhs_matrix = np.block([
                    [P_eq + sigma * np.eye(n), A_eq.T],
                    [A_eq, -R_inv]
                ])
                lu, D, perm = ldl(lhs_matrix)
                lu_perm = lu[perm] # lu_perm is lower triangular

                rhs = np.concatenate([
                    -q_eq + sigma * x_scaled,
                    y_scaled - R_inv @ u_dual
                ])
                rhs_perm = rhs[perm]
                temp1 = solve_triangular(lu_perm, rhs_perm, lower=True, unit_diagonal=True)
                temp2 = np.linalg.solve(D, temp1) # TODO: can be optimized since D is block diagonal
                output_perm = solve_triangular(lu_perm.T, temp2, unit_diagonal=True)
                output = np.zeros_like(output_perm)
                output[perm] = output_perm
                x_scaled = output[:n]
                v = output[n:]

            elif solving_method == 'new_fact':
                if k == 1:
                    lhs_matrix = np.block([
                        [P_eq + sigma * np.eye(n), A_eq.T],
                        [A_eq, -R_inv]
                    ])
                    H = P_eq + sigma * np.eye(n)
                    H_inv = np.linalg.inv(H)
                    L = np.block([
                        [np.eye(n), np.zeros((n, m))],
                        [A_eq @ H_inv, np.eye(m)]
                    ])
                    D = np.block([
                        [H, np.zeros((n, m))],
                        [np.zeros((m, n)), -(R_inv + A_eq @ H_inv @ A_eq.T)]
                    ])
                else:
                    lhs_matrix[n:, n:] += R_old_inv - R_inv
                    D[n:, n:] += R_old_inv - R_inv
                rhs = np.concatenate([
                    -q_eq + sigma * x_scaled,
                    y_scaled - R_inv @ u_dual
                ])
                temp1 = solve_triangular(L, rhs, lower=True, unit_diagonal=True)
                temp2 = np.linalg.solve(D, temp1)
                output = solve_triangular(L.T, temp2, unit_diagonal=True)
                x_scaled = output[:n]
                v = output[n:]
            
            elif solving_method == 'CG':
                lhs_matrix = self.P_sigma + A_eq.T @ (np.diag(R)[:, None] * A_eq)
                rhs = -q_eq + A_eq.T @ (np.diag(R) * y_scaled - u_dual) + sigma * x_scaled
                x_scaled = self._solve_indirect_cg(P_eq, A_eq, np.diag(R), sigma, x_scaled, y_scaled, u_dual, q_eq, r_norm=r_norm)

            elif solving_method == 'CG_precond':
                lhs_matrix = self.P_sigma + A_eq.T @ (np.diag(R)[:, None] * A_eq)
                rhs = -q_eq + A_eq.T @ (np.diag(R) * y_scaled - u_dual) + sigma * x_scaled
                x_scaled = self._solve_indirect_cg(P_eq, A_eq, np.diag(R), sigma, x_scaled, y_scaled, u_dual, q_eq, r_norm=r_norm, precond=True)
                
            else:
                raise ValueError(f"Unknown solving_method: {solving_method}")
                
            # y-update (in scaled coordinates)
            y_unconstrained = R_inv @ u_dual + A_eq @ x_scaled
            y_old = y_scaled.copy()
            y_scaled = np.clip(y_unconstrained, l_eq, u_eq)
            
            # dual update: u_dual = u_dual + (A x - y)
            residual = A_eq @ x_scaled - y_scaled
            u_dual = u_dual + R @ residual
            
            # Compute residuals for stopping criteria (in scaled coordinates)
            r_norm = np.linalg.norm(residual, ord=np.inf)
            s = A_eq.T @ R @ (y_scaled - y_old)
            s_norm = np.linalg.norm(s, ord=np.inf)

            # numerical error & b update
            if rhs.shape[0] == n:
                epsilon = np.linalg.norm(rhs - lhs_matrix @ x_scaled, ord=np.inf)
            else:
                epsilon = np.linalg.norm(rhs - lhs_matrix @ output, ord=np.inf)
            b = tau * b if epsilon >= r_norm else b

            # R matrix update
            update_mask = np.logical_or(np.abs(y_scaled - l_eq) < 1e-8, np.abs(y_scaled - u_eq) < 1e-8)
            for diag_idx in range(m):
                if update_mask[diag_idx]:
                    R[diag_idx, diag_idx] = min(R[diag_idx, diag_idx] * alpha, b)
                else:
                    R[diag_idx, diag_idx] = max(R[diag_idx, diag_idx] / alpha, 1 / b)
            R_old_inv = R_inv.copy()
            R_inv = 1.0 / np.diag(R) * np.eye(m)

            R_history.append(R.copy())
            b_history.append(b)
            
            if verbose and k % 100 == 0:
                print(f"ADMM iter {k:3d}: r_norm={r_norm:.4e}, s_norm={s_norm:.4e}")
            
            # Compute unscaled residuals for stopping criteria
            if use_preconditioning:
                # Transform back to original coordinates for stopping criteria
                x_unscaled = d * x_scaled  # x = D * x_tilde
                y_unscaled = y_scaled / e   # y = E^{-1} * y_tilde
                residual_orig = A @ x_unscaled - y_unscaled
                r_norm_orig = np.linalg.norm(residual_orig, ord=np.inf)
                
                # Use original scale tolerances
                eps_pri = np.sqrt(m) * abstol + reltol * np.maximum(np.linalg.norm(A @ x_unscaled), 
                                                                   np.linalg.norm(y_unscaled))
                eps_dual = np.sqrt(n) * abstol + reltol * np.linalg.norm(A.T @ (u_dual / e))
                
                if r_norm_orig <= eps_pri and s_norm <= eps_dual:
                    if verbose:
                        print(f"SuperADMM converged in {k} iterations (unscaled residuals)")
                    break
            else:
                # Standard stopping criteria
                eps_pri = np.sqrt(m) * abstol + reltol * np.maximum(np.linalg.norm(A_eq @ x_scaled), np.linalg.norm(y_scaled))
                eps_dual = np.sqrt(n) * abstol + reltol * np.linalg.norm(A_eq.T @ u_dual)
                
                if r_norm <= eps_pri and s_norm <= eps_dual:
                    if verbose:
                        print(f"SuperADMM converged in {k} iterations")
                    break
        
        # Transform solution back to original coordinates
        if use_preconditioning:
            x_sol = d * x_scaled  # x = D * x_tilde
            y_sol = y_scaled / e  # y = E^{-1} * y_tilde
            # Note: dual variables u_dual are already in the correct scale for original problem
        else:
            x_sol = x_scaled
            y_sol = y_scaled
        
        R_bounded_ratio = round(np.logical_or(np.diag(R) >= b, np.diag(R) <= 1 / b).sum() / m, 4)
        return x_sol, y_sol, u_dual, k, b, R_bounded_ratio, R_history, b_history
