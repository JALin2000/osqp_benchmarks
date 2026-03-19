"""
config.py — Hyperparameters for the learned OSQP pipeline.

All modules import Config from here. Change values here to tune the pipeline.
"""

from dataclasses import dataclass, field
import torch


# Constants mirroring _osqp.py
RHO_MIN = 1e-6
RHO_MAX = 1e6
RHO_EQ_OVER_RHO_INEQ = 1e3
OSQP_INFTY = 1e30
MIN_SCALING = 1e-4
RHO_TOL = 1e-4            # equality constraint threshold (u - l < RHO_TOL)


@dataclass
class Config:
    # ------------------------------------------------------------------ #
    # OSQP physics (must match _osqp.py defaults)
    # ------------------------------------------------------------------ #
    sigma: float = 1e-6          # KKT regularization (sigma*I in top-left block)
    rho: float = 0.1             # Initial penalty parameter
    rho_min: float = RHO_MIN     # Minimum rho (loose constraints)
    rho_eq_factor: float = RHO_EQ_OVER_RHO_INEQ   # rho multiplier for equality rows
    alpha_x: float = 1.6         # Fixed scalar over-relaxation for x-update

    # ------------------------------------------------------------------ #
    # Adaptive rho
    # ------------------------------------------------------------------ #
    adaptive_rho: bool = True
    adaptive_rho_tolerance: float = 5.0   # trigger if new_rho > tol*rho or < rho/tol

    # ------------------------------------------------------------------ #
    # Learned alpha range
    # sigmoid(0) * (alpha_max - alpha_min) + alpha_min
    #   = 0.5 * (1.95 - 1.25) + 1.25 = 0.5 * 0.7 + 1.25 = 1.6  ✓
    # ------------------------------------------------------------------ #
    alpha_min: float = 1.25
    alpha_max: float = 1.95

    # ------------------------------------------------------------------ #
    # Rollout / training loop
    # ------------------------------------------------------------------ #
    T: int = 10                  # OSQP iterations per stage
    max_stages: int = 2000         # Maximum stages per training episode

    # ------------------------------------------------------------------ #
    # Network architecture
    # ------------------------------------------------------------------ #
    feature_dim: int = 13         # Per-row feature vector length (see features.py)
    hidden_dim: int = 64
    n_layers: int = 3            # total layers = n_layers-1 hidden + 1 output

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    lr: float = 5e-5             # AdamW learning rate
    weight_decay: float = 1e-4   # AdamW weight decay
    batch_size: int = 16
    n_epochs: int = 100
    n_train: int = 160
    n_val: int = 80
    loss_eps: float = 1e-12      # Numerical floor in log-convergence loss
    grad_clip: float = 1.0       # max_norm for gradient clipping

    # ------------------------------------------------------------------ #
    # Data
    # ------------------------------------------------------------------ #
    n_fixed: int = 20            # QP primal dimension for random_qp (m = 10 * n_fixed)
    data_dir: str = 'learned_osqp/data'
    results_dir: str = 'learned_osqp/results'

    # Which QP types to include; each type needs an entry in qp_type_sizes.
    # Supported: 'random_qp', 'control', 'eq_qp', 'huber', 'lasso', 'portfolio'
    qp_types: list = field(default_factory=lambda: ['random_qp'])
    # Size parameter for each type (meaning varies by type):
    #   random_qp  → n (primal dim);  m = 10 * n
    #   control    → nx (state dim);  n_qp = 16*nx, m_qp ≈ 32*nx
    #   eq_qp      → n (primal dim);  m = n // 2
    #   huber      → n_feat;          n_qp = n_feat + 3*(100*n_feat), m_qp = 3*(100*n_feat)
    #   lasso      → n_feat;          n_qp = 2*n_feat + 100*n_feat,   m_qp = 102*n_feat
    #   portfolio  → k (factors);     n_qp ≈ 101*k, m_qp ≈ 101*k
    qp_type_sizes: dict = field(default_factory=lambda: {'random_qp': 20})

    # ------------------------------------------------------------------ #
    # Alpha mode
    # ------------------------------------------------------------------ #
    # 'vector' : per-row alpha_z (B, m) predicted by PerRowAlphaNet
    # 'scalar' : single scalar alpha replacing both alpha_x and alpha_z,
    #            predicted by ScalarAlphaNet from 7 global residual features
    alpha_mode: str = 'vector'
    scalar_feature_dim: int = 6   # dim of global features used by ScalarAlphaNet
    model_type: str = 'mlp'       # 'mlp' or 'gru' (only applies when alpha_mode='scalar')

    # ------------------------------------------------------------------ #
    # Dataset storage
    # ------------------------------------------------------------------ #
    store_spectral_matrices: bool = True  # store R, AR, ARAt (only needed for spectral_radius loss)

    # ------------------------------------------------------------------ #
    # Feature normalization
    # ------------------------------------------------------------------ #
    normalize_features: bool = False  # if True, normalize input features to zero mean / unit std

    # ------------------------------------------------------------------ #
    # Device and dtype
    # ------------------------------------------------------------------ #
    device: str = 'cpu'          # 'cpu' or 'cuda'
    dtype: str = 'float64'       # 'float64' or 'float32'

    # ------------------------------------------------------------------ #
    # Precision (OSQP convergence tolerance)
    # ------------------------------------------------------------------ #
    precision: str = 'low'       # 'low' → eps=1e-3; 'high' → eps=1e-5

    @property
    def eps_abs(self) -> float:
        """Absolute convergence tolerance (eps_abs for OSQP)."""
        return 1e-5 if self.precision == 'high' else 1e-3

    @property
    def eps_rel(self) -> float:
        """Relative convergence tolerance (eps_rel for OSQP)."""
        return 1e-5 if self.precision == 'high' else 1e-3

    @property
    def torch_dtype(self) -> torch.dtype:
        """Return the torch dtype corresponding to the dtype string."""
        return torch.float32 if self.dtype == 'float32' else torch.float64

    @property
    def torch_device(self) -> torch.device:
        """Return the torch device corresponding to the device string."""
        return torch.device(self.device)

    @property
    def m_fixed(self) -> int:
        """Number of constraint rows for fixed-size problems."""
        return self.n_fixed * 10
