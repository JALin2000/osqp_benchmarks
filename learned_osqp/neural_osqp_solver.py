"""
neural_osqp_solver.py — Plug PerRowAlphaNet into _osqp.py via learnt_component_callback.

Scaling notes
-------------
When scaling=True, _osqp.py stores:
    work.data.{P, q, A, l, u} : SCALED matrices/vectors
    work.{x, z, y}             : iterates in SCALED space
    work.info.pri_res_vec      : UNSCALED  = E_inv * (Ax_sc - z_sc)
    work.info.dua_res_vec      : UNSCALED  = cinv * Dinv * (Px_sc + q_sc + Asc^T * y_sc)

The network was trained on SCALED quantities.  Conversion rules:
    z_sc - Ax_sc  (scaled pri_res, sign matches features.py) = -E * pri_res_vec_unscaled
    Px_sc + q_sc + Asc^T * y_sc (scaled dua_res)            =  c * D * dua_res_vec_unscaled

For ratio features (f[9]-f[12]) the E/c/D factors cancel in numerator/denominator, so
unscaled residuals can be used directly.

All remaining features (f[0], f[1], f[4], f[7], f[8]) come from work.{z, data.l, data.u, y,
rho_vec, data.A} which are already in scaled space.
"""

from __future__ import annotations

import os
import numpy as np
import torch

from solvers.osqppurepy import OSQP as _OSQPInterface
from learned_osqp.config import Config
from learned_osqp.model import PerRowAlphaNet, ScalarAlphaNet, ScalarGRUNet

# ------------------------------------------------------------------ #
# Default checkpoint
# ------------------------------------------------------------------ #
_DEFAULT_CHECKPOINT = os.path.join(
    os.path.dirname(__file__),
    'checkpoints_new_feat_scaled',
    'best_model_n=100_long_train.pt',
)

_LOG_LOWER = 1e-6
_LOG_UPPER = 1e6
_EPS = 1e-8


def _log10c(v: np.ndarray) -> np.ndarray:
    """Element-wise clipped log10."""
    return np.log10(np.clip(v, _LOG_LOWER, _LOG_UPPER))


def _log10s(s: float) -> float:
    """Scalar clipped log10."""
    return float(np.log10(np.clip(s, _LOG_LOWER, _LOG_UPPER)))


def _load_model(checkpoint_path: str, cfg: Config) -> PerRowAlphaNet | ScalarAlphaNet | ScalarGRUNet:
    """Load a PerRowAlphaNet, ScalarAlphaNet, or ScalarGRUNet from a .pt checkpoint."""
    alpha_mode = getattr(cfg, 'alpha_mode', 'vector')
    model_type = getattr(cfg, 'model_type', 'mlp')
    if alpha_mode != 'scalar':
        model = PerRowAlphaNet(cfg)
    elif model_type == 'gru':
        model = ScalarGRUNet(cfg)
    else:
        model = ScalarAlphaNet(cfg)
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if isinstance(state, dict):
        for key in ('model_state', 'model_state_dict', 'model', 'state_dict'):
            if key in state:
                model.load_state_dict(state[key])
                break
        else:
            model.load_state_dict(state)
        # Restore normalization flag (buffers feat_mean/feat_std already in state_dict)
        model.feat_norm_active = state.get('feat_norm_active', False)
    else:
        model.load_state_dict(state)
    model.to(dtype=cfg.torch_dtype)
    model.eval()
    return model


class NeuralAlphaCallback:
    """
    Zero-argument callable installed as work.learnt_component_callback.

    Called once before each ADMM iteration.  Every T iterations it:
      1. Extracts 13-dim per-row features from the OSQP internal state.
      2. Runs PerRowAlphaNet → alpha_z in [alpha_min, alpha_max], shape (m,).
      3. Returns the numpy array to _osqp.update_alpha_z().

    Returns None on non-boundary iterations (keeps the previously set alpha_z).

    Feature scaling
    ---------------
    work.info.{pri_res_vec, dua_res_vec} are UNSCALED.  We rescale them back to
    scaled space for features that need absolute magnitude (f[2], f[5], f[6]).
    Ratio features (f[9]-f[12]) use unscaled values directly (scale cancels).
    """

    def __init__(self, work, model: PerRowAlphaNet, cfg: Config, T: int = 10):
        self.work = work
        self.model = model
        self.cfg = cfg
        self.T = T
        self.iter_count = 0
        self._torch_dtype = cfg.torch_dtype

        m = work.data.m
        self._m = m

        # State T steps ago — unscaled residuals (for ratio features, scale cancels)
        self._pri_res_unscaled_prev: np.ndarray = np.zeros(m)   # E_inv*(Ax-z) prev
        self._pri_res_inf_prev: float = 0.0
        self._dua_res_inf_prev: float = 0.0

        # --- Cache static quantities computed once ---
        # A row inf-norms (scaled A) — never changes after setup
        self._A_inf_norms = np.abs(work.data.A).max(axis=1).toarray().ravel()  # (m,)

        # Scaling vectors — never change after setup
        self._has_scaling = hasattr(work, 'scaling') and work.settings.scaling
        if self._has_scaling:
            self._e_vec = work.scaling.E.diagonal().copy()   # (m,)
            self._c = float(work.scaling.c)                  # scalar
            self._d_vec = work.scaling.D.diagonal().copy()   # (n,)

        # Pre-allocate feature buffer (m, 13) — reused every call
        self._feat_buf = np.empty((m, 13), dtype=np.float64)

    # ---------------------------------------------------------------- #
    # Feature computation
    # ---------------------------------------------------------------- #

    def _compute_features(self) -> np.ndarray:
        """
        Fills and returns self._feat_buf of shape (m, 13) in float64.

        All per-row/per-element quantities come from the SCALED workspace.
        work.info.{pri_res_vec, dua_res_vec} are UNSCALED and rescaled below.
        """
        work = self.work
        x  = work.x           # (n,) scaled
        z  = work.z           # (m,) scaled
        y  = work.y           # (m,) scaled
        l  = work.data.l      # (m,) scaled (may contain ±OSQP_INFTY)
        u  = work.data.u      # (m,) scaled
        rho_vec = work.rho_vec  # (m,)
        f = self._feat_buf

        # ---- Get unscaled residuals from work.info (populated every iter by update_info) ----
        pri_res_unscaled = getattr(work.info, 'pri_res_vec', None)
        dua_res_unscaled = getattr(work.info, 'dua_res_vec', None)

        if pri_res_unscaled is None:
            pri_res_unscaled = work.data.A.dot(x) - z
            pri_res_scaled = -pri_res_unscaled
        else:
            if self._has_scaling:
                pri_res_scaled = -self._e_vec * pri_res_unscaled
            else:
                pri_res_scaled = -pri_res_unscaled

        if dua_res_unscaled is None:
            dua_res_scaled = work.data.P.dot(x) + work.data.q + work.data.A.T.dot(y)
        else:
            if self._has_scaling:
                dua_res_scaled = self._c * self._d_vec * dua_res_unscaled
            else:
                dua_res_scaled = dua_res_unscaled

        abs_pri_res = np.abs(pri_res_scaled)              # (m,)
        pri_res_inf = float(np.max(abs_pri_res))
        dua_res_inf = float(np.max(np.abs(dua_res_scaled)))

        # T-step-ago (unscaled; ratios below cancel the scale factor E)
        abs_pri_res_pT = np.abs(self._pri_res_unscaled_prev)
        pri_res_inf_pT = self._pri_res_inf_prev
        dua_res_inf_pT = self._dua_res_inf_prev

        # Pre-compute scalar features (broadcast into columns below)
        s_pri_inf = _log10s(pri_res_inf)
        s_dua_inf = _log10s(dua_res_inf)
        s_pri_ratio = _log10s(pri_res_inf / (pri_res_inf_pT + _EPS))
        s_dua_ratio = _log10s(dua_res_inf / (dua_res_inf_pT + _EPS))
        s_imbalance = _log10s(pri_res_inf / (dua_res_inf + _EPS))

        # Fill feature buffer columns in-place
        f[:, 0] = _log10c(z - l)                                        # f[0]
        f[:, 1] = _log10c(u - z)                                        # f[1]
        f[:, 2] = _log10c(abs_pri_res)                                   # f[2]
        f[:, 3] = np.sign(pri_res_scaled)                                # f[3]
        f[:, 4] = _log10c(np.abs(y))                                     # f[4]
        f[:, 5] = s_pri_inf                                              # f[5] broadcast
        f[:, 6] = s_dua_inf                                              # f[6] broadcast
        f[:, 7] = _log10c(rho_vec)                                       # f[7]
        f[:, 8] = self._A_inf_norms                                      # f[8] cached
        f[:, 9] = _log10c(np.abs(pri_res_unscaled) / (abs_pri_res_pT + _EPS))  # f[9]
        f[:, 10] = s_pri_ratio                                           # f[10] broadcast
        f[:, 11] = s_dua_ratio                                           # f[11] broadcast
        f[:, 12] = s_imbalance                                           # f[12] broadcast

        return f

    # ---------------------------------------------------------------- #
    # __call__
    # ---------------------------------------------------------------- #

    def __call__(self) -> np.ndarray | None:
        """
        Returns (m,) alpha_z numpy array at stage boundaries, else None.
        """
        self.iter_count += 1

        # Only update alpha at the start of each T-block
        if (self.iter_count - 1) % self.T != 0:
            return None

        # Compute features FIRST (uses prev residuals from the previous stage boundary)
        feat_np = self._compute_features()                                   # (m, 13) float64
        feat_t  = torch.as_tensor(feat_np, dtype=self._torch_dtype).unsqueeze(0)  # (1, m, 13)

        with torch.no_grad():
            alpha_z_t = self.model(feat_t)                                   # (1, m)

        # THEN update prev residuals for the next stage boundary
        work = self.work
        pri_res_unscaled = getattr(work.info, 'pri_res_vec', None)
        dua_res_unscaled = getattr(work.info, 'dua_res_vec', None)

        if pri_res_unscaled is not None:
            self._pri_res_unscaled_prev = pri_res_unscaled.copy()
            self._pri_res_inf_prev = float(np.max(np.abs(pri_res_unscaled)))
        if dua_res_unscaled is not None:
            self._dua_res_inf_prev = float(np.max(np.abs(dua_res_unscaled)))

        return alpha_z_t.squeeze(0).numpy()                                  # (m,)


# ------------------------------------------------------------------ #
# NeuralScalarAlphaCallback — scalar alpha mode
# ------------------------------------------------------------------ #

class NeuralScalarAlphaCallback:
    """
    Scalar-alpha callback: computes 6 global residual features, runs
    ScalarAlphaNet → single scalar alpha, then applies it to BOTH
    alpha_x (via work.settings.alpha) and alpha_z (returned as constant
    (m,) array).

    Feature vector (6-dim, matches compute_global_features):
        f[0] log10(pri_res_inf_norm)        (unscaled, from work.info)
        f[1] log10(dua_res_inf_norm)        (unscaled, from work.info)
        f[2] log10(rho_scalar)              (work.settings.rho)
        f[3] log10(pri_res_inf_norm / prev)
        f[4] log10(dua_res_inf_norm / prev)
        f[5] log10(pri_res_inf_norm / dua_res_inf_norm)  (primal/dual imbalance)

    Note: the ratio features use unscaled residuals; since numerator and
    denominator come from the same space the scale factor cancels, matching
    the training pipeline exactly.
    """

    def __init__(self, work, model: ScalarAlphaNet | ScalarGRUNet, cfg: Config, T: int = 10):
        self.work  = work
        self.model = model
        self.cfg   = cfg
        self.T     = T
        self.iter_count = 0
        self._torch_dtype = cfg.torch_dtype
        self._m = work.data.m
        self._pri_res_inf_prev: float = 0.0
        self._dua_res_inf_prev: float = 0.0
        # Pre-allocate output buffer
        self._alpha_buf = np.empty(work.data.m, dtype=np.float64)
        # GRU hidden state — None means zeros (reset at start of each solve)
        self._is_gru: bool = isinstance(model, ScalarGRUNet)
        self._h: torch.Tensor | None = None  # (1, hidden_dim)

    def _compute_features(self) -> np.ndarray:
        """Returns (6,) float64 feature vector."""
        work = self.work

        pri_res_unscaled = getattr(work.info, 'pri_res_vec', None)
        dua_res_unscaled = getattr(work.info, 'dua_res_vec', None)

        if pri_res_unscaled is not None:
            pri_res_inf = float(np.max(np.abs(pri_res_unscaled)))
        else:
            pri_res_inf = float(np.max(np.abs(work.data.A.dot(work.x) - work.z)))

        if dua_res_unscaled is not None:
            dua_res_inf = float(np.max(np.abs(dua_res_unscaled)))
        else:
            dua = work.data.P.dot(work.x) + work.data.q + work.data.A.T.dot(work.y)
            dua_res_inf = float(np.max(np.abs(dua)))

        rho_val = float(work.settings.rho)

        return np.array([
            _log10s(pri_res_inf),
            _log10s(dua_res_inf),
            _log10s(rho_val),
            _log10s(pri_res_inf / (self._pri_res_inf_prev + _EPS)),
            _log10s(dua_res_inf / (self._dua_res_inf_prev + _EPS)),
            _log10s(pri_res_inf / (dua_res_inf + _EPS)),              # f[5] primal/dual imbalance
        ], dtype=np.float64)

    def __call__(self) -> np.ndarray | None:
        """Returns constant (m,) alpha array at stage boundaries, else None."""
        self.iter_count += 1
        if (self.iter_count - 1) % self.T != 0:
            return None

        work = self.work

        # Compute features FIRST (uses prev residuals from the previous stage boundary)
        feat_np = self._compute_features()                                   # (6,)
        feat_t  = torch.as_tensor(feat_np, dtype=self._torch_dtype).unsqueeze(0)  # (1, 6)

        with torch.no_grad():
            if self._is_gru:
                alpha_t, self._h = self.model(feat_t, self._h)  # (1,), (1, hidden_dim)
            else:
                alpha_t = self.model(feat_t)  # (1,)

        alpha_val = float(alpha_t.item())

        # THEN update prev residuals for the next stage boundary
        pri_res_unscaled = getattr(work.info, 'pri_res_vec', None)
        dua_res_unscaled = getattr(work.info, 'dua_res_vec', None)
        if pri_res_unscaled is not None:
            self._pri_res_inf_prev = float(np.max(np.abs(pri_res_unscaled)))
        if dua_res_unscaled is not None:
            self._dua_res_inf_prev = float(np.max(np.abs(dua_res_unscaled)))

        # Apply scalar alpha to alpha_x via settings
        work.settings.alpha = alpha_val

        # Return constant (m,) array for alpha_z — reuse buffer
        self._alpha_buf[:] = alpha_val
        return self._alpha_buf


# ------------------------------------------------------------------ #
# NeuralOSQPSolver — benchmark-compatible wrapper
# ------------------------------------------------------------------ #

class NeuralOSQPSolver:
    """
    Benchmark-compatible solver that wraps osqppurepy.OSQP and installs
    NeuralAlphaCallback before each solve.

    The solver name 'OSQP_python_neural' starts with 'OSQP_python', so
    benchmark_problems/example.py uses the OSQP_python branch:
        s = NeuralOSQPSolver()
        s.setup(P=..., q=..., A=..., l=..., u=..., **settings)
        results = s.solve()
    """

    def __init__(
        self,
        checkpoint_path: str = _DEFAULT_CHECKPOINT,
        cfg: Config | None = None,
        T: int = 10,
        alpha_mode: str = 'vector',
    ):
        self._cfg = cfg or Config()
        self._cfg.alpha_mode = alpha_mode
        self._nn = _load_model(checkpoint_path, self._cfg)
        self._T = T
        self._alpha_mode = alpha_mode
        self._osqp = _OSQPInterface()   # underlying solver
        self._callback: NeuralAlphaCallback | NeuralScalarAlphaCallback | None = None

    # Forward version / constant to osqp interface
    def version(self):
        return self._osqp.version()

    def constant(self, name):
        return self._osqp.constant(name)

    def setup(self, P=None, q=None, A=None, l=None, u=None, **settings):
        """Set up OSQP and install the neural alpha callback."""
        # Ensure scaling is on and learnt_component is 'alpha'
        settings.setdefault('scaling', 10)
        settings['learnt_component'] = 'alpha'

        self._osqp.setup(P=P, q=q, A=A, l=l, u=u, **settings)

        # Install callback on the internal _osqp.OSQP work object
        work = self._osqp._model.work
        if self._alpha_mode == 'scalar':
            self._callback = NeuralScalarAlphaCallback(work, self._nn, self._cfg, self._T)
        else:
            self._callback = NeuralAlphaCallback(work, self._nn, self._cfg, self._T)
        work.learnt_component_callback = self._callback

    def solve(self, total_iters=None):
        """Solve and return the Results object from _osqp."""
        return self._osqp.solve(total_iters=total_iters)

    def warm_start(self, x=None, y=None):
        return self._osqp.warm_start(x=x, y=y)

    def update(self, **kwargs):
        return self._osqp.update(**kwargs)

    def update_settings(self, **kwargs):
        return self._osqp.update_settings(**kwargs)
