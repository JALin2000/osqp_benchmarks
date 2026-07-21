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

Both callbacks rescale work.info residuals back to SCALED space for all features
that need absolute magnitude or inf-norm values.  For per-row ratio f[9] only,
unscaled residuals are used directly since E_inv cancels element-wise.

All remaining features (f[0], f[1], f[4], f[7], f[8]) come from work.{z, data.l, data.u, y,
rho_vec, data.A} which are already in scaled space.
"""

from __future__ import annotations

import os
import numpy as np
import torch

from solvers.osqppurepy import OSQP as _OSQPInterface
from learned_osqp.config import Config
from learned_osqp.model import PerRowAlphaNet, PerRowGRUNet, ScalarAlphaNet, ScalarGRUNet

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


def _load_model(checkpoint_path: str, cfg: Config) -> PerRowAlphaNet | PerRowGRUNet | ScalarAlphaNet | ScalarGRUNet:
    """Load a PerRowAlphaNet, PerRowGRUNet, ScalarAlphaNet, or ScalarGRUNet from a .pt checkpoint."""
    alpha_mode = getattr(cfg, 'alpha_mode', 'vector')
    model_type = getattr(cfg, 'model_type', 'mlp')
    if alpha_mode != 'scalar':
        if model_type == 'gru':
            model = PerRowGRUNet(cfg)
        else:
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
    scaled space for features that need absolute magnitude (f[2], f[5], f[6])
    and for inf-norm ratio features (f[10], f[11], f[12]).
    Per-row ratio f[9] uses unscaled values directly (E_inv cancels element-wise).
    """

    def __init__(self, work, model: PerRowAlphaNet | PerRowGRUNet, cfg: Config,
                 T: int = 10, record_history: bool = False):
        self.work = work
        self.model = model
        self.cfg = cfg
        self.T = T
        self.iter_count = 0
        self._torch_dtype = cfg.torch_dtype

        m = work.data.m
        self._m = m

        # GRU hidden state — None means zeros (reset at start of each solve)
        self._is_gru: bool = isinstance(model, PerRowGRUNet)
        self._h: torch.Tensor | None = None  # (1, m, hidden_dim)

        # State T steps ago
        self._pri_res_unscaled_prev: np.ndarray = np.zeros(m)   # E_inv*(Ax-z) prev (for f[9] per-row ratio, E cancels)
        self._pri_res_inf_prev: float = 0.0   # SCALED inf norm (for f[10] ratio)
        self._dua_res_inf_prev: float = 0.0   # SCALED inf norm (for f[11] ratio)

        # History recording (opt-in)
        self._record_history = record_history
        self._prev_alpha_z: np.ndarray | None = None   # (m,) previous alpha_z for change computation
        self._history: dict = {'iter': [], 'alpha_change': []}

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

    def get_history(self) -> dict | None:
        """Return recorded history dict, or None if record_history=False."""
        return self._history if self._record_history else None

    # ---------------------------------------------------------------- #
    # Feature computation
    # ---------------------------------------------------------------- #

    def _compute_features(self) -> tuple[np.ndarray, float, float]:
        """
        Fills and returns (self._feat_buf, pri_res_inf_scaled, dua_res_inf_scaled).

        All per-row/per-element quantities come from the SCALED workspace.
        work.info.{pri_res_vec, dua_res_vec} are UNSCALED and rescaled below.

        Returns:
            feat_buf: (m, 13) float64
            pri_res_inf: scaled primal residual inf norm (for prev storage)
            dua_res_inf: scaled dual residual inf norm (for prev storage)
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

        return f, pri_res_inf, dua_res_inf

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

        # Stop NN alpha updates after 500 iterations (keep last alpha)
        if self.iter_count > 500:
            return None

        # Compute features FIRST (uses prev residuals from the previous stage boundary)
        feat_np, pri_res_inf, dua_res_inf = self._compute_features()         # (m, 13), float, float
        feat_t  = torch.as_tensor(feat_np, dtype=self._torch_dtype).unsqueeze(0)  # (1, m, 13)

        with torch.no_grad():
            if self._is_gru:
                alpha_z_t, self._h = self.model(feat_t, self._h)             # (1, m), (1, m, hidden)
            else:
                alpha_z_t = self.model(feat_t)                               # (1, m)

        # THEN update prev residuals for the next stage boundary.
        # Inf norms are stored in SCALED space (from _compute_features) so that
        # f[10], f[11] ratios are scaled/scaled — matching training exactly.
        self._pri_res_inf_prev = pri_res_inf    # scaled inf norm
        self._dua_res_inf_prev = dua_res_inf    # scaled inf norm

        # Per-row prev (unscaled) for f[9] — E_inv cancels in per-row ratio
        work = self.work
        pri_res_unscaled = getattr(work.info, 'pri_res_vec', None)
        if pri_res_unscaled is not None:
            self._pri_res_unscaled_prev = pri_res_unscaled.copy()

        alpha_z_np = alpha_z_t.squeeze(0).numpy()  # (m,)

        # Record alpha change history (opt-in)
        if self._record_history:
            if self._prev_alpha_z is not None:
                change = float(np.max(np.abs(alpha_z_np - self._prev_alpha_z)))
            else:
                change = 0.0
            self._history['iter'].append(self.iter_count)
            self._history['alpha_change'].append(change)
            self._prev_alpha_z = alpha_z_np.copy()

        work.settings.alpha_z = alpha_z_np  # Set alpha_z for the next T iterations


# ------------------------------------------------------------------ #
# NeuralScalarAlphaCallback — scalar alpha mode
# ------------------------------------------------------------------ #

class NeuralScalarAlphaCallback:
    """
    Scalar-alpha callback: computes 6 global residual features, runs
    ScalarAlphaNet → single scalar alpha, then applies it to BOTH
    alpha_x (via work.settings.alpha) and alpha_z.

    Feature vector (6-dim, matches compute_global_features):
        f[0] log10(pri_res_inf_norm)        (SCALED, rescaled from work.info)
        f[1] log10(dua_res_inf_norm)        (SCALED, rescaled from work.info)
        f[2] log10(rho_scalar)              (work.settings.rho)
        f[3] log10(pri_res_inf_norm / prev) (SCALED / SCALED)
        f[4] log10(dua_res_inf_norm / prev) (SCALED / SCALED)
        f[5] log10(pri_res_inf_norm / dua_res_inf_norm)  (SCALED / SCALED)

    All residuals are rescaled to Ruiz-equilibrated (SCALED) space using
    cached E, c, D vectors, matching the training pipeline exactly.
    """

    def __init__(self, work, model: ScalarAlphaNet | ScalarGRUNet, cfg: Config,
                 T: int = 10, record_history: bool = False):
        self.work  = work
        self.model = model
        self.cfg   = cfg
        self.T     = T
        self.iter_count = 0
        self._torch_dtype = cfg.torch_dtype
        self._m = work.data.m
        self._pri_res_inf_prev: float = 0.0   # SCALED inf norm (for f[3] ratio)
        self._dua_res_inf_prev: float = 0.0   # SCALED inf norm (for f[4] ratio)
        # GRU hidden state — None means zeros (reset at start of each solve)
        self._is_gru: bool = isinstance(model, ScalarGRUNet)
        self._h: torch.Tensor | None = None  # (1, hidden_dim)

        # Scaling vectors — never change after setup
        self._has_scaling = hasattr(work, 'scaling') and work.settings.scaling
        if self._has_scaling:
            self._e_vec = work.scaling.E.diagonal().copy()   # (m,)
            self._c = float(work.scaling.c)                  # scalar
            self._d_vec = work.scaling.D.diagonal().copy()   # (n,)

        # History recording (opt-in; adds 2 sparse matvecs per stage boundary)
        self._record_history = record_history
        self._prev_alpha: float | None = None  # previous alpha for change computation
        self._q_inf = float(np.max(np.abs(work.data.q))) if work.data.q.size else 0.0
        self._history: dict = {'iter': [], 'alpha': [], 'alpha_change': [],
                               'scaled_prim': [], 'scaled_dual': []}

    def _compute_features(self) -> tuple[np.ndarray, float, float]:
        """Returns ((6,) feature vector, pri_res_inf_scaled, dua_res_inf_scaled).

        All residuals are rescaled to Ruiz-equilibrated (SCALED) space to match
        the training pipeline (compute_global_features in features.py).
        """
        work = self.work

        pri_res_unscaled = getattr(work.info, 'pri_res_vec', None)
        dua_res_unscaled = getattr(work.info, 'dua_res_vec', None)

        # Rescale to SCALED space: E * E_inv*(Ax-z) = Ax-z, c*D * c_inv*D_inv*(...) = Px+q+ATy
        if pri_res_unscaled is not None:
            if self._has_scaling:
                pri_res_scaled = self._e_vec * pri_res_unscaled       # (m,)
            else:
                pri_res_scaled = pri_res_unscaled
            pri_res_inf = float(np.max(np.abs(pri_res_scaled)))
        else:
            pri_res_inf = float(np.max(np.abs(work.data.A.dot(work.x) - work.z)))

        if dua_res_unscaled is not None:
            if self._has_scaling:
                dua_res_scaled = self._c * self._d_vec * dua_res_unscaled  # (n,)
            else:
                dua_res_scaled = dua_res_unscaled
            dua_res_inf = float(np.max(np.abs(dua_res_scaled)))
        else:
            dua = work.data.P.dot(work.x) + work.data.q + work.data.A.T.dot(work.y)
            dua_res_inf = float(np.max(np.abs(dua)))

        rho_val = float(work.settings.rho)

        features = np.array([
            _log10s(pri_res_inf),                                         # f[0] scaled
            _log10s(dua_res_inf),                                         # f[1] scaled
            _log10s(rho_val),                                             # f[2]
            _log10s(pri_res_inf / (self._pri_res_inf_prev + _EPS)),       # f[3] scaled/scaled
            _log10s(dua_res_inf / (self._dua_res_inf_prev + _EPS)),       # f[4] scaled/scaled
            _log10s(pri_res_inf / (dua_res_inf + _EPS)),                  # f[5] scaled/scaled
        ], dtype=np.float64)
        return features, pri_res_inf, dua_res_inf

    def _compute_scaled_residuals(self, pri_res_inf: float, dua_res_inf: float) -> tuple[float, float]:
        """
        Compute scaled_prim and scaled_dual using current work iterates.
        Requires 2 sparse matvecs (P @ x, A.T @ y); called only when record_history=True.
        """
        work = self.work

        # scaled_prim = ||Ax - z||_inf / max(||Ax||_inf, ||z||_inf)
        pri_res_vec = getattr(work.info, 'pri_res_vec', None)
        if pri_res_vec is not None:
            Ax_inf = float(np.max(np.abs(pri_res_vec + work.z)))  # Ax = r_prim + z
        else:
            Ax_inf = float(np.max(np.abs(work.data.A.dot(work.x))))
        z_inf = float(np.max(np.abs(work.z)))
        denom_prim = max(Ax_inf, z_inf, _EPS)
        sc_prim = pri_res_inf / denom_prim

        # scaled_dual = ||Px + q + ATy||_inf / max(||Px||_inf, ||ATy||_inf, ||q||_inf)
        Px  = work.data.P.dot(work.x)
        ATy = work.data.A.T.dot(work.y)
        Px_inf  = float(np.max(np.abs(Px)))  if Px.size > 0 else 0.0
        ATy_inf = float(np.max(np.abs(ATy)))
        denom_dual = max(Px_inf, ATy_inf, self._q_inf, _EPS)
        sc_dual = dua_res_inf / denom_dual

        return sc_prim, sc_dual

    def get_history(self) -> dict | None:
        """Return recorded history dict, or None if record_history=False."""
        return self._history if self._record_history else None

    def __call__(self) -> np.ndarray | None:
        """Returns constant (m,) alpha array at stage boundaries, else None."""
        self.iter_count += 1
        if (self.iter_count - 1) % self.T != 0:
            return None

        # Stop NN alpha updates after 500 iterations (keep last alpha)
        if self.iter_count > 500:
            return None

        work = self.work

        # Compute features FIRST (uses prev residuals from the previous stage boundary)
        feat_np, pri_res_inf, dua_res_inf = self._compute_features()         # (6,), float, float
        feat_t  = torch.as_tensor(feat_np, dtype=self._torch_dtype).unsqueeze(0)  # (1, 6)

        with torch.no_grad():
            if self._is_gru:
                alpha_t, self._h = self.model(feat_t, self._h)  # (1,), (1, hidden_dim)
            else:
                alpha_t = self.model(feat_t)  # (1,)

        alpha_val = float(alpha_t.item())

        # THEN update prev residuals (SCALED inf norms) for the next stage boundary
        self._pri_res_inf_prev = pri_res_inf    # scaled inf norm
        self._dua_res_inf_prev = dua_res_inf    # scaled inf norm

        # Record history at this stage boundary (opt-in)
        if self._record_history:
            sc_prim, sc_dual = self._compute_scaled_residuals(pri_res_inf, dua_res_inf)
            change = abs(alpha_val - self._prev_alpha) if self._prev_alpha is not None else 0.0
            self._prev_alpha = alpha_val
            self._history['iter'].append(self.iter_count)
            self._history['alpha'].append(alpha_val)
            self._history['alpha_change'].append(change)
            self._history['scaled_prim'].append(sc_prim)
            self._history['scaled_dual'].append(sc_dual)

        # Apply scalar alpha to alpha_x via settings
        work.settings.alpha_x = alpha_val
        work.settings.alpha_z = alpha_val

        # Return constant (m,) array for alpha_z — reuse buffer
        # self._alpha_buf[:] = alpha_val
        # return self._alpha_buf


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
        model_type: str = 'mlp',
        record_history: bool = False,
    ):
        self._cfg = cfg or Config()
        self._cfg.alpha_mode = alpha_mode
        self._cfg.model_type = model_type
        self._nn = _load_model(checkpoint_path, self._cfg)
        self._T = T
        self._alpha_mode = alpha_mode
        self._record_history = record_history
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
            self._callback = NeuralScalarAlphaCallback(
                work, self._nn, self._cfg, self._T,
                record_history=self._record_history,
            )
        else:
            self._callback = NeuralAlphaCallback(
                work, self._nn, self._cfg, self._T,
                record_history=self._record_history,
            )
        work.learnt_component_callback = self._callback

    def solve(self, total_iters=None):
        """Solve and return the Results object from _osqp."""
        return self._osqp.solve(total_iters=total_iters)

    def get_alpha_history(self) -> dict | None:
        """Return per-stage-boundary history dict from the callback.

        Returns None if record_history=False.
        Scalar keys: 'iter', 'alpha', 'alpha_change', 'scaled_prim', 'scaled_dual'.
        Vector keys: 'iter', 'alpha_change'.
        """
        if self._callback is not None and hasattr(self._callback, 'get_history'):
            return self._callback.get_history()
        return None

    def warm_start(self, x=None, y=None):
        return self._osqp.warm_start(x=x, y=y)

    def update(self, **kwargs):
        return self._osqp.update(**kwargs)

    def update_settings(self, **kwargs):
        return self._osqp.update_settings(**kwargs)
