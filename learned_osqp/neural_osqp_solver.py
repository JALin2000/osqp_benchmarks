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

For ratio features (f[9]-f[11]) the E/c/D factors cancel in numerator/denominator, so
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
from learned_osqp.model import PerRowAlphaNet

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


def _load_model(checkpoint_path: str, cfg: Config) -> PerRowAlphaNet:
    """Load a PerRowAlphaNet from a .pt checkpoint."""
    model = PerRowAlphaNet(cfg)
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
      1. Extracts 12-dim per-row features from the OSQP internal state.
      2. Runs PerRowAlphaNet → alpha_z in [alpha_min, alpha_max], shape (m,).
      3. Returns the numpy array to _osqp.update_alpha_z().

    Returns None on non-boundary iterations (keeps the previously set alpha_z).

    Feature scaling
    ---------------
    work.info.{pri_res_vec, dua_res_vec} are UNSCALED.  We rescale them back to
    scaled space for features that need absolute magnitude (f[2], f[5], f[6]).
    Ratio features (f[9]-f[11]) use unscaled values directly (scale cancels).
    """

    def __init__(self, work, model: PerRowAlphaNet, cfg: Config, T: int = 10):
        self.work = work
        self.model = model
        self.cfg = cfg
        self.T = T
        self.iter_count = 0

        # State T steps ago — unscaled residuals (for ratio features, scale cancels)
        m = work.data.m
        self._pri_res_unscaled_prev: np.ndarray = np.zeros(m)   # E_inv*(Ax-z) prev
        self._pri_res_inf_prev: float = 0.0
        self._dua_res_inf_prev: float = 0.0

    # ---------------------------------------------------------------- #
    # Feature computation
    # ---------------------------------------------------------------- #

    def _compute_features(self) -> np.ndarray:
        """
        Returns np.ndarray of shape (m, 12) in float64.

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
        m = len(z)

        # ---- A row inf-norms (scaled A) ----
        # Note: sparse.max(axis=1) returns sparse in newer scipy; use .toarray() to convert
        A_inf_norms = np.abs(work.data.A).max(axis=1).toarray().ravel()  # (m,)

        # ---- Get unscaled residuals from work.info (populated every iter by update_info) ----
        pri_res_unscaled = getattr(work.info, 'pri_res_vec', None)
        dua_res_unscaled = getattr(work.info, 'dua_res_vec', None)

        if pri_res_unscaled is None:
            # First iteration: info not yet populated — compute directly
            pri_res_unscaled = work.data.A.dot(x) - z   # Ax_sc - z_sc (unscaled = no E_inv yet)
            # In this case there's no scaling applied — treat as scaled directly
            pri_res_scaled = -pri_res_unscaled          # z_sc - Ax_sc
        else:
            # pri_res_unscaled = E_inv * (Ax_sc - z_sc)
            # scaled pri_res = z_sc - Ax_sc = -E * pri_res_unscaled
            if hasattr(work, 'scaling') and work.settings.scaling:
                e_vec = work.scaling.E.diagonal()       # (m,)
                pri_res_scaled = -e_vec * pri_res_unscaled
            else:
                pri_res_scaled = -pri_res_unscaled

        if dua_res_unscaled is None:
            dua_res_scaled = work.data.P.dot(x) + work.data.q + work.data.A.T.dot(y)
        else:
            # dua_res_unscaled = cinv * Dinv * (Px_sc + q_sc + ATy_sc)
            # scaled dua_res = c * D * dua_res_unscaled
            if hasattr(work, 'scaling') and work.settings.scaling:
                c     = work.scaling.c                  # scalar
                d_vec = work.scaling.D.diagonal()       # (n,)
                dua_res_scaled = c * d_vec * dua_res_unscaled
            else:
                dua_res_scaled = dua_res_unscaled

        abs_pri_res = np.abs(pri_res_scaled)              # (m,)
        sign_pri_res = np.sign(pri_res_scaled)            # (m,)
        pri_res_inf = float(np.max(abs_pri_res)) if m > 0 else 0.0
        dua_res_inf = float(np.max(np.abs(dua_res_scaled))) if len(dua_res_scaled) > 0 else 0.0

        # T-step-ago (unscaled; ratios below cancel the scale factor E)
        abs_pri_res_pT     = np.abs(self._pri_res_unscaled_prev)
        pri_res_inf_pT     = self._pri_res_inf_prev
        dua_res_inf_pT     = self._dua_res_inf_prev

        def _log10c(v: np.ndarray) -> np.ndarray:
            return np.log10(np.clip(v, _LOG_LOWER, _LOG_UPPER))

        def _log10s(s: float) -> float:
            return float(np.log10(np.clip(s, _LOG_LOWER, _LOG_UPPER)))

        f = np.stack([
            _log10c(z - l),                                                   # f[0] log dist to lower
            _log10c(u - z),                                                   # f[1] log dist to upper
            _log10c(abs_pri_res),                                              # f[2] log |pri_res| (scaled)
            sign_pri_res,                                                      # f[3] sign pri_res (scaled)
            _log10c(np.abs(y)),                                                # f[4] log |y| (scaled)
            np.full(m, _log10s(pri_res_inf)),                                  # f[5] log inf-norm pri_res (scaled)
            np.full(m, _log10s(dua_res_inf)),                                  # f[6] log inf-norm dua_res (scaled)
            _log10c(rho_vec),                                                  # f[7] log rho
            A_inf_norms,                                                       # f[8] A row inf-norms (scaled)
            # ratio features — scale cancels (unscaled/unscaled = scaled/scaled)
            _log10c(np.abs(pri_res_unscaled) / (abs_pri_res_pT + _EPS)),      # f[9]  per-row ratio
            np.full(m, _log10s(pri_res_inf / (pri_res_inf_pT + _EPS))),       # f[10] inf-norm ratio
            np.full(m, _log10s(dua_res_inf / (dua_res_inf_pT + _EPS))),       # f[11] dua inf-norm ratio
        ], axis=-1)   # (m, 12)

        return f.astype(np.float64)

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

        work = self.work

        # Save current UNSCALED residuals as T-step-ago state for the next stage
        pri_res_unscaled = getattr(work.info, 'pri_res_vec', None)
        dua_res_unscaled = getattr(work.info, 'dua_res_vec', None)

        if pri_res_unscaled is not None:
            self._pri_res_unscaled_prev = pri_res_unscaled.copy()
            self._pri_res_inf_prev = float(np.max(np.abs(pri_res_unscaled)))
        if dua_res_unscaled is not None:
            self._dua_res_inf_prev = float(np.max(np.abs(dua_res_unscaled)))

        # Compute features and run network
        feat_np = self._compute_features()                                   # (m, 12) float64
        feat_t  = torch.from_numpy(feat_np).to(dtype=self.cfg.torch_dtype).unsqueeze(0)  # (1, m, 12)

        with torch.no_grad():
            alpha_z_t = self.model(feat_t)                                   # (1, m)

        return alpha_z_t.squeeze(0).numpy()                                  # (m,)


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
    ):
        self._cfg = cfg or Config()
        self._nn = _load_model(checkpoint_path, self._cfg)
        self._T = T
        self._osqp = _OSQPInterface()   # underlying solver
        self._callback: NeuralAlphaCallback | None = None

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
