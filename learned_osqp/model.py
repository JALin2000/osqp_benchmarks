"""
model.py — PerRowAlphaNet: neural network predicting per-row alpha_z.

Architecture:
  - Shared-weight MLP applied pointwise to each constraint row.
  - Input:  (B, m, feature_dim)
  - Output: (B, m) alpha_z values in [alpha_min, alpha_max]

The network is row-equivariant: the same weights process every constraint row
independently. This generalises across problem sizes and is O(m) in depth.

Initialisation:
  The final linear layer is zero-initialised so that:
      sigmoid(0) * (alpha_max - alpha_min) + alpha_min
    = 0.5  * (1.99 - 1.21) + 1.21
    = 0.5  * 0.78 + 1.21
    = 0.39 + 1.21 = 1.60  ✓

  This means at epoch 0 the network behaves identically to the OSQP default
  alpha = 1.6, giving a stable starting point for training.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from learned_osqp.config import Config


class PerRowAlphaNet(nn.Module):
    """
    Pointwise MLP mapping per-row features → per-row alpha_z.

    Input shape:  (B, m, feature_dim)
    Output shape: (B, m)   values in [alpha_min, alpha_max]

    The MLP has shared weights across all rows (dim m is treated as the
    "instance" dimension, not a feature dimension). This makes the network
    equivariant to row permutations.

    Architecture:
      Linear(feature_dim → hidden_dim) + LayerNorm(hidden_dim) + ReLU
      ... (n_layers - 1 blocks total)
      Linear(hidden_dim → 1)
      Sigmoid → scale to [alpha_min, alpha_max]

    LayerNorm normalises over the hidden_dim axis independently per
    (batch, row), which is appropriate for this per-row structure.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        # Feature normalization: buffers are saved/loaded with state_dict.
        # feat_norm_active is saved separately in the checkpoint dict.
        self.register_buffer('feat_mean', torch.zeros(cfg.feature_dim))
        self.register_buffer('feat_std',  torch.ones(cfg.feature_dim))
        self.feat_norm_active: bool = False

        layers: list[nn.Module] = []
        in_dim = cfg.feature_dim

        # Hidden blocks
        for _ in range(cfg.n_layers - 1):
            layers.extend([
                nn.Linear(in_dim, cfg.hidden_dim),
                nn.LayerNorm(cfg.hidden_dim),
                nn.ELU(),
            ])
            in_dim = cfg.hidden_dim

        # Output layer: hidden_dim → 1
        self.output_layer = nn.Linear(in_dim, 1)
        self.hidden_net = nn.Sequential(*layers)

        self._init_weights()

    def set_feature_norm(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """Set per-feature normalization statistics and activate normalization.

        Args:
            mean : (feature_dim,) — per-feature mean computed from training data
            std  : (feature_dim,) — per-feature std; near-zero entries should
                   already be replaced with 1.0 by the caller.
        """
        self.feat_mean.copy_(mean.to(device=self.feat_mean.device, dtype=self.feat_mean.dtype))
        self.feat_std.copy_(std.to(device=self.feat_std.device,   dtype=self.feat_std.dtype))
        self.feat_norm_active = True

    def _init_weights(self) -> None:
        """
        Initialise hidden layers with small Xavier weights and zero bias.
        Initialise output layer with zero weight and zero bias so that
        sigmoid(0) → 1.6 on first forward pass.
        """
        for module in self.hidden_net.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                nn.init.zeros_(module.bias)

        # Zero-init output layer → sigmoid(0) = 0.5 → alpha = 1.6
        # nn.init.zeros_(self.output_layer.weight)
        nn.init.normal_(self.output_layer.weight, std=0.01)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features : (B, m, feature_dim) — per-row features

        Returns:
            alpha_z  : (B, m) in [cfg.alpha_min, cfg.alpha_max]
        """
        if self.feat_norm_active:
            features = (features - self.feat_mean) / (self.feat_std + 1e-8)
        h = self.hidden_net(features)          # (B, m, hidden_dim)
        raw = self.output_layer(h).squeeze(-1)  # (B, m)
        alpha = torch.sigmoid(raw)             # (B, m) in (0, 1)
        return (
            self.cfg.alpha_min
            + (self.cfg.alpha_max - self.cfg.alpha_min) * alpha
        )   # (B, m) in [alpha_min, alpha_max]

    def predict_scalar(
        self, B: int, m: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """
        Return the constant baseline alpha=1.6 as a tensor (no network eval).
        Useful for sanity-checking the initialisation.
        """
        return torch.full((B, m), 1.6, dtype=dtype, device=device)


class ScalarAlphaNet(nn.Module):
    """
    MLP mapping 5-dim global residual features → single scalar alpha per
    problem instance.

    Input shape:  (B, scalar_feature_dim)
    Output shape: (B,)   values in [alpha_min, alpha_max]

    The same scalar substitutes both alpha_x and alpha_z for every row.

    Architecture mirrors PerRowAlphaNet (shared init / norm interface) but
    operates on instance-level features rather than per-row features.

    Features (see features.py compute_global_features):
        f[0] log pri_res_inf_norm
        f[1] log dua_res_inf_norm
        f[2] log rho_scalar
        f[3] log(pri_res_inf_norm / pri_res_inf_norm_prev)
        f[4] log(dua_res_inf_norm / dua_res_inf_norm_prev)
    """

    def __init__(self, cfg: 'Config'):
        super().__init__()
        self.cfg = cfg

        in_dim = cfg.scalar_feature_dim

        # Feature normalization buffers — same interface as PerRowAlphaNet
        self.register_buffer('feat_mean', torch.zeros(in_dim))
        self.register_buffer('feat_std',  torch.ones(in_dim))
        self.feat_norm_active: bool = False

        layers: list[nn.Module] = []
        for _ in range(cfg.n_layers - 1):
            layers.extend([
                nn.Linear(in_dim, cfg.hidden_dim),
                nn.LayerNorm(cfg.hidden_dim),
                nn.ELU(),
            ])
            in_dim = cfg.hidden_dim

        self.output_layer = nn.Linear(in_dim, 1)
        self.hidden_net = nn.Sequential(*layers)

        self._init_weights()

    def set_feature_norm(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """Same interface as PerRowAlphaNet.set_feature_norm."""
        self.feat_mean.copy_(mean.to(device=self.feat_mean.device, dtype=self.feat_mean.dtype))
        self.feat_std.copy_(std.to(device=self.feat_std.device,   dtype=self.feat_std.dtype))
        self.feat_norm_active = True

    def _init_weights(self) -> None:
        for module in self.hidden_net.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                nn.init.zeros_(module.bias)
        nn.init.normal_(self.output_layer.weight, std=0.01)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features : (B, scalar_feature_dim) — global per-instance features

        Returns:
            alpha : (B,) in [cfg.alpha_min, cfg.alpha_max]
        """
        if self.feat_norm_active:
            features = (features - self.feat_mean) / (self.feat_std + 1e-8)
        h   = self.hidden_net(features)          # (B, hidden_dim)
        raw = self.output_layer(h).squeeze(-1)   # (B,)
        alpha = torch.sigmoid(raw)
        return (
            self.cfg.alpha_min
            + (self.cfg.alpha_max - self.cfg.alpha_min) * alpha
        )  # (B,) in [alpha_min, alpha_max]


class ScalarGRUNet(nn.Module):
    """
    GRU mapping sequential 6-dim global residual features → scalar alpha per
    problem instance, with memory across ADMM stages.

    At each stage boundary (every T=10 iterations), the GRU cell processes the
    current 6-dim feature vector and updates its hidden state h.  This lets the
    network detect multi-stage convergence patterns (oscillation, plateau,
    acceleration) that the memoryless ScalarAlphaNet cannot.

    Input shape:  (B, scalar_feature_dim) per stage + (B, hidden_dim) h state
    Output shape: (B,) alpha, (B, hidden_dim) h_new

    Hidden state is carried between stages during a solve and reset to zeros at
    the start of each new problem.  Gradients are detached between stages in
    training (same convention as x/z/y) — 1-step BPTT per backward pass.

    GRU vs LSTM: GRU has one fewer gate (no cell state c), ~25% fewer FLOPs,
    and equivalent performance on the short stage sequences seen here (5-30
    stages per solve).

    Initialisation:
      Output layer near-zero → sigmoid(0) = 0.5 → alpha = 1.6 at epoch 0,
      identical to the OSQP default.
    """

    def __init__(self, cfg: 'Config'):
        super().__init__()
        self.cfg = cfg
        in_dim = cfg.scalar_feature_dim

        # Feature normalisation — same interface as ScalarAlphaNet
        self.register_buffer('feat_mean', torch.zeros(in_dim))
        self.register_buffer('feat_std',  torch.ones(in_dim))
        self.feat_norm_active: bool = False

        self.gru_cell    = nn.GRUCell(in_dim, cfg.hidden_dim)
        self.post_gru    = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ELU(),
        )
        self.output_layer = nn.Linear(cfg.hidden_dim, 1)

        self._init_weights()

    def set_feature_norm(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """Same interface as ScalarAlphaNet.set_feature_norm."""
        self.feat_mean.copy_(mean.to(device=self.feat_mean.device, dtype=self.feat_mean.dtype))
        self.feat_std.copy_(std.to(device=self.feat_std.device,   dtype=self.feat_std.dtype))
        self.feat_norm_active = True

    def _init_weights(self) -> None:
        # GRU: Xavier for input-to-hidden, orthogonal for hidden-to-hidden
        for name, p in self.gru_cell.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(p)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(p)
            elif 'bias' in name:
                nn.init.zeros_(p)
        # post_gru linear: Xavier init
        for module in self.post_gru.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                nn.init.zeros_(module.bias)
        # Output layer: near-zero init → sigmoid(0) → alpha ≈ 1.6
        nn.init.normal_(self.output_layer.weight, std=0.01)
        nn.init.zeros_(self.output_layer.bias)

    def forward(
        self,
        features: torch.Tensor,         # (B, scalar_feature_dim)
        h: torch.Tensor | None = None,  # (B, hidden_dim)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features : (B, scalar_feature_dim) — current-stage global features
            h        : (B, hidden_dim) GRU hidden state; zeros if None (stage 0)

        Returns:
            alpha : (B,)            in [cfg.alpha_min, cfg.alpha_max]
            h_new : (B, hidden_dim) updated hidden state to carry to next stage

        Note: h_new is the raw GRU cell output. The post_gru layer only affects
        the alpha computation and does not alter the carried hidden state.
        """
        if self.feat_norm_active:
            features = (features - self.feat_mean) / (self.feat_std + 1e-8)
        B = features.shape[0]
        if h is None:
            h = torch.zeros(B, self.cfg.hidden_dim,
                            dtype=features.dtype, device=features.device)
        h_new = self.gru_cell(features, h)                    # (B, hidden_dim)
        raw   = self.output_layer(self.post_gru(h_new)).squeeze(-1)  # (B,)
        alpha = torch.sigmoid(raw)
        return (
            self.cfg.alpha_min + (self.cfg.alpha_max - self.cfg.alpha_min) * alpha,
            h_new,
        )
