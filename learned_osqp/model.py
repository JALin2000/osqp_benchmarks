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
