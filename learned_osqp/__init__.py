"""
learned_osqp — Learning to optimize OSQP per-row relaxation parameters.

Modules:
  config.py      — Config dataclass with all hyperparameters
  data.py        — QP dataset generation (RandomQPExample + cvxpy)
  osqp_torch.py  — Differentiable OSQP ADMM iterations in PyTorch
  features.py    — Per-row feature computation
  model.py       — PerRowAlphaNet (shared-weight MLP)
  loss.py        — Log-convergence-ratio loss
  train.py       — Multi-stage training loop
  eval.py        — Evaluation and comparison vs. baseline alpha=1.6
"""

from learned_osqp.config import Config
from learned_osqp.model import PerRowAlphaNet
