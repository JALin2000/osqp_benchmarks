"""Visualize training log files. Re-runs on all .log files each time."""
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


# ── regex patterns ────────────────────────────────────────────────────────────

RE_BASELINE = re.compile(
    r'Baseline.*?'
    r'train_iters=([0-9.]+)±([0-9.]+).*?'
    r'val_iters=([0-9.]+)±([0-9.]+).*?'
    r'train_rho_updates=([0-9.]+)±([0-9.]+).*?'
    r'val_rho_updates=([0-9.]+)±([0-9.]+)'
)

RE_EPOCH = re.compile(
    r'Epoch\s+(\d+)/\d+.*?'
    r'train_iters=([0-9.]+)±([0-9.]+).*?'
    r'val_iters=([0-9.]+)±([0-9.]+).*?'
    r'(?:train_alpha=([0-9.]+)±([0-9.]+).*?val_alpha=([0-9.]+)±([0-9.]+).*?)?'
    r'train_rho_updates=([0-9.]+)±([0-9.]+).*?'
    r'val_rho_updates=([0-9.]+)±([0-9.]+)'
)


def parse_log(path: Path):
    text = path.read_text(errors='replace')

    baseline = None
    m = RE_BASELINE.search(text)
    if m:
        vals = [float(v) for v in m.groups()]
        baseline = {
            'train_iters':       (vals[0], vals[1]),
            'val_iters':         (vals[2], vals[3]),
            'train_rho_updates': (vals[4], vals[5]),
            'val_rho_updates':   (vals[6], vals[7]),
        }

    epochs, train_iters, train_iters_std = [], [], []
    val_iters, val_iters_std = [], []
    train_alpha, train_alpha_std = [], []
    val_alpha,   val_alpha_std   = [], []
    train_rho, train_rho_std = [], []
    val_rho,   val_rho_std   = [], []

    for m in RE_EPOCH.finditer(text):
        gs = m.groups()  # (epoch, ti, ti_s, vi, vi_s, ta, ta_s, va, va_s, tr, tr_s, vr, vr_s)
        epochs.append(int(gs[0]))
        train_iters.append(float(gs[1]));      train_iters_std.append(float(gs[2]))
        val_iters.append(float(gs[3]));        val_iters_std.append(float(gs[4]))
        train_alpha.append(float(gs[5]) if gs[5] is not None else float('nan'))
        train_alpha_std.append(float(gs[6]) if gs[6] is not None else float('nan'))
        val_alpha.append(float(gs[7]) if gs[7] is not None else float('nan'))
        val_alpha_std.append(float(gs[8]) if gs[8] is not None else float('nan'))
        train_rho.append(float(gs[9]));        train_rho_std.append(float(gs[10]))
        val_rho.append(float(gs[11]));         val_rho_std.append(float(gs[12]))

    if not epochs:
        return None

    return {
        'baseline': baseline,
        'epochs':            np.array(epochs),
        'train_iters':       np.array(train_iters),
        'train_iters_std':   np.array(train_iters_std),
        'val_iters':         np.array(val_iters),
        'val_iters_std':     np.array(val_iters_std),
        'train_alpha':       np.array(train_alpha),
        'train_alpha_std':   np.array(train_alpha_std),
        'val_alpha':         np.array(val_alpha),
        'val_alpha_std':     np.array(val_alpha_std),
        'train_rho':         np.array(train_rho),
        'train_rho_std':     np.array(train_rho_std),
        'val_rho':           np.array(val_rho),
        'val_rho_std':       np.array(val_rho_std),
    }


def plot_log(path: Path):
    data = parse_log(path)
    if data is None:
        print(f'  [skip] no epoch data: {path.name}')
        return

    ep  = data['epochs']
    bl  = data['baseline']

    panels = [
        ('train_iters',       data['train_iters'],     data['train_iters_std'],
         'Train iterations',  'steelblue'),
        ('val_iters',         data['val_iters'],       data['val_iters_std'],
         'Val iterations',    'darkorange'),
        ('train_rho_updates', data['train_rho'],       data['train_rho_std'],
         'Train rho updates', 'seagreen'),
        ('val_rho_updates',   data['val_rho'],         data['val_rho_std'],
         'Val rho updates',   'tomato'),
        ('train_alpha',       data['train_alpha'],     data['train_alpha_std'],
         'Train alpha',       'mediumpurple'),
        ('val_alpha',         data['val_alpha'],       data['val_alpha_std'],
         'Val alpha',         'saddlebrown'),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    fig.suptitle(path.stem, fontsize=11, fontweight='bold')

    for ax, (key, mean, std, title, color) in zip(axes.flat, panels):
        ax.plot(ep, mean, color=color, linewidth=1.5, label='learned α')
        ax.fill_between(ep, mean - std, mean + std,
                        color=color, alpha=0.20)

        if bl is not None and key in bl:
            bm, bs = bl[key]
            ax.axhline(bm, color='gray', linewidth=1.2,
                       linestyle='--', label=f'baseline {bm:.2f}')
            ax.axhspan(bm - bs, bm + bs, color='gray', alpha=0.10)
        elif key in ('train_alpha', 'val_alpha'):
            # show fixed alpha=1.6 baseline
            ax.axhline(1.6, color='gray', linewidth=1.2,
                       linestyle='--', label='baseline 1.60')

        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.legend(fontsize=8)
        ax.grid(True, linewidth=0.4, alpha=0.6)

    plt.tight_layout()
    out = path.with_suffix('.png')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'  saved → {out.name}')


def main():
    # allow optional directory argument; default to the checkpoints folder
    if len(sys.argv) > 1:
        search_dir = Path(sys.argv[1])
    else:
        search_dir = Path(__file__).parent / 'learned_osqp' / 'checkpoints_arc/checkpoints'

    log_files = sorted(search_dir.glob('*.log'))
    if not log_files:
        print(f'No .log files found in {search_dir}')
        return

    print(f'Visualizing {len(log_files)} log file(s) in {search_dir}')
    for lf in log_files:
        plot_log(lf)
    print('Done.')


if __name__ == '__main__':
    main()
