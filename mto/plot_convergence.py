"""
plot_convergence.py
-------------------
Standalone convergence plotter for MTSO experiment results.

Reads pkl files from:
    DATA_ROOT/{prob_name}/{algo_name}/{algo_name}_{prob_name}_{run}.pkl

Plots mean ± 0.5*std best-so-far convergence curves for each task,
one subplot per task, all algorithms overlaid.

Usage:
    python plot_convergence.py                  # plots P1 with defaults
    python plot_convergence.py P3               # plots P3
    python plot_convergence.py P1 P3 P5         # plots multiple problems
"""

import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

# =============================================================================
# Configuration — edit here
# =============================================================================

DATA_ROOT   = './Data'
SAVE_ROOT   = './Results'
STD_SCALE   = 0.5        # fraction of std to shade (0.5 = half std)
FIG_FORMAT  = 'png'
DPI         = 150
ALPHA_BAND  = 0.15       # shading transparency

ALGO_ORDER = [
    'GA', 'BO', 'BO-LCB', 'MTBO', 'BO-LCB-BCKT',
    'BO-TFM', 'MTBO-TFM-Uni', 'MTBO-TFM-Elite',
    'MTBO-TFM-Uni-OH', 'MTBO-TFM-Elite-OH',
]

COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
]
MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', 'h', '<']

# =============================================================================
# Core helpers
# =============================================================================

def load_pkl(path: Path) -> dict:
    with open(path, 'rb') as f:
        return pickle.load(f)


def best_so_far(all_objs_task: list) -> np.ndarray:
    """
    Convert staircase all_objs[task] → best-so-far convergence curve.

    all_objs_task[g] has shape (g+1, n_objs).
    Returns array of shape (n_generations,) with the running minimum.
    """
    curve = np.array([np.min(gen[:, 0]) for gen in all_objs_task])
    # enforce monotone decrease (best-so-far)
    for i in range(1, len(curve)):
        curve[i] = min(curve[i], curve[i - 1])
    return curve


def load_algo_data(prob_dir: Path, algo_name: str, prob_name: str):
    """
    Load all run pkl files for one algorithm on one problem.

    Returns
    -------
    curves_per_task : list of list of np.ndarray
        curves_per_task[task][run] = convergence array of shape (n_nfes,)
    n_tasks : int
    """
    algo_dir = prob_dir / algo_name
    if not algo_dir.exists():
        return None, 0

    pkl_files = sorted(algo_dir.glob(f'{algo_name}_{prob_name}_*.pkl'))
    if not pkl_files:
        return None, 0

    all_runs = []
    for pkl_path in pkl_files:
        data = load_pkl(pkl_path)
        all_objs = data['all_objs']   # List[List[np.ndarray]]
        run_curves = [best_so_far(all_objs[t]) for t in range(len(all_objs))]
        all_runs.append(run_curves)

    n_tasks = len(all_runs[0])
    # curves_per_task[t] = list of convergence arrays (one per run)
    curves_per_task = []
    for t in range(n_tasks):
        task_curves = [all_runs[r][t] for r in range(len(all_runs))]
        curves_per_task.append(task_curves)

    return curves_per_task, n_tasks


def align_curves(curves: list) -> np.ndarray:
    """
    Stack curves of potentially different lengths by truncating to the shortest.
    Returns array of shape (n_runs, min_len).
    """
    min_len = min(len(c) for c in curves)
    return np.stack([c[:min_len] for c in curves], axis=0)


# =============================================================================
# Main plot function
# =============================================================================

def plot_problem(prob_name: str):
    prob_dir  = Path(DATA_ROOT) / prob_name
    save_dir  = Path(SAVE_ROOT) / prob_name
    save_dir.mkdir(parents=True, exist_ok=True)

    if not prob_dir.exists():
        print(f'[SKIP] Data folder not found: {prob_dir}')
        return

    # --- collect available algorithms (respect ALGO_ORDER, skip missing) ---
    present = [a for a in ALGO_ORDER if (prob_dir / a).exists()]
    extra   = sorted(set(d.name for d in prob_dir.iterdir() if d.is_dir())
                     - set(ALGO_ORDER))
    algos   = present + extra

    if not algos:
        print(f'[SKIP] No algorithm folders found in {prob_dir}')
        return

    # --- load data ---
    algo_data = {}
    n_tasks   = 0
    for algo in algos:
        curves, nt = load_algo_data(prob_dir, algo, prob_name)
        if curves is None:
            print(f'  [WARN] No pkl files found for {algo}, skipping.')
            continue
        algo_data[algo] = curves
        n_tasks = max(n_tasks, nt)

    if not algo_data:
        print(f'[SKIP] No data loaded for {prob_name}')
        return

    print(f'Plotting {prob_name}: {len(algo_data)} algorithms, {n_tasks} tasks')

    # --- figure: one subplot per task ---
    fig, axes = plt.subplots(
        1, n_tasks,
        figsize=(5 * n_tasks, 4),
        squeeze=False,
    )
    fig.suptitle(f'CEC17-MTSO-10D  —  {prob_name}', fontsize=13, fontweight='bold')

    color_map  = {a: COLORS[i % len(COLORS)]  for i, a in enumerate(ALGO_ORDER + list(algo_data.keys()))}
    marker_map = {a: MARKERS[i % len(MARKERS)] for i, a in enumerate(ALGO_ORDER + list(algo_data.keys()))}

    for t in range(n_tasks):
        ax = axes[0][t]
        ax.set_title(f'Task {t + 1}', fontsize=11)
        ax.set_xlabel('NFEs', fontsize=10)
        ax.set_ylabel('Best Objective Found', fontsize=10)

        for algo, curves_per_task in algo_data.items():
            if t >= len(curves_per_task):
                continue

            mat   = align_curves(curves_per_task[t])   # (n_runs, n_nfes)
            mean  = mat.mean(axis=0)
            std   = mat.std(axis=0)
            x     = np.arange(1, len(mean) + 1)

            color  = color_map.get(algo, '#333333')
            marker = marker_map.get(algo, 'o')

            # mark every ~10% of x range to avoid clutter
            mark_every = max(1, len(x) // 10)

            ax.plot(x, mean,
                    label=algo,
                    color=color,
                    marker=marker,
                    markevery=mark_every,
                    markersize=4,
                    linewidth=1.5)

            ax.fill_between(
                x,
                mean - STD_SCALE * std,
                mean + STD_SCALE * std,
                alpha=ALPHA_BAND,
                color=color,
            )

        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=6))
        ax.grid(True, linestyle='--', alpha=0.4)

    # shared legend below the subplots
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc='lower center',
        ncol=min(len(algo_data), 5),
        fontsize=9,
        bbox_to_anchor=(0.5, -0.12),
        frameon=True,
    )

    plt.tight_layout()
    out_path = save_dir / f'{prob_name}_convergence.{FIG_FORMAT}'
    fig.savefig(out_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved → {out_path}')


# =============================================================================
# Entry point
# =============================================================================

if __name__ == '__main__':
    problems = sys.argv[1:] if len(sys.argv) > 1 else ['P1']
    for prob in problems:
        plot_problem(prob)
