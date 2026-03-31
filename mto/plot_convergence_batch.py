"""
plot_convergence_batch.py
-------------------------
Convergence plotter for the Data_Batch folder layout.

Data layout (algo-first, problem files flat inside each folder):
    DATA_ROOT/{algo_name}/{algo_name}_{prob_name}_{run_id}.pkl

Output mirrors plot_convergence.py:
    SAVE_ROOT/{prob_name}/{prob_name}_convergence.png
    one subplot per task, all algorithms overlaid, mean ± STD_SCALE*std shading.

Usage:
    python plot_convergence_batch.py               # all discovered problems
    python plot_convergence_batch.py P1            # single problem
    python plot_convergence_batch.py P1 P3 P5      # subset
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

DATA_ROOT  = './Data_Batch'
SAVE_ROOT  = './Results_Batch'
STD_SCALE  = 0.5
FIG_FORMAT = 'png'
DPI        = 150
ALPHA_BAND = 0.15

# Algorithms to include and their display order.
# Any algo found in the folder but not listed here is appended at the end.
ALGO_ORDER = [
    'GA', 'BO', 'BO-LCB', 'MTBO', 'BO-LCB-BCKT',
    'BO-TFM', 'MTBO-TFM-Uni', 'MTBO-TFM-Elite',
    'MTBO-TFM-Uni-OH', 'MTBO-TFM-Elite-OH',
    'BO-TFM-CMA', 'MTBO-TFM-Uni-CMA', 'MTBO-TFM-Elite-CMA',
    'MTBO-TFM-Uni-OH-CMA', 'MTBO-TFM-Elite-OH-CMA',
]

COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
]
MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', 'h', '<', '>', 'p', 'H', '8', '+']


# =============================================================================
# Helpers
# =============================================================================

def load_pkl(path: Path) -> dict:
    with open(path, 'rb') as f:
        return pickle.load(f)


def best_so_far(all_objs_task: list) -> np.ndarray:
    """Staircase all_objs[task] → monotone best-so-far curve."""
    curve = np.array([np.min(gen[:, 0]) for gen in all_objs_task])
    for i in range(1, len(curve)):
        curve[i] = min(curve[i], curve[i - 1])
    return curve


def parse_prob_name(stem: str, algo_name: str) -> str:
    """
    Extract the problem name from a pkl stem.

    Stem format:  {algo_name}_{prob_name}_{run_id}
    e.g.  'MTBO-TFM-Uni_P3_2'  with algo_name='MTBO-TFM-Uni'  → 'P3'

    Works regardless of hyphens/underscores in algo_name because we strip
    the known prefix (algo_name + '_') and then split off the trailing run id.
    """
    suffix = stem[len(algo_name) + 1:]   # '{prob_name}_{run_id}'
    prob, _ = suffix.rsplit('_', 1)
    return prob


def discover_problems(data_root: Path, algo_names: list) -> list:
    """Scan all pkl files and collect unique problem names."""
    problems = set()
    for algo in algo_names:
        algo_dir = data_root / algo
        if not algo_dir.exists():
            continue
        for pkl in algo_dir.glob('*.pkl'):
            try:
                prob = parse_prob_name(pkl.stem, algo)
                problems.add(prob)
            except (ValueError, IndexError):
                pass
    return sorted(problems)


def load_algo_data(data_root: Path, algo_name: str, prob_name: str):
    """
    Load all run pkl files for (algo_name, prob_name) from the batch layout.

    Returns
    -------
    curves_per_task  : list[list[np.ndarray]]  — [task][run] = convergence array
    max_nfes_per_task: list[int]
    n_tasks          : int
    None, None, 0 if no data found.
    """
    algo_dir  = data_root / algo_name
    if not algo_dir.exists():
        return None, None, 0

    pkl_files = sorted(algo_dir.glob(f'{algo_name}_{prob_name}_*.pkl'))
    if not pkl_files:
        return None, None, 0

    all_runs          = []
    max_nfes_per_task = None

    for pkl_path in pkl_files:
        data      = load_pkl(pkl_path)
        all_objs  = data['all_objs']
        run_curves = [best_so_far(all_objs[t]) for t in range(len(all_objs))]
        all_runs.append(run_curves)
        if max_nfes_per_task is None:
            max_nfes_per_task = list(data['max_nfes'])

    n_tasks = len(all_runs[0])
    curves_per_task = [
        [all_runs[r][t] for r in range(len(all_runs))]
        for t in range(n_tasks)
    ]
    return curves_per_task, max_nfes_per_task, n_tasks


def align_curves(curves: list) -> np.ndarray:
    """Truncate curves to the shortest length and stack → (n_runs, min_len)."""
    min_len = min(len(c) for c in curves)
    return np.stack([c[:min_len] for c in curves], axis=0)


# =============================================================================
# Main plot function
# =============================================================================

def plot_problem(prob_name: str, data_root: Path, save_root: Path, algos: list):
    save_dir = save_root / prob_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # --- load data for every algorithm that has files for this problem ---
    algo_data     = {}
    algo_max_nfes = {}
    n_tasks       = 0

    for algo in algos:
        curves, max_nfes_pt, nt = load_algo_data(data_root, algo, prob_name)
        if curves is None:
            continue
        algo_data[algo]     = curves
        algo_max_nfes[algo] = max_nfes_pt
        n_tasks = max(n_tasks, nt)

    if not algo_data:
        print(f'  [SKIP] {prob_name}: no data found for any algorithm.')
        return

    n_runs_info = {a: len(algo_data[a][0]) for a in algo_data}
    print(f'  {prob_name}: {len(algo_data)} algos, {n_tasks} tasks — '
          f'runs: { {a: n for a, n in n_runs_info.items()} }')

    # --- colour / marker maps (stable across problems) ---
    all_algo_keys = list(algo_data.keys())
    color_map  = {}
    marker_map = {}
    for i, a in enumerate(ALGO_ORDER + [x for x in all_algo_keys if x not in ALGO_ORDER]):
        color_map[a]  = COLORS[i  % len(COLORS)]
        marker_map[a] = MARKERS[i % len(MARKERS)]

    # --- figure ---
    fig, axes = plt.subplots(1, n_tasks, figsize=(5 * n_tasks, 4), squeeze=False)
    fig.suptitle(f'CEC17-MTSO-10D  —  {prob_name}', fontsize=13, fontweight='bold')

    for t in range(n_tasks):
        ax = axes[0][t]
        ax.set_title(f'Task {t + 1}', fontsize=11)
        ax.set_xlabel('NFEs', fontsize=10)
        ax.set_ylabel('Best Objective Found', fontsize=10)

        for algo, curves_per_task in algo_data.items():
            if t >= len(curves_per_task):
                continue

            mat  = align_curves(curves_per_task[t])   # (n_runs, n_pts)
            mean = mat.mean(axis=0)
            std  = mat.std(axis=0)

            nfes       = algo_max_nfes[algo][t] if t < len(algo_max_nfes[algo]) else len(mean)
            x          = np.linspace(0, nfes, len(mean))
            color      = color_map.get(algo, '#333333')
            marker     = marker_map.get(algo, 'o')
            mark_every = max(1, len(x) // 10)

            ax.plot(x, mean,
                    label=algo,
                    color=color,
                    marker=marker,
                    markevery=mark_every,
                    markersize=4,
                    linewidth=1.5)

            if mat.shape[0] > 1:   # shade only when multiple runs exist
                ax.fill_between(
                    x,
                    mean - STD_SCALE * std,
                    mean + STD_SCALE * std,
                    alpha=ALPHA_BAND,
                    color=color,
                )

        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=6))
        ax.grid(True, linestyle='--', alpha=0.4)

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
    print(f'    Saved → {out_path}')


# =============================================================================
# Entry point
# =============================================================================

if __name__ == '__main__':
    data_root = Path(DATA_ROOT)
    save_root = Path(SAVE_ROOT)

    if not data_root.exists():
        print(f'ERROR: Data_Batch folder not found: {data_root.resolve()}')
        sys.exit(1)

    # Collect algo folders present on disk (preserve ALGO_ORDER, append extras)
    present_algos = {d.name for d in data_root.iterdir() if d.is_dir()}
    algos = [a for a in ALGO_ORDER if a in present_algos] + \
            sorted(present_algos - set(ALGO_ORDER))

    if not algos:
        print(f'ERROR: No algorithm folders found in {data_root.resolve()}')
        sys.exit(1)

    # Problems: from CLI args or auto-discovered from file names
    if len(sys.argv) > 1:
        problems = sys.argv[1:]
    else:
        problems = discover_problems(data_root, algos)
        if not problems:
            print('ERROR: Could not discover any problem names from pkl files.')
            sys.exit(1)
        print(f'Auto-discovered problems: {problems}')

    print(f'Algorithms found: {algos}')
    print(f'Problems to plot: {problems}\n')

    for prob in problems:
        plot_problem(prob, data_root, save_root, algos)

    print('\nAll done.')
