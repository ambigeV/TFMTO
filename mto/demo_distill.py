"""
First-run validation for the TFM-Distill variants.

Runs one problem (P1), one run, four algorithms:
  1. BO-TFM-1000        — random candidates baseline (existing)
  2. BO-TFM-Distill     — independent distill + L-BFGS, cold start
  3. BO-TFM-Distill-WS  — same but warm-started (fewer MLP epochs per step)
  4. MTBO-TFM-Uni-Distill — multi-task uniform distill + L-BFGS, cold start

Purpose: verify the distill pipeline runs end-to-end without crashing and
produces convergence curves that are plausible before committing to a full
N_RUNS=5 × 9-problem benchmark run.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ddmtolab.Problems.MTSO.cec17_mtso_10d import CEC17MTSO_10D
from ddmtolab.Algorithms.STSO.BO_TFM import BO_TFM
from ddmtolab.Algorithms.STSO.BO_TFM_Distill import BO_TFM_Distill
from ddmtolab.Algorithms.MTSO.MTBO_TFM_Distill import MTBO_TFM_Distill
from ddmtolab.Methods.data_analysis import DataAnalyzer

# =============================================================================
# Config — keep small for a quick sanity check
# =============================================================================

PROB_NAME    = 'P1'
N_INITIAL    = 20
MAX_NFES     = 100
BETA         = 1.0
N_ESTIMATORS = 8

benchmark = CEC17MTSO_10D()
problem   = benchmark.P1()

def data_path(algo_name):
    path = f'./Data/distill_demo/{algo_name}'
    os.makedirs(path, exist_ok=True)
    return path

# =============================================================================
# Run
# =============================================================================

print("1/4  BO-TFM-1000  (random baseline)")
BO_TFM(
    problem, n_initial=N_INITIAL, max_nfes=MAX_NFES,
    n_candidates=1000, n_estimators=N_ESTIMATORS,
    disable_tqdm=False,
    save_path=data_path('BO-TFM-1000'),
    name=f'BO-TFM-1000_{PROB_NAME}_1',
).optimize()

# Re-instantiate problem — each algorithm gets a fresh problem object
problem = benchmark.P1()

print("\n2/4  BO-TFM-Distill  (cold start, mse loss)")
BO_TFM_Distill(
    problem, n_initial=N_INITIAL, max_nfes=MAX_NFES,
    beta=BETA, n_estimators=N_ESTIMATORS,
    n_distill=1000, mlp_epochs=300,
    mlp_loss='mse', distill_model='mlp',
    warm_start=False,
    lbfgs_restarts=5, lbfgs_maxiter=100,
    disable_tqdm=False,
    save_path=data_path('BO-TFM-Distill'),
    name=f'BO-TFM-Distill_{PROB_NAME}_1',
).optimize()

problem = benchmark.P1()

print("\n3/4  BO-TFM-Distill-WS  (warm start, 50 finetune epochs)")
BO_TFM_Distill(
    problem, n_initial=N_INITIAL, max_nfes=MAX_NFES,
    beta=BETA, n_estimators=N_ESTIMATORS,
    n_distill=1000, mlp_epochs=300, mlp_finetune_epochs=50,
    mlp_loss='mse', distill_model='mlp',
    warm_start=True,
    lbfgs_restarts=5, lbfgs_maxiter=100,
    disable_tqdm=False,
    save_path=data_path('BO-TFM-Distill-WS'),
    name=f'BO-TFM-Distill-WS_{PROB_NAME}_1',
).optimize()

problem = benchmark.P1()

print("\n4/4  MTBO-TFM-Uni-Distill  (multi-task, cold start)")
MTBO_TFM_Distill(
    problem, n_initial=N_INITIAL, max_nfes=MAX_NFES,
    beta=BETA, n_estimators=N_ESTIMATORS,
    transfer='uniform', encoding='scalar',
    n_distill=1000, mlp_epochs=300,
    mlp_loss='mse', distill_model='mlp',
    warm_start=False,
    lbfgs_restarts=5, lbfgs_maxiter=100,
    disable_tqdm=False,
    save_path=data_path('MTBO-TFM-Uni-Distill'),
    name=f'MTBO-TFM-Uni-Distill_{PROB_NAME}_1',
).optimize()

# =============================================================================
# Plot
# =============================================================================

print("\nPlotting...")
DataAnalyzer(
    data_path  = './Data/distill_demo',
    save_path  = './Results/distill_demo',
    algorithm_order = [
        'BO-TFM-1000',
        'BO-TFM-Distill',
        'BO-TFM-Distill-WS',
        'MTBO-TFM-Uni-Distill',
    ],
    figure_format  = 'png',
    log_scale      = False,
    show_std_band  = False,   # single run — no std band
    best_so_far    = True,
    clear_results  = True,
).run()

print("\nDone. Check mto/Results/distill_demo/")
