"""
Batched Demo: MFEA-II vs ICL-MFEA-II on CEC17-MTSO 30D.

Compares vanilla MFEA-II (online-MLE random-mating probability) against
ICL-MFEA-II (the same RMP estimated under a MAP prior derived from TabPFN
cross-predictive task similarity, with a generation-decaying λ).

This is the evolutionary-computation instantiation of the ICL-MTBO paradigm:
the only changed coefficient is the off-diagonal RMP — MLE in MFEA-II,
MAP-toward-ICL-prior in ICL-MFEA-II.

Data layout (auto-managed by BatchExperiment):
    ./Data_MFEA_30D/{algo_name}/{algo_name}_{problem_name}_{run_id}.pkl
Results (mean ± 0.5*std convergence curves) → ./Results_MFEA_30D/.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ddmtolab.Problems.MTSO.cec17_mtso_30d import CEC17MTSO_30D
from ddmtolab.Algorithms.MTSO.MFEA_II import MFEA_II
from ddmtolab.Algorithms.MTSO.ICL_MFEA_II import ICL_MFEA_II
from ddmtolab.Methods.batch_experiment import BatchExperiment
from ddmtolab.Methods.data_analysis import DataAnalyzer

# =============================================================================
# Configuration
# =============================================================================

N_RUNS   = 5
N_POP    = 20          # population size per task (initial LHS count)
MAX_NFES = 500         # evaluation budget per task  (→ MAX_NFES/N_POP generations)
MAX_WORKERS = 4        # parallel processes — reduce if memory is tight

# --- ICL MAP config (locked defaults from the BO ablation) ---
LAMBDA_0     = 1.0
LAMBDA_DECAY = 0.05
TAU          = 1.0
N_CLASSES    = 2
N_ESTIMATORS = 1
DEVICE       = 'cpu'
# Space-filling LHS seed for the similarity archive (gives ρ signal from gen 0).
# Counts toward budget: ICL initial evals = N_POP + LHS_ARCHIVE per task.
LHS_ARCHIVE     = 200
RHO_ARCHIVE_CAP = 300

DATA_PATH    = './Data_MFEA_30D'
RESULTS_PATH = './Results_MFEA_30D'

ALGO_ORDER = ['MFEA-II', 'ICL-MFEA-II']

# =============================================================================
# Entry point — required on macOS/Windows (spawn-based multiprocessing)
# =============================================================================

if __name__ == '__main__':
    print('\n========== CEC17-MTSO 30D : MFEA-II vs ICL-MFEA-II ==========')

    batch_exp = BatchExperiment(base_path=DATA_PATH, clear_folder=False)

    # --- Problems: CEC17-MTSO 30D P1–P9 ---
    benchmark = CEC17MTSO_30D()
    for prob_name in ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9']:
        batch_exp.add_problem(getattr(benchmark, prob_name), prob_name)

    # --- Algorithms ---
    batch_exp.add_algorithm(MFEA_II, 'MFEA-II',
        n=N_POP, max_nfes=MAX_NFES, disable_tqdm=True)

    batch_exp.add_algorithm(ICL_MFEA_II, 'ICL-MFEA-II',
        n=N_POP, max_nfes=MAX_NFES,
        lambda_0=LAMBDA_0, lambda_decay=LAMBDA_DECAY, tau=TAU,
        n_classes=N_CLASSES, n_estimators=N_ESTIMATORS, device=DEVICE,
        lhs_archive=LHS_ARCHIVE, rho_archive_cap=RHO_ARCHIVE_CAP,
        disable_tqdm=True)

    # --- Run (parallel across workers) ---
    batch_exp.run(n_runs=N_RUNS, verbose=True, max_workers=MAX_WORKERS)

    # --- Results Analysis ---
    analyzer = DataAnalyzer(
        data_path=DATA_PATH,
        save_path=RESULTS_PATH,
        algorithm_order=ALGO_ORDER,
        figure_format='png',
        log_scale=False,
        show_std_band=True,
        std_scale=0.5,
        best_so_far=True,
        clear_results=True,
    )
    analyzer.run()
