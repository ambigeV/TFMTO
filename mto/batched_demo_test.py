"""
Batched Demo: Expensive Multi-Task Single-Objective Optimization

Runs all 8 algorithms on all 9 CEC17-MTSO-10D benchmark problems for 5
independent runs using BatchExperiment for parallel execution.

Data layout (auto-managed by BatchExperiment):
    ./Data/{algo_name}/{algo_name}_{problem_name}_{run_id}.pkl

Results (mean ± 0.5*std convergence curves) are saved to ./Results/.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ddmtolab.Problems.MTSO.cec17_mtso_10d import CEC17MTSO_10D
from ddmtolab.Algorithms.STSO.GA import GA
from ddmtolab.Algorithms.STSO.BO import BO
from ddmtolab.Algorithms.STSO.BOLCB import BOLCB
from ddmtolab.Algorithms.MTSO.MTBO import MTBO
from ddmtolab.Algorithms.MTSO.BO_LCB_BCKT import BO_LCB_BCKT
from ddmtolab.Algorithms.STSO.BO_TFM import BO_TFM
from ddmtolab.Algorithms.MTSO.MTBO_TFM_Uniform import MTBO_TFM_Uniform
from ddmtolab.Algorithms.MTSO.MTBO_TFM_Elite import MTBO_TFM_Elite
from ddmtolab.Methods.batch_experiment import BatchExperiment
from ddmtolab.Methods.data_analysis import DataAnalyzer

# =============================================================================
# Configuration
# =============================================================================

N_RUNS = 5
N_INITIAL = 20
MAX_NFES = 100
BETA = 1.0
N_ESTIMATORS = 4
MAX_WORKERS = 4          # parallel processes — reduce if memory is tight

ALGO_ORDER = ['GA', 'BO', 'BO-LCB', 'MTBO', 'BO-LCB-BCKT',
              'BO-TFM', 'MTBO-TFM-Uni', 'MTBO-TFM-Elite']

DATA_PATH = './Data'
RESULTS_PATH = './Results'

# =============================================================================
# Batch Experiment Setup
# =============================================================================

batch_exp = BatchExperiment(base_path=DATA_PATH, clear_folder=False)

# --- Problems ---
benchmark = CEC17MTSO_10D()
for prob_name in ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9']:
    batch_exp.add_problem(getattr(benchmark, prob_name), prob_name)

# --- Algorithms ---
batch_exp.add_algorithm(GA, 'GA',
    n=N_INITIAL, max_nfes=MAX_NFES, disable_tqdm=True)

batch_exp.add_algorithm(BO, 'BO',
    n_initial=N_INITIAL, max_nfes=MAX_NFES, disable_tqdm=True)

batch_exp.add_algorithm(BOLCB, 'BO-LCB',
    n_initial=N_INITIAL, max_nfes=MAX_NFES, beta=BETA, disable_tqdm=True)

batch_exp.add_algorithm(MTBO, 'MTBO',
    n_initial=N_INITIAL, max_nfes=MAX_NFES, disable_tqdm=True)

batch_exp.add_algorithm(BO_LCB_BCKT, 'BO-LCB-BCKT',
    n_initial=N_INITIAL, max_nfes=MAX_NFES, disable_tqdm=True)

batch_exp.add_algorithm(BO_TFM, 'BO-TFM',
    n_initial=N_INITIAL, max_nfes=MAX_NFES, beta=BETA,
    n_estimators=N_ESTIMATORS, disable_tqdm=True)

batch_exp.add_algorithm(MTBO_TFM_Uniform, 'MTBO-TFM-Uni',
    n_initial=N_INITIAL, max_nfes=MAX_NFES, beta=BETA,
    n_estimators=N_ESTIMATORS, disable_tqdm=True)

batch_exp.add_algorithm(MTBO_TFM_Elite, 'MTBO-TFM-Elite',
    n_initial=N_INITIAL, max_nfes=MAX_NFES, beta=BETA,
    n_estimators=N_ESTIMATORS, disable_tqdm=True)

# =============================================================================
# Run (parallel across workers)
# =============================================================================

batch_exp.run(n_runs=N_RUNS, verbose=True, max_workers=MAX_WORKERS)

# =============================================================================
# Results Analysis (all problems in one pass)
# =============================================================================

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
