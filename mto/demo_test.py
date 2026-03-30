"""
Demo: Expensive Multi-Task Single-Objective Optimization (5 Independent Runs)

Compares surrogate-assisted algorithms on all 9 CEC17-MTSO-10D benchmark problems.
Runs each algorithm 5 times independently per problem. Results (mean ± 0.5*std
convergence curves) are saved in a separate folder per problem.
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
from ddmtolab.Methods.data_analysis import DataAnalyzer

# =============================================================================
# Configuration
# =============================================================================

N_RUNS = 5
N_INITIAL = 20
MAX_NFES = 100
BETA = 1.0
N_ESTIMATORS = 4

ALGO_ORDER = ['GA', 'BO', 'BO-LCB', 'MTBO', 'BO-LCB-BCKT',
              'BO-TFM', 'MTBO-TFM-Uni', 'MTBO-TFM-Elite']

benchmark = CEC17MTSO_10D()
PROBLEMS = {
    'P1': benchmark.P1,
    'P2': benchmark.P2,
    'P3': benchmark.P3,
    'P4': benchmark.P4,
    'P5': benchmark.P5,
    'P6': benchmark.P6,
    'P7': benchmark.P7,
    'P8': benchmark.P8,
    'P9': benchmark.P9,
}

# =============================================================================
# Run Experiments
# =============================================================================

for prob_name, prob_fn in PROBLEMS.items():
    print(f"\n{'='*60}")
    print(f"Problem: CEC17-MTSO-10D-{prob_name}")
    print(f"{'='*60}")

    for run_id in range(1, N_RUNS + 1):
        print(f"\n--- Run {run_id}/{N_RUNS} ---")
        problem = prob_fn()

        def data_path(algo_name):
            path = f'./Data/{prob_name}/{algo_name}'
            os.makedirs(path, exist_ok=True)
            return path

        def run_name(algo_name):
            return f'{algo_name}_{prob_name}_{run_id}'

        GA(problem, n=10, max_nfes=MAX_NFES,
           save_path=data_path('GA'), name=run_name('GA')).optimize()

        BO(problem, n_initial=N_INITIAL, max_nfes=MAX_NFES,
           save_path=data_path('BO'), name=run_name('BO')).optimize()

        BOLCB(problem, n_initial=N_INITIAL, max_nfes=MAX_NFES, beta=BETA,
              save_path=data_path('BO-LCB'), name=run_name('BO-LCB')).optimize()

        MTBO(problem, n_initial=N_INITIAL, max_nfes=MAX_NFES,
             save_path=data_path('MTBO'), name=run_name('MTBO')).optimize()

        BO_LCB_BCKT(problem, n_initial=N_INITIAL, max_nfes=MAX_NFES,
                    save_path=data_path('BO-LCB-BCKT'), name=run_name('BO-LCB-BCKT')).optimize()

        BO_TFM(problem, n_initial=N_INITIAL, max_nfes=MAX_NFES, beta=BETA,
               n_estimators=N_ESTIMATORS,
               save_path=data_path('BO-TFM'), name=run_name('BO-TFM')).optimize()

        MTBO_TFM_Uniform(problem, n_initial=N_INITIAL, max_nfes=MAX_NFES, beta=BETA,
                         n_estimators=N_ESTIMATORS,
                         save_path=data_path('MTBO-TFM-Uni'), name=run_name('MTBO-TFM-Uni')).optimize()

        MTBO_TFM_Elite(problem, n_initial=N_INITIAL, max_nfes=MAX_NFES, beta=BETA,
                       n_estimators=N_ESTIMATORS,
                       save_path=data_path('MTBO-TFM-Elite'), name=run_name('MTBO-TFM-Elite')).optimize()

# =============================================================================
# Results Analysis (per problem)
# =============================================================================

for prob_name in PROBLEMS:
    print(f"\n{'='*60}")
    print(f"Analyzing: CEC17-MTSO-10D-{prob_name}")
    print(f"{'='*60}")

    analyzer = DataAnalyzer(
        data_path=f'./Data/{prob_name}',
        save_path=f'./Results/{prob_name}',
        algorithm_order=ALGO_ORDER,
        figure_format='png',
        log_scale=False,
        show_std_band=True,
        std_scale=0.5,
        best_so_far=True,
        clear_results=True,
    )
    analyzer.run()
