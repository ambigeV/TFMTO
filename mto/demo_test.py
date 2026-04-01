"""
Demo: Expensive Multi-Task Single-Objective Optimization (5 Independent Runs)

Compares surrogate-assisted algorithms on all 9 CEC17-MTSO-10D benchmark problems.
Runs each algorithm 5 times independently per problem. Results (mean ± 0.5*std
convergence curves) are saved in a separate folder per problem.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ddmtolab.Problems.MTSO.cec17_mtso import CEC17MTSO
from ddmtolab.Algorithms.STSO.GA import GA
from ddmtolab.Algorithms.STSO.BO import BO
from ddmtolab.Algorithms.STSO.BOLCB import BOLCB
from ddmtolab.Algorithms.MTSO.MTBO import MTBO
from ddmtolab.Algorithms.MTSO.BO_LCB_BCKT import BO_LCB_BCKT
from ddmtolab.Algorithms.STSO.BO_TFM import BO_TFM
from ddmtolab.Algorithms.MTSO.MTBO_TFM_Uniform import MTBO_TFM_Uniform
from ddmtolab.Algorithms.MTSO.MTBO_TFM_Elite import MTBO_TFM_Elite
from ddmtolab.Algorithms.MTSO.MTBO_TFM_Distill import MTBO_TFM_Distill
from ddmtolab.Methods.data_analysis import DataAnalyzer

# =============================================================================
# Configuration
# =============================================================================

N_RUNS = 5
N_INITIAL = 20
MAX_NFES = 100
BETA = 1.0
N_ESTIMATORS = 8
N_CANDIDATES = 1000
CMAES_POPSIZE = 20
CMAES_MAXITER = 50

ALGO_ORDER = ['GA', 'BO', 'BO-LCB', 'MTBO', 'BO-LCB-BCKT',
              'BO-TFM-{}'.format(N_CANDIDATES), 'MTBO-TFM-Uni-{}'.format(N_CANDIDATES),
              'MTBO-TFM-Elite-{}'.format(N_CANDIDATES),
              'MTBO-TFM-Uni-Distill', 'MTBO-TFM-Elite-Distill']

benchmark = CEC17MTSO()
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
    print(f"Problem: CEC17-MTSO-50D-{prob_name}")
    print(f"{'='*60}")

    for run_id in range(1, N_RUNS + 1):
        print(f"\n--- Run {run_id}/{N_RUNS} ---")
        problem = prob_fn()

        def data_path(algo_name):
            path = f'./Data_CEC17MTSO_50D/{prob_name}/{algo_name}'
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
               n_estimators=N_ESTIMATORS, n_candidates=N_CANDIDATES,
               save_path=data_path('BO-TFM-{}'.format(N_CANDIDATES)), name=run_name('BO-TFM-{}'.format(N_CANDIDATES))).optimize()

        MTBO_TFM_Uniform(problem, n_initial=N_INITIAL, max_nfes=MAX_NFES, beta=BETA,
                         n_estimators=N_ESTIMATORS, n_candidates=N_CANDIDATES,
                         save_path=data_path('MTBO-TFM-Uni-{}'.format(N_CANDIDATES)), name=run_name('MTBO-TFM-Uni-{}'.format(N_CANDIDATES))).optimize()

        MTBO_TFM_Elite(problem, n_initial=N_INITIAL, max_nfes=MAX_NFES, beta=BETA,
                       n_estimators=N_ESTIMATORS, n_candidates=N_CANDIDATES,
                       save_path=data_path('MTBO-TFM-Elite-{}'.format(N_CANDIDATES)), name=run_name('MTBO-TFM-Elite-{}'.format(N_CANDIDATES))).optimize()

        # ---------- Distill variants (warm-started, plain MLP, NLL loss) ----------
        MTBO_TFM_Distill(problem, n_initial=N_INITIAL, max_nfes=MAX_NFES, beta=BETA,
                         n_estimators=N_ESTIMATORS,
                         transfer='uniform', encoding='scalar',
                         mlp_loss='nll', distill_model='mlp', warm_start=True,
                         save_path=data_path('MTBO-TFM-Uni-Distill'),
                         name=run_name('MTBO-TFM-Uni-Distill')).optimize()

        MTBO_TFM_Distill(problem, n_initial=N_INITIAL, max_nfes=MAX_NFES, beta=BETA,
                         n_estimators=N_ESTIMATORS,
                         transfer='elite', encoding='scalar',
                         mlp_loss='nll', distill_model='mlp', warm_start=True,
                         save_path=data_path('MTBO-TFM-Elite-Distill'),
                         name=run_name('MTBO-TFM-Elite-Distill')).optimize()

        # # ---------- CMA-ES acquisition variants ----------
        # BO_TFM(problem, n_initial=N_INITIAL, max_nfes=MAX_NFES, beta=BETA,
        #        n_estimators=N_ESTIMATORS, acq_optimizer='cmaes',
        #        cmaes_popsize=CMAES_POPSIZE, cmaes_maxiter=CMAES_MAXITER,
        #        save_path=data_path('BO-TFM-CMA'), name=run_name('BO-TFM-CMA')).optimize()
        #
        # MTBO_TFM_Uniform(problem, n_initial=N_INITIAL, max_nfes=MAX_NFES, beta=BETA,
        #                  n_estimators=N_ESTIMATORS, acq_optimizer='cmaes',
        #                  cmaes_popsize=CMAES_POPSIZE, cmaes_maxiter=CMAES_MAXITER,
        #                  save_path=data_path('MTBO-TFM-Uni-CMA'), name=run_name('MTBO-TFM-Uni-CMA')).optimize()
        #
        # MTBO_TFM_Elite(problem, n_initial=N_INITIAL, max_nfes=MAX_NFES, beta=BETA,
        #                n_estimators=N_ESTIMATORS, acq_optimizer='cmaes',
        #                cmaes_popsize=CMAES_POPSIZE, cmaes_maxiter=CMAES_MAXITER,
        #                save_path=data_path('MTBO-TFM-Elite-CMA'), name=run_name('MTBO-TFM-Elite-CMA')).optimize()
        #
        # MTBO_TFM_Uniform_OH(problem, n_initial=N_INITIAL, max_nfes=MAX_NFES, beta=BETA,
        #                     n_estimators=N_ESTIMATORS, acq_optimizer='cmaes',
        #                     cmaes_popsize=CMAES_POPSIZE, cmaes_maxiter=CMAES_MAXITER,
        #                     save_path=data_path('MTBO-TFM-Uni-OH-CMA'), name=run_name('MTBO-TFM-Uni-OH-CMA')).optimize()
        #
        # MTBO_TFM_Elite_OH(problem, n_initial=N_INITIAL, max_nfes=MAX_NFES, beta=BETA,
        #                   n_estimators=N_ESTIMATORS, acq_optimizer='cmaes',
        #                   cmaes_popsize=CMAES_POPSIZE, cmaes_maxiter=CMAES_MAXITER,
        #                   save_path=data_path('MTBO-TFM-Elite-OH-CMA'), name=run_name('MTBO-TFM-Elite-OH-CMA')).optimize()

# =============================================================================
# Results Analysis (per problem)
# =============================================================================

for prob_name in PROBLEMS:
    print(f"\n{'='*60}")
    print(f"Analyzing: CEC17-MTSO-50D-{prob_name}")
    print(f"{'='*60}")

    analyzer = DataAnalyzer(
        data_path=f'./Data_CEC17MTSO_50D/{prob_name}',
        save_path=f'./Results_CEC17MTSO_50D/{prob_name}',
        algorithm_order=ALGO_ORDER,
        figure_format='png',
        log_scale=False,
        show_std_band=True,
        std_scale=0.5,
        best_so_far=True,
        clear_results=True,
    )
    analyzer.run()
