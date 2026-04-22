"""
Batched Demo: Expensive Multi-Task Single-Objective Optimization

Runs all algorithms on all 9 CEC17-MTSO-50D benchmark problems for 5
independent runs using BatchExperiment for parallel execution.

Data layout (auto-managed by BatchExperiment):
    ./Data_CEC17MTSO_50D/{algo_name}/{algo_name}_{problem_name}_{run_id}.pkl

Results (mean ± 0.5*std convergence curves) are saved to ./Results_CEC17MTSO_50D/.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ddmtolab.Problems.MTSO.cec17_mtso import CEC17MTSO
from ddmtolab.Problems.MTSO.cec17_mtso_10d import CEC17MTSO_10D
# from ddmtolab.Algorithms.STSO.GA import GA
# from ddmtolab.Algorithms.STSO.BO import BO
# from ddmtolab.Algorithms.STSO.BOLCB import BOLCB
# from ddmtolab.Algorithms.MTSO.MTBO import MTBO
# from ddmtolab.Algorithms.MTSO.BO_LCB_BCKT import BO_LCB_BCKT
# from ddmtolab.Algorithms.STSO.BO_TFM import BO_TFM
# from ddmtolab.Algorithms.MTSO.MTBO_TFM_Uniform import MTBO_TFM_Uniform
# from ddmtolab.Algorithms.MTSO.MTBO_TFM_Elite import MTBO_TFM_Elite
# from ddmtolab.Algorithms.MTSO.MTBO_TFM_Distill import MTBO_TFM_Distill
# from ddmtolab.Algorithms.STSO.BO_TFM_GPEmbed import BO_TFM_GPEmbed
from ddmtolab.Algorithms.STSO.BO_TFM_ResGP import BO_TFM_ResGP
from ddmtolab.Algorithms.MTSO.MTBO_TFM_Covar_Asym import MTBO_TFM_Covar_Asym
from ddmtolab.Algorithms.MTSO.MTBO_TFM_Covar_Cls import MTBO_TFM_Covar_Cls
from ddmtolab.Algorithms.MTSO.MTBO_TFM_Covar_Cls_Ranked import MTBO_TFM_Covar_Cls_Ranked
# from ddmtolab.Algorithms.MTSO.MTBO_TFM_Uniform_B import MTBO_TFM_Uniform_B
# from ddmtolab.Algorithms.MTSO.MTBO_TFM_Elite_B import MTBO_TFM_Elite_B
from ddmtolab.Algorithms.MTSO.MTBO_TFM_MAP_Sym import MTBO_TFM_MAP_Sym
from ddmtolab.Algorithms.MTSO.MTBO_TFM_MAP_Asym import MTBO_TFM_MAP_Asym
from ddmtolab.Methods.batch_experiment import BatchExperiment
from ddmtolab.Methods.data_analysis import DataAnalyzer

# =============================================================================
# Configuration
# =============================================================================

# --- Dimension switch: set to 10 for 10D benchmark, 50 for 50D benchmark ---
DIM = 50

N_RUNS = 5
N_INITIAL = 20
MAX_NFES = 100
BETA = 1.0          # for GP baselines (BOLCB, BO-LCB-BCKT)
TFM_BETA = 2.5      # for all TabPFN-based algorithms
N_ESTIMATORS = 1
N_CANDIDATES = 2000
CMAES_POPSIZE = 40
CMAES_MAXITER = 50
MAX_WORKERS = 4          # parallel processes — reduce if memory is tight

ALGO_ORDER = ['BO-TFM-ResGP',
              'MTBO-TFM-Covar-Asym', 'MTBO-TFM-Covar-Cls', 'MTBO-TFM-Covar-Cls-Ranked',
              'MTBO-TFM-MAP-Sym', 'MTBO-TFM-MAP-Asym']

DATA_PATH    = f'./Data_CEC17MTSO_{DIM}D'
RESULTS_PATH = f'./Results_CEC17MTSO_{DIM}D'

# =============================================================================
# Entry point — required on macOS/Windows (spawn-based multiprocessing)
# =============================================================================

if __name__ == '__main__':
    # -------------------------------------------------------------------------
    # Batch Experiment Setup
    # -------------------------------------------------------------------------
    batch_exp = BatchExperiment(base_path=DATA_PATH, clear_folder=False)

    # --- Problems ---
    benchmark = CEC17MTSO() if DIM == 50 else CEC17MTSO_10D()
    for prob_name in ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9']:
        batch_exp.add_problem(getattr(benchmark, prob_name), prob_name)

    # --- Algorithms ---
    # batch_exp.add_algorithm(GA, 'GA',
    #     n=N_INITIAL, max_nfes=MAX_NFES, disable_tqdm=True)

    # batch_exp.add_algorithm(BO, 'BO',
    #     n_initial=N_INITIAL, max_nfes=MAX_NFES, disable_tqdm=True)

    # batch_exp.add_algorithm(BOLCB, 'BO-LCB',
    #     n_initial=N_INITIAL, max_nfes=MAX_NFES, beta=BETA, disable_tqdm=True)

    # batch_exp.add_algorithm(MTBO, 'MTBO',
    #     n_initial=N_INITIAL, max_nfes=MAX_NFES, disable_tqdm=True)

    # batch_exp.add_algorithm(BO_LCB_BCKT, 'BO-LCB-BCKT',
    #     n_initial=N_INITIAL, max_nfes=MAX_NFES, disable_tqdm=True)

    # batch_exp.add_algorithm(BO_TFM, 'BO-TFM',
    #     n_initial=N_INITIAL, max_nfes=MAX_NFES, beta=TFM_BETA,
    #     n_estimators=N_ESTIMATORS, n_candidates=N_CANDIDATES, disable_tqdm=True)

    # batch_exp.add_algorithm(MTBO_TFM_Uniform, 'MTBO-TFM-Uni',
    #     n_initial=N_INITIAL, max_nfes=MAX_NFES, beta=TFM_BETA,
    #     n_estimators=N_ESTIMATORS, n_candidates=N_CANDIDATES, disable_tqdm=True)

    # batch_exp.add_algorithm(MTBO_TFM_Elite, 'MTBO-TFM-Elite',
    #     n_initial=N_INITIAL, max_nfes=MAX_NFES, beta=TFM_BETA,
    #     n_estimators=N_ESTIMATORS, n_candidates=N_CANDIDATES, disable_tqdm=True)

    # --- Distill variants (warm-started, plain MLP, NLL loss) ---
    # batch_exp.add_algorithm(MTBO_TFM_Distill, 'MTBO-TFM-Uni-Distill',
    #     n_initial=N_INITIAL, max_nfes=MAX_NFES, beta=TFM_BETA,
    #     n_estimators=N_ESTIMATORS,
    #     transfer='uniform', encoding='scalar',
    #     mlp_loss='nll', distill_model='mlp',
    #     warm_start=True, disable_tqdm=True)

    # batch_exp.add_algorithm(MTBO_TFM_Distill, 'MTBO-TFM-Elite-Distill',
    #     n_initial=N_INITIAL, max_nfes=MAX_NFES, beta=TFM_BETA,
    #     n_estimators=N_ESTIMATORS,
    #     transfer='elite', encoding='scalar',
    #     mlp_loss='nll', distill_model='mlp',
    #     warm_start=True, disable_tqdm=True)

    # --- CMA-ES acquisition variants ---
    # batch_exp.add_algorithm(MTBO_TFM_Uniform, 'MTBO-TFM-Uni-CMA',
    #     n_initial=N_INITIAL, max_nfes=MAX_NFES, beta=TFM_BETA,
    #     n_estimators=N_ESTIMATORS, acq_optimizer='cmaes',
    #     cmaes_popsize=CMAES_POPSIZE, cmaes_maxiter=CMAES_MAXITER, disable_tqdm=True)

    # batch_exp.add_algorithm(MTBO_TFM_Elite, 'MTBO-TFM-Elite-CMA',
    #     n_initial=N_INITIAL, max_nfes=MAX_NFES, beta=TFM_BETA,
    #     n_estimators=N_ESTIMATORS, acq_optimizer='cmaes',
    #     cmaes_popsize=CMAES_POPSIZE, cmaes_maxiter=CMAES_MAXITER, disable_tqdm=True)

    # batch_exp.add_algorithm(BO_TFM_GPEmbed, 'BO-TFM-GPEmbed',
    #     n_initial=N_INITIAL, max_nfes=MAX_NFES, beta=TFM_BETA,
    #     n_estimators=N_ESTIMATORS, disable_tqdm=True)

    batch_exp.add_algorithm(BO_TFM_ResGP, 'BO-TFM-ResGP',
        n_initial=N_INITIAL, max_nfes=MAX_NFES, beta=TFM_BETA,
        n_estimators=N_ESTIMATORS, disable_tqdm=True)

    # --- Fixed-covar variants (TabPFN-derived R frozen during MLL) ---
    batch_exp.add_algorithm(MTBO_TFM_Covar_Asym, 'MTBO-TFM-Covar-Asym',
        n_initial=N_INITIAL, max_nfes=MAX_NFES,
        n_estimators=N_ESTIMATORS, disable_tqdm=True)

    batch_exp.add_algorithm(MTBO_TFM_Covar_Cls, 'MTBO-TFM-Covar-Cls',
        n_initial=N_INITIAL, max_nfes=MAX_NFES,
        n_estimators=N_ESTIMATORS, disable_tqdm=True)

    batch_exp.add_algorithm(MTBO_TFM_Covar_Cls_Ranked, 'MTBO-TFM-Covar-Cls-Ranked',
        n_initial=N_INITIAL, max_nfes=MAX_NFES,
        n_estimators=N_ESTIMATORS, disable_tqdm=True)

    # --- Mean-only acquisition variants (no LCB uncertainty term) ---
    # batch_exp.add_algorithm(MTBO_TFM_Uniform_B, 'MTBO-TFM-Uni-B',
    #     n_initial=N_INITIAL, max_nfes=MAX_NFES,
    #     n_estimators=N_ESTIMATORS, n_candidates=N_CANDIDATES, disable_tqdm=True)

    # batch_exp.add_algorithm(MTBO_TFM_Elite_B, 'MTBO-TFM-Elite-B',
    #     n_initial=N_INITIAL, max_nfes=MAX_NFES,
    #     n_estimators=N_ESTIMATORS, n_candidates=N_CANDIDATES, disable_tqdm=True)

    # --- MAP-regularised variants (TFM Cls prior → decaying λ → MLL) ---
    batch_exp.add_algorithm(MTBO_TFM_MAP_Sym, 'MTBO-TFM-MAP-Sym',
        n_initial=N_INITIAL, max_nfes=MAX_NFES,
        n_estimators=N_ESTIMATORS, disable_tqdm=True)

    batch_exp.add_algorithm(MTBO_TFM_MAP_Asym, 'MTBO-TFM-MAP-Asym',
        n_initial=N_INITIAL, max_nfes=MAX_NFES,
        n_estimators=N_ESTIMATORS, disable_tqdm=True)

    # -------------------------------------------------------------------------
    # Run (parallel across workers)
    # -------------------------------------------------------------------------
    batch_exp.run(n_runs=N_RUNS, verbose=True, max_workers=MAX_WORKERS)

    # -------------------------------------------------------------------------
    # Results Analysis (all problems in one pass)
    # -------------------------------------------------------------------------
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
