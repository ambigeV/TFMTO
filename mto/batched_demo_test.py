"""
Batched Demo: Expensive Multi-Task Single-Objective Optimization

Runs BO / MTBO / MTBO-TFM-MAP-Asym with logEI and LCB acquisition
functions on the 9 SepArmMTSO benchmark problems (5D / 10D / 15D).

Data layout (auto-managed by BatchExperiment):
    ./Data_SepArmMTSO/{algo_name}/{algo_name}_{problem_name}_{run_id}.pkl

Results (mean ± 0.5*std convergence curves) are saved to ./Results_SepArmMTSO/.

--- CEC17 runs (commented out, preserved for reference) ---
# DIM switch: 10 → CEC17MTSO_10D_v2, 30 → CEC17MTSO_30D, 50 → CEC17MTSO
# DATA_PATH  = f'./Data_CEC17MTSO_{DIM}D'
# RESULTS_PATH = f'./Results_CEC17MTSO_{DIM}D'
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# --- CEC17 problem imports (kept for reference) ---
# from ddmtolab.Problems.MTSO.cec17_mtso import CEC17MTSO
# from ddmtolab.Problems.MTSO.cec17_mtso_10d_v2 import CEC17MTSO_10D_v2
# from ddmtolab.Problems.MTSO.cec17_mtso_30d import CEC17MTSO_30D

from ddmtolab.Problems.RWO.sep_arm_mtso import SepArmMTSO
# from ddmtolab.Algorithms.STSO.GA import GA
from ddmtolab.Algorithms.STSO.BO import BO
# from ddmtolab.Algorithms.STSO.BOLCB import BOLCB
from ddmtolab.Algorithms.MTSO.MTBO import MTBO
# from ddmtolab.Algorithms.MTSO.BO_LCB_BCKT import BO_LCB_BCKT
# from ddmtolab.Algorithms.STSO.BO_TFM import BO_TFM
# from ddmtolab.Algorithms.MTSO.MTBO_TFM_Uniform import MTBO_TFM_Uniform
# from ddmtolab.Algorithms.MTSO.MTBO_TFM_Elite import MTBO_TFM_Elite
# from ddmtolab.Algorithms.MTSO.MTBO_TFM_Distill import MTBO_TFM_Distill
# from ddmtolab.Algorithms.STSO.BO_TFM_GPEmbed import BO_TFM_GPEmbed
# from ddmtolab.Algorithms.STSO.BO_TFM_ResGP import BO_TFM_ResGP
# from ddmtolab.Algorithms.MTSO.MTBO_TFM_Covar_Asym import MTBO_TFM_Covar_Asym
# from ddmtolab.Algorithms.MTSO.MTBO_TFM_Covar_Cls import MTBO_TFM_Covar_Cls
# from ddmtolab.Algorithms.MTSO.MTBO_TFM_Covar_Cls_Ranked import MTBO_TFM_Covar_Cls_Ranked
# from ddmtolab.Algorithms.MTSO.MTBO_TFM_Uniform_B import MTBO_TFM_Uniform_B
# from ddmtolab.Algorithms.MTSO.MTBO_TFM_Elite_B import MTBO_TFM_Elite_B
# from ddmtolab.Algorithms.MTSO.MTBO_TFM_MAP_Sym import MTBO_TFM_MAP_Sym
from ddmtolab.Algorithms.MTSO.MTBO_TFM_MAP_Asym import MTBO_TFM_MAP_Asym
from ddmtolab.Methods.batch_experiment import BatchExperiment
from ddmtolab.Methods.data_analysis import DataAnalyzer

# =============================================================================
# Configuration
# =============================================================================

N_RUNS = 5
N_INITIAL = 20
MAX_NFES = 100
TFM_BETA = 2.5         # LCB exploration weight
MAP_LBFGS_ITER = 200   # L-BFGS iterations for MAP fitting
N_ESTIMATORS = 1
MAX_WORKERS = 4        # parallel processes — reduce if memory is tight

# MAP config
MAP_LAMBDA_0     = 1.0
MAP_LAMBDA_DECAY = 0.05

# Output normalization for MTBO and MAP-Asym: 'minmax' (default) or 'zscore'
OBJ_NORM = 'minmax'

ALGO_ORDER = [
    'BO-EI', 'BO-LCB',
    'MTBO-logEI', 'MTBO-LCB',
    f'MAP-Asym-{MAP_LAMBDA_0}-{MAP_LAMBDA_DECAY}-logEI',
    f'MAP-Asym-{MAP_LAMBDA_0}-{MAP_LAMBDA_DECAY}-LCB',
]

DATA_PATH    = './Data_SepArmMTSO'
RESULTS_PATH = './Results_SepArmMTSO'

# =============================================================================
# Entry point — required on macOS/Windows (spawn-based multiprocessing)
# =============================================================================

if __name__ == '__main__':
    # -------------------------------------------------------------------------
    # Batch Experiment Setup
    # -------------------------------------------------------------------------
    batch_exp = BatchExperiment(base_path=DATA_PATH, clear_folder=False)

    # --- Problems: SepArmMTSO P1–P9 (5D / 10D / 15D × HS / MS / LS) ---
    benchmark = SepArmMTSO()
    for prob_name in ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9']:
        batch_exp.add_problem(getattr(benchmark, prob_name), prob_name)

    # --- Algorithms ---

    # BO: standard EI and LCB (single-task baselines)
    batch_exp.add_algorithm(BO, 'BO-EI',
        n_initial=N_INITIAL, max_nfes=MAX_NFES,
        mode='ei', disable_tqdm=True)

    batch_exp.add_algorithm(BO, 'BO-LCB',
        n_initial=N_INITIAL, max_nfes=MAX_NFES,
        mode='lcb', disable_tqdm=True)

    # batch_exp.add_algorithm(BO, 'BO-TS',
    #     n_initial=N_INITIAL, max_nfes=MAX_NFES,
    #     mode='ts', disable_tqdm=True)

    # MTBO: logEI and LCB
    batch_exp.add_algorithm(MTBO, 'MTBO-logEI',
        n_initial=N_INITIAL, max_nfes=MAX_NFES,
        acq_fn='logEI', obj_norm=OBJ_NORM, disable_tqdm=True)

    batch_exp.add_algorithm(MTBO, 'MTBO-LCB',
        n_initial=N_INITIAL, max_nfes=MAX_NFES,
        acq_fn='LCB', beta=TFM_BETA, obj_norm=OBJ_NORM, disable_tqdm=True)

    # batch_exp.add_algorithm(MTBO, 'MTBO-TS',
    #     n_initial=N_INITIAL, max_nfes=MAX_NFES,
    #     acq_fn='TS', obj_norm=OBJ_NORM, disable_tqdm=True)

    # MAP-Asym: logEI and LCB
    batch_exp.add_algorithm(MTBO_TFM_MAP_Asym,
        f'MAP-Asym-{MAP_LAMBDA_0}-{MAP_LAMBDA_DECAY}-logEI',
        n_initial=N_INITIAL, max_nfes=MAX_NFES,
        lambda_0=MAP_LAMBDA_0, lambda_decay=MAP_LAMBDA_DECAY,
        n_estimators=N_ESTIMATORS,
        acq_fn='logEI', obj_norm=OBJ_NORM,
        lbfgs_iter=MAP_LBFGS_ITER, disable_tqdm=True)

    batch_exp.add_algorithm(MTBO_TFM_MAP_Asym,
        f'MAP-Asym-{MAP_LAMBDA_0}-{MAP_LAMBDA_DECAY}-LCB',
        n_initial=N_INITIAL, max_nfes=MAX_NFES,
        lambda_0=MAP_LAMBDA_0, lambda_decay=MAP_LAMBDA_DECAY,
        n_estimators=N_ESTIMATORS,
        acq_fn='LCB', beta=TFM_BETA, obj_norm=OBJ_NORM,
        lbfgs_iter=MAP_LBFGS_ITER, disable_tqdm=True)

    # batch_exp.add_algorithm(MTBO_TFM_MAP_Asym,
    #     f'MAP-Asym-{MAP_LAMBDA_0}-{MAP_LAMBDA_DECAY}-TS',
    #     n_initial=N_INITIAL, max_nfes=MAX_NFES,
    #     lambda_0=MAP_LAMBDA_0, lambda_decay=MAP_LAMBDA_DECAY,
    #     n_estimators=N_ESTIMATORS,
    #     acq_fn='TS', obj_norm=OBJ_NORM,
    #     lbfgs_iter=MAP_LBFGS_ITER, disable_tqdm=True)

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
