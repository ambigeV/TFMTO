"""
Demo 9: Expensive Single-Task Multi-Objective Optimization (Batch Experiment)

This demo runs batch experiments comparing surrogate-assisted MOEAs on
expensive multi-objective problems from the DTLZ benchmark suite.

Key concepts:
- Batch experiments for expensive STMO problems
- Comparing ParEGO, K-RVEA, and DSAEA-PS algorithms
- Statistical analysis with IGD metric

Note: Batch experiment code is commented out for reference; only analysis runs.
"""

from ddmtolab.Methods.data_analysis import DataAnalyzer
from ddmtolab.Methods.batch_experiment import BatchExperiment
from ddmtolab.Problems.STMO.DTLZ import DTLZ, SETTINGS
from ddmtolab.Algorithms.STMO.NSGA_II import NSGA_II
from ddmtolab.Algorithms.STMO.ParEGO import ParEGO
from ddmtolab.Algorithms.STMO.K_RVEA import K_RVEA
from ddmtolab.Algorithms.STMO.DSAEA_PS import DSAEA_PS

if __name__ == '__main__':
    # =========================================================================
    # Experiment Setup (Commented - for reference)
    # =========================================================================
    # Uncomment below to run batch experiments

    batch_exp = BatchExperiment(base_path='./Data', clear_folder=True)

    # Add DTLZ problems with 3 objectives
    problems = DTLZ()
    batch_exp.add_problem(problems.DTLZ1, 'DTLZ1', M=3, D=10)
    batch_exp.add_problem(problems.DTLZ2, 'DTLZ2', M=3, D=10)

    # Add surrogate-assisted algorithms
    # NSGA-II: baseline MOEA
    # ParEGO: efficient global optimization for MOO
    # K-RVEA: Kriging-assisted reference vector EA
    # DSAEA-PS: data-driven surrogate-assisted EA with Pareto selection
    batch_exp.add_algorithm(NSGA_II, 'NSGA-II', n=10, max_nfes=200)
    batch_exp.add_algorithm(ParEGO, 'ParEGO', n_initial=100, max_nfes=200, disable_tqdm=False)
    batch_exp.add_algorithm(K_RVEA, 'K-RVEA', n_initial=100, max_nfes=200, disable_tqdm=False)
    batch_exp.add_algorithm(DSAEA_PS, 'DSAEA-PS', n_initial=100, max_nfes=200, disable_tqdm=False)

    batch_exp.run(n_runs=3, verbose=True, max_workers=6)

    # =========================================================================
    # Statistical Analysis
    # =========================================================================
    # Analyze existing results with DTLZ reference Pareto fronts

    analyzer = DataAnalyzer(
        data_path='./Data',
        settings=SETTINGS,               # DTLZ reference PF settings
        algorithm_order=['NSGA-II','ParEGO','K-RVEA', 'DSAEA-PS'],
        save_path='./Results',
        table_format='latex',
        figure_format='pdf',
        statistic_type='mean',
        rank_sum_test=True,
        log_scale=True,
        show_pf=False,
        show_nd=True,
        merge_plots=True,
        merge_columns=2,                 # 2 columns for 2 problems
        best_so_far=True,
        clear_results=True,
        convergence_k=10
    )

    results = analyzer.run()
