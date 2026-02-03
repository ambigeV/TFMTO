"""
Demo 7: Multi-Task Multi-Objective Optimization (Batch Experiment)

This demo runs batch experiments comparing single-task and multi-task
MOEAs on the CEC17-MTMO benchmark suite. Generates statistical analysis
with IGD metric.

Key concepts:
- Batch experiments for MTMO problems
- SETTINGS configuration for benchmark IGD calculation
- Comparing NSGA-II, RVEA, and MTEA-D-DN algorithms
"""

from ddmtolab.Methods.data_analysis import DataAnalyzer
from ddmtolab.Methods.batch_experiment import BatchExperiment
from ddmtolab.Problems.MTMO.cec17_mtmo import CEC17MTMO, SETTINGS
from ddmtolab.Algorithms.STMO.NSGA_II import NSGA_II
from ddmtolab.Algorithms.STMO.RVEA import RVEA
from ddmtolab.Algorithms.MTMO.MTEA_D_DN import MTEA_D_DN


if __name__ == '__main__':
    # =========================================================================
    # Experiment Setup
    # =========================================================================

    batch_exp = BatchExperiment(base_path='./Data', clear_folder=True)

    # Add benchmark problems from CEC17-MTMO
    problems = CEC17MTMO()
    batch_exp.add_problem(problems.P1, 'P1')
    batch_exp.add_problem(problems.P2, 'P2')
    batch_exp.add_problem(problems.P8, 'P8')

    # Add algorithms to compare
    # NSGA-II: classic Pareto-based MOEA
    # RVEA: reference vector guided EA
    # MTEA-D-DN: multi-task MOEA with decomposition and dynamic neighbor
    batch_exp.add_algorithm(NSGA_II, 'NSGA-II', n=100, max_nfes=10000)
    batch_exp.add_algorithm(RVEA, 'RVEA', n=100, max_nfes=10000)
    batch_exp.add_algorithm(MTEA_D_DN, 'MTEA-D-DN', n=100, max_nfes=10000)

    # =========================================================================
    # Run Experiments
    # =========================================================================

    batch_exp.run(n_runs=5, verbose=True, max_workers=5)

    # =========================================================================
    # Statistical Analysis
    # =========================================================================
    # SETTINGS contains reference Pareto fronts for IGD calculation

    analyzer = DataAnalyzer(
        data_path='./Data',
        settings=SETTINGS,               # Reference PF settings from benchmark
        algorithm_order=['NSGA-II','RVEA','MTEA-D-DN'],
        save_path='./Results',
        table_format='latex',
        figure_format='pdf',
        statistic_type='mean',
        rank_sum_test=True,
        log_scale=True,
        show_pf=False,                   # PF not shown in batch convergence
        show_nd=True,
        merge_plots=True,
        merge_columns=4,
        best_so_far=True,
        clear_results=True,
        convergence_k=10
    )

    results = analyzer.run()
