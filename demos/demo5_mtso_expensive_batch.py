"""
Demo 5: Expensive Multi-Task Single-Objective Optimization (Batch Experiment)

This demo runs batch experiments comparing surrogate-assisted algorithms
on expensive multi-task problems. Includes both single-task and multi-task
Bayesian optimization variants.

Key concepts:
- Batch experiments with expensive surrogate-assisted algorithms
- Comparing BO, MTBO, and RAMTEA on limited budget problems
- Statistical analysis across multiple runs
"""

from ddmtolab.Methods.data_analysis import DataAnalyzer
from ddmtolab.Methods.batch_experiment import BatchExperiment
from ddmtolab.Problems.MTSO.cec17_mtso_10d import CEC17MTSO_10D
from ddmtolab.Algorithms.STSO.GA import GA
from ddmtolab.Algorithms.STSO.BO import BO
from ddmtolab.Algorithms.MTSO.MTBO import MTBO
from ddmtolab.Algorithms.MTSO.RAMTEA import RAMTEA


if __name__ == '__main__':
    # =========================================================================
    # Experiment Setup
    # =========================================================================

    batch_exp = BatchExperiment(base_path='./Data', clear_folder=False)

    # Add 10D problems for expensive optimization
    problems = CEC17MTSO_10D()
    batch_exp.add_problem(problems.P1, 'P1')
    batch_exp.add_problem(problems.P2, 'P2')
    batch_exp.add_problem(problems.P3, 'P3')

    # Add algorithms with limited evaluation budget
    # GA: baseline (small population due to limited budget)
    # BO: single-task Bayesian optimization
    # MTBO: multi-task Bayesian optimization
    # RAMTEA: ranking-based adaptive multi-task EA
    batch_exp.add_algorithm(GA, 'GA', n=10, max_nfes=100)
    batch_exp.add_algorithm(BO, 'BO', n_initial=20, max_nfes=100, disable_tqdm=False)
    batch_exp.add_algorithm(MTBO, 'MTBO', n_initial=20, max_nfes=100, disable_tqdm=False)
    batch_exp.add_algorithm(RAMTEA, 'RAMTEA', n_initial=20, max_nfes=100, disable_tqdm=False)

    # =========================================================================
    # Run Experiments
    # =========================================================================

    batch_exp.run(n_runs=3, verbose=True, max_workers=6)

    # =========================================================================
    # Statistical Analysis
    # =========================================================================

    analyzer = DataAnalyzer(
        data_path='./Data',
        settings=None,
        algorithm_order=['GA','BO','MTBO', 'RAMTEA'],
        save_path='./Results',
        table_format='latex',
        figure_format='pdf',
        statistic_type='mean',
        rank_sum_test=True,
        log_scale=True,
        show_pf=True,
        show_nd=True,
        merge_plots=True,
        merge_columns=4,
        best_so_far=True,
        clear_results=True,
        convergence_k=10
    )

    results = analyzer.run()
