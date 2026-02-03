"""
Demo 3: Multi-Task Single-Objective Optimization (Batch Experiment)

This demo runs batch experiments with multiple algorithms on multiple problems
from CEC17-MTSO benchmark. Generates statistical analysis with mean/std and
Wilcoxon rank-sum test.

Key concepts:
- BatchExperiment for parallel multi-run experiments
- DataAnalyzer for statistical comparison tables and plots
- LaTeX table generation for publication-ready results
"""

from ddmtolab.Methods.data_analysis import DataAnalyzer
from ddmtolab.Methods.batch_experiment import BatchExperiment
from ddmtolab.Problems.MTSO.cec17_mtso import CEC17MTSO
from ddmtolab.Algorithms.STSO.GA import GA
from ddmtolab.Algorithms.MTSO.MFEA import MFEA
from ddmtolab.Algorithms.MTSO.MTEA_AD import MTEA_AD


if __name__ == '__main__':
    # =========================================================================
    # Experiment Setup
    # =========================================================================
    # Initialize batch experiment manager
    batch_exp = BatchExperiment(base_path='./Data', clear_folder=False)

    # Add benchmark problems from CEC17-MTSO
    problems = CEC17MTSO()
    batch_exp.add_problem(problems.P1, 'P1')
    batch_exp.add_problem(problems.P2, 'P2')
    batch_exp.add_problem(problems.P3, 'P3')

    # Add algorithms to compare
    # GA: single-task genetic algorithm (baseline)
    # MFEA: multi-factorial evolutionary algorithm
    # MTEA-AD: multi-task EA with adaptive distribution
    batch_exp.add_algorithm(GA, 'GA', n=100, max_nfes=10000)
    batch_exp.add_algorithm(MFEA, 'MFEA', n=100, max_nfes=10000)
    batch_exp.add_algorithm(MTEA_AD, 'MTEA-AD', n=100, max_nfes=10000)

    # =========================================================================
    # Run Experiments
    # =========================================================================
    # n_runs: number of independent runs for statistical significance
    # max_workers: parallel processes for speedup
    batch_exp.run(n_runs=5, verbose=True, max_workers=5)

    # =========================================================================
    # Statistical Analysis
    # =========================================================================
    # Generate comparison tables (LaTeX/CSV) and convergence plots

    analyzer = DataAnalyzer(
        data_path='./Data',
        settings=None,                   # No reference PF for MTSO
        algorithm_order=['GA','MFEA','MTEA-AD'],
        save_path='./Results',
        table_format='latex',            # Publication-ready LaTeX tables
        figure_format='pdf',
        statistic_type='mean',           # Report mean values
        rank_sum_test=True,              # Wilcoxon rank-sum test
        log_scale=True,
        show_pf=True,
        show_nd=True,
        merge_plots=True,                # Combine plots for all problems
        merge_columns=3,                 # 3 columns per row in merged plot
        best_so_far=True,
        clear_results=True,
        convergence_k=10                 # Sample k points for convergence curves
    )

    results = analyzer.run()
