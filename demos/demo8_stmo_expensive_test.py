"""
Demo 8: Expensive Single-Task Multi-Objective Optimization (Single Run)

This demo compares surrogate-assisted MOEAs on expensive multi-objective
problems using the DTLZ2 benchmark. Suitable for scenarios where function
evaluations are computationally expensive.

Key concepts:
- Expensive multi-objective optimization
- Surrogate-assisted algorithms: ParEGO, K-RVEA
- DTLZ benchmark suite with configurable objectives
"""

from ddmtolab.Methods.test_data_analysis import TestDataAnalyzer
from ddmtolab.Methods.animation_generator import AnimationGenerator
from ddmtolab.Problems.STMO.DTLZ import DTLZ, SETTINGS
from ddmtolab.Algorithms.STMO.NSGA_II import NSGA_II
from ddmtolab.Algorithms.STMO.ParEGO import ParEGO
from ddmtolab.Algorithms.STMO.K_RVEA import K_RVEA

# =============================================================================
# Problem Setup and Optimization
# =============================================================================
# DTLZ2: scalable multi-objective benchmark
# M=4: 4 objectives, D=10: 10 decision variables

problem = DTLZ().DTLZ2(M=4, D=10)

# Compare algorithms with limited budget (200 evaluations)
# NSGA-II: standard MOEA (baseline)
# ParEGO: single-objective scalarization with GP surrogate
# K-RVEA: Kriging-assisted RVEA
NSGA_II(problem, n=20, max_nfes=200).optimize()
ParEGO(problem, n_initial=50, max_nfes=200, disable_tqdm=False).optimize()
K_RVEA(problem, n_initial=50, max_nfes=200, disable_tqdm=False).optimize()

# =============================================================================
# Settings Configuration for IGD Calculation
# =============================================================================

settings = SETTINGS.copy()
settings['problems'] = ['DTLZ2']  # One entry per task

# =============================================================================
# Results Analysis
# =============================================================================
# Analyze with IGD metric using DTLZ2 reference Pareto front

analyzer = TestDataAnalyzer(
    data_path='./Data',
    settings=settings,
    save_path='./Results',
    algorithm_order=['NSGA-II', 'ParEGO', 'K-RVEA'],
    figure_format='png',
    log_scale=False,
    show_pf=True,                 # Show reference Pareto front
    show_nd=True,
    best_so_far=True,
    clear_results=True
)
results = analyzer.run()

# =============================================================================
# Animation Generation
# =============================================================================

generator = AnimationGenerator(
    data_path='./Data',
    save_path='./Results',
    algorithm_order=['NSGA-II', 'ParEGO', 'K-RVEA'],
    title='NSGA-II, ParEGO, and K-RVEA on DTLZ2',
    merge=3,
    max_nfes=200,
    format='mp4',
    log_scale=False
)
generator.run()
