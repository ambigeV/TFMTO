"""
Demo 2: Multi-Task Single-Objective Optimization (Single Run)

This demo compares single-task GA with multi-task MFEA on the CEC17-MTSO
benchmark problem P1. Includes convergence analysis and optimization animation.

Key concepts:
- Loading benchmark problems from CEC17-MTSO suite
- Single-run experiment workflow
- Visualization with Pareto front and non-dominated solutions
"""

from ddmtolab.Methods.test_data_analysis import TestDataAnalyzer
from ddmtolab.Methods.animation_generator import AnimationGenerator
from ddmtolab.Problems.MTSO.cec17_mtso import CEC17MTSO
from ddmtolab.Algorithms.STSO.GA import GA
from ddmtolab.Algorithms.MTSO.MFEA import MFEA

# =============================================================================
# Problem Setup and Optimization
# =============================================================================
# CEC17-MTSO provides standardized multi-task benchmark problems

problem = CEC17MTSO().P1()  # Load problem P1 from CEC17-MTSO suite

# Run both algorithms with same budget
GA(problem, n=100, max_nfes=10000).optimize()    # Single-task baseline
MFEA(problem, n=100, max_nfes=10000).optimize()  # Multi-task algorithm

# =============================================================================
# Results Analysis
# =============================================================================
# Analyze convergence curves, runtime, and solution quality

analyzer = TestDataAnalyzer(
    data_path='./Data',
    save_path='./Results',
    algorithm_order=['GA', 'MFEA'],
    figure_format='png',
    log_scale=False,
    show_pf=True,                 # Show Pareto front reference
    show_nd=True,                 # Show non-dominated solutions
    best_so_far=True,
    clear_results=True
)
results = analyzer.run()

# =============================================================================
# Animation Generation
# =============================================================================
# Visualize optimization dynamics
# merge=2: both decision and objective spaces merged across tasks

generator = AnimationGenerator(
    data_path='./Data',
    save_path='./Results',
    algorithm_order=['GA', 'MFEA'],
    title='GA vs MFEA on CEC17-MTSO-P1',
    merge=2,
    max_nfes=10000,
    format='mp4',
    log_scale=False
)
generator.run()
