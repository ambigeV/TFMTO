"""
Demo 6: Multi-Task Multi-Objective Optimization (Single Run)

This demo compares single-task NSGA-II with multi-task MO-MFEA on the
CEC17-MTMO benchmark problem P1. Demonstrates IGD metric calculation
with reference Pareto fronts.

Key concepts:
- Multi-task multi-objective optimization
- Setting up SETTINGS for IGD metric calculation
- Reference Pareto front configuration per task
"""

from ddmtolab.Methods.test_data_analysis import TestDataAnalyzer
from ddmtolab.Methods.animation_generator import AnimationGenerator
from ddmtolab.Problems.MTMO.cec17_mtmo import CEC17MTMO, SETTINGS
from ddmtolab.Algorithms.STMO.NSGA_II import NSGA_II
from ddmtolab.Algorithms.MTMO.MO_MFEA import MO_MFEA

# =============================================================================
# Problem Setup and Optimization
# =============================================================================
# CEC17-MTMO: multi-task multi-objective benchmark suite

problem = CEC17MTMO().P1()

# Compare algorithms
# NSGA-II: single-task MOEA (solves each task independently)
# MO-MFEA: multi-task MOEA with implicit knowledge transfer
NSGA_II(problem, n=100, max_nfes=10000).optimize()
MO_MFEA(problem, n=100, max_nfes=10000).optimize()

# =============================================================================
# Settings Configuration for IGD Calculation
# =============================================================================
# 'problems' key maps each task to its problem key in SETTINGS
# P1 has 2 tasks, both use 'P1' to look up: SETTINGS['P1']['T1'], SETTINGS['P1']['T2']

settings = SETTINGS.copy()
settings['problems'] = ['P1', 'P1']  # One entry per task

# =============================================================================
# Results Analysis
# =============================================================================
# Analyze with IGD metric using reference Pareto fronts

analyzer = TestDataAnalyzer(
    data_path='./Data',
    settings=settings,            # Settings with reference PF for IGD
    save_path='./Results',
    algorithm_order=['NSGA-II', 'MO-MFEA'],
    figure_format='png',
    log_scale=False,
    show_pf=True,                 # Show reference Pareto front
    show_nd=True,                 # Show non-dominated solutions found
    best_so_far=True,
    clear_results=True
)
results = analyzer.run()

# =============================================================================
# Animation Generation
# =============================================================================
# merge=3: objective space separated by task for multi-objective visualization

generator = AnimationGenerator(
    data_path='./Data',
    save_path='./Results',
    algorithm_order=['NSGA-II', 'MO-MFEA'],
    title='NSGA-II vs MO-MFEA on CEC17-MTMO-P1',
    merge=3,
    max_nfes=10000,
    format='mp4',
    log_scale=False
)
generator.run()
