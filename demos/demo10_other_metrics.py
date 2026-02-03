"""
Demo 10: Using Hypervolume (HV) Metric for Multi-Objective Optimization

This demo shows how to configure the analysis to use Hypervolume (HV) instead
of IGD as the performance metric. HV requires reference points rather than
reference Pareto fronts.

Key concepts:
- Configuring HV metric with reference points
- Custom settings dictionary structure
- Reference point specification per task
"""

from ddmtolab.Methods.test_data_analysis import TestDataAnalyzer
from ddmtolab.Methods.animation_generator import AnimationGenerator
from ddmtolab.Problems.MTMO.cec17_mtmo import CEC17MTMO
from ddmtolab.Algorithms.STMO.NSGA_II import NSGA_II
from ddmtolab.Algorithms.MTMO.MTEA_D_DN import MTEA_D_DN

# =============================================================================
# Problem Setup and Optimization
# =============================================================================

problem = CEC17MTMO().P1()

# Run algorithms with extended budget
NSGA_II(problem, n=100, max_nfes=20000).optimize()
MTEA_D_DN(problem, n=100, max_nfes=20000).optimize()

# =============================================================================
# Settings Configuration for HV Metric
# =============================================================================
# HV requires reference points (upper bounds) instead of reference Pareto fronts
# Reference points should dominate all possible solutions
# Format: {problem_key: {task_key: [ref_point_per_objective]}}

settings = {
    'metric': 'HV',                      # Use Hypervolume instead of IGD
    'problems': ['P1', 'P1'],            # Map each task to problem key
    'P1': {
        'T1': [3, 3],                    # Reference point for Task 1 (2 objectives)
        'T2': [4, 4.5]                   # Reference point for Task 2 (2 objectives)
    }
}

# =============================================================================
# Results Analysis
# =============================================================================
# Analyze with HV metric (higher is better, unlike IGD)

analyzer = TestDataAnalyzer(
    data_path='./Data',
    settings=settings,
    save_path='./Results',
    algorithm_order=['NSGA-II', 'MTEA-D-DN'],
    figure_format='png',
    log_scale=False,
    show_pf=False,                       # No PF for HV visualization
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
    algorithm_order=['NSGA-II', 'MTEA-D-DN'],
    title='NSGA-II and MTEA-D-DN on CEC17-MTMO-P1',
    merge=3,
    max_nfes=20000,
    format='mp4',
    log_scale=False
)
generator.run()
