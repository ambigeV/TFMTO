"""
Demo: Expensive Multi-Task Single-Objective Optimization (Single Run)

Compares surrogate-assisted algorithms on expensive optimization problems
where function evaluations are costly. Uses CEC17-MTSO 10D variant with
limited evaluation budget.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ddmtolab.Methods.test_data_analysis import TestDataAnalyzer
from ddmtolab.Methods.animation_generator import AnimationGenerator
from ddmtolab.Problems.MTSO.cec17_mtso_10d import CEC17MTSO_10D
from ddmtolab.Algorithms.STSO.GA import GA
from ddmtolab.Algorithms.STSO.BO import BO
from ddmtolab.Algorithms.STSO.BOLCB import BOLCB
from ddmtolab.Algorithms.MTSO.MTBO import MTBO
from ddmtolab.Algorithms.MTSO.BO_LCB_BCKT import BO_LCB_BCKT

# =============================================================================
# Problem Setup and Optimization
# =============================================================================
# Use 10D variant of CEC17-MTSO for expensive optimization scenario

problem = CEC17MTSO_10D().P1()

# Compare algorithms with limited budget (50 evaluations)
# GA: baseline genetic algorithm
# BO: single-task Bayesian optimization with EI acquisition
# BO-LCB: single-task BO with LCB acquisition (beta=1)
# BO-LCB-BCKT: multi-task BO with LCB acquisition and knowledge transfer
GA(problem, n=10, max_nfes=50).optimize()
BO(problem, n_initial=10, max_nfes=50).optimize()
BOLCB(problem, n_initial=10, max_nfes=50, beta=1.0).optimize()
MTBO(problem, n_initial=10, max_nfes=50).optimize()
BO_LCB_BCKT(problem, n_initial=10, max_nfes=50).optimize()

# =============================================================================
# Results Analysis
# =============================================================================

analyzer = TestDataAnalyzer(
    data_path='./Data',
    save_path='./Results',
    algorithm_order=['GA', 'BO', 'BO-LCB', 'MTBO', 'BO-LCB-BCKT'],
    figure_format='png',
    log_scale=False,
    show_pf=True,
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
    algorithm_order=['GA', 'BO', 'BO-LCB', 'MTBO', 'BO-LCB-BCKT'],
    title='GA, BO, BO-LCB, MTBO, and BO-LCB-BCKT on CEC17-MTSO-10D-P1',
    merge=2,
    max_nfes=50,
    format='mp4',
    log_scale=False
)
generator.run()
