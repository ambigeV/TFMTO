"""
Demo 1: Custom Multi-Task Single-Objective Optimization Problem

This demo shows how to define custom objective functions and build a multi-task
problem from scratch. It compares single-task GA with multi-task MFEA algorithm.

Key concepts:
- Defining objective functions with normalized input [0,1]
- Building MTOP with tasks of different dimensions
- Running single-run experiments with analysis and animation
"""

import numpy as np
from ddmtolab.Methods.mtop import MTOP
from ddmtolab.Methods.test_data_analysis import TestDataAnalyzer
from ddmtolab.Methods.animation_generator import AnimationGenerator
from ddmtolab.Algorithms.STSO.GA import GA
from ddmtolab.Algorithms.MTSO.MFEA import MFEA


# =============================================================================
# Objective Function Definitions
# =============================================================================
# Note: Input x is normalized to [0,1], shape: (n_samples, dim)
# Functions should return shape: (n_samples, 1)

def sphere(x):
    """Sphere function: f(x) = sum(x^2), global minimum at origin."""
    x_scaled = x * 10 - 5  # Scale from [0,1] to [-5,5]
    return np.sum(x_scaled ** 2, axis=1, keepdims=True)


def rastrigin(x):
    """Rastrigin function: highly multimodal, global minimum at origin."""
    x_scaled = x * 10 - 5  # Scale from [0,1] to [-5,5]
    D = x.shape[1]
    return (10 * D + np.sum(x_scaled ** 2 - 10 * np.cos(2 * np.pi * x_scaled), axis=1, keepdims=True))


def rosenbrock(x):
    """Rosenbrock function: valley-shaped, global minimum at (1,1,...,1)."""
    x_scaled = x * 4 - 2  # Scale from [0,1] to [-2,2]
    result = np.zeros((x.shape[0], 1))
    for i in range(x.shape[1] - 1):
        result += 100 * (x_scaled[:, i+1:i+2] - x_scaled[:, i:i+1] ** 2) ** 2 + (x_scaled[:, i:i+1] - 1) ** 2
    return result


# =============================================================================
# Problem Definition
# =============================================================================
# Build multi-task problem with 3 tasks of different dimensions

problem = MTOP()
problem.add_task(sphere, dim=10)      # Task 1: 10D Sphere
problem.add_task(rastrigin, dim=20)   # Task 2: 20D Rastrigin
problem.add_task(rosenbrock, dim=30)  # Task 3: 30D Rosenbrock

# =============================================================================
# Run Optimization
# =============================================================================
# GA: solves each task independently
# MFEA: exploits inter-task similarities via implicit knowledge transfer

GA(problem, n=100, max_nfes=10000).optimize()
MFEA(problem, n=100, max_nfes=10000).optimize()

# =============================================================================
# Results Analysis
# =============================================================================
# Generate convergence curves and runtime comparison plots

analyzer = TestDataAnalyzer(
    data_path='./Data',           # Directory containing optimization results
    save_path='./Results',        # Output directory for analysis plots
    algorithm_order=['GA', 'MFEA'],
    figure_format='png',
    log_scale=True,               # Log scale for better visualization of convergence
    best_so_far=True,             # Plot best fitness found so far
    clear_results=True            # Clear previous results before saving
)
results = analyzer.run()

# =============================================================================
# Animation Generation
# =============================================================================
# Visualize optimization process over generations
# merge=1: decision space separated by task, objective space merged

generator = AnimationGenerator(
    data_path='./Data',
    save_path='./Results',
    algorithm_order=['GA', 'MFEA'],
    title='GA vs MFEA on Custom MTSO',
    merge=1,                      # Merge mode for visualization layout
    max_nfes=10000,
    format='mp4',
    log_scale=True
)
generator.run()
