"""Real-World Optimization (RWO) Problems.

This module contains real-world optimization problem benchmarks:
- PEPVM: Photovoltaic Model Parameter Extraction (MTSO)
- SOPM: Synchronous Optimal Pulse-width Modulation (MTMO, constrained)
- NN_Training: Neural Network Weight Optimization via Neuroevolution (STSO)
- TSP: Traveling Salesman Problem via Random Keys Encoding (STSO)
"""

from ddmtolab.Problems.RWO.pepvm import PEPVM
from ddmtolab.Problems.RWO.sopm import SOPM
from ddmtolab.Problems.RWO.nn_training import NN_Training
from ddmtolab.Problems.RWO.tsp import TSP

__all__ = ["PEPVM", "SOPM", "NN_Training", "TSP"]
