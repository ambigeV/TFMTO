"""Real-World Optimization (RWO) Problems.

This module contains real-world optimization problem benchmarks:
- PEPVM: Photovoltaic Model Parameter Extraction (MTSO)
- SOPM: Synchronous Optimal Pulse-width Modulation (MTMO, constrained)
- NN_Training: Neural Network Weight Optimization via Neuroevolution (STSO)
- TSP: Traveling Salesman Problem via Random Keys Encoding (STSO)
- SCP: Sensor Coverage Problem (MTSO)
- MO_SCP: Multi-Objective Sensor Coverage Problem (MTMO)
- PKACP: Planar Kinematic Arm Control Problem (MTSO)
- PINN_HPO: Physics-Informed Neural Network Hyperparameter Optimization (MTSO)
"""

from ddmtolab.Problems.RWO.pepvm import PEPVM
from ddmtolab.Problems.RWO.sopm import SOPM
from ddmtolab.Problems.RWO.nn_training import NN_Training
from ddmtolab.Problems.RWO.tsp import TSP
from ddmtolab.Problems.RWO.scp import SCP
from ddmtolab.Problems.RWO.mo_scp import MO_SCP
from ddmtolab.Problems.RWO.pkacp import PKACP
from ddmtolab.Problems.RWO.pinn_hpo import PINN_HPO

__all__ = ["PEPVM", "SOPM", "NN_Training", "TSP", "SCP", "MO_SCP", "PKACP", "PINN_HPO"]
