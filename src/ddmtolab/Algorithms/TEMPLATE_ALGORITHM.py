"""
Algorithm Template for DDMTOLab
===============================

This is a template for creating new optimization algorithms that will be
automatically detected by the UI. Follow this pattern to ensure your
algorithm appears in the UI.

Usage:
------
1. Copy this file to the appropriate category folder:
   - STSO/ - Single-Task Single-Objective algorithms
   - STMO/ - Single-Task Multi-Objective algorithms
   - MTSO/ - Multi-Task Single-Objective algorithms
   - MTMO/ - Multi-Task Multi-Objective algorithms

2. Rename the file and class (e.g., my_algo.py, MyAlgo)
   - File name and class name should match (my_algo.py -> MyAlgo or MY_ALGO)

3. Implement the optimize() method

4. Run the UI - your algorithm will be automatically detected!

Naming Conventions:
-------------------
- File name: snake_case or UPPER_CASE (my_algo.py or MY_ALGO.py)
- Class name: Same as file name (MyAlgo or MY_ALGO)
- Special characters: Use underscore in file, hyphen in display
  - CMA_ES.py -> displays as "CMA-ES"
  - NSGA_II.py -> displays as "NSGA-II"
  - MOEA_D.py -> displays as "MOEA/D"

Required Methods:
-----------------
- __init__(): Initialize with problem and parameters
- optimize(): Run the optimization and return Results object

Required Attributes:
--------------------
- algorithm_information: Dict describing algorithm capabilities
"""

import numpy as np
from ddmtolab.Methods.mtop import MTOP


class TemplateAlgorithm:
    """
    Template optimization algorithm for DDMTOLab.

    This class demonstrates the standard pattern for algorithm implementation.
    The UI will automatically detect this class and its parameters.

    Parameters
    ----------
    problem : MTOP
        The optimization problem to solve.
    n : int, default=100
        Population size.
    max_nfes : int, default=10000
        Maximum number of function evaluations.
    save_data : bool, default=False
        Whether to save optimization history.
    save_path : str, default='./'
        Directory to save data.
    name : str, default='TemplateAlgorithm'
        Name for saved files.
    disable_tqdm : bool, default=False
        Whether to disable progress bar.

    Attributes
    ----------
    algorithm_information : dict
        Describes algorithm capabilities for UI display.
    """

    # =========================================================================
    # Algorithm Information (REQUIRED for UI detection)
    # =========================================================================
    algorithm_information = {
        'n_tasks': 'single',      # 'single' or 'multi' or 'any'
        'dims': 'equal',          # 'equal' or 'unequal' (for multi-task)
        'objs': 'single',         # 'single' or 'multi'
        'n_objs': 'equal',        # 'equal' or 'unequal' (for multi-task MO)
        'cons': 'none',           # 'none' or 'supported'
        'expensive': False,       # True for surrogate-assisted algorithms
        'knowledge_transfer': False,  # True for transfer learning algorithms
    }

    def __init__(
        self,
        problem: MTOP,
        n: int = 100,
        max_nfes: int = 10000,
        # Add your algorithm-specific parameters here
        pc: float = 0.9,          # Crossover probability
        pm: float = None,         # Mutation probability (None = 1/D)
        muc: int = 20,            # Crossover distribution index
        mum: int = 20,            # Mutation distribution index
        # Standard parameters (keep these)
        save_data: bool = False,
        save_path: str = './',
        name: str = 'TemplateAlgorithm',
        disable_tqdm: bool = False,
    ):
        """Initialize the algorithm."""
        # Store problem
        self.problem = problem
        self.n_tasks = problem.n_tasks
        self.dims = problem.dims
        self.n_objs = problem.n_objs

        # Store parameters
        self.n = n
        self.max_nfes = max_nfes
        self.pc = pc
        self.pm = pm
        self.muc = muc
        self.mum = mum

        # Store save settings
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

        # Initialize tracking
        self.nfes = 0
        self.gen = 0

    def optimize(self):
        """
        Run the optimization algorithm.

        Returns
        -------
        Results
            Object containing optimization results:
            - best_decs: Best decision variables found
            - best_objs: Best objective values found
            - all_decs: History of all decision variables (if save_data=True)
            - all_objs: History of all objective values (if save_data=True)
            - runtime: Total optimization time
        """
        import time
        from tqdm import tqdm

        start_time = time.time()

        # Get problem bounds
        lb = self.problem.lower_bounds[0]
        ub = self.problem.upper_bounds[0]
        dim = self.dims[0]

        # Initialize population randomly
        pop = np.random.rand(self.n, dim) * (ub - lb) + lb

        # Evaluate initial population
        objs = self.problem.evaluate(pop, task=0)
        self.nfes += self.n

        # History tracking
        all_decs = [pop.copy()] if self.save_data else None
        all_objs = [objs.copy()] if self.save_data else None

        # Main optimization loop
        max_gen = (self.max_nfes - self.n) // self.n
        pbar = tqdm(range(max_gen), disable=self.disable_tqdm, desc=self.name)

        for gen in pbar:
            self.gen = gen

            # =================================================================
            # YOUR ALGORITHM LOGIC HERE
            # =================================================================

            # Example: Simple random search (replace with your algorithm)
            offspring = np.random.rand(self.n, dim) * (ub - lb) + lb

            # Evaluate offspring
            off_objs = self.problem.evaluate(offspring, task=0)
            self.nfes += self.n

            # Selection (example: keep better solutions)
            combined_pop = np.vstack([pop, offspring])
            combined_objs = np.vstack([objs, off_objs])

            # Sort by objective and select best n
            indices = np.argsort(combined_objs[:, 0])
            pop = combined_pop[indices[:self.n]]
            objs = combined_objs[indices[:self.n]]

            # =================================================================
            # END OF YOUR ALGORITHM LOGIC
            # =================================================================

            # Save history
            if self.save_data:
                all_decs.append(pop.copy())
                all_objs.append(objs.copy())

            # Update progress bar
            best_obj = np.min(objs)
            pbar.set_postfix({'best': f'{best_obj:.6f}', 'nfes': self.nfes})

            # Check termination
            if self.nfes >= self.max_nfes:
                break

        # Build results
        runtime = time.time() - start_time
        best_idx = np.argmin(objs[:, 0])

        results = Results(
            best_decs=[pop[best_idx:best_idx+1]],
            best_objs=[objs[best_idx:best_idx+1]],
            all_decs=[all_decs] if self.save_data else None,
            all_objs=[all_objs] if self.save_data else None,
            runtime=runtime,
        )

        # Save to file if requested
        if self.save_data:
            results.save(self.save_path, self.name)

        return results


class Results:
    """
    Container for optimization results.

    Attributes
    ----------
    best_decs : List[np.ndarray]
        Best decision variables for each task.
    best_objs : List[np.ndarray]
        Best objective values for each task.
    all_decs : List[List[np.ndarray]], optional
        History of decision variables per generation per task.
    all_objs : List[List[np.ndarray]], optional
        History of objective values per generation per task.
    runtime : float
        Total optimization time in seconds.
    """

    def __init__(self, best_decs, best_objs, all_decs=None, all_objs=None, runtime=0.0):
        self.best_decs = best_decs
        self.best_objs = best_objs
        self.all_decs = all_decs
        self.all_objs = all_objs
        self.runtime = runtime

    def save(self, path: str, name: str):
        """Save results to pickle file."""
        import pickle
        import os

        os.makedirs(path, exist_ok=True)
        filepath = os.path.join(path, f'{name}.pkl')

        with open(filepath, 'wb') as f:
            pickle.dump({
                'best_decs': self.best_decs,
                'best_objs': self.best_objs,
                'all_decs': self.all_decs,
                'all_objs': self.all_objs,
                'runtime': self.runtime,
            }, f)


# =============================================================================
# Multi-Task Algorithm Template
# =============================================================================

class TemplateMultiTaskAlgorithm:
    """
    Template for multi-task optimization algorithms.

    For MTSO/MTMO algorithms that handle multiple tasks with knowledge transfer.
    """

    algorithm_information = {
        'n_tasks': 'multi',       # Handles multiple tasks
        'dims': 'equal',          # 'equal' or 'unequal' dimensions across tasks
        'objs': 'single',         # 'single' or 'multi'
        'n_objs': 'equal',
        'cons': 'none',
        'expensive': False,
        'knowledge_transfer': True,  # Uses transfer learning
    }

    def __init__(
        self,
        problem: MTOP,
        n: int = 100,
        max_nfes: int = 10000,
        rmp: float = 0.3,         # Random mating probability (key MT parameter)
        save_data: bool = False,
        save_path: str = './',
        name: str = 'TemplateMultiTask',
        disable_tqdm: bool = False,
    ):
        self.problem = problem
        self.n_tasks = problem.n_tasks
        self.dims = problem.dims
        self.n_objs = problem.n_objs

        self.n = n
        self.max_nfes = max_nfes
        self.rmp = rmp

        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def optimize(self):
        """Run multi-task optimization."""
        # Implementation for multi-task algorithm
        # Handle multiple tasks, knowledge transfer, etc.
        pass


# =============================================================================
# Surrogate-Assisted Algorithm Template
# =============================================================================

class TemplateSurrogateAlgorithm:
    """
    Template for surrogate-assisted (expensive) optimization algorithms.

    For algorithms that use surrogate models (GP, RBF, etc.) to reduce
    function evaluations.
    """

    algorithm_information = {
        'n_tasks': 'single',
        'dims': 'equal',
        'objs': 'single',
        'n_objs': 'equal',
        'cons': 'none',
        'expensive': True,        # Uses surrogate model
        'knowledge_transfer': False,
    }

    def __init__(
        self,
        problem: MTOP,
        n: int = 100,
        n_initial: int = 50,      # Initial samples for surrogate
        max_nfes: int = 200,      # Typically small for expensive problems
        mode: str = 'ei',         # Acquisition function: 'ei', 'lcb', 'pi'
        save_data: bool = False,
        save_path: str = './',
        name: str = 'TemplateSurrogate',
        disable_tqdm: bool = False,
    ):
        self.problem = problem
        self.n = n
        self.n_initial = n_initial
        self.max_nfes = max_nfes
        self.mode = mode

        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def optimize(self):
        """Run surrogate-assisted optimization."""
        # Implementation using GP/RBF surrogate
        pass
