.. _algorithms:

Algorithms
==========

This chapter introduces the algorithm design philosophy and construction rules in **DDMTOLab**, providing comprehensive guidance for implementing custom optimization algorithms.

Algorithm Construction
----------------------

Considering the complexity and diversity of data-driven multitask optimization, **DDMTOLab** adopts a **loosely-coupled algorithm design philosophy**. The platform does not mandate algorithms to inherit specific base classes or implement fixed interface methods, thereby avoiding restrictions on algorithm flexibility. This design approach offers the following advantages:

1. **Enhanced Platform Compatibility**: Traditional gradient-based methods, evolutionary algorithms, advanced data-driven multitask optimization algorithms, and hybrid innovative architectures can all be seamlessly integrated into the platform.

2. **Improved Development Convenience**: Users can quickly implement algorithms across the full spectrum—from inexpensive single-task single-objective unconstrained optimization to expensive multitask multiobjective constrained optimization—without understanding complex class inheritance hierarchies.

3. **Guaranteed Algorithm Freedom**: Users are free to design data structures, optimization workflows, and knowledge transfer strategies according to specific problem characteristics and algorithm mechanisms, without framework constraints.

To facilitate subsequent data processing and efficient coordination with the platform's experiment modules and data analysis modules, **DDMTOLab** imposes only **3 basic rules** on algorithm construction, ensuring normal platform functionality while maximizing algorithm development flexibility.

Rule 1: Algorithm Framework
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Algorithms must be implemented as **classes** and include the following core components:

1. **Algorithm Metadata**: Class attribute ``algorithm_information`` dictionary declaring the algorithm's basic characteristics
2. **Metadata Access Method**: Class method ``get_algorithm_information`` for retrieving and displaying algorithm metadata
3. **Initialization Method**: ``__init__`` method that must accept a ``problem`` (MTOP instance) as the first parameter
4. **Optimization Method**: ``optimize`` method that executes the optimization process and returns a ``Results`` object

**Example Structure**:

.. code-block:: python

    class AlgorithmName:
        # Component 1: Algorithm metadata (required)
        algorithm_information = {
            'n_tasks': '1-K',               # Supported task number types
            'dims': 'unequal',              # Decision variable dimension constraint
            'objs': 'unequal',              # Objective number constraint
            'n_objs': '1-M',                # Objective quantity type
            'cons': 'unequal',              # Constraint number constraint
            'n_cons': '0-C',                # Constraint quantity type
            'expensive': 'False',           # Whether expensive optimization
            'knowledge_transfer': 'False',  # Whether knowledge transfer involved
            'param': 'unequal'              # Algorithm parameter constraint
        }

        # Component 2: Metadata access method (required)
        @classmethod
        def get_algorithm_information(cls, print_info=True):
            return get_algorithm_information(cls, print_info)

        # Component 3: Initialization method (required)
        def __init__(self, problem, n=None, max_nfes=None, ...):
            self.problem = problem
            # Other parameter initialization

        # Component 4: Optimization method (required)
        def optimize(self):
            # Algorithm implementation
            return results

Rule 2: Algorithm Input
~~~~~~~~~~~~~~~~~~~~~~~~

Algorithms must accept an ``MTOP`` instance as an input parameter. The ``MTOP`` instance encapsulates complete information about the optimization problem, through which the algorithm obtains all problem information. Other parameters can be freely designed according to algorithm requirements.

**Example**:

.. code-block:: python

    def __init__(self, problem, n=None, max_nfes=None, ...):
        """
        Args:
            problem: MTOP instance (required parameter)
            n: Population size per task (custom parameter)
            ...: Other algorithm-specific parameters
        """
        self.problem = problem  # Store problem instance
        # Other parameter initialization

Rule 3: Algorithm Output
~~~~~~~~~~~~~~~~~~~~~~~~~

Algorithms must return a result object conforming to the ``Results`` dataclass specification. The ``Results`` class encapsulates complete information about the optimization process:

**Results Dataclass Definition**:

.. code-block:: python

    @dataclass
    class Results:
        """Optimization results container"""
        best_decs: List[np.ndarray]      # Best decision variables for each task
        best_objs: List[np.ndarray]      # Best objective values for each task
        all_decs: List[List[np.ndarray]] # Decision variable evolution history
        all_objs: List[List[np.ndarray]] # Objective value evolution history
        runtime: float                    # Total runtime (seconds)
        max_nfes: List[int]              # Max function evaluations per task
        best_cons: Optional[List[np.ndarray]] = None  # Best constraint values
        all_cons: Optional[List[List[np.ndarray]]] = None  # Constraint history

**Results Fields Description**

.. list-table::
   :header-rows: 1
   :widths: 20 25 55

   * - Field
     - Data Type
     - Description
   * - ``best_decs``
     - ``List[np.ndarray]``
     - **Best decision variables**. List length is the number of tasks K. ``best_decs[i]`` is the best decision variable for task i. Shape is :math:`(n, D^i)`, where n is the number of optimal solutions (n=1 for single-objective; n≥2 for multiobjective)
   * - ``best_objs``
     - ``List[np.ndarray]``
     - **Best objective values**. List length is K. ``best_objs[i]`` is the best objective value for task i. Shape is :math:`(n, M^i)`
   * - ``all_decs``
     - ``List[List[np.ndarray]]``
     - **Decision variable history**. ``all_decs[i][g]`` represents all decision variables of task i at generation g. Shape is :math:`(n, D^i)`
   * - ``all_objs``
     - ``List[List[np.ndarray]]``
     - **Objective value history**. ``all_objs[i][g]`` represents all objective values of task i at generation g. Shape is :math:`(n, M^i)`
   * - ``runtime``
     - ``float``
     - **Total runtime** (seconds). Records total time from start to end for performance evaluation
   * - ``max_nfes``
     - ``List[int]``
     - **Maximum function evaluations**. List length is K. ``max_nfes[i]`` is the maximum number of function evaluations for task i
   * - ``best_cons``
     - ``Optional[List[np.ndarray]]``
     - **Best constraint values** (optional). Used only in constrained optimization. ``best_cons[i]`` is the constraint value corresponding to the best solution of task i. Shape is :math:`(n, C^i)`. None for unconstrained problems
   * - ``all_cons``
     - ``Optional[List[List[np.ndarray]]]``
     - **Constraint evolution history** (optional). ``all_cons[i][g]`` represents all constraint values of task i at generation g. Shape is :math:`(n, C^i)`. None for unconstrained problems

The input/output structure is straightforward: **input must include an MTOP instance, and output must follow the specified data structure**.

Algorithm Metadata
------------------

Algorithms must declare their basic characteristics through the ``algorithm_information`` class attribute dictionary to facilitate algorithm management, experiment matching, and performance analysis. The key fields are described below:

**Metadata Fields**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Field
     - Description
   * - ``n_tasks``
     - Supported task numbers. ``1`` means single-task only, ``K`` means multitask only (K≥2), ``1-K`` means both single and multitask supported
   * - ``dims``
     - Decision variable dimension constraint. ``equal`` requires same dimensions across tasks, ``unequal`` supports unequal-dimension tasks
   * - ``objs``
     - Objective number constraint. ``equal`` requires same number of objectives across tasks, ``unequal`` supports unequal objective numbers
   * - ``n_objs``
     - Objective quantity type. ``1`` means single-objective only, ``M`` means multiobjective only (M≥2), ``1-M`` means both supported
   * - ``cons``
     - Constraint number constraint. ``equal`` requires same number of constraints across tasks, ``unequal`` supports unequal constraint numbers
   * - ``n_cons``
     - Constraint quantity type. ``0`` means unconstrained, ``C`` means constrained only (C≥1), ``0-C`` means both supported
   * - ``expensive``
     - Whether expensive optimization (involving surrogate models). ``True`` uses surrogate models, ``False`` does not
   * - ``knowledge_transfer``
     - Whether inter-task knowledge transfer involved. ``True`` means the algorithm includes knowledge transfer mechanisms, ``False`` means tasks are optimized independently
   * - ``param``
     - Algorithm parameter constraint. ``equal`` requires same parameters (e.g., population size, evaluation count) across tasks, ``unequal`` allows different parameters per task

**Example: GA Metadata Declaration**:

.. code-block:: python

    class GA:
        algorithm_information = {
            'n_tasks': '1-K',           # Supports single and multitask
            'dims': 'unequal',          # Supports unequal dimensions
            'objs': 'unequal',          # Supports unequal objectives
            'n_objs': '1',              # Single-objective only
            'cons': 'unequal',          # Supports unequal constraints
            'n_cons': '0',              # Unconstrained only
            'expensive': 'False',       # Not expensive (no surrogate)
            'knowledge_transfer': 'False',  # No knowledge transfer
            'n': 'unequal',             # Different population sizes
            'max_nfes': 'unequal'       # Different max evaluations
        }

        @classmethod
        def get_algorithm_information(cls, print_info=True):
            """Get and print algorithm metadata"""
            return get_algorithm_information(cls, print_info)

Viewing Algorithm Metadata
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**DDMTOLab** provides the ``get_algorithm_information`` class method for each algorithm to retrieve and display metadata:

.. code-block:: python

    from Algorithms.STSO.GA import GA

    # Call class method to view GA metadata
    GA.get_algorithm_information()

**Output**:

.. code-block:: none

    🤖️ GA
    Algorithm Information:
      - n_tasks: 1-K
      - dims: unequal
      - objs: unequal
      - n_objs: 1
      - cons: unequal
      - n_cons: 0
      - expensive: False
      - knowledge_transfer: False
      - param: unequal

This method prints the algorithm name and all metadata fields in a structured format, helping users quickly understand the algorithm's scope and characteristic constraints. By viewing the metadata, users can determine whether an algorithm is suitable for their optimization problem.

The method also supports returning metadata as a dictionary for programmatic processing:

.. code-block:: python

    from Algorithms.STSO.GA import GA
    info = GA.get_algorithm_information(print_info=False)
    print(info)

**Output**:

.. code-block:: python

    {'n_tasks': '1-K', 'dims': 'unequal', 'objs': 'unequal', 'n_objs': '1',
     'cons': 'unequal', 'n_cons': '0', 'expensive': 'False',
     'knowledge_transfer': 'False', 'n': 'unequal', 'max_nfes': 'unequal'}

Using Algorithms
----------------

Basic Usage
~~~~~~~~~~~

**Example: Single-Task Optimization**:

.. code-block:: python

    from ddmtolab.Methods.mtop import MTOP
    from ddmtolab.Algorithms.STSO.GA import GA
    import numpy as np

    # Define objective function
    def sphere(x):
        return np.sum(x**2, axis=1)

    # Create problem instance using MTOP
    problem = MTOP()
    problem.add_task(sphere, dim=30)

    # Initialize algorithm
    algorithm = GA(
        problem=problem,
        n=100,             # Population size
        max_nfes=10000,    # Max function evaluations
        pc=0.9,            # Crossover probability
        pm=0.1             # Mutation probability
    )

    # Run optimization
    results = algorithm.optimize()

    # Access results
    print(f"Best objective: {results.best_objs[0]}")
    print(f"Runtime: {results.runtime:.2f}s")

**Example: Multitask Optimization**:

.. code-block:: python

    from ddmtolab.Methods.mtop import MTOP
    from ddmtolab.Algorithms.MTSO.MFEA import MFEA
    import numpy as np

    # Define objective functions
    def sphere(x):
        return np.sum(x**2, axis=1)

    def rosenbrock(x):
        return np.sum(100*(x[:, 1:] - x[:, :-1]**2)**2 + (1 - x[:, :-1])**2, axis=1)

    def rastrigin(x):
        return 10*x.shape[1] + np.sum(x**2 - 10*np.cos(2*np.pi*x), axis=1)

    # Create multitask problem using MTOP
    problem = MTOP()
    problem.add_task(sphere, dim=30)
    problem.add_task(rosenbrock, dim=30)
    problem.add_task(rastrigin, dim=30)

    # Initialize MFEA
    algorithm = MFEA(
        problem=problem,
        n=100,
        max_nfes=10000,
        rmp=0.3  # Random mating probability
    )

    # Run optimization
    results = algorithm.optimize()

    # Compare task performance
    for i in range(problem.n_tasks):
        print(f"Task {i+1} best: {results.best_objs[i]}")

Advanced Configuration
~~~~~~~~~~~~~~~~~~~~~~

**Custom Parameter Settings**:

.. code-block:: python

    # Configure algorithm with custom parameters
    algorithm = GA(
        problem=problem,
        n=[200],              # Larger population
        max_nfes=[50000],     # More evaluations
        pc=0.85,              # Custom crossover rate
        pm=0.15,              # Custom mutation rate
        selection='tournament',  # Selection method
        tournament_size=3     # Tournament size
    )

**Accessing Optimization History**:

.. code-block:: python

    results = algorithm.optimize()

    # Get evolution trajectory for task 0
    obj_history = results.all_objs[0]

    # Plot convergence curve
    import matplotlib.pyplot as plt

    best_per_gen = [min(gen_objs) for gen_objs in obj_history]
    plt.plot(best_per_gen)
    plt.xlabel('Generation')
    plt.ylabel('Best Objective Value')
    plt.title('Convergence Curve')
    plt.show()

Implementing Custom Algorithms
-------------------------------

You can easily implement custom algorithms by following the three construction rules:

**Example: Simple Custom Algorithm**:

.. code-block:: python

    import numpy as np
    import time
    from ddmtolab.Methods.Algo_Methods.algo_utils import (
        Results, get_algorithm_information,
        initialization, evaluation,
        init_history, append_history, build_save_results
    )

    class MyCustomAlgorithm:
        # Rule 1: Algorithm metadata
        algorithm_information = {
            'n_tasks': '1',
            'dims': 'unequal',
            'objs': 'unequal',
            'n_objs': '1',
            'cons': 'unequal',
            'n_cons': '0',
            'expensive': 'False',
            'knowledge_transfer': 'False',
            'param': 'unequal'
        }

        @classmethod
        def get_algorithm_information(cls, print_info=True):
            return get_algorithm_information(cls, print_info)

        # Rule 2: Accept MTOP instance
        def __init__(self, problem, n=100, max_nfes=10000,
                     save_data=True, save_path='./Data', name='MyAlgo'):
            self.problem = problem
            self.n = n
            self.max_nfes = max_nfes
            self.save_data = save_data
            self.save_path = save_path
            self.name = name

        # Rule 3: Return Results object
        def optimize(self):
            start_time = time.time()

            # Initialize population using algo_utils
            decs = initialization(self.problem, self.n)
            objs, cons = evaluation(self.problem, decs)

            # Initialize history tracking
            all_decs, all_objs, all_cons = init_history(decs, objs, cons)

            # Main optimization loop
            nfes = self.n
            while nfes < self.max_nfes:
                # Your optimization logic here
                # Generate new solutions, evaluate, select...

                # Track history
                append_history(all_decs, all_objs, all_cons, decs, objs, cons)
                nfes += self.n

            runtime = time.time() - start_time

            # Build and save results using utility function
            return build_save_results(
                problem=self.problem,
                all_decs=all_decs,
                all_objs=all_objs,
                all_cons=all_cons,
                runtime=runtime,
                max_nfes=self.max_nfes,
                save_data=self.save_data,
                save_path=self.save_path,
                name=self.name
            )

Available Algorithms
--------------------

DDMTOLab provides 90+ optimization algorithms organized into four categories:

STSO (Single-Task Single-Objective)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Classical evolutionary algorithms and surrogate-assisted methods for single-objective optimization.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Algorithm
     - Description
   * - ``GA``
     - Genetic Algorithm
   * - ``DE``
     - Differential Evolution
   * - ``PSO``
     - Particle Swarm Optimization
   * - ``CMA_ES``
     - Covariance Matrix Adaptation Evolution Strategy
   * - ``IPOP_CMA_ES``
     - CMA-ES with Increasing Population
   * - ``sep_CMA_ES``
     - Separable CMA-ES
   * - ``MA_ES``
     - Matrix Adaptation Evolution Strategy
   * - ``OpenAI_ES``
     - OpenAI Evolution Strategy
   * - ``xNES``
     - Exponential Natural Evolution Strategy
   * - ``CSO``
     - Competitive Swarm Optimizer
   * - ``SL_PSO``
     - Social Learning PSO
   * - ``KLPSO``
     - Knowledge Learning PSO
   * - ``SHPSO``
     - Self-adaptive Hierarchical PSO
   * - ``GWO``
     - Grey Wolf Optimizer
   * - ``AO``
     - Aquila Optimizer
   * - ``EO``
     - Equilibrium Optimizer
   * - ``GL_SADE``
     - Gaussian Local Search with Self-adaptive DE
   * - ``SA_COSO``
     - Surrogate-Assisted Competitive Swarm Optimizer
   * - ``BO``
     - Bayesian Optimization
   * - ``EEI_BO``
     - Expected Exploration Improvement BO
   * - ``ESAO``
     - Efficient Surrogate-Assisted Optimization
   * - ``TLRBF``
     - Two-Layer RBF Surrogate-Assisted Optimization
   * - ``AutoSAEA``
     - Surrogate-Assisted EA with Auto-Configuration
   * - ``DDEA_MESS``
     - Data-Driven EA with Multi-Evolutionary Sampling Strategy
   * - ``LSADE``
     - Lipschitz Surrogate-Assisted Differential Evolution

STMO (Single-Task Multiobjective)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Multiobjective evolutionary algorithms and surrogate-assisted methods.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Algorithm
     - Description
   * - ``NSGA_II``
     - Non-dominated Sorting Genetic Algorithm II
   * - ``NSGA_III``
     - Non-dominated Sorting Genetic Algorithm III
   * - ``NSGA_II_SDR``
     - NSGA-II with Stochastic Dominance Ranking
   * - ``MOEA_D``
     - Multiobjective Evolutionary Algorithm based on Decomposition
   * - ``MOEA_DD``
     - MOEA/D with Diversity Enhancement
   * - ``MOEA_D_STM``
     - MOEA/D with Stable Matching
   * - ``MOEA_D_FRRMAB``
     - MOEA/D with Fitness-Rate-Rank Multi-Armed Bandit
   * - ``MCEA_D``
     - Multi-Criteria Evolutionary Algorithm based on Decomposition
   * - ``RVEA``
     - Reference Vector Guided Evolutionary Algorithm
   * - ``K_RVEA``
     - Kriging-assisted RVEA
   * - ``IBEA``
     - Indicator-Based Evolutionary Algorithm
   * - ``SPEA2``
     - Strength Pareto Evolutionary Algorithm 2
   * - ``TwoArch2``
     - Two-Archive Algorithm 2
   * - ``CCMO``
     - Coevolutionary Constrained Multiobjective Optimization
   * - ``C_TAEA``
     - Constrained Two-Archive Evolutionary Algorithm
   * - ``CPS_MOEA``
     - Constrained Push and Search MOEA
   * - ``KTA2``
     - Kriging-assisted Two-Archive Algorithm
   * - ``ParEGO``
     - Pareto Efficient Global Optimization
   * - ``MSEA``
     - Multi-Surrogate Evolutionary Algorithm
   * - ``REMO``
     - Reference-based Multiobjective Optimization
   * - ``DSAEA_PS``
     - Data-driven Surrogate-Assisted EA with Pareto Selection
   * - ``ADSAPSO``
     - Adaptive Dropout Surrogate-Assisted PSO
   * - ``CSEA``
     - Classification-based Surrogate-assisted EA
   * - ``DISK``
     - Distribution-Informed Surrogate-assisted Kriging
   * - ``DRLSAEA``
     - Deep Reinforcement Learning Surrogate-Assisted EA
   * - ``DirHV_EI``
     - Direction-based Hypervolume Expected Improvement
   * - ``EDN_ARMOEA``
     - Efficient Dropout Neural Network based AR-MOEA
   * - ``EIM_EGO``
     - Expected Improvement Matrix based EGO
   * - ``EM_SAEA``
     - Ensemble Model Surrogate-Assisted EA
   * - ``KTS``
     - Kriging-Assisted Two-Archive Search
   * - ``MGSAEA``
     - Multigranularity Surrogate-Assisted EA
   * - ``MMRAEA``
     - Multi-Model Ranking Aggregation EA
   * - ``MOEA_D_EGO``
     - MOEA/D with Efficient Global Optimization
   * - ``MultiObjectiveEGO``
     - Multiobjective Efficient Global Optimization
   * - ``PCSAEA``
     - Pairwise Comparison Surrogate-Assisted EA
   * - ``PEA``
     - Pareto-based Efficient Algorithm
   * - ``PIEA``
     - Performance Indicator-based EA
   * - ``SAEA_DBLL``
     - Surrogate-Assisted EA with Direction-Based Local Learning
   * - ``SSDE``
     - Self-Organizing Surrogate-Assisted Non-Dominated Sorting DE
   * - ``TEA``
     - Two-phase EA with Probabilistic Dominance

MTSO (Multitask Single-Objective)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Multitask evolutionary algorithms with knowledge transfer for single-objective optimization.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Algorithm
     - Description
   * - ``MFEA``
     - Multi-Factorial Evolutionary Algorithm
   * - ``MFEA_II``
     - Multi-Factorial Evolutionary Algorithm II
   * - ``G_MFEA``
     - Generalized MFEA
   * - ``MTEA_AD``
     - Multitask Evolutionary Algorithm with Adaptive Distribution
   * - ``MTEA_SaO``
     - Multitask EA with Surrogate-assisted Optimization
   * - ``EMEA``
     - Evolutionary Multitask Evolutionary Algorithm
   * - ``MKTDE``
     - Multi-Knowledge Transfer Differential Evolution
   * - ``SREMTO``
     - Self-Regulated Evolutionary Multitask Optimization
   * - ``RAMTEA``
     - Resource Allocation Multitask Evolutionary Algorithm
   * - ``SELF``
     - Self-adaptive Evolutionary Learning Framework
   * - ``EBS``
     - Evolution by Similarity
   * - ``MTBO``
     - Multitask Bayesian Optimization
   * - ``MUMBO``
     - Multitask Multiobjective Bayesian Optimization
   * - ``LCB_EMT``
     - Lower Confidence Bound Evolutionary Multitasking
   * - ``BO_LCB_CKT``
     - BO with LCB and Curriculum Knowledge Transfer
   * - ``BO_LCB_BCKT``
     - BO with LCB and Bidirectional Curriculum Knowledge Transfer
   * - ``EEI_BO_plus``
     - Enhanced EEI-BO for Multitask Optimization

MTMO (Multitask Multiobjective)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Multitask multiobjective evolutionary algorithms with knowledge transfer.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Algorithm
     - Description
   * - ``MO_MFEA``
     - Multiobjective Multi-Factorial Evolutionary Algorithm
   * - ``MO_MFEA_II``
     - Multiobjective MFEA II
   * - ``MO_EMEA``
     - Multiobjective Evolutionary Multitask EA
   * - ``MO_MTEA_SaO``
     - Multiobjective MTEA with Surrogate-assisted Optimization
   * - ``MTEA_D_DN``
     - Multitask EA/D with Dynamic Neighborhood
   * - ``MTDE_MKTA``
     - Multitask DE with Multi-Knowledge Transfer Adaptation
   * - ``EMT_ET``
     - Evolutionary Multitasking with Explicit Transfer
   * - ``EMT_PD``
     - Evolutionary Multitasking with Probabilistic Distribution
   * - ``ParEGO_KT``
     - ParEGO with Knowledge Transfer

See Also
--------

* :ref:`api` - Complete API documentation
* :ref:`demos` - Diverse demonstrations