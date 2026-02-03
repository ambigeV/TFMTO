.. _api:

API Reference
=============

This page provides API documentation for the main classes and methods in DDMTOLab.

Algorithms
----------

All algorithms follow a consistent interface:

.. code-block:: python

   from ddmtolab.Algorithms.STSO.DE import DE
   from ddmtolab.Methods.mtop import MTOP

   problem = MTOP()
   problem.add_task(objective_func, dim=10)

   optimizer = DE(problem, n=50, max_nfes=1000)
   results = optimizer.optimize()

   print(results.best_decs, results.best_objs)

Single-Task Single-Objective (STSO)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ddmtolab.Algorithms.STSO.DE.DE
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. autoclass:: ddmtolab.Algorithms.STSO.BO.BO
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. autoclass:: ddmtolab.Algorithms.STSO.PSO.PSO
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. autoclass:: ddmtolab.Algorithms.STSO.GA.GA
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

Multi-Task Single-Objective (MTSO)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ddmtolab.Algorithms.MTSO.MFEA.MFEA
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. autoclass:: ddmtolab.Algorithms.MTSO.MTBO.MTBO
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. autoclass:: ddmtolab.Algorithms.MTSO.EMEA.EMEA
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

Single-Task Multi-Objective (STMO)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ddmtolab.Algorithms.STMO.NSGA_II.NSGA_II
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. autoclass:: ddmtolab.Algorithms.STMO.MOEA_D.MOEA_D
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. autoclass:: ddmtolab.Algorithms.STMO.RVEA.RVEA
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. autoclass:: ddmtolab.Algorithms.STMO.IBEA.IBEA
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

Multi-Task Multi-Objective (MTMO)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ddmtolab.Algorithms.MTMO.MO_MFEA.MO_MFEA
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. autoclass:: ddmtolab.Algorithms.MTMO.MO_MFEA_II.MO_MFEA_II
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

Problems
--------

MTOP Class
~~~~~~~~~~

The **MTOP (Multi-Task Optimization Problem)** class is the core component for defining optimization problems.

.. autoclass:: ddmtolab.Methods.mtop.MTOP
   :members: add_task, add_tasks, evaluate_task, evaluate_tasks, get_task_info, set_unified_eval_mode
   :undoc-members:
   :show-inheritance:

Benchmark Problem Suites
~~~~~~~~~~~~~~~~~~~~~~~~

**STSO Problems:**

.. autoclass:: ddmtolab.Problems.STSO.classical_so.CLASSICALSO
   :members:
   :undoc-members:

**STMO Problems:**

.. autoclass:: ddmtolab.Problems.STMO.ZDT.ZDT
   :members:
   :undoc-members:

.. autoclass:: ddmtolab.Problems.STMO.DTLZ.DTLZ
   :members:
   :undoc-members:

**MTSO Problems:**

.. autoclass:: ddmtolab.Problems.MTSO.cec17_mtso.CEC17MTSO
   :members:
   :undoc-members:

**MTMO Problems:**

.. autoclass:: ddmtolab.Problems.MTMO.cec17_mtmo.CEC17MTMO
   :members:
   :undoc-members:

Methods and Utilities
---------------------

Batch Experiment
~~~~~~~~~~~~~~~~

.. autoclass:: ddmtolab.Methods.batch_experiment.BatchExperiment
   :members: add_problem, add_algorithm, run
   :undoc-members:
   :show-inheritance:

Data Analysis
~~~~~~~~~~~~~

.. autoclass:: ddmtolab.Methods.data_analysis.DataAnalyzer
   :members: run
   :undoc-members:

.. autoclass:: ddmtolab.Methods.test_data_analysis.TestDataAnalyzer
   :members: run
   :undoc-members:

Performance Metrics
~~~~~~~~~~~~~~~~~~~

.. automodule:: ddmtolab.Methods.metrics
   :members: IGD, GD, IGDp, HV, DeltaP, Spacing, Spread
   :undoc-members:

Algorithm Utilities
~~~~~~~~~~~~~~~~~~~

.. automodule:: ddmtolab.Methods.Algo_Methods.algo_utils
   :members: Results, initialization, evaluation, nd_sort, crowding_distance, tournament_selection, ga_generation, de_generation
   :undoc-members:

Bayesian Optimization Utilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: ddmtolab.Methods.Algo_Methods.bo_utils
   :members:
   :undoc-members:

Uniform Point Generation
~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: ddmtolab.Methods.Algo_Methods.uniform_point.uniform_point
