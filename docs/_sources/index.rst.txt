Data-Driven Multitask Optimization Laboratory
==============================================

.. image:: _static/logo.svg
   :alt: DDMTOLab Logo
   :width: 500
   :align: center

|

.. image:: https://img.shields.io/badge/docs-latest-blue.svg
   :target: https://jiangtaoshen.github.io/DDMTOLab/
   :alt: Documentation

.. image:: https://img.shields.io/pypi/v/ddmtolab.svg
   :target: https://pypi.org/project/ddmtolab/
   :alt: PyPI version

.. image:: https://img.shields.io/pypi/dm/ddmtolab.svg
   :target: https://pypi.org/project/ddmtolab/
   :alt: Downloads

.. image:: https://img.shields.io/github/stars/JiangtaoShen/DDMTOLab?style=social
   :target: https://github.com/JiangtaoShen/DDMTOLab/stargazers
   :alt: GitHub Stars

.. image:: https://img.shields.io/badge/Python-3.10+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python Version

.. image:: https://img.shields.io/badge/license-MIT-green.svg
   :target: https://github.com/JiangtaoShen/DDMTOLab/blob/main/LICENSE
   :alt: License

----

Overview
--------

**DDMTOLab (Data-Driven Multitask Optimization Laboratory)** is a comprehensive Python platform for optimization research, featuring **60+ algorithms**, **180+ benchmark problems**, and powerful tools for problem definition, algorithm development, and performance evaluation.

Whether you're working on expensive black-box optimization, multi-objective optimization, or complex multi-task scenarios, DDMTOLab provides a flexible and extensible framework to accelerate your **research** and support real-world **applications**.

Features
--------

* **Comprehensive Algorithms** - 60+ algorithms for expensive/inexpensive, single/multi-task, single/multi-objective optimization
* **Rich Problem Suites** - 180+ benchmark problems and real-world applications
* **Data-Driven Optimization** - Surrogate modelling (GP, BO) for expensive optimization
* **Flexible Framework** - Simple API and intuitive workflow for rapid prototyping
* **Fully Extensible** - Easy to add custom algorithms and problems
* **Powerful Analysis Tools** - Built-in visualization and statistical analysis
* **Parallel Computing** - Multi-core support for batch experiments

Documentation
-------------

.. toctree::
   :maxdepth: 1

   installation
   demos
   algorithms
   methods
   problems
   api

Quick Start
-----------

Installation
^^^^^^^^^^^^

.. code-block:: bash

   pip install ddmtolab

**Requirements:** Python 3.10+, PyTorch 2.5+, NumPy 2.0+

Basic Usage
^^^^^^^^^^^

.. code-block:: python

   import numpy as np
   from ddmtolab.Methods.mtop import MTOP
   from ddmtolab.Algorithms.MTSO.MTBO import MTBO

   # Define objective function
   def forrester(x):
       return (6 * x - 2) ** 2 * np.sin(12 * x - 4)

   # Create and solve optimization problem
   problem = MTOP()
   problem.add_task(forrester, dim=1)
   results = MTBO(problem).optimize()

   print(f"Best solution: {results.best_decs}")
   print(f"Best objective: {results.best_objs}")

Batch Experiments
^^^^^^^^^^^^^^^^^

.. code-block:: python

   from ddmtolab.Methods.batch_experiment import BatchExperiment
   from ddmtolab.Methods.data_analysis import DataAnalyzer
   from ddmtolab.Algorithms.STSO.BO import BO
   from ddmtolab.Algorithms.MTSO.MTBO import MTBO
   from ddmtolab.Problems.MTSO.cec17_mtso_10d import CEC17MTSO_10D

   if __name__ == '__main__':
       batch_exp = BatchExperiment(base_path='./Data', clear_folder=True)

       prob = CEC17MTSO_10D()
       batch_exp.add_problem(prob.P1, 'P1')
       batch_exp.add_problem(prob.P2, 'P2')

       batch_exp.add_algorithm(BO, 'BO', n_initial=20, max_nfes=100)
       batch_exp.add_algorithm(MTBO, 'MTBO', n_initial=20, max_nfes=100)

       batch_exp.run(n_runs=20, max_workers=8)
       DataAnalyzer().run()

Optimization Visualization
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. image:: _static/animation.gif
   :alt: Optimization Animation
   :width: 100%
   :align: center

Key Components
--------------

Algorithms (60+)
^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Category
     - Algorithms
   * - **STSO**
     - GA, DE, PSO, SL_PSO, KL_PSO, CSO, CMA_ES, IPOP_CMA_ES, sep_CMA_ES, MA_ES, xNES, OpenAI_ES, AO, GWO, EO, BO, EEI_BO, ESAO, SHPSO, SA_COSO, TLRBF, GL_SADE
   * - **STMO**
     - NSGA_II, NSGA_III, NSGA_II_SDR, SPEA2, MOEA_D, MOEA_DD, MOEA_D_FRRMAB, MOEA_D_STM, RVEA, IBEA, TwoArch2, MSEA, C_TAEA, CCMO, MCEA_D, CPS_MOEA, ParEGO, K_RVEA, DSAEA_PS, KTA2, REMO
   * - **MTSO**
     - MFEA, MFEA_II, EMEA, EBS, G_MFEA, MTEA_AD, MKTDE, MTEA_SaO, SREMTO, LCB_EMT, MTBO, RAMTEA, SELF, EEI_BO_plus, MUMBO, BO_LCB_CKT, BO_LCB_BCKT
   * - **MTMO**
     - MO_MFEA, MO_MFEA_II, MO_EMEA, MO_MTEA_SaO, MTDE_MKTA, MTEA_D_DN, EMT_ET, EMT_PD, ParEGO_KT

Problems (180+)
^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Category
     - Problem Suites
   * - **STSO**
     - CLASSICALSO (8), CEC10_CSO (20)
   * - **STMO**
     - ZDT (6), DTLZ (9), WFG (9), UF (10), CF (10), MW (14)
   * - **MTSO**
     - CEC17_MTSO (9), CEC17_MTSO_10D (9), CEC19_MaTSO, CMT (9), STOP (12)
   * - **MTMO**
     - CEC17_MTMO (9), CEC19_MTMO (10), CEC19_MaTMO, CEC21_MTMO (10), MTMO_DTLZ
   * - **RWO**
     - PEPVM, PINN_HPO (12), SOPM, SCP, MO_SCP, PKACP

Citation
--------

If you use DDMTOLab in your research, please cite:

.. code-block:: bibtex

   @software{ddmtolab2025,
     author = {Jiangtao Shen},
     title = {DDMTOLab: A Python Platform for Data-Driven Multitask Optimization},
     year = {2025},
     url = {https://github.com/JiangtaoShen/DDMTOLab}
   }

Contact
-------

* **Author**: Jiangtao Shen
* **Email**: j.shen5@exeter.ac.uk
* **GitHub**: `JiangtaoShen/DDMTOLab <https://github.com/JiangtaoShen/DDMTOLab>`_
* **Issues**: `GitHub Issues <https://github.com/JiangtaoShen/DDMTOLab/issues>`_

License
-------

This project is licensed under the MIT License.
