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

.. autoclass:: ddmtolab.Algorithms.STSO.GA.GA
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. autoclass:: ddmtolab.Algorithms.STSO.DE.DE
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

.. autoclass:: ddmtolab.Algorithms.STSO.CMA_ES.CMA_ES
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. autoclass:: ddmtolab.Algorithms.STSO.IPOP_CMA_ES.IPOP_CMA_ES
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. autoclass:: ddmtolab.Algorithms.STSO.sep_CMA_ES.sep_CMA_ES
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. autoclass:: ddmtolab.Algorithms.STSO.MA_ES.MA_ES
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. autoclass:: ddmtolab.Algorithms.STSO.OpenAI_ES.OpenAI_ES
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. autoclass:: ddmtolab.Algorithms.STSO.xNES.xNES
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. autoclass:: ddmtolab.Algorithms.STSO.CSO.CSO
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. autoclass:: ddmtolab.Algorithms.STSO.SL_PSO.SL_PSO
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. autoclass:: ddmtolab.Algorithms.STSO.KL_PSO.KL_PSO
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. autoclass:: ddmtolab.Algorithms.STSO.SHPSO.SHPSO
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. autoclass:: ddmtolab.Algorithms.STSO.GWO.GWO
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. autoclass:: ddmtolab.Algorithms.STSO.AO.AO
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. autoclass:: ddmtolab.Algorithms.STSO.EO.EO
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. autoclass:: ddmtolab.Algorithms.STSO.GL_SADE.GL_SADE
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

.. autoclass:: ddmtolab.Algorithms.STSO.EEI_BO.EEI_BO
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. autoclass:: ddmtolab.Algorithms.STSO.ESAO.ESAO
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. autoclass:: ddmtolab.Algorithms.STSO.SA_COSO.SA_COSO
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. autoclass:: ddmtolab.Algorithms.STSO.TLRBF.TLRBF
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. autoclass:: ddmtolab.Algorithms.STSO.AutoSAEA.AutoSAEA
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. autoclass:: ddmtolab.Algorithms.STSO.DDEA_MESS.DDEA_MESS
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. autoclass:: ddmtolab.Algorithms.STSO.LSADE.LSADE
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

Single-Task Multiobjective (STMO)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ddmtolab.Algorithms.STMO.NSGA_II.NSGA_II
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. autoclass:: ddmtolab.Algorithms.STMO.NSGA_III.NSGA_III
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. autoclass:: ddmtolab.Algorithms.STMO.NSGA_II_SDR.NSGA_II_SDR
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

.. autoclass:: ddmtolab.Algorithms.STMO.MOEA_DD.MOEA_DD
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. autoclass:: ddmtolab.Algorithms.STMO.MOEA_D_STM.MOEA_D_STM
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. autoclass:: ddmtolab.Algorithms.STMO.MOEA_D_FRRMAB.MOEA_D_FRRMAB
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. autoclass:: ddmtolab.Algorithms.STMO.MCEA_D.MCEA_D
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

.. autoclass:: ddmtolab.Algorithms.STMO.K_RVEA.K_RVEA
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

.. autoclass:: ddmtolab.Algorithms.STMO.SPEA2.SPEA2
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. autoclass:: ddmtolab.Algorithms.STMO.TwoArch2.TwoArch2
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. autoclass:: ddmtolab.Algorithms.STMO.CCMO.CCMO
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. autoclass:: ddmtolab.Algorithms.STMO.C_TAEA.C_TAEA
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. autoclass:: ddmtolab.Algorithms.STMO.CPS_MOEA.CPS_MOEA
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. autoclass:: ddmtolab.Algorithms.STMO.ParEGO.ParEGO
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. autoclass:: ddmtolab.Algorithms.STMO.MSEA.MSEA
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. autoclass:: ddmtolab.Algorithms.STMO.REMO.REMO
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. autoclass:: ddmtolab.Algorithms.STMO.KTA2.KTA2
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. autoclass:: ddmtolab.Algorithms.STMO.DSAEA_PS.DSAEA_PS
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. autoclass:: ddmtolab.Algorithms.STMO.ADSAPSO.ADSAPSO
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. autoclass:: ddmtolab.Algorithms.STMO.CSEA.CSEA
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. autoclass:: ddmtolab.Algorithms.STMO.DISK.DISK
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. autoclass:: ddmtolab.Algorithms.STMO.DRLSAEA.DRLSAEA
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. autoclass:: ddmtolab.Algorithms.STMO.DirHV_EI.DirHV_EI
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. autoclass:: ddmtolab.Algorithms.STMO.EDN_ARMOEA.EDN_ARMOEA
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. autoclass:: ddmtolab.Algorithms.STMO.EIM_EGO.EIM_EGO
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. autoclass:: ddmtolab.Algorithms.STMO.EM_SAEA.EM_SAEA
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. autoclass:: ddmtolab.Algorithms.STMO.KTS.KTS
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. autoclass:: ddmtolab.Algorithms.STMO.MGSAEA.MGSAEA
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. autoclass:: ddmtolab.Algorithms.STMO.MMRAEA.MMRAEA
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. autoclass:: ddmtolab.Algorithms.STMO.MOEA_D_EGO.MOEA_D_EGO
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. autoclass:: ddmtolab.Algorithms.STMO.MultiObjectiveEGO.MultiObjectiveEGO
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. autoclass:: ddmtolab.Algorithms.STMO.PCSAEA.PCSAEA
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. autoclass:: ddmtolab.Algorithms.STMO.PEA.PEA
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. autoclass:: ddmtolab.Algorithms.STMO.PIEA.PIEA
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. autoclass:: ddmtolab.Algorithms.STMO.SAEA_DBLL.SAEA_DBLL
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. autoclass:: ddmtolab.Algorithms.STMO.SSDE.SSDE
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. autoclass:: ddmtolab.Algorithms.STMO.TEA.TEA
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

Multitask Single-Objective (MTSO)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ddmtolab.Algorithms.MTSO.MFEA.MFEA
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. autoclass:: ddmtolab.Algorithms.MTSO.MFEA_II.MFEA_II
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. autoclass:: ddmtolab.Algorithms.MTSO.G_MFEA.G_MFEA
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. autoclass:: ddmtolab.Algorithms.MTSO.MTEA_AD.MTEA_AD
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. autoclass:: ddmtolab.Algorithms.MTSO.MTEA_SaO.MTEA_SaO
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

.. autoclass:: ddmtolab.Algorithms.MTSO.MKTDE.MKTDE
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. autoclass:: ddmtolab.Algorithms.MTSO.SREMTO.SREMTO
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. autoclass:: ddmtolab.Algorithms.MTSO.RAMTEA.RAMTEA
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. autoclass:: ddmtolab.Algorithms.MTSO.SELF.SELF
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. autoclass:: ddmtolab.Algorithms.MTSO.EBS.EBS
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

.. autoclass:: ddmtolab.Algorithms.MTSO.MUMBO.MUMBO
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. autoclass:: ddmtolab.Algorithms.MTSO.LCB_EMT.LCB_EMT
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. autoclass:: ddmtolab.Algorithms.MTSO.BO_LCB_CKT.BO_LCB_CKT
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. autoclass:: ddmtolab.Algorithms.MTSO.BO_LCB_BCKT.BO_LCB_BCKT
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. autoclass:: ddmtolab.Algorithms.MTSO.EEI_BO_plus.EEI_BO_plus
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

Multitask Multiobjective (MTMO)
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

.. autoclass:: ddmtolab.Algorithms.MTMO.MO_EMEA.MO_EMEA
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. autoclass:: ddmtolab.Algorithms.MTMO.MO_MTEA_SaO.MO_MTEA_SaO
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. autoclass:: ddmtolab.Algorithms.MTMO.MTEA_D_DN.MTEA_D_DN
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. autoclass:: ddmtolab.Algorithms.MTMO.MTDE_MKTA.MTDE_MKTA
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. autoclass:: ddmtolab.Algorithms.MTMO.EMT_ET.EMT_ET
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. autoclass:: ddmtolab.Algorithms.MTMO.EMT_PD.EMT_PD
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

.. autoclass:: ddmtolab.Algorithms.MTMO.ParEGO_KT.ParEGO_KT
   :members: optimize
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: algorithm_information, get_algorithm_information

Problems
--------

MTOP Class
~~~~~~~~~~

The **MTOP (Multitask Optimization Problem)** class is the core component for defining optimization problems.

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

.. autoclass:: ddmtolab.Problems.STSO.cec10_cso.CEC10_CSO
   :members:
   :undoc-members:

.. autoclass:: ddmtolab.Problems.STSO.stsotest.STSOtest
   :members:
   :undoc-members:

**STMO Problems:**

.. autoclass:: ddmtolab.Problems.STMO.ZDT.ZDT
   :members:
   :undoc-members:

.. autoclass:: ddmtolab.Problems.STMO.DTLZ.DTLZ
   :members:
   :undoc-members:

.. autoclass:: ddmtolab.Problems.STMO.WFG.WFG
   :members:
   :undoc-members:

.. autoclass:: ddmtolab.Problems.STMO.UF.UF
   :members:
   :undoc-members:

.. autoclass:: ddmtolab.Problems.STMO.CF.CF
   :members:
   :undoc-members:

.. autoclass:: ddmtolab.Problems.STMO.MW.MW
   :members:
   :undoc-members:

**MTSO Problems:**

.. autoclass:: ddmtolab.Problems.MTSO.cec17_mtso.CEC17MTSO
   :members:
   :undoc-members:

.. autoclass:: ddmtolab.Problems.MTSO.cec17_mtso_10d.CEC17MTSO_10D
   :members:
   :undoc-members:

.. autoclass:: ddmtolab.Problems.MTSO.cec19_matso.CEC19MaTSO
   :members:
   :undoc-members:

.. autoclass:: ddmtolab.Problems.MTSO.cmt.CMT
   :members:
   :undoc-members:

.. autoclass:: ddmtolab.Problems.MTSO.stop.STOP
   :members:
   :undoc-members:

**MTMO Problems:**

.. autoclass:: ddmtolab.Problems.MTMO.cec17_mtmo.CEC17MTMO
   :members:
   :undoc-members:

.. autoclass:: ddmtolab.Problems.MTMO.cec19_mtmo.CEC19MTMO
   :members:
   :undoc-members:

.. autoclass:: ddmtolab.Problems.MTMO.cec19_matmo.CEC19_MaTMO
   :members:
   :undoc-members:

.. autoclass:: ddmtolab.Problems.MTMO.cec21_mtmo.CEC21MTMO
   :members:
   :undoc-members:

.. autoclass:: ddmtolab.Problems.MTMO.mtmo_dtlz.MTMO_DTLZ
   :members:
   :undoc-members:

**Real-World Optimization (RWO) Problems:**

.. autoclass:: ddmtolab.Problems.RWO.pepvm.PEPVM
   :members:
   :undoc-members:

.. autoclass:: ddmtolab.Problems.RWO.sopm.SOPM
   :members:
   :undoc-members:

.. autoclass:: ddmtolab.Problems.RWO.nn_training.NN_Training
   :members:
   :undoc-members:

.. autoclass:: ddmtolab.Problems.RWO.tsp.TSP
   :members:
   :undoc-members:

.. autoclass:: ddmtolab.Problems.RWO.scp.SCP
   :members:
   :undoc-members:

.. autoclass:: ddmtolab.Problems.RWO.mo_scp.MO_SCP
   :members:
   :undoc-members:

.. autoclass:: ddmtolab.Problems.RWO.pkacp.PKACP
   :members:
   :undoc-members:

.. autoclass:: ddmtolab.Problems.RWO.pinn_hpo.PINN_HPO
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
   :members: IGD, GD, IGDp, HV, DeltaP, Spacing, Spread, FR, CV
   :undoc-members:

Algorithm Utilities
~~~~~~~~~~~~~~~~~~~

.. automodule:: ddmtolab.Methods.Algo_Methods.algo_utils
   :members: Results, initialization, evaluation, nd_sort, crowding_distance, tournament_selection, ga_generation, de_generation, init_history, append_history, build_save_results, rbf_build, rbf_predict, dsmerge, merge_archive, spea2_fitness, spea2_truncation, spea2_truncation_fast, reorganize_initial_data
   :undoc-members:

Bayesian Optimization Utilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: ddmtolab.Methods.Algo_Methods.bo_utils
   :members:
   :undoc-members:

Uniform Point Generation
~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: ddmtolab.Methods.Algo_Methods.uniform_point.uniform_point
