# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DDMTOLab (Data-Driven Multitask Optimization Laboratory) is a Python platform for optimization research featuring 60+ algorithms and 180+ benchmark problems. The library supports single/multi-task and single/multi-objective optimization with surrogate-assisted methods.

**Requirements:** Python 3.10+, PyTorch 2.5+, NumPy 2.0+

## Build and Development Commands

```bash
# Install in development mode
pip install -e .

# Install with dev dependencies (pytest, black, flake8, sphinx)
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run single test with verbose output
pytest tests/test.py -v

# Run tests with coverage
pytest --cov=src/ddmtolab tests/

# Format code
black src/ddmtolab/

# Lint code
flake8 src/ddmtolab/

# Build documentation
cd docs && make html

# Build package
python -m build
```

## Architecture

The codebase follows a src-layout with `src/ddmtolab/` as the main package:

### Core Modules

- **`Algorithms/`** - 60+ optimization algorithms organized by category:
  - `STSO/` - Single-Task Single-Objective (GA, DE, PSO, BO, CMA-ES, etc.)
  - `STMO/` - Single-Task Multi-Objective (NSGA-II, NSGA-III, MOEA/D, ParEGO, etc.)
  - `MTSO/` - Multi-Task Single-Objective (MFEA, MTBO, MTEA-AD, etc.)
  - `MTMO/` - Multi-Task Multi-Objective (MO-MFEA, MTDE-MKTA, etc.)

- **`Problems/`** - 180+ benchmark problems:
  - `STSO/` - Classical functions, CEC10-CSO
  - `STMO/` - ZDT, DTLZ, WFG, UF, CF, MW test suites
  - `MTSO/` - CEC17-MTSO, CEC19-MaTSO, CMT, STOP
  - `MTMO/` - CEC17-MTMO, CEC19-MTMO, CEC21-MTMO
  - `RWO/` - Real-world optimization problems
  - `BasicFunctions/` - Base function implementations

- **`Methods/`** - Core utilities:
  - `mtop.py` - MTOP class for problem definition
  - `batch_experiment.py` - Parallel experiment framework
  - `data_analysis.py` - Statistical analysis for batch experiments
  - `test_data_analysis.py` - Single experiment analysis
  - `metrics.py` - Performance metrics (IGD, HV, etc.)
  - `animation_generator.py` - Optimization visualization
  - `Algo_Methods/` - Reusable algorithm components (BO utilities, operators, etc.)

### Key Patterns

**Algorithm Implementation**: All algorithms follow a consistent pattern:
- `algorithm_information` dict describing capabilities
- Constructor accepts `problem`, parameters, and save options
- `.optimize()` method returns a `Results` object with `best_decs`, `best_objs`, histories, runtime

**Problem Definition**: Use `MTOP` class to define problems:
```python
from ddmtolab.Methods.mtop import MTOP
problem = MTOP()
problem.add_task(objective_fn, dim=D)
```

**Batch Experiments**: For running multiple algorithms on multiple problems:
```python
from ddmtolab.Methods.batch_experiment import BatchExperiment
from ddmtolab.Methods.data_analysis import DataAnalyzer

batch_exp = BatchExperiment(base_path='./Data', clear_folder=True)
batch_exp.add_problem(problem, 'name')
batch_exp.add_algorithm(AlgorithmClass, 'name', **params)
batch_exp.run(n_runs=20, max_workers=8)
DataAnalyzer().run()
```

## Code Style

- NumPy-style docstrings
- Type hints throughout
- Black for formatting
- Flake8 for linting

## Algorithm Development Reference

When implementing or modifying optimization algorithms, **MUST read** the following documentation:

- **`.claude/skills/algo_utils/SKILL.md`** - Comprehensive API reference for `algo_utils` module
- **`.claude/skills/experiment_stso/SKILL.md`** - Workflow for STSO batch experiments with statistical analysis

**Critical import statement:**
```python
from ddmtolab.Methods.Algo_Methods.algo_utils import *
```

This provides all essential utilities: `initialization`, `evaluation`, `ga_generation`, `de_generation`, `nd_sort`, `crowding_distance`, `tournament_selection`, `init_history`, `append_history`, `build_save_results`, etc.

**Key conventions:**
- Decision variables are scaled to [0, 1] internally
- Multi-task data uses `List[np.ndarray]` structure
- History tracking: `all_decs[task_idx][generation]` → `np.ndarray`

## Experimental Execution Notes (Windows)

### Required UTF-8 Encoding

Windows defaults to GBK encoding, causing Unicode character errors. Always run this before scripts:

```bash
set PYTHONIOENCODING=utf-8
python D:/DDMTOLab/tests/experiment_test1.py
```

### Path Specifications

Use **forward slashes (/) or double backslashes (\\)**:
```python
# Correct
script_path = "D:/DDMTOLab/tests/experiment_test1.py"
script_path = "D:\\DDMTOLab\\tests\\experiment_test1.py"
```

**Correct way to execute scripts:**
```bash
python D:/DDMTOLab/tests/experiment_test1.py
```



