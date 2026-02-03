# Algo Utils - Optimization Algorithm Utilities

## Overview

Collection of commonly used components for implementing optimization algorithms, particularly for multi-task and multi-objective optimization.

## Import Statement

**CRITICAL:** Always use this import path:

```python
from ddmtolab.Methods.Algo_Methods.algo_utils import *
```

This imports all utility functions including:
- Population initialization and evaluation
- Genetic operators (crossover, mutation, GA/DE generation)
- Multi-objective utilities (nd_sort, crowding_distance, IBEA)
- Selection operations (tournament, elite)
- Data management (history, results)
- Space transformations and normalization
- Array utilities

## Core Data Structures

### Results (dataclass)
**Purpose:** Container for optimization results

**Attributes:**
- `best_decs`: List[np.ndarray] - Best decision variables per task
- `best_objs`: List[np.ndarray] - Best objective values per task
- `all_decs`: List[List[np.ndarray]] - Decision history across generations
- `all_objs`: List[List[np.ndarray]] - Objective history across generations
- `runtime`: float - Total runtime in seconds
- `max_nfes`: List[int] - Max function evaluations per task
- `best_cons`: Optional[List[np.ndarray]] - Best constraint values per task
- `all_cons`: Optional[List[List[np.ndarray]]] - Constraint history
- `bounds`: Optional[List[np.ndarray]] - Bounds per task

## Function Reference

### Population Initialization & Evaluation

#### `initialization(problem, n, method='random', the_same=False)`
**Purpose:** Initialize decision variables for multiple tasks

**Parameters:**
- `problem`: MTOP instance
- `n`: int or List[int] - Number of samples per task
- `method`: str - 'random' or 'lhs' (Latin Hypercube Sampling)
- `the_same`: bool - Whether all tasks share same samples

**Returns:**
- `decs`: List[np.ndarray] - Decision matrices, each shape (n_i, d_i)

**Example:**
```python
# Random initialization with different sample sizes per task
decs = initialization(problem, n=[30, 50], method='random')

# LHS initialization with same samples across tasks
decs = initialization(problem, n=100, method='lhs', the_same=True)
```

---

#### `evaluation(problem, decs, unified=False, fill_value=0.0, eval_objectives=True, eval_constraints=True)`
**Purpose:** Evaluate decision variables on multiple tasks

**Parameters:**
- `problem`: MTOP instance
- `decs`: List[np.ndarray] - Decision matrices scaled in [0,1]
- `unified`: bool - Pad to max dimensions if True
- `fill_value`: float - Padding value for unified mode
- `eval_objectives`: bool or List - Which objectives to evaluate
- `eval_constraints`: bool or List - Which constraints to evaluate

**Returns:**
- `objs`: List[np.ndarray] - Objective values per task
- `cons`: List[np.ndarray] - Constraint values per task

**Example:**
```python
# Evaluate all objectives and constraints
objs, cons = evaluation(problem, decs)

# Evaluate only first objective of each task
objs, cons = evaluation(problem, decs, eval_objectives=0)

# Unified mode (pad to max dimensions)
objs, cons = evaluation(problem, decs, unified=True)
```

---

#### `evaluation_single(problem, decs, index, unified=False, fill_value=0.0, eval_objectives=True, eval_constraints=True)`
**Purpose:** Evaluate decision variables on a specific task

**Parameters:**
- `problem`: MTOP instance
- `decs`: np.ndarray - Decision matrix for one task, shape (n, d)
- `index`: int - Task index to evaluate
- `unified`: bool - Pad to max dimensions if True
- `eval_objectives`: bool, int, or List[int] - Objectives to evaluate
- `eval_constraints`: bool, int, or List[int] - Constraints to evaluate

**Returns:**
- `objs`: np.ndarray - Objective values, shape (n, m)
- `cons`: np.ndarray - Constraint values, shape (n, c)

**Example:**
```python
# Evaluate task 0
objs, cons = evaluation_single(problem, decs, index=0)

# Evaluate only objectives [0, 2] for task 1
objs, cons = evaluation_single(problem, decs, index=1, eval_objectives=[0, 2])
```

---

### Genetic Operators

#### `crossover(par_dec1, par_dec2, mu=2)`
**Purpose:** Simulated Binary Crossover (SBX)

**Parameters:**
- `par_dec1`: np.ndarray - First parent, shape (d,), scaled in [0,1]
- `par_dec2`: np.ndarray - Second parent, shape (d,), scaled in [0,1]
- `mu`: float - Distribution index

**Returns:**
- `off_dec1`: np.ndarray - First offspring, shape (d,)
- `off_dec2`: np.ndarray - Second offspring, shape (d,)

**Example:**
```python
parent1 = np.random.rand(10)
parent2 = np.random.rand(10)
child1, child2 = crossover(parent1, parent2, mu=2)
```

---

#### `mutation(dec, mu=5)`
**Purpose:** Polynomial mutation

**Parameters:**
- `dec`: np.ndarray - Parent decision vector, shape (d,), scaled in [0,1]
- `mu`: float - Distribution index

**Returns:**
- `mutated_dec`: np.ndarray - Mutated decision vector, shape (d,)

**Example:**
```python
parent = np.random.rand(10)
child = mutation(parent, mu=5)
```

---

#### `ga_generation(parents, muc, mum)`
**Purpose:** Generate offspring using GA operators (SBX + polynomial mutation)

**Parameters:**
- `parents`: np.ndarray - Parent population, shape (n, d)
- `muc`: float - Distribution index for crossover
- `mum`: float - Distribution index for mutation

**Returns:**
- `offdecs`: np.ndarray - Offspring population, shape (n, d)

**Example:**
```python
parents = np.random.rand(100, 10)
offspring = ga_generation(parents, muc=2, mum=5)
```

---

#### `de_generation(parents, F, CR)`
**Purpose:** Generate offspring using Differential Evolution (DE/rand/1/bin)

**Parameters:**
- `parents`: np.ndarray - Parent population, shape (n, d)
- `F`: float - Differential weight (mutation scale factor)
- `CR`: float - Crossover rate in [0, 1]

**Returns:**
- `offdecs`: np.ndarray - Offspring population, shape (n, d)

**Example:**
```python
parents = np.random.rand(100, 10)
offspring = de_generation(parents, F=0.5, CR=0.9)
```

---

### Multi-Objective Optimization

#### `nd_sort(objs, *args)`
**Purpose:** Non-dominated sorting for multi-objective optimization

**Parameters:**
- `objs`: np.ndarray - Objective values, shape (n, m)
- `*args`: Optional (n_sort,) or (cons, n_sort)
  - `n_sort`: int - Number of solutions to sort
  - `cons`: np.ndarray - Constraint values for constrained domination

**Returns:**
- `front_no`: np.ndarray - Front number for each solution, shape (n,)
- `max_fno`: int - Maximum front number assigned

**Example:**
```python
# Basic non-dominated sorting
front_no, max_fno = nd_sort(objs, n_sort=100)

# With constraints
front_no, max_fno = nd_sort(objs, cons, n_sort=100)
```

---

#### `crowding_distance(pop_obj, front_no=None)`
**Purpose:** Calculate crowding distance for diversity preservation

**Parameters:**
- `pop_obj`: np.ndarray - Objective values, shape (n, m)
- `front_no`: np.ndarray - Front numbers, shape (n,). If None, assumes single front

**Returns:**
- `crowd_dis`: np.ndarray - Crowding distance, shape (n,). Boundary solutions have inf

**Example:**
```python
# Calculate crowding distance for each front
front_no, _ = nd_sort(objs, n_sort=100)
crowd_dis = crowding_distance(objs, front_no)

# For single front
crowd_dis = crowding_distance(objs)
```

---

#### `ibea_fitness(objs, kappa)`
**Purpose:** Calculate IBEA (Indicator-Based Evolutionary Algorithm) fitness

**Parameters:**
- `objs`: np.ndarray - Objective values, shape (n, m)
- `kappa`: float - Fitness scaling factor

**Returns:**
- `fitness`: np.ndarray - Fitness values, shape (n,)
- `I`: np.ndarray - Indicator matrix, shape (n, n)
- `C`: np.ndarray - Normalization constants, shape (n,)

**Example:**
```python
fitness, I, C = ibea_fitness(objs, kappa=0.05)
```

---

### Selection Operations

#### `tournament_selection(K, N, *fitness_arrays, rng=None)`
**Purpose:** Tournament selection based on fitness

**Parameters:**
- `K`: int - Tournament size (K<=1 means random selection)
- `N`: int - Number of individuals to select
- `*fitness_arrays`: np.ndarray - One or more fitness arrays (higher is better)
- `rng`: np.random.Generator - Random number generator

**Returns:**
- `selected`: np.ndarray - Selected individual indices, shape (N,)

**Example:**
```python
# Binary tournament selection
indices = tournament_selection(K=2, N=50, fitness_array)

# Multi-criteria tournament (lexicographic)
indices = tournament_selection(K=2, N=50, front_no, -crowd_dis)
```

---

#### `selection_elit(objs, n, cons=None, epsilon=0)`
**Purpose:** Elite selection for single-objective optimization

**Parameters:**
- `objs`: np.ndarray - Objective values (minimize), shape (n, 1)
- `n`: int - Number to select
- `cons`: np.ndarray - Constraint violations, shape (n, c)
- `epsilon`: float - Feasibility threshold

**Returns:**
- `indices`: np.ndarray - Selected indices, shape (n,)

**Example:**
```python
# Select top 50 solutions
indices = selection_elit(objs, n=50)

# With constraints
indices = selection_elit(objs, n=50, cons=cons, epsilon=1e-6)
```

---

### Data Management

#### `init_history(decs, objs, cons=None)`
**Purpose:** Initialize storage for population history

**Parameters:**
- `decs`: List[np.ndarray] - Initial decision variables per task
- `objs`: List[np.ndarray] - Initial objective values per task
- `cons`: List[np.ndarray] - Initial constraint values (optional)

**Returns:**
- `all_decs`: List[List[np.ndarray]] - Decision history storage
- `all_objs`: List[List[np.ndarray]] - Objective history storage
- `all_cons`: List[List[np.ndarray]] - Constraint history (if cons provided)

**Example:**
```python
# Without constraints
all_decs, all_objs = init_history(decs, objs)

# With constraints
all_decs, all_objs, all_cons = init_history(decs, objs, cons)
```

---

#### `append_history(*pairs)`
**Purpose:** Append current generation data to history

**Parameters:**
- `*pairs`: Alternating (history_list, current_data) pairs

**Returns:**
- Tuple of updated history lists

**Example:**
```python
# Single task
all_decs, all_objs = append_history(
    all_decs, new_decs,
    all_objs, new_objs
)

# Multi-task with constraints
all_decs, all_objs, all_cons = append_history(
    all_decs, [task0_decs, task1_decs],
    all_objs, [task0_objs, task1_objs],
    all_cons, [task0_cons, task1_cons]
)
```

---

#### `build_save_results(all_decs, all_objs, runtime, max_nfes, all_cons=None, bounds=None, save_path=None, filename=None, save_data=True, **kwargs)`
**Purpose:** Extract best solutions, build Results object, and save to file

**Parameters:**
- `all_decs`: List[List[np.ndarray]] - Decision history
- `all_objs`: List[List[np.ndarray]] - Objective history
- `runtime`: float - Total runtime
- `max_nfes`: List[int] - Max evaluations per task
- `all_cons`: List[List[np.ndarray]] - Constraint history (optional)
- `bounds`: List[Tuple] - Bounds per task (optional)
- `save_path`: str - Directory to save
- `filename`: str - Filename without extension
- `save_data`: bool - Whether to save
- `**kwargs`: Additional data to include

**Returns:**
- `results`: Results object

**Example:**
```python
results = build_save_results(
    all_decs, all_objs, runtime, max_nfes,
    save_path='./Data/GA',
    filename='GA_P1_run1',
    save_data=True
)
```

---

#### `trim_excess_evaluations(all_decs, all_objs, nt, max_nfes_per_task, nfes_per_task, all_cons=None)`
**Purpose:** Trim excess evaluations when budget is exceeded

**Parameters:**
- `all_decs`: List[List[np.ndarray]] - Decision history
- `all_objs`: List[List[np.ndarray]] - Objective history
- `nt`: int - Number of tasks
- `max_nfes_per_task`: List[int] - Budget per task
- `nfes_per_task`: List[int] - Actual evaluations used
- `all_cons`: List[List[np.ndarray]] - Constraint history (optional)

**Returns:**
- `all_decs`: Trimmed decision history
- `all_objs`: Trimmed objective history
- `nfes_per_task`: Updated evaluation counts
- `all_cons`: Trimmed constraint history (if provided)

**Example:**
```python
all_decs, all_objs, nfes_per_task = trim_excess_evaluations(
    all_decs, all_objs, nt=2,
    max_nfes_per_task=[10000, 10000],
    nfes_per_task=[10050, 9980]
)
```

---

### Array Utilities

#### `vstack_groups(*args)`
**Purpose:** Stack population arrays vertically

**Parameters:**
- `*args`: List[np.ndarray] or Tuple[np.ndarray, ...] - Arrays to stack

**Returns:**
- `results`: np.ndarray or List[np.ndarray] - Stacked array(s)

**Example:**
```python
# Stack single group
combined = vstack_groups([arr1, arr2, arr3])

# Stack multiple groups
decs, objs = vstack_groups(
    (decs1, decs2),
    (objs1, objs2)
)
```

---

#### `select_by_index(index, *arrays)`
**Purpose:** Select rows from arrays by index

**Parameters:**
- `index`: np.ndarray - Indices to select
- `*arrays`: np.ndarray or None - Arrays to select from

**Returns:**
- Selected array(s), None values passed through

**Example:**
```python
# Select from multiple arrays
selected_decs, selected_objs = select_by_index(
    index, decs, objs
)

# With None values
selected_decs, selected_objs, selected_cons = select_by_index(
    index, decs, objs, None
)
```

---

#### `par_list(par, n_tasks)`
**Purpose:** Convert parameter to list for multi-task scenarios

**Parameters:**
- `par`: int or List[int] - Parameter value(s)
- `n_tasks`: int - Number of tasks

**Returns:**
- `par_per_task`: List[int] - Parameter list

**Example:**
```python
# Single value to list
n_per_task = par_list(100, n_tasks=3)  # [100, 100, 100]

# List remains unchanged
n_per_task = par_list([50, 100, 150], n_tasks=3)  # [50, 100, 150]
```

---

#### `reorganize_initial_data(data, nt, n_initial_per_task, interval=1)`
**Purpose:** Reorganize initial data by task and number of points

**Parameters:**
- `data`: List[np.ndarray] - Original data per task
- `nt`: int - Number of tasks
- `n_initial_per_task`: List[int] - Initial points per task
- `interval`: int - Selection interval (1 for all, 2 for every 2nd, etc.)

**Returns:**
- `all_data`: List[List[np.ndarray]] - Reorganized data

**Example:**
```python
# Reorganize with all points
all_data = reorganize_initial_data(data, nt=2, n_initial_per_task=[10, 20])

# With interval=2 (2, 4, 6, ... points)
all_data = reorganize_initial_data(data, nt=2, n_initial_per_task=[10, 20], interval=2)
```

---

### Space Transformations

#### `space_transfer(problem, decs, objs=None, cons=None, type='real', padding='zero')`
**Purpose:** Transfer between unified and real spaces

**Parameters:**
- `problem`: MTOP instance
- `decs`: List[np.ndarray] - Decision variables
- `objs`: List[np.ndarray] - Objectives (optional)
- `cons`: List[np.ndarray] - Constraints (optional)
- `type`: str - 'uni' (pad to max) or 'real' (truncate to original)
- `padding`: str - 'zero' or 'random' for unified mode

**Returns:**
- `new_decs`: Transformed decision variables
- `new_objs`: Transformed objectives (if provided)
- `new_cons`: Transformed constraints (if provided)

**Example:**
```python
# Convert to unified space (pad to max dimensions)
uni_decs, uni_objs = space_transfer(
    problem, decs, objs, type='uni', padding='zero'
)

# Convert back to real space (truncate)
real_decs, real_objs = space_transfer(
    problem, uni_decs, uni_objs, type='real'
)
```

---

#### `normalize(data, axis=0, method='minmax')`
**Purpose:** Normalize data to standard range

**Parameters:**
- `data`: np.ndarray or List - Data to normalize
- `axis`: int - 0 (column-wise) or 1 (row-wise)
- `method`: str - 'minmax' ([0,1]) or 'zscore' (mean=0, std=1)

**Returns:**
- `normalized`: Normalized data
- `stat1`: Min (minmax) or mean (zscore)
- `stat2`: Max (minmax) or std (zscore)

**Example:**
```python
# Min-max normalization
norm_data, mins, maxs = normalize(data, method='minmax')

# Z-score normalization
norm_data, means, stds = normalize(data, method='zscore')

# Multiple arrays
norm_list, mins, maxs = normalize([data1, data2], method='minmax')
```

---

#### `denormalize(data, stat1, stat2, axis=0, method='minmax')`
**Purpose:** Restore data to original scale

**Parameters:**
- `data`: Normalized data
- `stat1`: Min or mean from normalize()
- `stat2`: Max or std from normalize()
- `axis`: int - Must match normalize() call
- `method`: str - Must match normalize() call

**Returns:**
- `restored`: Data in original scale

**Example:**
```python
# Restore min-max normalized data
original = denormalize(norm_data, mins, maxs, method='minmax')

# Restore z-score normalized data
original = denormalize(norm_data, means, stds, method='zscore')
```

---

### Utility Functions

#### `remove_duplicates(new_decs, existing_decs=None, tol=1e-6)`
**Purpose:** Remove duplicate solutions

**Parameters:**
- `new_decs`: np.ndarray - New solutions, shape (n, d)
- `existing_decs`: np.ndarray - Existing solutions to check against (optional)
- `tol`: float - Tolerance for duplicate detection

**Returns:**
- `unique_decs`: np.ndarray - Unique solutions

**Example:**
```python
# Remove duplicates within new solutions
unique = remove_duplicates(new_solutions)

# Remove duplicates against existing population
unique = remove_duplicates(new_solutions, existing_population)
```

---

#### `kmeans_clustering(data, k)`
**Purpose:** K-means clustering using sklearn

**Parameters:**
- `data`: np.ndarray - Data points, shape (n, d)
- `k`: int - Number of clusters

**Returns:**
- `labels`: np.ndarray - Cluster labels, shape (n,), values in [0, k-1]

**Example:**
```python
labels = kmeans_clustering(data, k=5)
```

---

#### `is_duplicate(x, X, epsilon=1e-10)`
**Purpose:** Check if position(s) are duplicates

**Parameters:**
- `x`: np.ndarray - Position(s) to check, shape (d,) or (n, d)
- `X`: np.ndarray - Existing positions, shape (m, d)
- `epsilon`: float - Tolerance

**Returns:**
- bool or List[bool] - Duplicate status

**Example:**
```python
# Check single point
is_dup = is_duplicate(new_point, population)

# Check multiple points
dup_flags = is_duplicate(new_points, population)
```

---

#### `get_algorithm_information(algorithm_class, print_info=True)`
**Purpose:** Get algorithm metadata

**Parameters:**
- `algorithm_class`: type - Algorithm class with 'algorithm_information' attribute
- `print_info`: bool - Whether to print

**Returns:**
- `algo_info`: dict - Algorithm information

**Example:**
```python
info = get_algorithm_information(GA, print_info=True)
```

---

## Common Usage Patterns

### Pattern 1: Basic GA Evolution Loop

```python
from ddmtolab.Methods.Algo_Methods.algo_utils import *

# Initialize
decs = initialization(problem, n=100, method='lhs')
objs, cons = evaluation(problem, decs)
all_decs, all_objs, all_cons = init_history(decs, objs, cons)

# Evolution
for gen in range(max_gen):
    # Generate offspring
    offspring = ga_generation(decs, muc=2, mum=5)
    off_objs, off_cons = evaluation(problem, offspring)
    
    # Combine and select
    combined_decs = vstack_groups((decs, offspring))
    combined_objs = vstack_groups((objs, off_objs))
    
    # Environmental selection
    front_no, _ = nd_sort(combined_objs, n_sort=100)
    crowd_dis = crowding_distance(combined_objs, front_no)
    indices = tournament_selection(2, 100, -front_no, crowd_dis)
    
    # Update population
    decs, objs, cons = select_by_index(indices, combined_decs, combined_objs, combined_cons)
    
    # Record history
    all_decs, all_objs, all_cons = append_history(
        all_decs, decs, all_objs, objs, all_cons, cons
    )
```

### Pattern 2: Multi-Task Optimization

```python
from ddmtolab.Methods.Algo_Methods.algo_utils import *

# Initialize for multiple tasks
decs = initialization(problem, n=[100, 100], method='random')
objs, cons = evaluation(problem, decs)

# Unified space (for knowledge transfer)
uni_decs, uni_objs = space_transfer(
    problem, decs, objs, type='uni', padding='zero'
)

# Transfer knowledge across tasks
# ... (algorithm-specific operations)

# Convert back to real space
real_decs, real_objs = space_transfer(
    problem, uni_decs, uni_objs, type='real'
)
```

### Pattern 3: Save and Load Results

```python
from ddmtolab.Methods.Algo_Methods.algo_utils import *
import time

start_time = time.time()

# Run optimization
# ...

runtime = time.time() - start_time

# Save results
results = build_save_results(
    all_decs, all_objs, runtime, max_nfes,
    all_cons=all_cons,
    bounds=problem.bounds,
    save_path='./Data/MyAlgo',
    filename='MyAlgo_P1_run1',
    save_data=True,
    # Optional extra data
    algorithm_params={'muc': 2, 'mum': 5}
)

# Access best solutions
best_objs_task0 = results.best_objs[0]
best_decs_task0 = results.best_decs[0]
```

### Pattern 4: Constrained Optimization

```python
from ddmtolab.Methods.Algo_Methods.algo_utils import *

# Evaluate with constraints
objs, cons = evaluation(problem, decs)

# Constrained domination sorting
front_no, _ = nd_sort(objs, cons, n_sort=100)

# Elite selection with constraint handling
indices = selection_elit(objs, n=50, cons=cons, epsilon=1e-6)
```

### Pattern 5: Normalization Pipeline

```python
from ddmtolab.Methods.Algo_Methods.algo_utils import *

# Normalize objectives for selection
norm_objs, mins, maxs = normalize(objs, axis=0, method='minmax')

# Perform selection on normalized objectives
# ...

# Denormalize if needed
original_objs = denormalize(norm_objs, mins, maxs, method='minmax')
```

## Key Design Principles

1. **Multi-Task First**: All functions handle lists of arrays for multi-task scenarios
2. **Scaled in [0,1]**: Decision variables internally use [0,1] scale
3. **Unified vs Real**: Support both unified space (max dimensions) and real space (original dimensions)
4. **Optional Constraints**: All functions handle both constrained and unconstrained problems
5. **History Tracking**: Efficient storage of population evolution

## Important Notes

### Decision Variable Scaling
- Internal: [0, 1] scale for all decision variables
- External: Real bounds via problem.bounds
- Use `evaluation()` for automatic scaling

### Multi-Task Array Structure
```python
# Decision variables
decs[task_idx] -> np.ndarray, shape (n_samples, dim)

# History
all_decs[task_idx][generation] -> np.ndarray, shape (n_samples, dim)

# Results
results.best_decs[task_idx] -> np.ndarray for single-obj, or full population for multi-obj
```

### Non-Dominated Sorting Modes
```python
# Basic sorting
front_no, max_fno = nd_sort(objs, n_sort=100)

# With constraints (constrained domination)
front_no, max_fno = nd_sort(objs, cons, n_sort=100)
```

### Space Transfer Modes
- `type='uni'`: Pad to max dimensions (for knowledge transfer)
- `type='real'`: Truncate to original dimensions (for evaluation)
- Tasks with 0 constraints handled correctly in both modes
