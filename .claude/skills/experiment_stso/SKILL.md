# STSO Batch Experiment

## Overview

This skill provides a structured workflow for setting up and running batch experiments comparing Single-Task Single-Objective (STSO) optimization algorithms on benchmark problems. It includes automatic statistical analysis with Wilcoxon rank-sum tests, algorithm ranking, LaTeX table generation, and convergence curve visualization.

## Workflow

### 1. Initialize Batch Experiment

Set up the experiment with a base path and optional folder clearing:

```python
from ddmtolab.Methods.batch_experiment import BatchExperiment

batch_exp = BatchExperiment(
    base_path='./Data',        # Where experimental data will be stored
    clear_folder=True          # Clear existing data before running
)
```

### 2. Add Problems

Add benchmark problems to test. Common sources include CEC benchmark suites and STSOtest:

```python
from ddmtolab.Problems.STSO.stsotest import STSOtest

# Initialize problem suite with dimension
prob = STSOtest(dim=50)

# Add specific problems (P1-P9)
for i in range(1, 10):
    batch_exp.add_problem(getattr(prob, f'P{i}'), f'P{i}')

# Or add individual problems
batch_exp.add_problem(prob.P1, 'P1')
batch_exp.add_problem(prob.P2, 'P2')
```

**Alternative problem sources:**

```python
from ddmtolab.Problems.MTSO.cec17_mtso import CEC17MTSO

prob = CEC17MTSO()
for i in range(1, 10):
    batch_exp.add_problem(getattr(prob, f'P{i}'), f'P{i}')
```

**Key parameters:**
- First argument: Problem instance (callable)
- Second argument: Problem name (string identifier)
- `dim`: Problem dimension (for STSOtest)

### 3. Add Algorithms

Configure algorithms to compare with consistent parameters. The framework supports various evolutionary algorithms and CMA-ES variants:

```python
from ddmtolab.Algorithms.STSO.CMAES import CMAES
from ddmtolab.Algorithms.STSO.IPOPCMAES import IPOPCMAES
from ddmtolab.Algorithms.STSO.sepCMAES import sepCMAES
from ddmtolab.Algorithms.STSO.MAES import MAES
from ddmtolab.Algorithms.STSO.PSO import PSO
from ddmtolab.Algorithms.STSO.CSO import CSO
from ddmtolab.Algorithms.STSO.DE import DE
from ddmtolab.Algorithms.STSO.GA import GA

# Common parameters
n = 100              # Population size
max_nfes = 50000     # Maximum function evaluations

# Add CMA-ES variants
batch_exp.add_algorithm(CMAES, 'CMA-ES', n=n, max_nfes=max_nfes)
batch_exp.add_algorithm(IPOPCMAES, 'IPOP-CMA-ES', n=n, max_nfes=max_nfes)
batch_exp.add_algorithm(sepCMAES, 'sep-CMA-ES', n=n, max_nfes=max_nfes)
batch_exp.add_algorithm(MAES, 'MA-ES', n=n, max_nfes=max_nfes)

# Add evolutionary algorithms
batch_exp.add_algorithm(GA, 'GA', n=n, max_nfes=max_nfes)
batch_exp.add_algorithm(DE, 'DE', n=n, max_nfes=max_nfes)
batch_exp.add_algorithm(PSO, 'PSO', n=n, max_nfes=max_nfes)
batch_exp.add_algorithm(CSO, 'CSO', n=n, max_nfes=max_nfes)
```

**Available algorithms:**
- **CMA-ES variants**: CMAES, IPOPCMAES (Increasing Population), sepCMAES (Separable), MAES (Matrix Adaptation)
- **Evolutionary algorithms**: GA (Genetic Algorithm), DE (Differential Evolution), PSO (Particle Swarm Optimization), CSO (Competitive Swarm Optimizer)

**Key parameters:**
- First argument: Algorithm class (not instance)
- Second argument: Algorithm name for results
- Remaining kwargs: Algorithm-specific parameters (n, max_nfes, etc.)

### 4. Run Experiments

Execute the batch experiment with multiple independent runs:

```python
batch_exp.run(
    n_runs=2,           # Number of independent runs per algorithm-problem pair
    verbose=True,       # Show progress information
    max_workers=6       # Number of parallel workers (adjust based on CPU cores)
)
```

**Performance tips:**
- `max_workers` should be ≤ number of CPU cores
- Higher `n_runs` provides better statistical significance (typically 30-50 for publication)
- `verbose=True` helps monitor long-running experiments
- For quick testing: use `n_runs=2-3`
- For publication: use `n_runs=30-50`

### 5. Analyze Results

Configure and run comprehensive statistical analysis with LaTeX table generation:

```python
from ddmtolab.Methods.data_analysis import DataAnalyzer

analyzer = DataAnalyzer(
    data_path='./Data',                                    
    algorithm_order=['DE', 'PSO', 'CSO', 'CMA-ES', 'IPOP-CMA-ES', 'sep-CMA-ES', 'MA-ES', 'GA'],
    save_path='./Results',                                 
    table_format='latex',                                  
    figure_format='pdf',                                   
    statistic_type='mean',                                 
    significance_level=0.05,                               
    rank_sum_test=True,                                    
    log_scale=True,                                        
    best_so_far=True,                                      
    clear_results=True                                     
)

results = analyzer.run()
```

**Key parameters:**

**Paths:**
- `data_path`: Where `BatchExperiment` saved the data
- `save_path`: Where analysis outputs will be saved

**Algorithm ordering:**
- `algorithm_order`: Controls the column order in tables and legend order in plots
- Must match the algorithm names used in `add_algorithm()`
- Typically ordered by algorithm family or performance

**Output formats:**
- `table_format`: Choose based on intended use
  - `'latex'`: For academic papers (generates publication-ready tables)
  - `'markdown'`: For documentation/README
  - `'csv'`: For further processing
- `figure_format`: Choose based on publication requirements
  - `'pdf'`: Vector graphics for papers
  - `'png'`: Raster graphics for web
  - `'svg'`: Vector graphics for web

**Statistical options:**
- `statistic_type`: `'mean'` (standard) or `'median'` (robust to outliers)
- `significance_level`: Typically 0.05 (5% significance)
- `rank_sum_test`: Enables Wilcoxon rank-sum pairwise comparisons
  - Results shown as +/=/- symbols in tables
  - Compares each algorithm against the base algorithm (last in order)

**Visualization options:**
- `log_scale`: Recommended for problems with wide fitness ranges
- `best_so_far`: Shows cumulative best fitness over evaluations

## Complete Example

```python
from ddmtolab.Methods.batch_experiment import BatchExperiment
from ddmtolab.Methods.data_analysis import DataAnalyzer
from ddmtolab.Problems.STSO.stsotest import STSOtest
from ddmtolab.Algorithms.STSO.CMAES import CMAES
from ddmtolab.Algorithms.STSO.IPOPCMAES import IPOPCMAES
from ddmtolab.Algorithms.STSO.sepCMAES import sepCMAES
from ddmtolab.Algorithms.STSO.MAES import MAES
from ddmtolab.Algorithms.STSO.PSO import PSO
from ddmtolab.Algorithms.STSO.CSO import CSO
from ddmtolab.Algorithms.STSO.DE import DE
from ddmtolab.Algorithms.STSO.GA import GA

if __name__ == '__main__':
    # Step 1: Initialize experiment
    batch_exp = BatchExperiment(base_path='./Data', clear_folder=True)

    # Step 2: Add problems (50-dimensional)
    prob = STSOtest(dim=50)
    for i in range(1, 10):
        batch_exp.add_problem(getattr(prob, f'P{i}'), f'P{i}')

    # Step 3: Configure and add algorithms
    n = 100
    max_nfes = 50000
    
    # CMA-ES variants
    batch_exp.add_algorithm(CMAES, 'CMA-ES', n=n, max_nfes=max_nfes)
    batch_exp.add_algorithm(IPOPCMAES, 'IPOP-CMA-ES', n=n, max_nfes=max_nfes)
    batch_exp.add_algorithm(sepCMAES, 'sep-CMA-ES', n=n, max_nfes=max_nfes)
    batch_exp.add_algorithm(MAES, 'MA-ES', n=n, max_nfes=max_nfes)
    
    # Evolutionary algorithms
    batch_exp.add_algorithm(GA, 'GA', n=n, max_nfes=max_nfes)
    batch_exp.add_algorithm(DE, 'DE', n=n, max_nfes=max_nfes)
    batch_exp.add_algorithm(PSO, 'PSO', n=n, max_nfes=max_nfes)
    batch_exp.add_algorithm(CSO, 'CSO', n=n, max_nfes=max_nfes)

    # Step 4: Run experiments
    batch_exp.run(n_runs=2, verbose=True, max_workers=6)

    # Step 5: Analyze and visualize results
    analyzer = DataAnalyzer(
        data_path='./Data',
        algorithm_order=['DE', 'PSO', 'CSO', 'CMA-ES', 'IPOP-CMA-ES', 'sep-CMA-ES', 'MA-ES', 'GA'],
        save_path='./Results',
        table_format='latex',
        figure_format='pdf',
        statistic_type='mean',
        significance_level=0.05,
        rank_sum_test=True,
        log_scale=True,
        best_so_far=True,
        clear_results=True
    )
    
    results = analyzer.run()
```

## Understanding the Results

### LaTeX Table Output

The analyzer generates a publication-ready LaTeX table with the following structure:

```latex
\begin{table*}[htbp]
\renewcommand{\arraystretch}{1.2}
\centering
\caption{Your caption here}
\label{tab:results}
\resizebox{1.0\textwidth}{!}{
\begin{tabular}{|c|c|c|c|c|c|c|c|c|c|}
\hline
Problem & Task & DE & PSO & CSO & CMA-ES & IPOP-CMA-ES & sep-CMA-ES & MA-ES & GA \\
\hline
P1 & 1 & 2.13e+01(1.14e-02)~= & ... & \textbf{2.59e-07(1.79e-07)~=} & ... \\
\hline
...
\hline
\multicolumn{2}{|c|}{+/$-$/=} & 0/0/9 & 0/0/9 & 0/0/9 & ... & Base \\
\hline
\multicolumn{2}{|c|}{Average Rank} & 6.33 & 5.78 & 6.00 & \textbf{2.11} & ... \\
\hline
\end{tabular}}
\end{table*}
```

**Table components:**
- **Problem/Task columns**: Problem identifier
- **Algorithm columns**: Mean fitness (standard deviation) followed by significance symbol
- **Significance symbols**: 
  - `+`: Significantly better than base algorithm
  - `-`: Significantly worse than base algorithm
  - `=`: No significant difference
- **Bold values**: Best performance on that problem
- **+/-/= row**: Summary of pairwise comparisons (wins/losses/ties)
- **Average Rank row**: Mean rank across all problems (lower is better)
- **Base**: Last algorithm in the order serves as the baseline for comparisons

### Interpreting Results

**Example from the table:**
- CMA-ES achieves an average rank of 2.11 (best overall)
- On problem P1, CMA-ES gets 2.59e-07 ± 1.79e-07 (best result, in bold)
- The `=` symbol indicates no significant difference from the baseline (GA)
- Scientific notation: 2.59e-07 means 2.59 × 10⁻⁷

## Common Variations

### Quick Test Run (Fast Development)
```python
# Reduced parameters for quick testing
prob = STSOtest(dim=10)  # Lower dimension
n = 30
max_nfes = 5000
batch_exp.run(n_runs=2, verbose=True, max_workers=4)
```

### Publication-Quality Run (High Statistical Power)
```python
# Increased runs for better statistics
prob = STSOtest(dim=50)
n = 100
max_nfes = 100000
batch_exp.run(n_runs=30, verbose=True, max_workers=12)
```

### Different Problem Dimensions
```python
# Test on multiple dimensions
for dim in [10, 30, 50]:
    batch_exp = BatchExperiment(base_path=f'./Data_D{dim}', clear_folder=True)
    prob = STSOtest(dim=dim)
    for i in range(1, 10):
        batch_exp.add_problem(getattr(prob, f'P{i}'), f'P{i}')
    # Add algorithms and run...
```

### Comparing CMA-ES Variants Only
```python
# Focus on CMA-ES family comparison
batch_exp.add_algorithm(CMAES, 'CMA-ES', n=100, max_nfes=50000)
batch_exp.add_algorithm(IPOPCMAES, 'IPOP-CMA-ES', n=100, max_nfes=50000)
batch_exp.add_algorithm(sepCMAES, 'sep-CMA-ES', n=100, max_nfes=50000)
batch_exp.add_algorithm(MAES, 'MA-ES', n=100, max_nfes=50000)

analyzer = DataAnalyzer(
    algorithm_order=['CMA-ES', 'IPOP-CMA-ES', 'sep-CMA-ES', 'MA-ES'],
    # ... other parameters
)
```

### Algorithm-Specific Parameters
```python
# Different parameters for different algorithms
batch_exp.add_algorithm(GA, 'GA', n=100, max_nfes=50000, pc=0.9, pm=0.1)
batch_exp.add_algorithm(DE, 'DE', n=50, max_nfes=50000, F=0.5, CR=0.9)
batch_exp.add_algorithm(PSO, 'PSO', n=100, max_nfes=50000, w=0.7, c1=2.0, c2=2.0)
batch_exp.add_algorithm(CMAES, 'CMA-ES', n=100, max_nfes=50000)
```

### Custom Analysis Settings
```python
# Median-based analysis (more robust to outliers)
analyzer = DataAnalyzer(
    data_path='./Data',
    algorithm_order=['CMA-ES', 'DE', 'PSO', 'GA'],
    save_path='./Results',
    statistic_type='median',  # Use median instead of mean
    rank_sum_test=True,
    log_scale=False,          # Linear scale
    best_so_far=False,        # Don't plot best-so-far curves
)
```

## Output Structure

After running the complete workflow, expect the following output structure:

```
./Data/                          # Experimental data
├── CMA-ES/
│   ├── P1_run_0.pkl
│   ├── P1_run_1.pkl
│   ├── P2_run_0.pkl
│   └── ...
├── IPOP-CMA-ES/
├── sep-CMA-ES/
├── MA-ES/
├── DE/
├── PSO/
├── CSO/
└── GA/

./Results/                       # Analysis outputs
├── tables/
│   ├── mean_fitness.tex         # LaTeX table with statistics
│   ├── mean_fitness.csv         # CSV version (if requested)
│   └── rankings.tex             # Algorithm rankings table
├── figures/
│   ├── P1_convergence.pdf       # Convergence curves per problem
│   ├── P2_convergence.pdf
│   ├── P3_convergence.pdf
│   └── ...
└── summary_report.txt           # Text summary of results
```

## Best Practices

1. **Always use `if __name__ == '__main__':`** for parallel execution safety
2. **Set appropriate `max_workers`**: Don't exceed CPU core count
3. **Use `clear_folder=True` cautiously**: It deletes existing data
4. **Match `algorithm_order` to added algorithms**: Ensures proper analysis ordering
5. **Keep problem names consistent**: Use same naming in both experiment and analysis
6. **Use sufficient `n_runs`**: 
   - At least 2 for quick testing
   - 10-20 for preliminary results
   - 30-50 for publication
7. **Monitor disk space**: Large experiments can generate substantial data
8. **Choose appropriate dimensions**: 
   - dim=10 for quick tests
   - dim=30 for standard benchmarks
   - dim=50-100 for challenging problems
9. **Set `max_nfes` appropriately**: Should allow algorithms to converge
   - 5,000-10,000 for quick tests
   - 50,000-100,000 for publication
10. **Order algorithms logically**: Group by family (e.g., CMA-ES variants together)

## Troubleshooting

**Issue: Parallel execution hangs**
- Reduce `max_workers` to 1 for debugging
- Ensure algorithm classes are properly importable
- Check if algorithms have unpicklable attributes
- Verify `if __name__ == '__main__':` is used

**Issue: Analysis fails to find data**
- Verify `data_path` matches `BatchExperiment.base_path`
- Check that algorithms were named consistently
- Ensure experiments completed without errors
- Look for .pkl files in Data folder

**Issue: Memory errors during execution**
- Reduce `max_workers`
- Reduce population size `n`
- Reduce problem dimension `dim`
- Process fewer problems at once

**Issue: Statistical tests show no significant differences**
- Increase `n_runs` for better statistical power
- Check if algorithms actually differ in performance
- Consider using `statistic_type='median'` for robustness
- Reduce `significance_level` (e.g., to 0.01) for stricter tests

**Issue: LaTeX table formatting issues**
- Check if algorithm names contain special characters
- Ensure `algorithm_order` matches actual algorithm names
- Verify LaTeX packages are installed (for rendering)
- Use `table_format='csv'` for easier debugging

**Issue: Convergence curves not generated**
- Verify `figure_format` is supported ('pdf', 'png', 'svg')
- Check `log_scale` setting (try both True and False)
- Ensure matplotlib backend is properly configured
- Check disk space in save_path

**Issue: "Base algorithm not found" error**
- Ensure the last algorithm in `algorithm_order` exists in the data
- Verify algorithm names match exactly (case-sensitive)
- Check that all algorithms completed successfully

## Advanced Tips

### Comparing on Multiple Problem Sets
```python
# Run experiments on both STSOtest and CEC17
problem_sets = [
    ('STSOtest', STSOtest(dim=50)),
    ('CEC17', CEC17MTSO())
]

for name, prob_suite in problem_sets:
    batch_exp = BatchExperiment(base_path=f'./Data_{name}', clear_folder=True)
    for i in range(1, 10):
        batch_exp.add_problem(getattr(prob_suite, f'P{i}'), f'P{i}')
    # Add algorithms and run...
```

### Saving Intermediate Results
```python
# Don't clear folder to preserve previous runs
batch_exp = BatchExperiment(base_path='./Data', clear_folder=False)
# New runs will be added to existing data
```

### Custom Table Captions and Labels
After generating the LaTeX table, manually edit:
```latex
\caption{Performance comparison of CMA-ES variants and evolutionary algorithms on 50-dimensional STSOtest problems}
\label{tab:stso_results_dim50}
```

### Extracting Specific Statistics
```python
# Access the results dictionary
results = analyzer.run()
# results contains detailed statistics, rankings, and test results
# Can be used for custom analysis or plotting
```