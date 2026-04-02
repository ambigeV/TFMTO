"""
Timing Benchmark: Per-Step Wall-Clock Cost Across All Algorithms

Runs every algorithm on CEC17-MTSO P1 at both 10D and 50D with a minimal
budget (N_INITIAL=10, MAX_NFES=12 → 2 BO steps) to isolate per-step cost.

Each algorithm is timed N_REPEATS times; mean and std are reported.
No results are saved — this script is purely for timing.

Output: a formatted table printed to stdout.

Usage:
    cd mto/
    python timing_benchmark.py
"""
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ddmtolab.Problems.MTSO.cec17_mtso_10d import CEC17MTSO_10D
from ddmtolab.Problems.MTSO.cec17_mtso    import CEC17MTSO
from ddmtolab.Algorithms.STSO.GA           import GA
from ddmtolab.Algorithms.STSO.BO           import BO
from ddmtolab.Algorithms.STSO.BOLCB        import BOLCB
from ddmtolab.Algorithms.MTSO.MTBO         import MTBO
from ddmtolab.Algorithms.MTSO.BO_LCB_BCKT  import BO_LCB_BCKT
from ddmtolab.Algorithms.STSO.BO_TFM       import BO_TFM
from ddmtolab.Algorithms.MTSO.MTBO_TFM_Uniform import MTBO_TFM_Uniform
from ddmtolab.Algorithms.MTSO.MTBO_TFM_Elite   import MTBO_TFM_Elite
from ddmtolab.Algorithms.MTSO.MTBO_TFM_Distill import MTBO_TFM_Distill

# =============================================================================
# Configuration
# =============================================================================

N_INITIAL   = 10
MAX_NFES    = 12        # N_INITIAL + 2 BO steps
N_STEPS     = MAX_NFES - N_INITIAL   # = 2
N_REPEATS   = 3         # timed repetitions per algorithm — take mean ± std

BETA        = 1.0
TFM_BETA    = 2.5
N_ESTIMATORS = 1
N_CANDIDATES = 2000
CMAES_POPSIZE = 40
CMAES_MAXITER = 50

# =============================================================================
# Algorithm factory — returns a fresh instance for each repeat
# =============================================================================

def make_algorithms(problem):
    """Return list of (label, algo_instance) for one problem instance."""
    return [
        ("GA",
         GA(problem, n=N_INITIAL, max_nfes=MAX_NFES, disable_tqdm=True)),

        ("BO",
         BO(problem, n_initial=N_INITIAL, max_nfes=MAX_NFES, disable_tqdm=True)),

        ("BO-LCB",
         BOLCB(problem, n_initial=N_INITIAL, max_nfes=MAX_NFES, beta=BETA,
               disable_tqdm=True)),

        ("MTBO",
         MTBO(problem, n_initial=N_INITIAL, max_nfes=MAX_NFES, disable_tqdm=True)),

        ("BO-LCB-BCKT",
         BO_LCB_BCKT(problem, n_initial=N_INITIAL, max_nfes=MAX_NFES,
                     disable_tqdm=True)),

        ("BO-TFM",
         BO_TFM(problem, n_initial=N_INITIAL, max_nfes=MAX_NFES, beta=TFM_BETA,
                n_estimators=N_ESTIMATORS, n_candidates=N_CANDIDATES,
                disable_tqdm=True)),

        ("MTBO-TFM-Uni",
         MTBO_TFM_Uniform(problem, n_initial=N_INITIAL, max_nfes=MAX_NFES,
                          beta=TFM_BETA, n_estimators=N_ESTIMATORS,
                          n_candidates=N_CANDIDATES, disable_tqdm=True)),

        ("MTBO-TFM-Elite",
         MTBO_TFM_Elite(problem, n_initial=N_INITIAL, max_nfes=MAX_NFES,
                        beta=TFM_BETA, n_estimators=N_ESTIMATORS,
                        n_candidates=N_CANDIDATES, disable_tqdm=True)),

        ("MTBO-TFM-Uni-Distill",
         MTBO_TFM_Distill(problem, n_initial=N_INITIAL, max_nfes=MAX_NFES,
                          beta=TFM_BETA, n_estimators=N_ESTIMATORS,
                          transfer='uniform', encoding='scalar',
                          mlp_loss='nll', distill_model='mlp',
                          warm_start=True, disable_tqdm=True)),

        ("MTBO-TFM-Elite-Distill",
         MTBO_TFM_Distill(problem, n_initial=N_INITIAL, max_nfes=MAX_NFES,
                          beta=TFM_BETA, n_estimators=N_ESTIMATORS,
                          transfer='elite', encoding='scalar',
                          mlp_loss='nll', distill_model='mlp',
                          warm_start=True, disable_tqdm=True)),

        ("MTBO-TFM-Uni-CMA",
         MTBO_TFM_Uniform(problem, n_initial=N_INITIAL, max_nfes=MAX_NFES,
                          beta=TFM_BETA, n_estimators=N_ESTIMATORS,
                          acq_optimizer='cmaes',
                          cmaes_popsize=CMAES_POPSIZE,
                          cmaes_maxiter=CMAES_MAXITER,
                          disable_tqdm=True)),

        ("MTBO-TFM-Elite-CMA",
         MTBO_TFM_Elite(problem, n_initial=N_INITIAL, max_nfes=MAX_NFES,
                        beta=TFM_BETA, n_estimators=N_ESTIMATORS,
                        acq_optimizer='cmaes',
                        cmaes_popsize=CMAES_POPSIZE,
                        cmaes_maxiter=CMAES_MAXITER,
                        disable_tqdm=True)),
    ]

# =============================================================================
# Timing helper
# =============================================================================

def time_algorithm(label, algo):
    """Run algo.optimize(), return elapsed seconds."""
    t0 = time.perf_counter()
    algo.optimize()
    return time.perf_counter() - t0


def run_timing(problem_fn, tag):
    """
    Time all algorithms on a fresh problem instance, N_REPEATS times each.

    Returns list of dicts with keys:
        label, total_mean, total_std, step_mean, step_std
    """
    print(f"\n  [{tag}]  N_INITIAL={N_INITIAL}  MAX_NFES={MAX_NFES}"
          f"  ({N_STEPS} BO steps)  repeats={N_REPEATS}")
    print(f"  {'Algorithm':<26}  {'total(s)':>10}  {'±':>4}  {'per-step(s)':>12}  {'±':>4}")
    print(f"  {'-'*26}  {'-'*10}  {'-'*4}  {'-'*12}  {'-'*4}")

    rows = []
    # Use a fixed set of algorithm labels from the first problem instance
    # to infer which algorithms to run — instantiate fresh each repeat
    first_algos = make_algorithms(problem_fn())
    labels = [lbl for lbl, _ in first_algos]

    for label in labels:
        times = []
        for rep in range(N_REPEATS):
            problem  = problem_fn()              # fresh problem per repeat
            algos    = dict(make_algorithms(problem))
            elapsed  = time_algorithm(label, algos[label])
            times.append(elapsed)

        import statistics
        mean_t = statistics.mean(times)
        std_t  = statistics.stdev(times) if len(times) > 1 else 0.0
        mean_s = mean_t / N_STEPS
        std_s  = std_t  / N_STEPS

        print(f"  {label:<26}  {mean_t:>10.3f}  {std_t:>4.2f}  {mean_s:>12.3f}  {std_s:>4.2f}")
        rows.append(dict(label=label,
                         total_mean=mean_t, total_std=std_t,
                         step_mean=mean_s,  step_std=std_s))

    return rows


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    bm_10d = CEC17MTSO_10D()
    bm_50d = CEC17MTSO()

    print("=" * 66)
    print("  Timing Benchmark — CEC17-MTSO P1")
    print(f"  Config: N_ESTIMATORS={N_ESTIMATORS}  N_CANDIDATES={N_CANDIDATES}")
    print(f"          CMAES_POPSIZE={CMAES_POPSIZE}  CMAES_MAXITER={CMAES_MAXITER}")
    print(f"          LM-CMA buffer m=max(4,ceil(log2(d))): d=10→m=4, d=50→m=6")
    print("=" * 66)

    rows_10d = run_timing(bm_10d.P1, "10D  P1")
    rows_50d = run_timing(bm_50d.P1, "50D  P1")

    # --- side-by-side comparison table ---
    print("\n")
    print("=" * 80)
    print("  Per-step cost comparison (seconds/BO-step, mean over repeats)")
    print("=" * 80)
    fmt = "  {:<26}  {:>12}  {:>12}  {:>10}"
    print(fmt.format("Algorithm", "10D step(s)", "50D step(s)", "50D/10D"))
    print("  " + "-" * 76)

    r10 = {r['label']: r for r in rows_10d}
    r50 = {r['label']: r for r in rows_50d}
    for label in [r['label'] for r in rows_10d]:
        s10 = r10[label]['step_mean']
        s50 = r50[label]['step_mean']
        ratio = f"{s50/s10:.1f}x" if s10 > 1e-4 else "—"
        print(fmt.format(label,
                         f"{s10:.3f} ±{r10[label]['step_std']:.2f}",
                         f"{s50:.3f} ±{r50[label]['step_std']:.2f}",
                         ratio))

    print("=" * 80)
    print(f"\n  Note: 'per-step' = total_time / {N_STEPS}  (includes init amortised over steps).")
    print("  For pure step cost, subtract init time measured separately.")
