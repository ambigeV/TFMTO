# Research Proposal: Rethinking TabPFN in Bayesian Optimisation Loops

**Status:** Draft — April 2026  
**Context:** Motivated by empirical observations in CEC17-MTSO 10D/50D benchmarks showing that
TabPFN-as-surrogate produces wiggly, non-smooth LCB landscapes in sparse high-dimensional regimes,
causing both random candidate-pool and distill-MLP acquisition optimisers to degrade or collapse.

---

## 1. Problem Statement

Using TabPFN directly as a drop-in GP surrogate violates a key assumption of gradient-based and
candidate-pool acquisition optimisers: that the surrogate produces a **smooth, calibrated uncertainty
landscape** over the continuous search space. TabPFN's ensemble of transformers produces predictions
that are:

1. **Discontinuous as a function of x** — the ensemble mean/std is not kernel-smoothed, so the
   LCB landscape has high-frequency wiggles with no correspondence to the true function.
2. **Unreliable in sparse high-D regimes** — at 50D with n ≈ 100 training points, each estimator
   interpolates between very distant neighbours, causing large-amplitude hallucinated structure.
3. **Not differentiable w.r.t. x by design** — TabPFN was built for tabular prediction, not for
   landscape optimisation; backpropagating through it per candidate is impractical.

The central question is therefore not *whether* to use TabPFN, but **where in the BO loop to place it**
so that its genuine strengths (meta-learned priors, distributional robustness, cross-task ranking) are
exploited without exposing its weaknesses (smoothness, calibration at high d).

---

## 2. Proposed Method 1 — TabPFN-Screened GP-BO (TFM-Screen)

### 2.1 Motivation

TabPFN is strong at the **relative ranking** question: *"given (X_obs, y_obs) as in-context examples,
which of these new candidates is most likely to have low y?"* This is a classification/ranking task
the model was pre-trained on across millions of datasets. It does not require a smooth landscape — only
a correct ordering over a finite candidate set.

The acquisition function optimisation (the step that requires smoothness) is then delegated entirely
to a GP fitted in the usual way — but only over a **pre-filtered** candidate set, dramatically reducing
the search space.

### 2.2 Algorithm

```
Given: observed data D = {(x_i, y_i)}, current best y*

1. Sample C = 10,000 LHS candidates in [0,1]^d

2. SCREEN via TabPFN
   - For MT setting: pool all task data with task-ID, predict LCB on C
   - LCB_TFM(x) = μ_TFM(x) - β · σ_TFM(x)
   - Select top-K candidates by LCB_TFM:   K = 200

3. FIT GP on D (standard SingleTaskGP or MultiTaskGP, ARD Matérn-5/2)

4. ACQUIRE via GP on the K-point filtered set
   - Option A (fast):   argmin_{x ∈ top-K}  LCB_GP(x)             — O(K · n²)
   - Option B (precise): use top-K as warm starts for optimize_acqf — L-BFGS-B from K initialisations

5. Evaluate f at the selected x_new, update D
```

### 2.3 Time Complexity

**TabPFN inference** is the dominant cost. With the adaptive chunking heuristic
`chunk = max(n_train, 20)`, the minimum attention cost per forward pass is:

```
Cost_TFM = (n_train + chunk)^2 × (n_test / chunk)

With chunk = n_train (optimal):
Cost_TFM = 4 · n_train · n_test · n_estimators
```

This is **linear in n_test** — the optimal chunk derivation shows the minimum is achieved at
`chunk* = n_train`, which the current heuristic already satisfies for n_train ≥ 20.

| n_candidates | n_train=100, n_est=4 | Relative cost |
|---|---|---|
| 2,000 (current) | 4 × 100 × 2000 × 4 = 3.2M ops | 1× |
| 10,000 (proposed) | 4 × 100 × 10000 × 4 = 16M ops | 5× |

**GP inference on K=200 filtered candidates** is negligible:

```
Cost_GP = O(n_train^2 + K · n_train) = O(100^2 + 200 × 100) = O(30,000)
```

**Per-step total:**

| Component | 10K-screen | 2K-random (current) |
|---|---|---|
| Candidate generation | O(d × 10K) — fast | O(d × 2K) — fast |
| TabPFN inference | **5× current cost** | baseline |
| GP predict/optimise | O(n² + K·n) — negligible | O(n² + C·n) — moderate |
| **Total overhead** | **~5× TabPFN inference only** | baseline |

If N_CANDIDATES=2000 costs ~1s per step, 10K costs ~5s per step. Over 80 BO steps:
**+320s (≈5 min) per run** — acceptable given the quality improvement expected.

The critical point: **scaling candidates is O(n_candidates) in TabPFN time, not O(n_candidates²)**.
The quadratic cost is in `n_train`, which is fixed. Large candidate sets are cheap.

### 2.4 Expected Benefit

- At 10D: TabPFN's meta-learned ranking over 10K candidates may surface better regions than
  uniform 2K LHS → GP then exploits them precisely. Closes the BO-TFM vs BO gap.
- At 50D: The screening step uses TabPFN for ranking (robust) rather than landscape optimisation
  (fragile). The GP handles smoothness in 50D without the 51D-input problem because
  the GP operates on the original d-dimensional x, not the task-appended (d+1)-dimensional one.
  For MT setting, task-ID is only used by TabPFN for screening; the GP can be a separate per-task
  model or a full MultiTaskGP.

---

## 3. Proposed Method 2 — GP with TabPFN Prior Mean (TFM-Prior)

### 3.1 Motivation

A GP is fully specified by its mean function `μ₀(x)` and kernel `k(x, x')`. The standard BO
convention sets `μ₀(x) = 0` (zero prior mean), which is appropriate when no prior knowledge exists.
TabPFN, having been meta-trained on millions of datasets, encodes a structured prior over regression
functions. This prior can be injected into the GP as a **non-zero, data-informed mean function**
without changing any other property of the GP.

### 3.2 Model Formulation

Let `μ_TFM(x)` denote TabPFN's predicted mean at x given the current dataset D.

**Step 1 — Compute residuals:**
```
r_i = y_i - μ_TFM(x_i)    for all (x_i, y_i) ∈ D
```

**Step 2 — Fit zero-mean GP on residuals:**
```
r(x) ~ GP(0, k_ARD(x, x'))
```
Fit hyperparameters (lengthscales, noise) by MLL on residuals in the usual way.

**Step 3 — Posterior predictive for a new point x:**
```
μ_hybrid(x) = μ_TFM(x) + μ_GP(x | r_obs)
σ_hybrid(x) = σ_GP(x | r_obs)          ← unchanged from standard GP
```

The posterior variance is purely from the GP — smooth, calibrated, kernel-defined. TabPFN only
contributes to the **mean**, not the uncertainty. This is the key property that makes acquisition
function optimisation tractable.

### 3.3 Why the Additive Prior Mean is Not the Right Architecture

The additive hybrid `LCB(x) = μ_TFM(x) + μ_GP(x|r) - β·σ_GP(x|r)` has a fundamental problem:
`μ_TFM(x)` is **not differentiable w.r.t. x** in a practical sense — backpropagating through
the full transformer at every gradient step is prohibitively expensive. Treating `μ_TFM` as
constant during L-BFGS-B is not an approximation — it means the optimizer follows a completely
different gradient than the true acquisition landscape. The additive prior mean formulation is
therefore only usable with candidate-pool evaluation (no gradient refinement), which limits it
to the same coarse resolution as Method 1.

### 3.4 Correct Architecture: DKL-Style Feature Embedding (TFM-DKL)

**How DKL solves this** (Wilson et al. 2016): the neural network operates on the **input**, not
the output. The GP is placed in the feature space of a differentiable encoder φ:

```
k(x, x') = k_GP( φ(x; w),  φ(x'; w) )

LCB(x) = μ_GP(φ(x)) - β · σ_GP(φ(x))

dLCB/dx = (dLCB/dφ) · (dφ/dx)    ← both terms exist, chain rule applies
```

The acquisition function is smooth and differentiable w.r.t. x through the encoder.
Standard L-BFGS-B or CMA-ES on x works correctly.

**TabPFN as the encoder:** TabPFN's penultimate layer representation `φ_TFM(x)` given
in-context data `(X_train, y_train)` is a meta-learned, task-aware embedding of x. Placing
a GP on top of these embeddings gives a DKL model where the kernel geometry is informed by
observed function values — the GP's lengthscales operate in a space where similar-valued
inputs are already close together.

```
φ_TFM(x) = TabPFN penultimate embedding of x | (X_train, y_train)
K_ij = k_ARD( φ_TFM(x_i), φ_TFM(x_j) )    ← fitted once per BO step
LCB(x) = μ_GP(φ_TFM(x)) - β · σ_GP(φ_TFM(x))
```

**Problem:** direct autograd through TabPFN costs a full transformer backward pass per
gradient step during acquisition optimisation — with n_train=100 and quadratic attention,
this is ~100× more expensive than differentiating through a small MLP.

### 3.5 Practical Solution: FSBO-Style Amortised Embedding (TFM-FSBO)

Inspired by FSBO (Wistuba & Grabocka 2021), the expensive TabPFN encoder is **amortised**
into a lightweight MLP that is cheap to differentiate:

```
Algorithm — per BO step:

1. TabPFN FORWARD (once):
   Compute φ_TFM(x_i) for all x_i ∈ X_train   [expensive, done once]

2. DISTIL embedding into φ_MLP:
   Train small MLP s.t. φ_MLP(x) ≈ φ_TFM(x) on X_train
   Loss: MSE( φ_MLP(x_i), φ_TFM(x_i) )
   Architecture: 2–3 layer MLP, output dim = embedding_dim (e.g. 32)

3. FIT GP in embedding space:
   K_ij = k_ARD( φ_MLP(x_i), φ_MLP(x_j) )
   Fit GP hyperparameters by MLL as usual

4. OPTIMISE acquisition via full gradient:
   dLCB/dx = (dLCB/dφ_MLP) · (dφ_MLP/dx)   ← cheap, MLP backward
   Use L-BFGS-B or Adam directly on x ∈ [0,1]^d
```

The TabPFN forward pass happens **once per BO step**, not per gradient step. The MLP
approximation is cheap to differentiate. The GP sits on a task-aware, meta-learned feature
space without requiring repeated TabPFN inference during acquisition optimisation.

### 3.6 Multi-Task Extension

For MTSO, the feature encoder sees pooled cross-task data as in-context examples:

```
φ_TFM(x, task_id) = TabPFN embedding of (x, task_id) | (X_all, y_all, task_ids)
```

The task-ID is passed to TabPFN (as it is now) but only for computing the embedding —
the GP operates in the resulting feature space. For the Elite variant, only elite-filtered
cross-task data enters the TabPFN context, preserving the negative-transfer protection.

### 3.7 Acquisition Function and Optimisation (Corrected)

```
LCB(x) = μ_GP(φ_MLP(x)) - β · σ_GP(φ_MLP(x))

Optimisation:
  Option A: L-BFGS-B with multi-start (5–10 random initialisations in [0,1]^d)
            Gradient: backprop through φ_MLP into GP posterior
  Option B: CMA-ES on x directly (no gradient needed, φ_MLP evaluated per candidate)
```

Both options are fully consistent — no hidden constant-offset approximations.
The acquisition landscape is smooth because the GP kernel enforces it in φ_MLP space.

### 3.4 Multi-Task Extension

For MTSO, the prior mean is computed from pooled multi-task data:
```
μ_TFM(x, task_id) = TabPFN prediction using ALL tasks as context, with scalar task-ID
r_i^(t) = y_i^(t) - μ_TFM(x_i^(t), t)
GP^(t) fitted on {(x_i^(t), r_i^(t))} — separate per-task OR joint MultiTaskGP on residuals
```

For the **Elite variant**: compute `μ_TFM` using elite-filtered cross-task data (top elite_ratio of
other tasks) rather than full pooling. This directly addresses negative transfer:
if tasks are dissimilar, TabPFN's pooled prediction for task t will be poor → large residuals →
GP learns most of the signal → effectively falls back to single-task GP. Graceful degradation.

### 3.5 Expected Benefit

- **10D heterogeneous problems (negative transfer regime):** TabPFN's pooled prior mean captures
  shared structure where it exists; GP residual captures task-specific deviation. Better than MTBO's
  IndexKernel, which can over-couple dissimilar tasks.
- **50D:** TabPFN prior mean is evaluated on candidates (ranking-robust), not optimised through.
  GP in 50D now operates on residuals — which may have lower variance and simpler structure than
  raw y, making ARD kernel fitting easier in sparse regimes.
- **Clean ablation:** `MTBO` vs `TFM-Prior-MTBO` isolates exactly TabPFN's contribution to transfer.

---

## 4. Proposed Method 3 — TFM-Derived Task Covariance for MultiTask GP (TFM-MTBO-Covar)

### 4.1 Motivation: The IndexKernel's Failure Mode

Standard MTBO uses BoTorch's `MultiTaskGP` with an `IndexKernel` to model inter-task covariance:

```
k_MT((x,t), (x',t')) = k_ARD(x, x') × B_tt'

B = W W^T + diag(v)    [low-rank + diagonal, learned by MLL]
```

`B` is a fixed T×T matrix fitted once per step by MLL on typically 40–200 data points.
At small n, this estimate is unreliable — the MLL landscape for `B` is flat, and the optimizer
may converge to a degenerate solution (fully correlated or fully independent). More critically,
`B` cannot distinguish *directional* transfer: task 1 may usefully inform task 2 without the
reverse being true. The IndexKernel is symmetric by construction.

**The core idea:** replace `B` with a data-driven task covariance matrix derived from TabPFN's
penultimate embeddings at every BO step. TabPFN has seen millions of regression datasets and
has learned what "similar functions" look like from data alone — its embeddings are a principled
measure of functional similarity that the MLL cannot recover from 40 observations.

### 4.2 Task Embedding via Cross-Prediction Quality

Rather than using a simple cosine similarity of aggregated embeddings (which conflates location
with function shape), the most principled approach is **cross-predictive compatibility**:

> Task t1 and task t2 are functionally similar if t1's data is a good in-context predictor of t2's values.

For each ordered pair (t1 → t2):

```
NLL(t1 → t2) = -log p_TFM( y_t2 | X_t2,  X_t1, y_t1 )
```

This is a single TabPFN forward pass: context = (X_t1, y_t1), query = X_t2.
Low NLL means task t1's observations predict task t2's values well → high functional similarity.

Convert to a bounded similarity score:

```
s(t1 → t2) = exp( -NLL(t1 → t2) / τ )    τ: temperature hyperparameter (e.g. 1.0)
```

Symmetrise to obtain a valid correlation coefficient:

```
ρ_t1t2 = ( s(t1→t2) + s(t2→t1) ) / 2     ∈ (0, 1]
```

Build the task covariance matrix (for T tasks):

```
R[i, j] = ρ_ij     (R is symmetric, diagonal = 1 by construction)
Σ_task  = diag(σ) · R · diag(σ)    [σ_t^2 still fitted by MLL — only R is fixed from TFM]
```

Ensure PSD: if R has negative eigenvalues due to numerical error, project via
`R ← R + max(0, -λ_min + ε) · I` before use.

### 4.3 Algorithm

```
TFM-MTBO-Covar — per BO step:

1. COMPUTE cross-predictive similarity (T×(T-1) TabPFN calls):
   For each ordered pair (t1, t2), t1 ≠ t2:
     NLL(t1→t2) = TabPFN NLL of y_t2 given context (X_t1, y_t1), query X_t2
   ρ_t1t2 = symmetrised similarity (see above)
   R = correlation matrix from {ρ_t1t2}

2. FIT MultiTaskGP with FIXED task correlation R:
   k_MT((x,t),(x',t')) = k_ARD(x, x') × Σ_task[t,t']
   where Σ_task = diag(σ) · R · diag(σ),  R fixed, σ optimised by MLL
   → MLL only fits: lengthscales, noise, per-task signal variance σ_t²
   → MLL does NOT fit the off-diagonal correlation — that comes from TFM

3. ACQUIRE with standard optimize_acqf:
   GP is a fully valid MultiTaskGP → gradient-based acquisition works unchanged
   Use LogExpectedImprovement or LCB as normal
```

**Computational cost:** For T=2 tasks, only 2 extra TabPFN forward passes per BO step
(one per ordered pair). Each pass has the same cost as the current TFM inference call.
Total overhead: 2× current TabPFN cost per step. For T tasks: T(T-1) calls — scales
quadratically in T but is trivial for the typical T=2 MTSO setting.

### 4.4 Why This is Correct

- **Acquisition optimisation is fully standard.** The GP is a valid kernel machine with a
  properly defined covariance structure. `optimize_acqf` works unchanged — no non-differentiable
  components in the acquisition landscape.
- **MLL fitting is simpler.** With R fixed, MLL only optimises lengthscales + per-task σ²
  (2+d parameters instead of 2+d+T² parameters). Less overfitting on small datasets.
- **Adaptive transfer.** At early iterations (few observations), NLL scores are noisy → ρ
  values regress toward 0.5 (moderate transfer). As data grows, ρ sharpens toward 0 or 1
  depending on true task similarity. Transfer strength is self-calibrating.
- **Negative transfer protection.** When tasks are dissimilar, ρ → 0 → Σ_task → diagonal →
  GP degenerates to T independent GPs. No forced coupling between unrelated tasks.
- **Asymmetry captured then symmetrised.** The raw NLL scores s(t1→t2) and s(t2→t1) may
  differ. The symmetrisation step preserves this signal (asymmetric raw scores → moderate ρ)
  while maintaining a valid covariance matrix.

### 4.5 Variant: Asymmetric Transfer via Directional Weighting

For MTSO where one task may dominate (e.g. a source task is well-optimised and reliably
informs the target), the symmetric ρ discards directional information. An asymmetric extension:

```
When optimising task t_target, weight source task t_source data by:
   w(t_source → t_target) = s(t_source → t_target)   [unnormalised, asymmetric]

This replaces the Elite ratio (fixed 10%) with a TFM-derived, per-step transfer weight.
```

This is a natural replacement for the Elite transfer heuristic — instead of fixed top-10%,
each source point is weighted by the predicted cross-task predictive compatibility.

### 4.6 Comparison to Existing Methods

| Property | MTBO (IndexKernel) | MTBO-TFM-Uni/Elite | **TFM-MTBO-Covar** |
|---|---|---|---|
| Task covariance source | MLL on current data | Not applicable (no GP) | TabPFN cross-prediction |
| Adapts per step? | Yes (MLL refitted) | N/A | Yes (NLL recomputed) |
| Negative transfer protection | Weak (MLL may overfit ρ) | Elite heuristic | Automatic (ρ → 0) |
| Acquisition optimisation | Standard gradient | Candidate pool | **Standard gradient** |
| Extra cost per step | Included in MLL | TabPFN inference | 2× TabPFN inference |
| Handles 50D? | Degrades (50 ARD + T² params) | Collapses | Yes — MLL simpler |
| Novelty | Baseline | Current work | **New** |

---

## 5. Comparison of All Three Proposals

| Property | Method 1: TFM-Screen | Method 2: TFM-FSBO | Method 3: TFM-MTBO-Covar |
|---|---|---|---|
| Role of TabPFN | Candidate ranker | Feature encoder (distilled) | Task covariance estimator |
| Role of GP | Full surrogate | GP in feature space | Full MultiTaskGP |
| Acq optimisation | GP `optimize_acqf` on K=200 | Gradient through φ_MLP + GP | Standard `optimize_acqf` |
| Gradient correct? | Yes | Yes | Yes |
| Handles 50D? | Yes | Yes | Yes (simpler MLL) |
| Neg. transfer protection | Partial | Via elite context | Automatic (ρ → 0) |
| Extra TabPFN calls/step | 1 (screening) | 1 (embedding) | T(T-1) = 2 for T=2 |
| Implementation effort | Low | High (MLP distill + GP kernel) | Medium (custom kernel) |
| Novelty | Moderate | High | **High** |
| Closest prior work | — | FSBO (Wistuba 2021) | MTGP + dataset2vec |

**Method 3 is the strongest contribution:** it intervenes at the exact failure point of MTBO
(unreliable IndexKernel on small data), is fully compatible with standard GP acquisition
optimisation, and produces an interpretable task similarity matrix as a diagnostic byproduct.

---

## 6. Experimental Plan

### 6.1 Baselines
- `GA`, `BO`, `BO-LCB`, `MTBO`, `BO-LCB-BCKT` (existing)
- `BO-TFM`, `MTBO-TFM-Uni`, `MTBO-TFM-Elite` (existing)

### 6.2 Proposed Algorithms
- `MTBO-TFM-Screen` — Method 1, TabPFN screens 10K candidates → GP acquires on top-200
- `MTBO-TFM-FSBO` — Method 2, distilled φ_MLP feature space, GP + gradient acquisition
- `MTBO-TFM-Covar` — Method 3, TabPFN cross-predictive task covariance, standard MultiTaskGP
- `MTBO-TFM-Covar-Asymm` — Method 3 variant, directional transfer weighting

### 6.3 Benchmark
- CEC17-MTSO, all 9 problems × {10D, 50D}
- N_RUNS=5, N_INITIAL=20, MAX_NFES=100
- Problems stratified: CI (P1–P3), PI (P4–P6), NI (P7–P9)

### 6.4 Metrics
- Best-found objective per task at termination (mean ± 0.5 std)
- Wilcoxon rank-sum test (α=0.05) vs MTBO per problem
- Per-step wallclock breakdown: TabPFN inference / MLL fit / acquisition optimisation
- **Method 3 diagnostic:** plot ρ trajectory over BO steps for CI vs NI problems —
  expect ρ → 1 for CI, ρ → 0 for NI

### 6.5 Key Ablations
- Method 3: fixed ρ (computed at initialisation only) vs adaptive ρ (recomputed per step)
- Method 3: temperature τ sensitivity (τ ∈ {0.5, 1.0, 2.0})
- Method 3 vs MTBO: per-task convergence on P5 (known negative transfer case)

---

## 7. Anticipated Contributions

1. **Empirical finding:** TabPFN-as-surrogate produces non-smooth acquisition landscapes in
   sparse high-d regimes; this is the primary failure mode, not model capacity.
2. **Method 1 (TFM-Screen):** TabPFN as large-candidate acquisition pre-filter; O(n_train × n_candidates)
   cost; delegates smoothness to GP; minimal architectural change.
3. **Method 2 (TFM-FSBO):** Distilled TabPFN embedding as GP feature space; fully differentiable
   acquisition; high novelty but high implementation cost.
4. **Method 3 (TFM-MTBO-Covar):** TabPFN cross-predictive NLL as data-driven task covariance
   replacing the IndexKernel; automatic negative transfer protection; standard GP acquisition;
   interpretable ρ trajectory as a research diagnostic.
5. **Unified insight:** The correct role for TabPFN in BO loops is as a **meta-learned prior
   encoder** over task relationships or candidate rankings — not as a landscape optimisation target.

---

## 7. Open Questions

- Does Method 2's residual GP still suffer underdetermined ARD at 50D, or does the prior mean
  reduce residual variance enough that fewer lengthscales are needed?
- At what n_train does TabPFN's ranking quality (for Method 1) degrade — is there a threshold
  below which the screening step adds noise rather than signal?
- Can the warm-start selection in Method 2 Strategy A be further improved by using TabPFN's
  uncertainty (not just mean) to select diverse warm starts covering different promising regions?
