"""
Utilities for MTBO_TFM_Covar.

Method summary (Research Proposal §4)
--------------------------------------
Replace BoTorch's IndexKernel (B = WW^T + diag(v), learned by MLL on ~40 points)
with a TabPFN-derived task correlation matrix R that is fixed during MLL fitting:

    B_tt' = σ_t · R_tt' · σ_t'          R fixed,  σ_t² learnable by MLL

R is computed per BO step via cross-predictive NLL:

    NLL(t1 → t2) = -log p_TFM(y_t2 | X_t2,  context=(X_t1, y_t1))
                 ≈ Gaussian NLL using TabPFN's predicted (mean, std)

    s(t1 → t2)  = exp(-NLL(t1→t2) / τ)        τ: temperature
    ρ_t1t2      = (s(t1→t2) + s(t2→t1)) / 2   symmetrised → [0, 1]
    R[i, i]     = 1   (diagonal fixed)

Public API
----------
  tabpfn_cross_pred_nll           per-sample Gaussian NLL of t2 under t1's context
  tabpfn_cross_pred_ce            per-sample CE of t2 classes under t1's context
  compute_task_similarity_matrix  full T×T correlation matrix from cross-pred NLLs
  compute_task_similarity_matrix_directed_classification
                                  full T×T directed similarity from cross-pred CE
  make_psd                        project R to PSD via eigenvalue clipping
  FixedCorrelationTaskKernel      gpytorch.Kernel  B = diag(σ) · R · diag(σ)
"""

import numpy as np
import torch
import torch.nn as nn
import gpytorch

_LOG2PI = float(np.log(2 * np.pi))


# =============================================================================
# 1.  Cross-predictive NLL via TabPFN
# =============================================================================

def tabpfn_cross_pred_nll(
    X_ctx: np.ndarray,
    y_ctx: np.ndarray,
    X_qry: np.ndarray,
    y_qry: np.ndarray,
    *,
    n_estimators: int = 1,
    device: str = 'cpu',
) -> float:
    """
    Compute the mean per-sample Gaussian NLL of y_qry under a TabPFN model
    fitted on context (X_ctx, y_ctx).

    Model:   p(y | x) ≈ N(μ_TFM(x), σ_TFM²(x))
    where μ and σ are TabPFN's predicted mean and std.

    NLL per sample = log σ + (y - μ)² / (2σ²) + ½ log(2π)
    Returns the mean over all query points.

    Parameters
    ----------
    X_ctx, y_ctx : context (task t1) — already min-max normalised y
    X_qry, y_qry : query  (task t2) — already min-max normalised y
    n_estimators : TabPFN ensemble size
    device       : 'cuda' or 'cpu'

    Returns
    -------
    nll : float  — mean per-sample Gaussian NLL
    """
    from ddmtolab.Methods.Algo_Methods.tfm_utils import tabpfn_predict

    mu, sigma = tabpfn_predict(
        X_ctx, y_ctx, X_qry,
        return_std=True,
        n_estimators=n_estimators,
        device=device,
    )
    sigma = np.maximum(sigma, 1e-6)
    nll   = np.log(sigma) + (y_qry - mu) ** 2 / (2.0 * sigma ** 2) + 0.5 * _LOG2PI
    return float(nll.mean())


def tabpfn_cross_pred_ce(
    X_ctx: np.ndarray,
    y_ctx_cls: np.ndarray,
    X_qry: np.ndarray,
    y_qry_cls: np.ndarray,
    *,
    n_estimators: int = 1,
    device: str = 'cpu',
    eps: float = 1e-8,
) -> float:
    """
    Compute mean cross-entropy of query class labels under a TabPFN classifier
    trained on context (X_ctx, y_ctx_cls).
    """
    from ddmtolab.Methods.Algo_Methods.tfm_utils import tabpfn_predict_proba

    proba = tabpfn_predict_proba(
        X_ctx, y_ctx_cls, X_qry,
        n_estimators=n_estimators,
        device=device,
    )

    yq = y_qry_cls.astype(int).ravel()
    n = yq.shape[0]
    p = np.full(n, eps, dtype=np.float64)
    valid = (yq >= 0) & (yq < proba.shape[1])
    if np.any(valid):
        rows = np.arange(n)[valid]
        p[rows] = proba[rows, yq[valid]]
    return float((-np.log(np.clip(p, eps, 1.0))).mean())


# =============================================================================
# 2.  Task similarity matrix
# =============================================================================

def _normalize_y_01(y: np.ndarray) -> np.ndarray:
    """Min-max normalise to [0, 1]; returns zeros if range is degenerate."""
    lo, hi = y.min(), y.max()
    rng = hi - lo
    return (y - lo) / rng if rng > 1e-12 else np.zeros_like(y)


def _rank_quantile_labels(y: np.ndarray, n_classes: int = 3) -> np.ndarray:
    """
    Build ordinal class labels by rank-quantile binning.

    This avoids dependence on absolute objective scale and makes labels
    comparable across tasks.
    """
    n = y.shape[0]
    if n == 0:
        return np.array([], dtype=np.int64)
    ranks = np.argsort(np.argsort(y, kind='mergesort'), kind='mergesort')
    labels = np.floor(ranks * n_classes / n).astype(np.int64)
    return np.clip(labels, 0, n_classes - 1)


def compute_task_similarity_matrix(
    decs: list,
    objs: list,
    *,
    n_estimators: int = 1,
    device: str = 'cpu',
    tau: float = 1.0,
    return_raw_s: bool = False,
):
    """
    Compute the T×T task correlation matrix R from cross-predictive NLLs.

    For each ordered pair (t1, t2)  t1 ≠ t2:
        NLL[t1, t2] = tabpfn_cross_pred_nll(
                          X_t1, y_t1_norm,
                          X_t2, y_t2_norm)
        s[t1, t2]   = exp(-NLL[t1, t2] / τ)

    Symmetrised correlation:
        R[i, j] = (s[i,j] + s[j,i]) / 2      for i ≠ j
        R[i, i] = 1.0

    The matrix is then projected to PSD via eigenvalue clipping (make_psd).

    Parameters
    ----------
    decs         : list of (n_t, d_t) arrays — decision variables per task
    objs         : list of (n_t, 1) or (n_t,) arrays — objectives per task
                   (passed as-is; normalised internally)
    n_estimators : TabPFN ensemble size (1 = fast)
    device       : TabPFN device string
    tau          : temperature for similarity conversion
                   small τ → sharp discrimination;  large τ → flat/uniform

    Returns
    -------
    R : np.ndarray (T, T)  — symmetric, diagonal-1, PSD correlation matrix
    s : np.ndarray (T, T)  — raw directed similarity scores (only if return_raw_s=True)
    """
    T = len(decs)
    y_norm = [_normalize_y_01(o.ravel()) for o in objs]

    # T×T raw similarity scores (diagonal unused)
    s = np.zeros((T, T), dtype=np.float64)
    for t1 in range(T):
        for t2 in range(T):
            if t1 == t2:
                continue
            nll = tabpfn_cross_pred_nll(
                decs[t1], y_norm[t1],
                decs[t2], y_norm[t2],
                n_estimators=n_estimators,
                device=device,
            )
            s[t1, t2] = np.exp(-nll / tau)

    # Symmetrise
    R = np.eye(T, dtype=np.float64)
    for i in range(T):
        for j in range(i + 1, T):
            rho = (s[i, j] + s[j, i]) / 2.0
            R[i, j] = rho
            R[j, i] = rho

    R_psd = make_psd(R)
    if return_raw_s:
        s_full = s.copy()
        np.fill_diagonal(s_full, 1.0)
        return R_psd, s_full
    return R_psd


# =============================================================================
# 3.  Directed similarity (asymmetric): regression and classification
# =============================================================================

def compute_task_similarity_matrix_directed(
    decs: list,
    objs: list,
    *,
    n_estimators: int = 1,
    device: str = 'cpu',
    tau: float = 1.0,
) -> np.ndarray:
    """
    Compute the directed/asymmetric T×T TFN similarity matrix S.

    For each ordered pair (t1, t2):
        S[t1, t2] = exp(-NLL(t1->t2) / tau),  t1 != t2
        S[t,  t]  = 1

    Notes
    -----
    - S is generally asymmetric because t1->t2 and t2->t1 can differ.
    - This matrix itself is NOT directly usable as a GP covariance matrix.
    """
    T = len(decs)
    y_norm = [_normalize_y_01(o.ravel()) for o in objs]

    S = np.eye(T, dtype=np.float64)
    for t1 in range(T):
        for t2 in range(T):
            if t1 == t2:
                continue
            nll = tabpfn_cross_pred_nll(
                decs[t1], y_norm[t1],
                decs[t2], y_norm[t2],
                n_estimators=n_estimators,
                device=device,
            )
            S[t1, t2] = np.exp(-nll / tau)
    return S


def compute_task_similarity_matrix_directed_classification(
    decs: list,
    objs: list,
    *,
    n_classes: int = 3,
    n_estimators: int = 1,
    device: str = 'cpu',
    tau: float = 1.0,
) -> np.ndarray:
    """
    Compute directed/asymmetric T×T similarity matrix S using classification CE.

    Steps
    -----
    1) Convert each task objective to ordinal class labels via rank-quantile bins.
    2) For each ordered pair (t1, t2):
         CE[t1, t2] = cross-entropy of y_t2_cls predicted by classifier fit on t1
         S[t1, t2]  = exp(-CE[t1, t2] / tau)
    3) Set diagonal to 1.
    """
    T = len(decs)
    cls = [_rank_quantile_labels(o.ravel(), n_classes=n_classes) for o in objs]

    S = np.eye(T, dtype=np.float64)
    for t1 in range(T):
        for t2 in range(T):
            if t1 == t2:
                continue
            ce = tabpfn_cross_pred_ce(
                decs[t1], cls[t1],
                decs[t2], cls[t2],
                n_estimators=n_estimators,
                device=device,
            )
            S[t1, t2] = np.exp(-ce / tau)
    return S


# =============================================================================
# 4.  Mapping directed S -> GP-valid correlation
# =============================================================================

def directed_similarity_to_correlation(
    S: np.ndarray,
    *,
    method: str = 'gram',
    eps: float = 1e-4,
) -> np.ndarray:
    """
    Map a directed TFN matrix S into a symmetric PSD correlation matrix R.

    Parameters
    ----------
    S      : (T, T) directed similarity matrix (typically diag=1)
    method : mapping strategy
             - 'gram': R_raw = S @ S.T      (default; preserves directionality
                       implicitly via row embeddings)
             - 'avg' : R_raw = 0.5 * (S + S.T)
    eps    : minimum eigenvalue used in final PSD projection

    Returns
    -------
    R : (T, T) symmetric PSD correlation matrix with unit diagonal.
    """
    if method == 'gram':
        R_raw = S @ S.T
    elif method == 'avg':
        R_raw = 0.5 * (S + S.T)
    else:
        raise ValueError(f"Unknown directed->correlation method: {method}")

    d = np.sqrt(np.maximum(np.diag(R_raw), 1e-12))
    R = R_raw / np.outer(d, d)
    return make_psd(R, eps=eps)


# =============================================================================
# 5.  PSD projection
# =============================================================================

def make_psd(R: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """
    Project a symmetric matrix to PSD by clipping negative eigenvalues.

    R_psd = V · max(Λ, eps·I) · V^T,  normalised so diagonal stays 1.

    Parameters
    ----------
    R   : (T, T) symmetric matrix (diagonal should be 1)
    eps : minimum eigenvalue after projection

    Returns
    -------
    R_psd : (T, T) PSD matrix with unit diagonal
    """
    R = (R + R.T) / 2.0                         # enforce exact symmetry
    eigvals, V = np.linalg.eigh(R)
    eigvals_clipped = np.maximum(eigvals, eps)
    R_psd = V @ np.diag(eigvals_clipped) @ V.T
    # Re-normalise diagonal to 1
    d = np.sqrt(np.diag(R_psd))
    d = np.where(d < 1e-12, 1.0, d)
    R_psd = R_psd / np.outer(d, d)
    return R_psd


# =============================================================================
# 6.  FixedCorrelationTaskKernel
# =============================================================================

class FixedCorrelationTaskKernel(gpytorch.kernels.Kernel):
    """
    Drop-in replacement for gpytorch's IndexKernel inside a MultiTaskGP.

    Task covariance:
        B_tt' = σ_t · R_tt' · σ_t'
        R  — fixed correlation matrix  (registered as a buffer, no grad)
        σ  — learnable per-task scales (log_sigma parameter, init = 0 → σ=1)

    Interface
    ---------
    Inherits from gpytorch.kernels.Kernel and follows the same
    forward(i1, i2, **params) contract as IndexKernel:
      i1 : (..., n1, 1) float tensor  — task feature values
      i2 : (..., n2, 1) float tensor  — task feature values
      returns : (..., n1, n2) dense kernel matrix

    Task-index convention
    ---------------------
    The existing mtgp_build uses torch.linspace(0, 1, T) task values.
    This kernel converts them to integer indices via:
        task_int = round( task_float × (T - 1) )
    so task 0 → 0.0, task 1 → 1.0 (for T=2) or 0.5 (for T=3), etc.
    Integer task IDs (0, 1, ..., T-1) are also accepted unchanged.
    """

    def __init__(self, R_fixed: torch.Tensor, active_dims=None):
        super().__init__(active_dims=active_dims)
        T = R_fixed.shape[0]
        self.T = T
        # Fixed correlation — not a parameter
        self.register_buffer('R', R_fixed.detach().clone())
        # Learnable per-task log-scale (init 0 → σ = 1)
        self.register_parameter(
            'log_sigma',
            nn.Parameter(torch.zeros(T, dtype=self.R.dtype, device=self.R.device)),
        )

    # ------------------------------------------------------------------

    @property
    def covariance_matrix(self) -> torch.Tensor:
        """B = diag(σ) · R · diag(σ),  shape (T, T)."""
        sigma = torch.exp(self.log_sigma)          # (T,)
        return sigma.unsqueeze(-1) * self.R * sigma.unsqueeze(0)

    # ------------------------------------------------------------------

    def _to_int_index(self, i: torch.Tensor) -> torch.Tensor:
        """
        Convert float task-feature values (linspace convention) to
        integer task indices 0 … T-1.
        """
        i_val = i.squeeze(-1)

        if torch.is_floating_point(i_val):
            i_float = i_val.float()

            # Accept direct integer-like task ids in [0, T-1].
            is_int_like = torch.all(
                (i_float - torch.round(i_float)).abs() < 1e-6
            )
            in_task_range = torch.all((i_float >= -1e-6) & (i_float <= (self.T - 1) + 1e-6))
            if bool(is_int_like and in_task_range):
                idx = torch.round(i_float).long()
            else:
                # linspace(0,1,T): task_int = round(val × (T-1))
                idx = torch.round(i_float * (self.T - 1)).long()
        else:
            idx = i_val.long()

        return idx.clamp(0, self.T - 1)

    # ------------------------------------------------------------------

    def forward(
        self,
        i1: torch.Tensor,
        i2: torch.Tensor,
        diag: bool = False,
        last_dim_is_batch: bool = False,
        **params,
    ) -> torch.Tensor:
        """
        Returns B[i1, i2] as a dense (n1, n2) tensor (or batched).

        i1 : (..., n1, 1)  task feature values (float or int)
        i2 : (..., n2, 1)  task feature values (float or int)
        """
        if last_dim_is_batch:
            raise RuntimeError("FixedCorrelationTaskKernel does not support last_dim_is_batch=True.")
        if i1.size(-1) != 1 or i2.size(-1) != 1:
            raise RuntimeError(
                f"FixedCorrelationTaskKernel expects task-feature tensors with last dim 1, "
                f"got i1 shape {tuple(i1.shape)} and i2 shape {tuple(i2.shape)}. "
                f"This usually means active_dims was not set to the task feature."
            )

        B = self.covariance_matrix                 # (T, T)
        i1_idx = self._to_int_index(i1)            # (..., n1)
        i2_idx = self._to_int_index(i2)            # (..., n2)

        if diag:
            return B[i1_idx, i2_idx]

        # Advanced indexing: res[..., k, l] = B[i1[k], i2[l]]
        return B[i1_idx.unsqueeze(-1), i2_idx.unsqueeze(-2)]  # (..., n1, n2)


# =============================================================================
# 7.  ρ CSV logger  (T=2 only)
# =============================================================================

def write_rho_csv(
    save_path: str,
    name: str,
    s_history: list,
    rho_history: list,
    asymmetric: bool = False,
) -> None:
    """
    Write per-step task-similarity logging CSV for T=2 problems.

    Columns
    -------
    step       : BO step index (0-based)
    s_0to1    : S[0,1] — raw directed similarity task-0 → task-1
    s_1to0    : S[1,0] — raw directed similarity task-1 → task-0
    rho_task0 : ρ used in task-0's GP  (R_0[0,1] = S[1→0] for asym; avg for sym)
    rho_task1 : ρ used in task-1's GP  (R_1[0,1] = S[0→1] for asym; avg for sym)

    Parameters
    ----------
    s_history   : list of (T,T) raw directed S arrays
    rho_history : list of (T,T) R arrays  [symmetric]
                  OR list of {int: (T,T)} dicts  [asymmetric / per-target]
    asymmetric  : True for per-target methods (Covar_Asym, Covar_Cls, MAP_Asym)
    """
    import csv
    import os

    if not s_history or s_history[0].shape[0] != 2:
        return   # only log for T=2

    os.makedirs(save_path, exist_ok=True)
    csv_path = os.path.join(save_path, f'{name}_rho_log.csv')

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 's_0to1', 's_1to0', 'rho_task0', 'rho_task1'])
        for step, (S, R_entry) in enumerate(zip(s_history, rho_history)):
            s01 = float(S[0, 1])
            s10 = float(S[1, 0])
            if asymmetric:
                # rho_history entry is a dict {task_id: R_i}
                rho_t0 = float(R_entry[0][0, 1]) if 0 in R_entry else float('nan')
                rho_t1 = float(R_entry[1][0, 1]) if 1 in R_entry else float('nan')
            else:
                # rho_history entry is a single symmetric (T,T) array
                rho_t0 = float(R_entry[0, 1])
                rho_t1 = float(R_entry[0, 1])
            writer.writerow([step,
                             f'{s01:.6f}', f'{s10:.6f}',
                             f'{rho_t0:.6f}', f'{rho_t1:.6f}'])


# =============================================================================
# 8.  RhoRecorder — per-iteration ρ accumulator
# =============================================================================

class RhoRecorder:
    """
    Accumulates per-step directed similarity S and task-correlation R matrices
    during BO and saves them to a CSV at the end of the run.

    Usage
    -----
    Instantiate once per algorithm run:

        recorder = RhoRecorder(asymmetric=False)  # or True for per-target methods

    Call ``record`` at the end of every BO step:

        recorder.record(S_np, R_entry)

    where
      - S_np     : (T, T) directed raw similarity array for this step
      - R_entry  : (T, T) numpy array      — for symmetric methods (Covar, MAP-Sym)
                   OR {task_id: (T,T) array} — for per-target methods (Covar-Asym/Cls, MAP-Asym)

    At the end of the run:

        recorder.save(save_path, name)   # writes  <save_path>/<name>_rho_log.csv

    Attributes
    ----------
    s_history   : list[(T, T)]             raw directed S per step (copies)
    rho_history : list[(T, T)] or list[dict]  R entry per step (copies)
    asymmetric  : bool  — mirrors the flag passed at construction
    """

    def __init__(self, asymmetric: bool = False):
        self.asymmetric: bool = asymmetric
        self.s_history:   list = []
        self.rho_history: list = []

    def record(self, S_np: np.ndarray, R_entry) -> None:
        """Append one BO-step snapshot.

        Parameters
        ----------
        S_np    : (T, T) raw directed similarity matrix for this step
        R_entry : (T, T) array  [symmetric mode]
                  OR {task_id: (T, T) array}  [asymmetric mode]
        """
        self.s_history.append(S_np.copy())
        if isinstance(R_entry, dict):
            self.rho_history.append({k: v.copy() for k, v in R_entry.items()})
        else:
            self.rho_history.append(np.array(R_entry))

    def save(self, save_path: str, name: str) -> None:
        """Write <save_path>/<name>_rho_log.csv (no-op if no data or T != 2)."""
        write_rho_csv(save_path, name, self.s_history, self.rho_history, self.asymmetric)
