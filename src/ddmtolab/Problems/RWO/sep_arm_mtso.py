"""
Separable-Arm Multi-Task Single-Objective Benchmark (SepArmMTSO)

3-task problems where tasks share the same target [0.5, 0.5] but differ
in arm configuration: maximum angular range (amax) only; link length (lmax)
is fixed at 0.75 across all tasks.

Design rationale: with amax centred at 0.3, the optimal joint angle maps to
x* ≈ 0.92 — deep in the corner of [0,1]^N. This makes the problem genuinely
hard for single-task BO at low N_INITIAL, while all three tasks still share
the same corner region, creating meaningful inter-task transfer.

Decision variables (per task):
    x[0 : n_joints]  — joint angles in [0, 1]

Fixed task parameters (different per task):
    amax  — maximum angular range factor, centred at 0.3
    lmax  — total arm length factor, fixed at 0.75 for all tasks

Objective:
    f(x) = 1 − exp(−‖end_effector(x) − [0.5, 0.5]‖)  ∈ [0, 1)

Optimal x* (per joint, analytically): x* = 0.5 + 1 / (8 · amax)
    HS tasks: x* ∈ [0.897, 0.938]  — nearly identical optima
    MS tasks: x* ∈ [0.879, 0.963]  — noticeable but positive transfer
    LS tasks: x* ∈ [0.862, 0.990]  — spread, but all in upper corner

Problems (all 3-task, equal dims = n_joints):

    P1   5D HS  — amax ±5%  of 0.3    very similar
    P2   5D MS  — amax ±10% of 0.3    moderately similar
    P3   5D LS  — amax ±15% of 0.3    low similarity
    P4  10D HS  — same config triplets, 10 joints
    P5  10D MS
    P6  10D LS
    P7  15D HS  — same config triplets, 15 joints
    P8  15D MS
    P9  15D LS
"""

import math
import numpy as np
from ddmtolab.Methods.mtop import MTOP


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _forward_kinematics(command: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Return end-effector (x, y) for commanded joint angles and link lengths."""
    n = len(command)
    angles = np.append(command, 0.0)
    lens   = np.insert(lengths, 0, 0.0)
    mat = np.eye(4)
    ef  = np.zeros(2)
    for i in range(n + 1):
        c, s = math.cos(angles[i]), math.sin(angles[i])
        m = np.array([
            [c, -s, 0, lens[i]],
            [s,  c, 0, 0],
            [0,  0, 1, 0],
            [0,  0, 0, 1],
        ])
        mat = mat @ m
        ef = mat[:2, 3]
    return ef


_TARGET = np.array([0.5, 0.5])


class _SepArmTask:
    """
    One arm-reaching task with fixed configuration.

    Parameters
    ----------
    n_joints : int
        Number of joints. Decision dim = n_joints.
    amax : float
        Maximum angular range (controls how wide each joint can sweep).
    lmax : float
        Total arm length (divided equally across joints).
    """

    def __init__(self, n_joints: int, amax: float, lmax: float):
        self.n_joints = n_joints
        self.amax     = amax
        self.lmax     = lmax

    def __call__(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        nj = self.n_joints
        angular_range = self.amax / nj
        lengths = np.full(nj, self.lmax / nj)
        results = np.empty(len(X))
        for k, x in enumerate(X):
            command = (x[:nj] - 0.5) * angular_range * math.pi * 2
            ef = _forward_kinematics(command, lengths)
            results[k] = 1.0 - math.exp(-np.linalg.norm(ef - _TARGET))
        return results.reshape(-1, 1)


# ---------------------------------------------------------------------------
# Benchmark class
# ---------------------------------------------------------------------------

class SepArmMTSO:
    """
    Separable-Arm Multi-Task Single-Objective Benchmark.

    Nine 3-task problems spanning three dimension groups (5/10/15D) and
    three similarity levels (HS/MS/LS: ±5%/±10%/±15% of amax=0.3).
    All tasks share target [0.5, 0.5] and lmax=0.75; tasks differ only
    in amax, placing x* deep in the upper corner of [0,1]^N. Call P1–P9
    to get MTOP instances.

    Examples
    --------
    >>> bm = SepArmMTSO()
    >>> prob = bm.P1()   # 3 tasks, dim=5,  HS
    >>> prob = bm.P5()   # 3 tasks, dim=10, MS
    >>> prob = bm.P9()   # 3 tasks, dim=15, LS
    """

    problem_information = {
        'n_cases': 9,
        'n_tasks': '3',
        'n_dims': '[5, 10, 15]',
        'n_objs': '1',
        'n_cons': '0',
        'type': 'real_world',
    }

    # amax centred at 0.3; lmax fixed at 0.75 for all tasks.
    # x* = 0.5 + 1/(8*amax) per joint — all tasks share the upper-corner region.
    _CONFIGS = {
        'HS': [(0.285, 0.75), (0.300, 0.75), (0.315, 0.75)],  # ±5%  → x* ∈ [0.897, 0.938]
        'MS': [(0.270, 0.75), (0.300, 0.75), (0.330, 0.75)],  # ±10% → x* ∈ [0.879, 0.963]
        'LS': [(0.255, 0.75), (0.300, 0.75), (0.345, 0.75)],  # ±15% → x* ∈ [0.862, 0.990]
    }

    def __init__(self):
        pass

    # ------------------------------------------------------------------
    # Internal builder
    # ------------------------------------------------------------------

    def _make_problem(self, n_joints: int, configs: list) -> MTOP:
        """Build a 3-task MTOP from n_joints and a list of (amax, lmax) pairs."""
        problem = MTOP()
        for amax, lmax in configs:
            task = _SepArmTask(n_joints, amax, lmax)
            problem.add_task(task, dim=n_joints,
                             lower_bound=np.zeros(n_joints),
                             upper_bound=np.ones(n_joints))
        return problem

    # ------------------------------------------------------------------
    # Problems — 5D
    # ------------------------------------------------------------------

    def P1(self) -> MTOP:
        """5D HS — amax ±5% of 0.3: very similar tasks, x* ∈ [0.897, 0.938]."""
        return self._make_problem(5, self._CONFIGS['HS'])

    def P2(self) -> MTOP:
        """5D MS — amax ±10% of 0.3: moderately similar tasks, x* ∈ [0.879, 0.963]."""
        return self._make_problem(5, self._CONFIGS['MS'])

    def P3(self) -> MTOP:
        """5D LS — amax ±15% of 0.3: low-similarity tasks, x* ∈ [0.862, 0.990]."""
        return self._make_problem(5, self._CONFIGS['LS'])

    # ------------------------------------------------------------------
    # Problems — 10D
    # ------------------------------------------------------------------

    def P4(self) -> MTOP:
        """10D HS — amax ±5% of 0.3: very similar tasks, x* ∈ [0.897, 0.938]."""
        return self._make_problem(10, self._CONFIGS['HS'])

    def P5(self) -> MTOP:
        """10D MS — amax ±10% of 0.3: moderately similar tasks, x* ∈ [0.879, 0.963]."""
        return self._make_problem(10, self._CONFIGS['MS'])

    def P6(self) -> MTOP:
        """10D LS — amax ±15% of 0.3: low-similarity tasks, x* ∈ [0.862, 0.990]."""
        return self._make_problem(10, self._CONFIGS['LS'])

    # ------------------------------------------------------------------
    # Problems — 15D
    # ------------------------------------------------------------------

    def P7(self) -> MTOP:
        """15D HS — amax ±5% of 0.3: very similar tasks, x* ∈ [0.897, 0.938]."""
        return self._make_problem(15, self._CONFIGS['HS'])

    def P8(self) -> MTOP:
        """15D MS — amax ±10% of 0.3: moderately similar tasks, x* ∈ [0.879, 0.963]."""
        return self._make_problem(15, self._CONFIGS['MS'])

    def P9(self) -> MTOP:
        """15D LS — amax ±15% of 0.3: low-similarity tasks, x* ∈ [0.862, 0.990]."""
        return self._make_problem(15, self._CONFIGS['LS'])
