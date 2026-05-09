"""
Separable-Arm Multi-Task Single-Objective Benchmark (SepArmMTSO)

3-task problems where tasks share the same target [0.5, 0.5] but differ
in arm configuration: maximum angular range (amax) and total link length
(lmax). Task similarity is controlled by how close the 3 (amax, lmax)
triplets are in parameter space.

Decision variables (per task):
    x[0 : n_joints]  — joint angles in [0, 1]

Fixed task parameters (different per task):
    amax  — maximum angular range factor, in [0.5, 1.0]
              (mirrors x[n_joints] ∈ [0,1] via amax = x*0.5 + 0.5)
    lmax  — total arm length factor, in [0.5, 1.0]
              (mirrors x[n_joints+1] ∈ [0,1] via lmax = x*0.5 + 0.5)

Objective:
    f(x) = 1 − exp(−‖end_effector(x) − [0.5, 0.5]‖)  ∈ [0, 1)

Problems (all 3-task, equal dims = n_joints):

    P1  5D  HS  — configs ±5%  of midpoint    very similar
    P2  5D  MS  — configs ±16% of midpoint    moderately similar
    P3  5D  LS  — configs spread [0.5, 1.0]   low similarity
    P4  10D HS  — same config triplet, 10 joints
    P5  10D MS
    P6  10D LS
    P7  15D HS  — same config triplet, 15 joints
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

    Three 3-task problems with decreasing inter-task similarity.
    All tasks share target [0.5, 0.5]; tasks differ by arm configuration
    (amax, lmax). Call P1/P2/P3 to get MTOP instances.

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

    # Task parameter triplets: list of 3 (amax, lmax) per problem.
    # Both amax and lmax are in [0.5, 1.0], matching the original ActualArm
    # formula: amax = x[n_dim] * 0.5 + 0.5, lmax = x[n_dim+1] * 0.5 + 0.5.
    # Midpoint of the valid range is 0.75.
    # Same triplets are reused across all three dimension groups.
    _CONFIGS = {
        'HS': [(0.75, 0.75), (0.79, 0.75), (0.75, 0.79)],   # ±5%  of midpoint
        'MS': [(0.75, 0.75), (0.87, 0.75), (0.75, 0.87)],   # ±16% of midpoint
        'LS': [(0.55, 0.55), (0.75, 0.75), (0.95, 0.95)],   # spread across [0.5,1.0]
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
        """5D HS — configs within ±5% of midpoint: very similar tasks."""
        return self._make_problem(5, self._CONFIGS['HS'])

    def P2(self) -> MTOP:
        """5D MS — configs within ±16% of midpoint: moderately similar tasks."""
        return self._make_problem(5, self._CONFIGS['MS'])

    def P3(self) -> MTOP:
        """5D LS — configs spread across [0.5, 1.0]: low-similarity tasks."""
        return self._make_problem(5, self._CONFIGS['LS'])

    # ------------------------------------------------------------------
    # Problems — 10D
    # ------------------------------------------------------------------

    def P4(self) -> MTOP:
        """10D HS — configs within ±5% of midpoint: very similar tasks."""
        return self._make_problem(10, self._CONFIGS['HS'])

    def P5(self) -> MTOP:
        """10D MS — configs within ±16% of midpoint: moderately similar tasks."""
        return self._make_problem(10, self._CONFIGS['MS'])

    def P6(self) -> MTOP:
        """10D LS — configs spread across [0.5, 1.0]: low-similarity tasks."""
        return self._make_problem(10, self._CONFIGS['LS'])

    # ------------------------------------------------------------------
    # Problems — 15D
    # ------------------------------------------------------------------

    def P7(self) -> MTOP:
        """15D HS — configs within ±5% of midpoint: very similar tasks."""
        return self._make_problem(15, self._CONFIGS['HS'])

    def P8(self) -> MTOP:
        """15D MS — configs within ±16% of midpoint: moderately similar tasks."""
        return self._make_problem(15, self._CONFIGS['MS'])

    def P9(self) -> MTOP:
        """15D LS — configs spread across [0.5, 1.0]: low-similarity tasks."""
        return self._make_problem(15, self._CONFIGS['LS'])
