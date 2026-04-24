import pkgutil
import scipy.io
import io
from ddmtolab.Problems.BasicFunctions.basic_functions import *
from ddmtolab.Methods.mtop import MTOP
import numpy as np

_D    = 30          # dimensions exposed to the optimiser
_TAIL = 50 - _D    # dimensions fixed at their global-optimum value (= 20)

# Schwefel's global optimum is at z_i = 420.9687... (not at z=0).
# With identity rotation and go=zeros, the minimiser in x-space is this constant.
_SCHWEFEL_OPT = 420.9687462275636

# Rosenbrock's global optimum is at z=(1,...,1) (not z=0).
# With identity rotation and go=zeros, the minimiser in x-space is ones.
_ROSENBROCK_OPT = 1.0


class CEC17MTSO_30D:
    """
    30-Dimensional version of the CEC 2017 MTSO benchmark (P1–P9).

    Uses the same 50D rotation matrices and global optima from the original
    CEC17 competition .mat files.  The optimiser sees only the first 30
    decision variables; the remaining 20 are fixed at go[30:] (the global
    optimum coordinates for those dimensions) before each function call.

    This preserves the full 50D landscape geometry — the CI/PI/NI and
    HS/MS/LS characteristics are identical to the 50D benchmark.  The
    effective search space is a 30D affine slice passing through the 50D
    global optimum.

    Notes
    -----
    - D_exposed = 30,  D_fixed = 20
    - Reuses data_cec17mtso/ .mat files — no new data files required
    - P6 Task 2 stays at dim=25 (original dim < 30, no fixing applied)
    """

    problem_information = {
        'n_cases': 9,
        'n_tasks': '2',
        'n_dims': '30',
        'n_objs': '1',
        'n_cons': '0',
        'type': 'synthetic',
    }

    def __init__(self):
        self.data_dir = 'data_cec17mtso'

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    @staticmethod
    def _fix_tail(fn50, tail):
        """
        Wrap a 50D callable to accept _D-dimensional input.

        Handles both vectorised calls (n_samples, _D) and single-row calls
        (_D,) by tiling the fixed tail along the sample axis before
        reconstructing the full (n_samples, 50) array for fn50.
        """
        tail = np.asarray(tail, dtype=float).copy()
        def inner(x):
            x2 = np.atleast_2d(x)                          # (n, _D)
            tail_block = np.tile(tail, (x2.shape[0], 1))   # (n, _TAIL)
            return fn50(np.hstack([x2, tail_block]))        # (n, 50)
        return inner

    def _load(self, filename):
        data_bytes = pkgutil.get_data('ddmtolab.Problems.MTSO',
                                      f'{self.data_dir}/{filename}')
        return scipy.io.loadmat(io.BytesIO(data_bytes))

    # ------------------------------------------------------------------
    # Problems
    # ------------------------------------------------------------------

    def P1(self) -> MTOP:
        """CI-HS (30D): T1 Griewank + T2 Rastrigin — complete intersection, high similarity."""
        mat = self._load('CI_H.mat')
        rot1 = mat['Rotation_Task1'].squeeze()
        go1  = mat['GO_Task1'].squeeze()
        rot2 = mat['Rotation_Task2'].squeeze()
        go2  = mat['GO_Task2'].squeeze()

        T1 = self._fix_tail(lambda x: Griewank(x, rot1, go1, 0.), go1[_D:])
        T2 = self._fix_tail(lambda x: Rastrigin(x, rot2, go2, 0.), go2[_D:])

        problem = MTOP()
        problem.add_task(T1, dim=_D, lower_bound=np.full(_D, -100), upper_bound=np.full(_D, 100))
        problem.add_task(T2, dim=_D, lower_bound=np.full(_D,  -50), upper_bound=np.full(_D,  50))
        return problem

    def P2(self) -> MTOP:
        """CI-MS (30D): T1 Ackley + T2 Rastrigin — complete intersection, medium similarity."""
        mat = self._load('CI_M.mat')
        rot1 = mat['Rotation_Task1'].squeeze()
        go1  = mat['GO_Task1'].squeeze()
        rot2 = mat['Rotation_Task2'].squeeze()
        go2  = mat['GO_Task2'].squeeze()

        T1 = self._fix_tail(lambda x: Ackley(x, rot1, go1, 0.), go1[_D:])
        T2 = self._fix_tail(lambda x: Rastrigin(x, rot2, go2, 0.), go2[_D:])

        problem = MTOP()
        problem.add_task(T1, dim=_D, lower_bound=np.full(_D, -50), upper_bound=np.full(_D, 50))
        problem.add_task(T2, dim=_D, lower_bound=np.full(_D, -50), upper_bound=np.full(_D, 50))
        return problem

    def P3(self) -> MTOP:
        """CI-LS (30D): T1 Ackley + T2 Schwefel — complete intersection, low similarity."""
        mat  = self._load('CI_L.mat')
        rot1 = mat['Rotation_Task1'].squeeze()
        go1  = mat['GO_Task1'].squeeze()
        # T2: identity rotation, origin optimum (same as 50D)
        rot2 = np.eye(50, dtype=float)
        go2  = np.zeros(50, dtype=float)

        T1 = self._fix_tail(lambda x: Ackley(x, rot1, go1, 0.), go1[_D:])
        T2 = self._fix_tail(lambda x: Schwefel(x, rot2, go2, 0.),
                            np.full(_TAIL, _SCHWEFEL_OPT))

        problem = MTOP()
        problem.add_task(T1, dim=_D, lower_bound=np.full(_D,  -50), upper_bound=np.full(_D,  50))
        problem.add_task(T2, dim=_D, lower_bound=np.full(_D, -500), upper_bound=np.full(_D, 500))
        return problem

    def P4(self) -> MTOP:
        """PI-HS (30D): T1 Rastrigin + T2 Sphere — partial intersection, high similarity."""
        mat  = self._load('PI_H.mat')
        rot1 = mat['Rotation_Task1'].squeeze()
        go1  = mat['GO_Task1'].squeeze()
        go2  = mat['GO_Task2'].squeeze()
        # T2: identity rotation (same as 50D)
        rot2 = np.eye(50, dtype=float)

        T1 = self._fix_tail(lambda x: Rastrigin(x, rot1, go1, 0.), go1[_D:])
        T2 = self._fix_tail(lambda x: Sphere(x, rot2, go2, 0),     go2[_D:])

        problem = MTOP()
        problem.add_task(T1, dim=_D, lower_bound=np.full(_D,  -50), upper_bound=np.full(_D,  50))
        problem.add_task(T2, dim=_D, lower_bound=np.full(_D, -100), upper_bound=np.full(_D, 100))
        return problem

    def P5(self) -> MTOP:
        """PI-MS (30D): T1 Ackley + T2 Rosenbrock — partial intersection, medium similarity."""
        mat  = self._load('PI_M.mat')
        rot1 = mat['Rotation_Task1'].squeeze()
        go1  = mat['GO_Task1'].squeeze()
        # T2: identity rotation, origin optimum (same as 50D)
        rot2 = np.eye(50, dtype=float)
        go2  = np.zeros(50, dtype=float)

        T1 = self._fix_tail(lambda x: Ackley(x, rot1, go1, 0.),     go1[_D:])
        T2 = self._fix_tail(lambda x: Rosenbrock(x, rot2, go2, 0.),
                            np.full(_TAIL, _ROSENBROCK_OPT))

        problem = MTOP()
        problem.add_task(T1, dim=_D, lower_bound=np.full(_D, -50), upper_bound=np.full(_D, 50))
        problem.add_task(T2, dim=_D, lower_bound=np.full(_D, -50), upper_bound=np.full(_D, 50))
        return problem

    def P6(self) -> MTOP:
        """
        PI-LS (30D vs 25D): T1 Ackley (30D) + T2 Weierstrass (25D) — partial intersection,
        low similarity, unequal dimensions.

        T2 is already 25D in the original benchmark (< 30), so it is kept at full 25D
        with no tail-fixing applied.  Only T1 (originally 50D) is reduced to 30D.
        """
        mat  = self._load('PI_L.mat')
        rot1 = mat['Rotation_Task1'].squeeze()
        go1  = mat['GO_Task1'].squeeze()
        rot2 = mat['Rotation_Task2'].squeeze()   # (25, 25)
        go2  = mat['GO_Task2'].squeeze()          # (25,)

        T1 = self._fix_tail(lambda x: Ackley(x, rot1, go1, 0.),        go1[_D:])
        # T2 is 25D — evaluate directly, no wrapper needed
        def T2(x):
            return Weierstrass(x, rot2, go2, 0.)

        problem = MTOP()
        problem.add_task(T1, dim=_D, lower_bound=np.full(_D, -50),   upper_bound=np.full(_D,  50))
        problem.add_task(T2, dim=25, lower_bound=np.full(25, -0.5),  upper_bound=np.full(25,  0.5))
        return problem

    def P7(self) -> MTOP:
        """NI-HS (30D): T1 Rosenbrock + T2 Rastrigin — no intersection, high similarity."""
        mat  = self._load('NI_H.mat')
        rot2 = mat['Rotation_Task2'].squeeze()
        go2  = mat['GO_Task2'].squeeze()
        # T1: identity rotation, origin optimum (same as 50D)
        rot1 = np.eye(50, dtype=float)
        go1  = np.zeros(50, dtype=float)

        T1 = self._fix_tail(lambda x: Rosenbrock(x, rot1, go1, 0.),
                            np.full(_TAIL, _ROSENBROCK_OPT))
        T2 = self._fix_tail(lambda x: Rastrigin(x, rot2, go2, 0.),  go2[_D:])

        problem = MTOP()
        problem.add_task(T1, dim=_D, lower_bound=np.full(_D, -50), upper_bound=np.full(_D, 50))
        problem.add_task(T2, dim=_D, lower_bound=np.full(_D, -50), upper_bound=np.full(_D, 50))
        return problem

    def P8(self) -> MTOP:
        """NI-MS (30D): T1 Griewank + T2 Weierstrass — no intersection, medium similarity."""
        mat  = self._load('NI_M.mat')
        rot1 = mat['Rotation_Task1'].squeeze()
        go1  = mat['GO_Task1'].squeeze()
        rot2 = mat['Rotation_Task2'].squeeze()
        go2  = mat['GO_Task2'].squeeze()

        T1 = self._fix_tail(lambda x: Griewank(x, rot1, go1, 0.),     go1[_D:])
        T2 = self._fix_tail(lambda x: Weierstrass(x, rot2, go2, 0.),  go2[_D:])

        problem = MTOP()
        problem.add_task(T1, dim=_D, lower_bound=np.full(_D, -100), upper_bound=np.full(_D, 100))
        problem.add_task(T2, dim=_D, lower_bound=np.full(_D,  -0.5), upper_bound=np.full(_D,  0.5))
        return problem

    def P9(self) -> MTOP:
        """NI-LS (30D): T1 Rastrigin + T2 Schwefel — no intersection, low similarity."""
        mat  = self._load('NI_L.mat')
        rot1 = mat['Rotation_Task1'].squeeze()
        go1  = mat['GO_Task1'].squeeze()
        # T2: identity rotation, origin optimum (same as 50D)
        rot2 = np.eye(50, dtype=float)
        go2  = np.zeros(50, dtype=float)

        T1 = self._fix_tail(lambda x: Rastrigin(x, rot1, go1, 0.),  go1[_D:])
        T2 = self._fix_tail(lambda x: Schwefel(x, rot2, go2, 0.),
                            np.full(_TAIL, _SCHWEFEL_OPT))

        problem = MTOP()
        problem.add_task(T1, dim=_D, lower_bound=np.full(_D,  -50), upper_bound=np.full(_D,  50))
        problem.add_task(T2, dim=_D, lower_bound=np.full(_D, -500), upper_bound=np.full(_D, 500))
        return problem
