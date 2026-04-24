import pkgutil
import scipy.io
import io
from ddmtolab.Problems.BasicFunctions.basic_functions import *
from ddmtolab.Methods.mtop import MTOP
import numpy as np

_D    = 10          # dimensions exposed to the optimiser
_TAIL = 50 - _D    # dimensions fixed at their global-optimum value (= 40)

# Schwefel's global optimum is at z_i = 420.9687... (not at z=0).
# With identity rotation and go=zeros, the minimiser in x-space is this constant.
_SCHWEFEL_OPT = 420.9687462275636

# Rosenbrock's global optimum is at z=(1,...,1) (not z=0).
# With identity rotation and go=zeros, the minimiser in x-space is ones.
_ROSENBROCK_OPT = 1.0


class CEC17MTSO_10D_v2:
    """
    10-Dimensional version of the CEC 2017 MTSO benchmark (P1–P9).

    Uses the same 50D rotation matrices and global optima from the original
    CEC17 competition .mat files.  The optimiser sees only the first 10
    decision variables; the remaining 40 are fixed at the true global-optimum
    coordinates for those dimensions before each function call.

    This is the slice-based counterpart of CEC17MTSO_30D, applied at D=10.
    The CI/PI/NI and HS/MS/LS task-relationship characteristics are preserved
    from the 50D benchmark.

    Notes
    -----
    - D_exposed = 10,  D_fixed = 40
    - Reuses data_cec17mtso/ .mat files — no new data files required
    - P6 Task 2 (originally 25D) is reduced to 10D; tail = go2[10:] (zeros)
    - Rosenbrock tails fixed at ones (true x* for identity rotation + go=zeros)
    - Schwefel tails fixed at 420.9687·ones (true x* for identity rotation + go=zeros)
    """

    problem_information = {
        'n_cases': 9,
        'n_tasks': '2',
        'n_dims': '10',
        'n_objs': '1',
        'n_cons': '0',
        'type': 'synthetic',
    }

    def __init__(self):
        self.data_dir = 'data_cec17mtso'

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fix_tail(fn_full, tail):
        """
        Wrap a full-dimensional callable to accept _D-dimensional input.

        Handles both vectorised (n, _D) and single-row (_D,) calls by tiling
        the fixed tail before reconstructing the full input vector.
        """
        tail = np.asarray(tail, dtype=float).copy()
        def inner(x):
            x2 = np.atleast_2d(x)
            tail_block = np.tile(tail, (x2.shape[0], 1))
            return fn_full(np.hstack([x2, tail_block]))
        return inner

    def _load(self, filename):
        data_bytes = pkgutil.get_data('ddmtolab.Problems.MTSO',
                                      f'{self.data_dir}/{filename}')
        return scipy.io.loadmat(io.BytesIO(data_bytes))

    # ------------------------------------------------------------------
    # Problems
    # ------------------------------------------------------------------

    def P1(self) -> MTOP:
        """CI-HS (10D): T1 Griewank + T2 Rastrigin — complete intersection, high similarity."""
        mat  = self._load('CI_H.mat')
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
        """CI-MS (10D): T1 Ackley + T2 Rastrigin — complete intersection, medium similarity."""
        mat  = self._load('CI_M.mat')
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
        """CI-LS (10D): T1 Ackley + T2 Schwefel — complete intersection, low similarity."""
        mat  = self._load('CI_L.mat')
        rot1 = mat['Rotation_Task1'].squeeze()
        go1  = mat['GO_Task1'].squeeze()
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
        """PI-HS (10D): T1 Rastrigin + T2 Sphere — partial intersection, high similarity."""
        mat  = self._load('PI_H.mat')
        rot1 = mat['Rotation_Task1'].squeeze()
        go1  = mat['GO_Task1'].squeeze()
        go2  = mat['GO_Task2'].squeeze()
        rot2 = np.eye(50, dtype=float)

        T1 = self._fix_tail(lambda x: Rastrigin(x, rot1, go1, 0.), go1[_D:])
        T2 = self._fix_tail(lambda x: Sphere(x, rot2, go2, 0),     go2[_D:])

        problem = MTOP()
        problem.add_task(T1, dim=_D, lower_bound=np.full(_D,  -50), upper_bound=np.full(_D,  50))
        problem.add_task(T2, dim=_D, lower_bound=np.full(_D, -100), upper_bound=np.full(_D, 100))
        return problem

    def P5(self) -> MTOP:
        """PI-MS (10D): T1 Ackley + T2 Rosenbrock — partial intersection, medium similarity."""
        mat  = self._load('PI_M.mat')
        rot1 = mat['Rotation_Task1'].squeeze()
        go1  = mat['GO_Task1'].squeeze()
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
        PI-LS (10D): T1 Ackley (10D) + T2 Weierstrass (10D) — partial intersection,
        low similarity.

        T2 was originally 25D; reduced to 10D with tail = go2[10:] (zeros).
        """
        mat  = self._load('PI_L.mat')
        rot1 = mat['Rotation_Task1'].squeeze()   # (50, 50)
        go1  = mat['GO_Task1'].squeeze()          # (50,)
        rot2 = mat['Rotation_Task2'].squeeze()   # (25, 25)
        go2  = mat['GO_Task2'].squeeze()          # (25,)

        T1 = self._fix_tail(lambda x: Ackley(x, rot1, go1, 0.),       go1[_D:])
        T2 = self._fix_tail(lambda x: Weierstrass(x, rot2, go2, 0.),  go2[_D:])

        problem = MTOP()
        problem.add_task(T1, dim=_D, lower_bound=np.full(_D, -50),   upper_bound=np.full(_D,  50))
        problem.add_task(T2, dim=_D, lower_bound=np.full(_D, -0.5),  upper_bound=np.full(_D,  0.5))
        return problem

    def P7(self) -> MTOP:
        """NI-HS (10D): T1 Rosenbrock + T2 Rastrigin — no intersection, high similarity."""
        mat  = self._load('NI_H.mat')
        rot2 = mat['Rotation_Task2'].squeeze()
        go2  = mat['GO_Task2'].squeeze()
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
        """NI-MS (10D): T1 Griewank + T2 Weierstrass — no intersection, medium similarity."""
        mat  = self._load('NI_M.mat')
        rot1 = mat['Rotation_Task1'].squeeze()
        go1  = mat['GO_Task1'].squeeze()
        rot2 = mat['Rotation_Task2'].squeeze()
        go2  = mat['GO_Task2'].squeeze()

        T1 = self._fix_tail(lambda x: Griewank(x, rot1, go1, 0.),    go1[_D:])
        T2 = self._fix_tail(lambda x: Weierstrass(x, rot2, go2, 0.), go2[_D:])

        problem = MTOP()
        problem.add_task(T1, dim=_D, lower_bound=np.full(_D, -100), upper_bound=np.full(_D, 100))
        problem.add_task(T2, dim=_D, lower_bound=np.full(_D, -0.5), upper_bound=np.full(_D,  0.5))
        return problem

    def P9(self) -> MTOP:
        """NI-LS (10D): T1 Rastrigin + T2 Schwefel — no intersection, low similarity."""
        mat  = self._load('NI_L.mat')
        rot1 = mat['Rotation_Task1'].squeeze()
        go1  = mat['GO_Task1'].squeeze()
        rot2 = np.eye(50, dtype=float)
        go2  = np.zeros(50, dtype=float)

        T1 = self._fix_tail(lambda x: Rastrigin(x, rot1, go1, 0.),  go1[_D:])
        T2 = self._fix_tail(lambda x: Schwefel(x, rot2, go2, 0.),
                            np.full(_TAIL, _SCHWEFEL_OPT))

        problem = MTOP()
        problem.add_task(T1, dim=_D, lower_bound=np.full(_D,  -50), upper_bound=np.full(_D,  50))
        problem.add_task(T2, dim=_D, lower_bound=np.full(_D, -500), upper_bound=np.full(_D, 500))
        return problem
