from ddmtolab.Problems.BasicFunctions.basic_functions import *
from ddmtolab.Methods.mtop import MTOP
import numpy as np


class STSOtest:
    """
    Modified Single-Task Optimization (STSOtest) benchmark problems.

    This class provides a set of standard single-objective optimization
    benchmark functions (e.g., Ackley, Rastrigin, Sphere) configured as
    Multi-Task Optimization Problems (MTOPs) with only one task.
    Unlike CLASSICALSO, this class uses non-identity rotation matrices (M)
    and non-zero offset vectors (o) to create different but corresponding
    problem instances.

    Parameters
    ----------
    dim : int, optional
        The dimensionality of the search space for all tasks (default is 50).

    Attributes
    ----------
    dim : int
        The dimensionality of the problem.
    M : numpy.ndarray
        Rotation matrix for transformation (orthogonal matrix).
    o : numpy.ndarray
        Offset vector for shifting the optimum.
    """

    def __init__(self, dim=50):
        self.dim = dim
        np.random.seed(1234)
        random_matrix = np.random.randn(self.dim, self.dim)
        self.M, _ = np.linalg.qr(random_matrix)
        self.o = np.random.uniform(0., 10., (1, self.dim))

    def P1(self) -> MTOP:
        """
        Generates a single-task MTOP based on the **Ackley** function.

        The search space is set to [-100.0, 100.0] in all dimensions.
        Uses fixed rotation matrix M and offset vector o.

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing the modified Ackley task.
        """

        def Task(x):
            x = np.atleast_2d(x)
            return Ackley(x, self.M, self.o, 0.0)

        problem = MTOP()
        lb = np.full(self.dim, -100.0)
        ub = np.full(self.dim, 100.0)
        problem.add_task(Task, dim=self.dim, lower_bound=lb, upper_bound=ub)
        return problem

    def P2(self) -> MTOP:
        """
        Generates a single-task MTOP based on the **Elliptic** function.

        The search space is set to [-100.0, 100.0] in all dimensions.
        Uses fixed rotation matrix M and offset vector o.

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing the modified Elliptic task.
        """

        def Task(x):
            x = np.atleast_2d(x)
            return Elliptic(x, self.M, self.o, 0.0)

        problem = MTOP()
        lb = np.full(self.dim, -100.0)
        ub = np.full(self.dim, 100.0)
        problem.add_task(Task, dim=self.dim, lower_bound=lb, upper_bound=ub)
        return problem

    def P3(self) -> MTOP:
        """
        Generates a single-task MTOP based on the **Griewank** function.

        The search space is set to [-100.0, 100.0] in all dimensions.
        Uses fixed rotation matrix M and offset vector o.

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing the modified Griewank task.
        """

        def Task(x):
            x = np.atleast_2d(x)
            return Griewank(x, self.M, self.o, 0.0)

        problem = MTOP()
        lb = np.full(self.dim, -100.0)
        ub = np.full(self.dim, 100.0)
        problem.add_task(Task, dim=self.dim, lower_bound=lb, upper_bound=ub)
        return problem

    def P4(self) -> MTOP:
        """
        Generates a single-task MTOP based on the **Rastrigin** function.

        The search space is set to [-100.0, 100.0] in all dimensions.
        Uses fixed rotation matrix M and offset vector o.

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing the modified Rastrigin task.
        """

        def Task(x):
            x = np.atleast_2d(x)
            return Rastrigin(x, self.M, self.o, 0.0)

        problem = MTOP()
        lb = np.full(self.dim, -100.0)
        ub = np.full(self.dim, 100.0)
        problem.add_task(Task, dim=self.dim, lower_bound=lb, upper_bound=ub)
        return problem

    def P5(self) -> MTOP:
        """
        Generates a single-task MTOP based on the **Rosenbrock** function.

        The search space is set to [-100.0, 100.0] in all dimensions.
        Uses fixed rotation matrix M and offset vector o.

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing the modified Rosenbrock task.
        """

        def Task(x):
            x = np.atleast_2d(x)
            return Rosenbrock(x, self.M, self.o, 0.0)

        problem = MTOP()
        lb = np.full(self.dim, -100.0)
        ub = np.full(self.dim, 100.0)
        problem.add_task(Task, dim=self.dim, lower_bound=lb, upper_bound=ub)
        return problem

    def P6(self) -> MTOP:
        """
        Generates a single-task MTOP based on the **Schwefel** function (F6).

        The search space is set to [-500.0, 500.0] in all dimensions.
        Uses fixed rotation matrix M and offset vector o.

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing the modified Schwefel task.
        """

        def Task(x):
            x = np.atleast_2d(x)
            return Schwefel(x, self.M, self.o, 0.0)

        problem = MTOP()
        lb = np.full(self.dim, -500.0)
        ub = np.full(self.dim, 500.0)
        problem.add_task(Task, dim=self.dim, lower_bound=lb, upper_bound=ub)
        return problem

    def P7(self) -> MTOP:
        """
        Generates a single-task MTOP based on the **Schwefel 2.22** function (F7).

        The search space is set to [-100.0, 100.0] in all dimensions.
        Uses fixed rotation matrix M and offset vector o.

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing the modified Schwefel 2.22 task.
        """

        def Task(x):
            x = np.atleast_2d(x)
            return Schwefel2(x, self.M, self.o, 0.0)

        problem = MTOP()
        lb = np.full(self.dim, -100.0)
        ub = np.full(self.dim, 100.0)
        problem.add_task(Task, dim=self.dim, lower_bound=lb, upper_bound=ub)
        return problem

    def P8(self) -> MTOP:
        """
        Generates a single-task MTOP based on the **Sphere** function.

        The search space is set to [-100.0, 100.0] in all dimensions.
        Uses fixed rotation matrix M and offset vector o.

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing the modified Sphere task.
        """

        def Task(x):
            x = np.atleast_2d(x)
            return Sphere(x, self.M, self.o, 0.0)

        problem = MTOP()
        lb = np.full(self.dim, -100.0)
        ub = np.full(self.dim, 100.0)
        problem.add_task(Task, dim=self.dim, lower_bound=lb, upper_bound=ub)
        return problem

    def P9(self) -> MTOP:
        """
        Generates a single-task MTOP based on the **Weierstrass** function.

        The search space is set to [-0.5, 0.5] in all dimensions.
        Uses fixed rotation matrix M and offset vector o.

        Returns
        -------
        MTOP
            A Multi-Task Optimization Problem instance containing the modified Weierstrass task.
        """

        def Task(x):
            x = np.atleast_2d(x)
            return Weierstrass(x, self.M, self.o, 0.0)

        problem = MTOP()
        lb = np.full(self.dim, -0.5)
        ub = np.full(self.dim, 0.5)
        problem.add_task(Task, dim=self.dim, lower_bound=lb, upper_bound=ub)
        return problem