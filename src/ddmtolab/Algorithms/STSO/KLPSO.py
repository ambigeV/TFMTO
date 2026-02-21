"""
Knowledge Learning PSO (KLPSO)

This module implements Knowledge Learning PSO for single-objective optimization problems.
A neural network learns successful movement patterns from improved particles.
With probability LR, particles move using the learned knowledge instead of
standard PSO velocity update.

References
----------
    [1] Jiang, Yi, et al. "Knowledge Learning for Evolutionary Computation."
        IEEE Transactions on Evolutionary Computation, 2023.
        DOI: 10.1109/TEVC.2023.3278132

Notes
-----
Author: Jiangtao Shen
Email: j.shen5@exeter.ac.uk
Date: 2026.02.21
Version: 1.0
"""
import time
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from ddmtolab.Methods.Algo_Methods.algo_utils import *


class _KLNet(nn.Module):
    """
    Feedforward network for learning position changes.

    Architecture: dim -> 16 (sigmoid) -> 16 (sigmoid) -> dim (linear)
    Matches MATLAB: newff(..., [16 16 D], {'logsig' 'logsig' 'purelin'})
    """

    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 16),
            nn.Sigmoid(),
            nn.Linear(16, 16),
            nn.Sigmoid(),
            nn.Linear(16, dim),
        )

    def forward(self, x):
        return self.net(x)


class KLPSO:
    """
    Knowledge Learning PSO algorithm for single-objective optimization.

    Attributes
    ----------
    algorithm_information : dict
        Dictionary containing algorithm capabilities and requirements
    """

    algorithm_information = {
        'n_tasks': '[1, K]',
        'dims': 'unequal',
        'objs': 'equal',
        'n_objs': '1',
        'cons': 'unequal',
        'n_cons': '[0, C]',
        'expensive': 'False',
        'knowledge_transfer': 'False',
        'n': 'unequal',
        'max_nfes': 'unequal'
    }

    @classmethod
    def get_algorithm_information(cls, print_info=True):
        return get_algorithm_information(cls, print_info)

    def __init__(self, problem, n=None, max_nfes=None, lr=0.2, ep=10,
                 min_w=0.4, max_w=0.9, c1=0.2, c2=0.2, save_data=True,
                 save_path='./Data', name='KLPSO', disable_tqdm=True):
        """
        Initialize KLPSO algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n : int or List[int], optional
            Population size per task (default: 100)
        max_nfes : int or List[int], optional
            Maximum number of function evaluations per task (default: 10000)
        lr : float, optional
            Learning rate - probability of using neural net prediction (default: 0.2)
        ep : int, optional
            Number of training epochs per generation (default: 10)
        min_w : float, optional
            Minimum inertia weight (default: 0.4)
        max_w : float, optional
            Maximum inertia weight (default: 0.9)
        c1 : float, optional
            Cognitive coefficient (default: 0.2)
        c2 : float, optional
            Social coefficient (default: 0.2)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './Data')
        name : str, optional
            Name for the experiment (default: 'KLPSO')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n = n if n is not None else 100
        self.max_nfes = max_nfes if max_nfes is not None else 10000
        self.lr = lr
        self.ep = ep
        self.min_w = min_w
        self.max_w = max_w
        self.c1 = c1
        self.c2 = c2
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def optimize(self):
        """
        Execute the KLPSO algorithm.

        Returns
        -------
        Results
            Optimization results containing decision variables, objectives, and runtime
        """
        start_time = time.time()
        problem = self.problem
        dims = problem.dims
        nt = problem.n_tasks
        n_per_task = par_list(self.n, nt)
        max_nfes_per_task = par_list(self.max_nfes, nt)

        # Initialize population in [0,1] space and evaluate for each task
        decs = initialization(problem, n_per_task)
        objs, cons = evaluation(problem, decs)
        nfes_per_task = n_per_task.copy()
        all_decs, all_objs, all_cons = init_history(decs, objs, cons)

        # Initialize particle velocities to zero
        vel = [np.zeros_like(d) for d in decs]

        # Initialize personal best positions, objectives, and constraints
        pbest_decs = [d.copy() for d in decs]
        pbest_objs = [o.copy() for o in objs]
        pbest_cons = [c.copy() for c in cons]

        # Initialize global best for each task
        gbest_decs = []
        gbest_objs = []
        gbest_cons = []

        for i in range(nt):
            cvs = np.sum(np.maximum(0, cons[i]), axis=1)
            sort_indices = np.lexsort((objs[i].flatten(), cvs))
            best_idx = sort_indices[0]

            gbest_decs.append(decs[i][best_idx:best_idx + 1, :])
            gbest_objs.append(objs[i][best_idx:best_idx + 1, :])
            gbest_cons.append(cons[i][best_idx:best_idx + 1, :])

        # Neural networks for knowledge learning (one per task)
        nets = [None] * nt
        optimizers = [None] * nt
        trained = [False] * nt

        total_nfes = sum(max_nfes_per_task)
        pbar = tqdm(total=total_nfes, initial=sum(n_per_task), desc=f"{self.name}",
                    disable=self.disable_tqdm)

        while sum(nfes_per_task) < total_nfes:
            # Skip tasks that have exhausted their evaluation budget
            active_tasks = [i for i in range(nt) if nfes_per_task[i] < max_nfes_per_task[i]]
            if not active_tasks:
                break

            for i in active_tasks:
                # Linearly decrease inertia weight from max_w to min_w
                w = self.max_w - (self.max_w - self.min_w) * nfes_per_task[i] / max_nfes_per_task[i]

                # Save old state for training data collection
                old_decs = decs[i].copy()
                old_objs = objs[i].copy()
                old_cons = cons[i].copy()

                # Determine which particles use neural net vs standard PSO
                if trained[i]:
                    use_net = np.random.rand(n_per_task[i]) < self.lr
                else:
                    use_net = np.zeros(n_per_task[i], dtype=bool)

                # Standard PSO update for non-net particles
                pso_idx = np.where(~use_net)[0]
                if len(pso_idx) > 0:
                    r1 = np.random.rand(len(pso_idx), 1)
                    r2 = np.random.rand(len(pso_idx), 1)
                    vel[i][pso_idx] = (w * vel[i][pso_idx] +
                                      self.c1 * r1 * (pbest_decs[i][pso_idx] - decs[i][pso_idx]) +
                                      self.c2 * r2 * (gbest_decs[i] - decs[i][pso_idx]))
                    decs[i][pso_idx] = decs[i][pso_idx] + vel[i][pso_idx]

                # Neural net update for selected particles
                net_idx = np.where(use_net)[0]
                if len(net_idx) > 0:
                    with torch.no_grad():
                        x_t = torch.tensor(decs[i][net_idx], dtype=torch.float32)
                        y_pred = nets[i](x_t).numpy()
                    r_scale = 2 * np.random.rand(len(net_idx), 1)
                    decs[i][net_idx] = decs[i][net_idx] + r_scale * y_pred

                # Clip to [0, 1] boundary
                decs[i] = np.clip(decs[i], 0, 1)

                # Evaluation
                objs[i], cons[i] = evaluation_single(problem, decs[i], i)

                # PBest update
                current_cvs = np.sum(np.maximum(0, cons[i]), axis=1)
                pbest_cvs = np.sum(np.maximum(0, pbest_cons[i]), axis=1)

                improved_pb = (current_cvs < pbest_cvs) | \
                              ((current_cvs == pbest_cvs) & (objs[i].flatten() < pbest_objs[i].flatten()))

                pbest_decs[i][improved_pb] = decs[i][improved_pb]
                pbest_objs[i][improved_pb] = objs[i][improved_pb]
                pbest_cons[i][improved_pb] = cons[i][improved_pb]

                # GBest update
                pbest_cvs = np.sum(np.maximum(0, pbest_cons[i]), axis=1)
                sort_indices = np.lexsort((pbest_objs[i].flatten(), pbest_cvs))
                best_idx = sort_indices[0]

                gbest_cv = np.sum(np.maximum(0, gbest_cons[i]))
                best_cv = pbest_cvs[best_idx]

                if (best_cv < gbest_cv) or \
                        (best_cv == gbest_cv and pbest_objs[i][best_idx] < gbest_objs[i][0]):
                    gbest_decs[i] = pbest_decs[i][best_idx:best_idx + 1, :]
                    gbest_objs[i] = pbest_objs[i][best_idx:best_idx + 1, :]
                    gbest_cons[i] = pbest_cons[i][best_idx:best_idx + 1, :]

                # Identify improved particles for training data collection
                # Matches MATLAB Selection_Tournament with Ep=0:
                #   both feasible AND new has lower obj, OR
                #   both infeasible AND new has lower CV
                old_cvs = np.sum(np.maximum(0, old_cons), axis=1)
                new_cvs = np.sum(np.maximum(0, cons[i]), axis=1)
                both_feasible = (old_cvs <= 0) & (new_cvs <= 0)
                both_infeasible = (old_cvs > 0) & (new_cvs > 0)
                replace = (both_feasible & (objs[i].flatten() < old_objs.flatten())) | \
                          (both_infeasible & (new_cvs < old_cvs))

                if np.any(replace):
                    in_data = old_decs[replace]
                    out_data = decs[i][replace] - old_decs[replace]

                    # Initialize network on first improvement
                    if not trained[i]:
                        nets[i] = _KLNet(dims[i])
                        optimizers[i] = torch.optim.SGD(nets[i].parameters(), lr=0.1, momentum=0.9)
                        trained[i] = True

                    # Train network on this generation's improved particles
                    in_tensor = torch.tensor(in_data, dtype=torch.float32)
                    out_tensor = torch.tensor(out_data, dtype=torch.float32)
                    loss_fn = nn.MSELoss()
                    nets[i].train()
                    for _ in range(self.ep):
                        pred = nets[i](in_tensor)
                        loss = loss_fn(pred, out_tensor)
                        optimizers[i].zero_grad()
                        loss.backward()
                        optimizers[i].step()

                nfes_per_task[i] += n_per_task[i]
                pbar.update(n_per_task[i])

                # Append current population to history
                append_history(all_decs[i], decs[i], all_objs[i], objs[i], all_cons[i], cons[i])

        pbar.close()
        runtime = time.time() - start_time

        # Save results
        results = build_save_results(all_decs=all_decs, all_objs=all_objs, runtime=runtime, max_nfes=nfes_per_task,
                                     all_cons=all_cons, bounds=problem.bounds, save_path=self.save_path,
                                     filename=self.name, save_data=self.save_data)

        return results
