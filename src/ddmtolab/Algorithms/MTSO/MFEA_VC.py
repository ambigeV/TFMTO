"""
Multifactorial Evolutionary Algorithm with Variational Crossover (MFEA-VC)

This module implements MFEA-VC for multi-task optimization using a contrastive
Variational Auto-Encoder (VAE) to guide knowledge transfer in early generations.

References
----------
    [1] Wang, Ruilin, et al. "Contrastive Variational Auto-Encoder Driven
        Convergence Guidance in Evolutionary Multitasking." Applied Soft
        Computing, 163: 111883, 2024.

Notes
-----
Author: Jiangtao Shen (DDMTOLab adaptation)
Date: 2026.02.22
Version: 1.0
"""
import time
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from ddmtolab.Methods.Algo_Methods.algo_utils import *


class MFEA_VC:
    """
    Multifactorial Evolutionary Algorithm with Variational Crossover.

    Uses a VAE (with random weights, no training) to generate cross-task
    individuals for the first `vae_gens` generations. The VAE encodes both
    tasks' population data into a shared latent space and decodes to produce
    mixed-task offspring used as SBX crossover partners.

    After `vae_gens` generations, reverts to standard MFEA behavior with
    SBX crossover and polynomial mutation.

    Attributes
    ----------
    algorithm_information : dict
        Dictionary containing algorithm capabilities and requirements
    """

    algorithm_information = {
        'n_tasks': '[2, K]',
        'dims': 'unequal',
        'objs': 'equal',
        'n_objs': '1',
        'cons': 'unequal',
        'n_cons': '[0, C]',
        'expensive': 'False',
        'knowledge_transfer': 'True',
        'n': 'equal',
        'max_nfes': 'equal'
    }

    @classmethod
    def get_algorithm_information(cls, print_info=True):
        return get_algorithm_information(cls, print_info)

    def __init__(self, problem, n=None, max_nfes=None, rmp=0.3, muc=2, mum=5,
                 vae_gens=25, lam=0.8, save_data=True, save_path='./Data',
                 name='MFEA-VC', disable_tqdm=True):
        """
        Initialize MFEA-VC algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n : int, optional
            Population size per task (default: 100)
        max_nfes : int, optional
            Maximum number of function evaluations per task (default: 10000)
        rmp : float, optional
            Random mating probability (default: 0.3)
        muc : float, optional
            Distribution index for SBX crossover (default: 2)
        mum : float, optional
            Distribution index for polynomial mutation (default: 5)
        vae_gens : int, optional
            Number of generations to use VAE-guided crossover (default: 25)
        lam : float, optional
            Lambda scaling factor for VAE latent space (default: 0.8)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './Data')
        name : str, optional
            Name for the experiment (default: 'MFEA-VC')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n = n if n is not None else 100
        self.max_nfes = max_nfes if max_nfes is not None else 10000
        self.rmp = rmp
        self.muc = muc
        self.mum = mum
        self.vae_gens = vae_gens
        self.lam = lam
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def optimize(self):
        """
        Execute the MFEA-VC algorithm.

        Returns
        -------
        Results
            Optimization results containing decision variables, objectives, and runtime
        """
        start_time = time.time()
        problem = self.problem
        nt = problem.n_tasks
        dims = problem.dims
        n = self.n
        max_nfes_per_task = par_list(self.max_nfes, nt)
        max_nfes = self.max_nfes * nt
        pop_size = n * nt

        # Initialize population and evaluate
        decs = initialization(problem, n)
        objs, cons = evaluation(problem, decs)
        nfes = n * nt
        all_decs, all_objs, all_cons = init_history(decs, objs, cons)

        # Transform to unified space
        pop_decs, pop_cons = space_transfer(problem=problem, decs=decs, cons=cons,
                                            type='uni', padding='mid')
        pop_objs = objs
        pop_sfs = [np.full((n, 1), fill_value=i) for i in range(nt)]

        pbar = tqdm(total=max_nfes, initial=nfes, desc=f"{self.name}",
                    disable=self.disable_tqdm)

        gen = 1
        while nfes < max_nfes:
            # --- VAE generation (first vae_gens generations, 2-task only) ---
            vae_decs = None
            if gen <= self.vae_gens and nt == 2:
                vae_decs = _generate_vae_individuals(
                    pop_decs, pop_objs, pop_sfs, nt, self.lam)

            # Merge populations
            m_decs, m_objs, m_cons, m_sfs = vstack_groups(
                pop_decs, pop_objs, pop_cons, pop_sfs)

            maxD = m_decs.shape[1]

            # --- Generation ---
            off_decs = np.zeros_like(m_decs)
            off_objs = np.full_like(m_objs, np.inf)
            off_cons = np.zeros_like(m_cons)
            off_sfs = np.zeros_like(m_sfs)

            shuffled = np.random.permutation(pop_size)

            for pair_idx in range(pop_size // 2):
                p1 = shuffled[pair_idx]
                p2 = shuffled[pair_idx + pop_size // 2]
                sf1 = m_sfs[p1].item()
                sf2 = m_sfs[p2].item()
                idx1 = pair_idx * 2
                idx2 = pair_idx * 2 + 1

                if sf1 == sf2 or np.random.rand() < self.rmp:
                    # --- Transfer: crossover ---
                    if vae_decs is not None and len(vae_decs) > 0:
                        # VAE-guided crossover: crossover with VAE individual
                        vi = np.random.randint(len(vae_decs))
                        vae_dec = vae_decs[vi]

                        off_decs[idx1], _ = crossover(
                            m_decs[p1], vae_dec, mu=self.muc)
                        off_decs[idx2], _ = crossover(
                            m_decs[p2], vae_dec, mu=self.muc)

                        # Trim to proper length (in unified space, already maxD)
                        off_decs[idx1] = off_decs[idx1][:maxD]
                        off_decs[idx2] = off_decs[idx2][:maxD]
                    else:
                        # Standard SBX crossover
                        off_decs[idx1], off_decs[idx2] = crossover(
                            m_decs[p1], m_decs[p2], mu=self.muc)

                    # Task imitation: random parent's MFFactor
                    off_sfs[idx1] = np.random.choice([sf1, sf2])
                    off_sfs[idx2] = np.random.choice([sf1, sf2])
                else:
                    # --- No transfer: polynomial mutation ---
                    off_decs[idx1] = mutation(m_decs[p1], mu=self.mum)
                    off_decs[idx2] = mutation(m_decs[p2], mu=self.mum)
                    off_sfs[idx1] = sf1
                    off_sfs[idx2] = sf2

                # Clip to [0, 1]
                off_decs[idx1] = np.clip(off_decs[idx1], 0, 1)
                off_decs[idx2] = np.clip(off_decs[idx2], 0, 1)

            # --- Evaluation ---
            for idx in range(pop_size):
                t = off_sfs[idx].item()
                off_objs[idx], off_cons[idx] = evaluation_single(
                    problem, off_decs[idx, :dims[t]], t)

            nfes += pop_size
            pbar.update(pop_size)

            # --- Selection ---
            merged_decs = np.vstack([m_decs, off_decs])
            merged_objs = np.vstack([m_objs, off_objs])
            merged_cons = np.vstack([m_cons, off_cons])
            merged_sfs = np.vstack([m_sfs, off_sfs])

            pop_decs, pop_objs, pop_cons, pop_sfs = [], [], [], []
            for t in range(nt):
                indices = np.where(merged_sfs.flatten() == t)[0]
                t_decs, t_objs, t_cons = select_by_index(
                    indices, merged_decs, merged_objs, merged_cons)
                sel = selection_elit(objs=t_objs, n=n, cons=t_cons)
                pop_decs.append(t_decs[sel])
                pop_objs.append(t_objs[sel])
                pop_cons.append(t_cons[sel])
                pop_sfs.append(np.full((n, 1), t))

            # Record history
            real_decs, real_cons = space_transfer(
                problem, decs=pop_decs, cons=pop_cons, type='real')
            append_history(all_decs, real_decs, all_objs, pop_objs, all_cons, real_cons)

            gen += 1

        pbar.close()
        runtime = time.time() - start_time

        results = build_save_results(
            all_decs=all_decs, all_objs=all_objs, runtime=runtime,
            max_nfes=max_nfes_per_task, all_cons=all_cons,
            bounds=problem.bounds, save_path=self.save_path,
            filename=self.name, save_data=self.save_data)

        return results


# ============================================================
# VAE model and generation helpers
# ============================================================

class _SimpleVAE(nn.Module):
    """
    VAE matching the MATLAB implementation architecture.

    Encoder: input → FC(H) → ReLU → FC(H) → ReLU → FC(H) → ReLU →
             FC(H) → Sigmoid → FC_mean(L) → FC_logvar(L)
    Decoder: FC(H) → ReLU → FC(H) → ReLU → FC(H) → ReLU →
             FC(H) → Sigmoid → FC(input_size)

    Note: fc_logvar takes fc_mean's output as input (sequential in MATLAB).
    """

    def __init__(self, input_size, hidden_size=256, latent_size=200):
        super().__init__()
        self.encoder_body = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid(),
        )
        # Sequential: sigmoid → fc_mean → fc_logvar (logvar takes mean as input)
        self.fc_mean = nn.Linear(hidden_size, latent_size)
        self.fc_logvar = nn.Linear(latent_size, latent_size)

        self.decoder = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, input_size),
        )

    @torch.no_grad()
    def generate(self, x, lam=0.8):
        """
        Encode, reparameterize, and decode.

        Parameters
        ----------
        x : torch.Tensor
            Combined data from both tasks, shape (n1+n2, input_size).
            First n1 rows = task 1, last n2 rows = task 2.
        lam : float
            Lambda scaling factor for latent space.

        Returns
        -------
        output : np.ndarray
            Decoded output, shape (n_generated, input_size)
        """
        h = self.encoder_body(x)
        z_mean_all = self.fc_mean(h)
        z_logvar_all = self.fc_logvar(z_mean_all)

        n_total = x.shape[0]
        n_half = n_total // 2

        # Task 1's encoded output → mean, Task 2's → logvar
        z_mean = z_mean_all[:n_half]
        z_logvar = z_logvar_all[n_half:]

        # Reparameterization
        eps = torch.randn_like(z_mean)
        z = z_mean + torch.exp(0.5 * z_logvar) * eps

        # Decode with lambda scaling
        output = self.decoder(z * lam)
        return output.numpy()


def _generate_vae_individuals(pop_decs, pop_objs, pop_sfs, nt, lam):
    """
    Generate VAE-guided individuals for knowledge transfer.

    Prepares population data, passes through untrained VAE, and extracts
    decision variables for use as crossover partners.

    Parameters
    ----------
    pop_decs : list of np.ndarray
        Population decision variables per task (unified space)
    pop_objs : list of np.ndarray
        Population objective values per task
    pop_sfs : list of np.ndarray
        Skill factors per task
    nt : int
        Number of tasks (must be 2)
    lam : float
        Lambda scaling for VAE latent space

    Returns
    -------
    vae_decs : list of np.ndarray
        VAE-generated decision vectors for crossover
    """
    if nt != 2:
        return []

    maxD = pop_decs[0].shape[1]
    desired_cols = 100

    # Build data matrices: [Dec * 10000; MFObj; TaskLabel * 10000]
    data_tasks = []
    for t in range(nt):
        n_t = pop_decs[t].shape[0]
        dec_scaled = pop_decs[t] * 10000.0  # (n_t, maxD)
        obj_vals = pop_objs[t]  # (n_t, n_objs)
        task_label = np.full((n_t, 1), (t + 1) * 10000.0)
        # data per individual: [Dec*10000, MFObj, TaskLabel*10000]
        data_t = np.hstack([dec_scaled, obj_vals, task_label])  # (n_t, maxD+n_objs+1)
        data_tasks.append(data_t)

    # Pad or truncate to desired_cols per task
    for t in range(nt):
        n_t = data_tasks[t].shape[0]
        if n_t > desired_cols:
            data_tasks[t] = data_tasks[t][:desired_cols]
        elif n_t < desired_cols:
            extra_idx = np.random.randint(0, n_t, size=desired_cols - n_t)
            data_tasks[t] = np.vstack([data_tasks[t], data_tasks[t][extra_idx]])

    # 50/50 train split (train = test in MATLAB code)
    n_train = desired_cols // 2
    for t in range(nt):
        perm = np.random.permutation(desired_cols)
        data_tasks[t] = data_tasks[t][perm[:n_train]]  # (50, features)

    # Remove TaskLabel column for encoder input
    X1 = data_tasks[0][:, :-1]  # (50, maxD + n_objs)
    X2 = data_tasks[1][:, :-1]  # (50, maxD + n_objs)
    input_size = X1.shape[1]

    # Combine for encoding
    x_combined = np.vstack([X1, X2]).astype(np.float32)  # (100, input_size)

    # Build VAE with random weights (no training, matching MATLAB istraining=false)
    vae = _SimpleVAE(input_size, hidden_size=256, latent_size=200)
    vae.eval()

    # Generate
    x_tensor = torch.from_numpy(x_combined)
    output = vae.generate(x_tensor, lam=lam)  # (50, input_size)

    # Split into task 1 and task 2
    n_gen = output.shape[0]
    n_half = n_gen // 2
    new_x1 = output[:n_half]   # (25, input_size)
    new_x2 = output[n_half:]   # (25, input_size)

    # Extract Dec values (first maxD columns)
    vae_decs = []
    for row in new_x1:
        vae_decs.append(row[:maxD].copy())
    for row in new_x2:
        vae_decs.append(row[:maxD].copy())

    return vae_decs
