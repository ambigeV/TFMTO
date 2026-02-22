"""
Evolutionary Multitasking for Multi-objective Optimization Based on Generative Strategies (EMT-GS)

This module implements EMT-GS for multi-task multi-objective optimization problems.
EMT-GS uses Generative Adversarial Networks (GANs) to transfer knowledge between tasks.

References
----------
    [1] Z. Liang, Y. Zhu, X. Wang, Z. Li, and Z. Zhu, "Evolutionary Multitasking for Multi-objective Optimization Based on Generative Strategies," IEEE Transactions on Evolutionary Computation, 2022.

Notes
-----
Author: Jiangtao Shen
Email: j.shen5@exeter.ac.uk
Date: 2026.02.22
Version: 1.0
"""
import time
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from ddmtolab.Algorithms.STMO.NSGA_II import nsga2_sort
from ddmtolab.Methods.Algo_Methods.algo_utils import *


class _Generator(nn.Module):
    """Generator network: maps source task solutions to target task space."""

    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.fc_out = nn.Linear(dim, dim)

        # Small initialization like MATLAB (sigma=0.03 for fc1, 0.06 for others)
        nn.init.normal_(self.fc1.weight, 0, 0.03)
        nn.init.zeros_(self.fc1.bias)
        nn.init.normal_(self.fc2.weight, 0, 0.06)
        nn.init.zeros_(self.fc2.bias)
        nn.init.normal_(self.fc_out.weight, 0, 0.06)
        nn.init.zeros_(self.fc_out.bias)

    def forward(self, x):
        h = torch.nn.functional.leaky_relu(self.fc1(x), 0.5)
        h = self.bn1(h)
        h = torch.nn.functional.leaky_relu(self.fc2(h), 0.5)
        h = self.bn2(h)
        out = torch.sigmoid(self.fc_out(h))
        return out


class _Discriminator(nn.Module):
    """Discriminator network: distinguishes real target solutions from generated ones."""

    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc_out = nn.Linear(dim, 1)
        self.dropout = nn.Dropout(0.5)

        nn.init.normal_(self.fc1.weight, 0, 0.03)
        nn.init.zeros_(self.fc1.bias)
        nn.init.normal_(self.fc_out.weight, 0, 0.06)
        nn.init.zeros_(self.fc_out.bias)

    def forward(self, x):
        h = torch.nn.functional.leaky_relu(self.fc1(x), 0.5)
        h = self.dropout(h)
        h = self.bn1(h)
        out = torch.sigmoid(self.fc_out(h))
        return out


class EMT_GS:
    """
    Evolutionary Multitasking for Multi-objective Optimization Based on Generative Strategies.

    This algorithm features:
    - GAN-based cross-task knowledge transfer
    - Generator maps source task solutions to target task space
    - DE mutation with rand-or-best strategy
    - NSGA-II based environmental selection per task

    Attributes
    ----------
    algorithm_information : dict
        Dictionary containing algorithm capabilities and requirements
    """

    algorithm_information = {
        'n_tasks': '[2, K]',
        'dims': 'unequal',
        'objs': 'unequal',
        'n_objs': '[2, M]',
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

    def __init__(self, problem, n=None, max_nfes=None,
                 G=10, lrD=0.0002, lrG=0.0003, BS=10,
                 pp=0.5, CR=0.6,
                 save_data=True, save_path='./Data',
                 name='EMT-GS', disable_tqdm=True):
        """
        Initialize EMT-GS algorithm.

        Parameters
        ----------
        problem : MTOP
            Multi-task optimization problem instance
        n : int, optional
            Population size per task (default: 100)
        max_nfes : int, optional
            Maximum number of function evaluations per task (default: 10000)
        G : int, optional
            GAN training gap in generations (default: 10)
        lrD : float, optional
            Learning rate for discriminator (default: 0.0002)
        lrG : float, optional
            Learning rate for generator (default: 0.0003)
        BS : int, optional
            Batch size for GAN training (default: 10)
        pp : float, optional
            Probability of using random (vs best) base vector (default: 0.5)
        CR : float, optional
            Crossover rate for DE (default: 0.6)
        save_data : bool, optional
            Whether to save optimization data (default: True)
        save_path : str, optional
            Path to save results (default: './Data')
        name : str, optional
            Name for the experiment (default: 'EMT-GS')
        disable_tqdm : bool, optional
            Whether to disable progress bar (default: True)
        """
        self.problem = problem
        self.n = n if n is not None else 100
        self.max_nfes = max_nfes if max_nfes is not None else 10000
        self.G = G
        self.lrD = lrD
        self.lrG = lrG
        self.BS = BS
        self.pp = pp
        self.CR = CR
        self.save_data = save_data
        self.save_path = save_path
        self.name = name
        self.disable_tqdm = disable_tqdm

    def optimize(self):
        """
        Execute the EMT-GS algorithm.

        Returns
        -------
        Results
            Optimization results containing decision variables, objectives, constraints, and runtime
        """
        start_time = time.time()
        problem = self.problem
        nt = problem.n_tasks
        n = self.n
        dims = problem.dims
        d_uni = max(dims)
        max_nfes_per_task = par_list(self.max_nfes, nt)
        max_nfes = self.max_nfes * nt

        # Initialize population in native space, then transfer to unified
        decs = initialization(problem, n)
        objs, cons = evaluation(problem, decs)
        nfes = n * nt

        # Transfer to unified space for operations
        decs, objs, cons = space_transfer(problem, decs, objs, cons, type='uni')

        # NSGA-II sorting per task
        for t in range(nt):
            rank_t, _, _ = nsga2_sort(objs[t], cons[t])
            sorted_idx = np.argsort(rank_t)
            decs[t] = decs[t][sorted_idx]
            objs[t] = objs[t][sorted_idx]
            cons[t] = cons[t][sorted_idx]

        # Transfer back to native for history
        decs_native, objs_native, cons_native = space_transfer(
            problem, decs, objs, cons, type='real')
        all_decs, all_objs, all_cons = init_history(decs_native, objs_native, cons_native)

        # Save previous population for DE mutation
        prepop = [d.copy() for d in decs]

        # Initialize GANs: GAN[t][k] maps task k solutions -> task t space
        generators = {}
        discriminators = {}
        opt_G = {}
        opt_D = {}
        gan_outputs = {}

        device = torch.device('cpu')

        for t in range(nt):
            for k in range(nt):
                if t == k:
                    continue
                gen, disc, o_g, o_d = self._init_gan(d_uni, device)
                generators[(t, k)] = gen
                discriminators[(t, k)] = disc
                opt_G[(t, k)] = o_g
                opt_D[(t, k)] = o_d
                # Initial training (20 epochs)
                gan_out = self._train_gan(
                    generators[(t, k)], discriminators[(t, k)],
                    opt_G[(t, k)], opt_D[(t, k)],
                    decs[t], decs[k], epochs=20, device=device
                )
                gan_outputs[(t, k)] = gan_out

        pbar = tqdm(total=max_nfes, initial=nfes, desc=f"{self.name}",
                    disable=self.disable_tqdm)

        gen_count = 0
        while nfes < max_nfes:
            gen_count += 1

            # Update GANs
            for t in range(nt):
                for k in range(nt):
                    if t == k:
                        continue
                    if gen_count % self.G == 0:
                        # Retrain GAN (2 epochs)
                        gan_out = self._train_gan(
                            generators[(t, k)], discriminators[(t, k)],
                            opt_G[(t, k)], opt_D[(t, k)],
                            decs[t], decs[k], epochs=2, device=device
                        )
                        gan_outputs[(t, k)] = gan_out
                    else:
                        # Generate using existing GAN
                        if np.random.rand() < 0.5:
                            gan_outputs[(t, k)] = self._generate_gan(
                                generators[(t, k)], decs[k], device)
                        else:
                            # Use GAN{k,t} instead (swapped generator)
                            gan_outputs[(t, k)] = self._generate_gan(
                                generators[(k, t)], decs[k], device)

            # Generation: create offspring via DE with GAN transfer
            off_decs, off_sfs = self._generation(
                decs, prepop, gan_outputs, n, nt, d_uni)

            # Save current as prepop for next generation
            prepop = [d.copy() for d in decs]

            # Evaluate and select per task
            for t in range(nt):
                # Get offspring for this task
                mask = off_sfs == t
                off_decs_t = off_decs[mask]
                if len(off_decs_t) == 0:
                    continue

                # Trim to task dimension for evaluation, then evaluate
                off_decs_t_native = off_decs_t[:, :dims[t]]
                off_objs_t, off_cons_t = evaluation_single(
                    problem, off_decs_t_native, t)
                nfes += len(off_decs_t)
                pbar.update(len(off_decs_t))

                # Pad objs/cons back to unified shape
                off_objs_uni = np.zeros((len(off_decs_t), objs[t].shape[1]))
                off_objs_uni[:, :off_objs_t.shape[1]] = off_objs_t
                off_cons_uni = np.zeros((len(off_decs_t), cons[t].shape[1]))
                if off_cons_t.shape[1] > 0:
                    off_cons_uni[:, :off_cons_t.shape[1]] = off_cons_t

                # Merge parent + offspring
                merged_decs = np.vstack([decs[t], off_decs_t])
                merged_objs = np.vstack([objs[t], off_objs_uni])
                merged_cons = np.vstack([cons[t], off_cons_uni])

                # NSGA-II selection
                rank_t, _, _ = nsga2_sort(merged_objs, merged_cons)
                sorted_idx = np.argsort(rank_t)[:n]

                decs[t] = merged_decs[sorted_idx]
                objs[t] = merged_objs[sorted_idx]
                cons[t] = merged_cons[sorted_idx]

                # Save native-space history
                decs_t_native = decs[t][:, :dims[t]]
                objs_t_native = objs[t][:, :problem.n_objs[t]]
                cons_t_native = cons[t][:, :problem.n_cons[t]] if problem.n_cons[t] > 0 else cons[t][:, :0]
                append_history(all_decs[t], decs_t_native,
                               all_objs[t], objs_t_native,
                               all_cons[t], cons_t_native)

            if nfes >= max_nfes:
                break

        pbar.close()
        runtime = time.time() - start_time

        results = build_save_results(
            all_decs=all_decs, all_objs=all_objs, runtime=runtime,
            max_nfes=max_nfes_per_task, all_cons=all_cons,
            bounds=problem.bounds, save_path=self.save_path,
            filename=self.name, save_data=self.save_data)

        return results

    def _init_gan(self, dim, device):
        """Initialize Generator and Discriminator networks with Adam optimizers."""
        gen = _Generator(dim).to(device)
        disc = _Discriminator(dim).to(device)
        opt_g = torch.optim.Adam(gen.parameters(), lr=self.lrG,
                                 betas=(0.7, 0.9))
        opt_d = torch.optim.Adam(disc.parameters(), lr=self.lrD,
                                 betas=(0.7, 0.9))
        return gen, disc, opt_g, opt_d

    def _train_gan(self, gen, disc, opt_g, opt_d,
                   target_data, source_data, epochs, device):
        """
        Train GAN to map source task solutions to target task distribution.

        Parameters
        ----------
        gen : _Generator
            Generator network
        disc : _Discriminator
            Discriminator network
        opt_g : torch.optim.Adam
            Generator optimizer
        opt_d : torch.optim.Adam
            Discriminator optimizer
        target_data : np.ndarray
            Target task population (real data) of shape (N, dim)
        source_data : np.ndarray
            Source task population (noise input) of shape (N, dim)
        epochs : int
            Number of training epochs
        device : torch.device
            Device for computation

        Returns
        -------
        generated : np.ndarray
            Generated solutions of shape (N, dim)
        """
        gen.train()
        disc.train()
        n = target_data.shape[0]
        bs = min(self.BS, n)
        n_iter = max(1, n // bs)

        # Shuffle source data
        noise_data = source_data[np.random.permutation(n)]

        for epoch in range(epochs):
            perm = np.random.permutation(n)
            target_shuffled = target_data[perm]
            for i in range(n_iter):
                idx_start = i * bs
                idx_end = min(idx_start + bs, n)
                if idx_end - idx_start < 2:
                    continue

                real_batch = torch.tensor(
                    target_shuffled[idx_start:idx_end],
                    dtype=torch.float32, device=device)
                noise_batch = torch.tensor(
                    noise_data[idx_start:idx_end],
                    dtype=torch.float32, device=device)

                # Generate fake samples
                fake_batch = gen(noise_batch)

                # Discriminator loss: -mean(0.9*log(D(real)) + log(1-D(fake)))
                d_real = disc(real_batch)
                d_fake = disc(fake_batch.detach())
                d_loss = -torch.mean(
                    0.9 * torch.log(d_real + 1e-8) +
                    torch.log(1 - d_fake + 1e-8))

                opt_d.zero_grad()
                d_loss.backward()
                opt_d.step()

                # Generator loss: -mean(log(D(fake)))
                d_fake_for_g = disc(fake_batch)
                g_loss = -torch.mean(torch.log(d_fake_for_g + 1e-8))

                opt_g.zero_grad()
                g_loss.backward()
                opt_g.step()

        # Generate output
        gen.eval()
        with torch.no_grad():
            noise_all = torch.tensor(
                noise_data, dtype=torch.float32, device=device)
            generated = gen(noise_all).cpu().numpy()

        return generated

    def _generate_gan(self, gen, source_data, device):
        """
        Generate solutions using a trained generator.

        Parameters
        ----------
        gen : _Generator
            Trained generator network
        source_data : np.ndarray
            Source task population of shape (N, dim)
        device : torch.device
            Device for computation

        Returns
        -------
        generated : np.ndarray
            Generated solutions of shape (N, dim)
        """
        gen.eval()
        with torch.no_grad():
            noise = torch.tensor(
                source_data, dtype=torch.float32, device=device)
            generated = gen(noise).cpu().numpy()
        return generated

    def _generation(self, population, prepop, gan_outputs, Np, nt, d_uni):
        """
        Generate offspring using DE mutation with GAN-based cross-task transfer.

        Same-task pairs: standard DE/rand-or-best/1
        Cross-task pairs: GAN-transferred solutions + DE

        Parameters
        ----------
        population : list of np.ndarray
            Current populations per task, each of shape (Np, d_uni)
        prepop : list of np.ndarray
            Previous generation populations per task
        gan_outputs : dict
            GAN outputs: gan_outputs[(t,k)] = generated from task k to task t
        Np : int
            Population size per task
        nt : int
            Number of tasks
        d_uni : int
            Unified dimension

        Returns
        -------
        off_decs : np.ndarray
            Offspring decision variables of shape (total, d_uni)
        off_sfs : np.ndarray
            Skill factors of offspring
        """
        # Merge all populations and assign skill factors
        parent_list = []
        sf_list = []
        for t in range(nt):
            parent_list.append(population[t])
            sf_list.append(np.full(Np, t))

        all_parent = np.vstack(parent_list)
        all_sf = np.concatenate(sf_list)
        total = len(all_parent)

        # Random permutation for pairing
        rndper = np.random.permutation(total)
        all_parent = all_parent[rndper]
        # Track original position within task for GAN indexing
        orig_idx = np.arange(total)[rndper]
        all_sf = all_sf[rndper]

        off_decs = np.zeros((total, d_uni))
        off_sfs = np.zeros(total, dtype=int)

        for i in range(0, total - 1, 2):
            p1, p2 = i, i + 1
            sf1 = int(all_sf[p1])
            sf2 = int(all_sf[p2])

            # Sample F ~ N(0.5, 0.2) clipped to (0, 1)
            F = np.random.normal(0.5, 0.2)
            while F > 1 or F < 0:
                F = np.random.normal(0.5, 0.2)

            if sf1 == sf2:
                # Same-task pair: standard DE/rand-or-best/1
                # Offspring 1
                r1 = np.random.randint(Np)
                r2 = np.random.randint(Np)
                if np.random.rand() < self.pp:
                    x1 = population[sf1][r1]  # rand
                else:
                    x1 = population[sf1][0]  # best (index 0 = best after sorting)
                mutant1 = x1 + F * (all_parent[p1] - prepop[sf1][r2])
                off1 = self._de_crossover(mutant1, all_parent[p1], self.CR)

                # Offspring 2
                r1 = np.random.randint(Np)
                r2 = np.random.randint(Np)
                if np.random.rand() < self.pp:
                    x1 = population[sf2][r1]
                else:
                    x1 = population[sf2][0]
                mutant2 = x1 + F * (all_parent[p2] - prepop[sf2][r2])
                off2 = self._de_crossover(mutant2, all_parent[p2], self.CR)
            else:
                # Cross-task pair: GAN-transferred + DE
                # p1r, p2r: index within task population (mod Np)
                p1r = orig_idx[p1] % Np
                p2r = orig_idx[p2] % Np

                # c1: GAN{sf1, sf2} output at p1r (map sf2 -> sf1 for parent p1)
                c1 = gan_outputs.get((sf1, sf2), None)
                if c1 is not None and p1r < len(c1):
                    c1_dec = c1[p1r]
                else:
                    c1_dec = all_parent[p1]

                # c2: GAN{sf2, sf1} output at p2r (map sf1 -> sf2 for parent p2)
                c2 = gan_outputs.get((sf2, sf1), None)
                if c2 is not None and p2r < len(c2):
                    c2_dec = c2[p2r]
                else:
                    c2_dec = all_parent[p2]

                # Offspring 1: x1 + F*(c1 - prepop[sf1][r2])
                r1 = np.random.randint(Np)
                r2 = np.random.randint(Np)
                if np.random.rand() < self.pp:
                    x1 = population[sf1][r1]
                else:
                    x1 = population[sf1][0]
                mutant1 = x1 + F * (c1_dec - prepop[sf1][r2])
                off1 = self._de_crossover(mutant1, x1, self.CR)

                # Offspring 2: x1 + F*(c2 - prepop[sf2][r2])
                r1 = np.random.randint(Np)
                r2 = np.random.randint(Np)
                if np.random.rand() < self.pp:
                    x1 = population[sf2][r1]
                else:
                    x1 = population[sf2][0]
                mutant2 = x1 + F * (c2_dec - prepop[sf2][r2])
                off2 = self._de_crossover(mutant2, x1, self.CR)

            # Assign skill factors (imitation: offspring inherits parent's task)
            off_sfs[p1] = sf1
            off_sfs[p2] = sf2

            # Clip to [0, 1]
            off_decs[p1] = np.clip(off1, 0, 1)
            off_decs[p2] = np.clip(off2, 0, 1)

        # Handle odd last individual
        if total % 2 == 1:
            off_decs[-1] = all_parent[-1]
            off_sfs[-1] = int(all_sf[-1])

        return off_decs, off_sfs

    def _de_crossover(self, mutant, target, CR):
        """DE binomial crossover."""
        dim = len(mutant)
        j_rand = np.random.randint(dim)
        mask = np.random.rand(dim) < CR
        mask[j_rand] = True
        return np.where(mask, mutant, target)
