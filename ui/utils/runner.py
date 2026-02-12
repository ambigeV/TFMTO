"""Background runners for optimization tasks."""

import threading
import time
import glob
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from multiprocessing import cpu_count


@dataclass
class RunStatus:
    """Shared status object for monitoring a single run."""
    running: bool = False
    finished: bool = False
    cancelled: bool = False
    error: Optional[str] = None
    result: object = None
    start_time: float = 0.0
    elapsed: float = 0.0
    current_algo: str = ""
    current_idx: int = 0
    total_algos: int = 0


@dataclass
class BatchStatus:
    """Shared status for batch experiment monitoring."""
    running: bool = False
    finished: bool = False
    cancelled: bool = False
    error: Optional[str] = None
    total_tasks: int = 0
    completed_tasks: int = 0
    start_time: float = 0.0
    elapsed: float = 0.0
    current_phase: str = "idle"  # idle, running, analyzing, complete
    analysis_result: object = None


def run_single_algorithm(algorithm, status: RunStatus):
    """Run algorithm.optimize() in a thread, updating status."""
    status.running = True
    status.start_time = time.time()
    try:
        result = algorithm.optimize()
        status.result = result
        status.finished = True
    except Exception as e:
        status.error = str(e)
        status.finished = True
    finally:
        status.running = False
        status.elapsed = time.time() - status.start_time


def start_algorithm_thread(algorithm) -> tuple:
    """Start an algorithm in a background thread. Returns (thread, status)."""
    status = RunStatus()
    t = threading.Thread(target=run_single_algorithm, args=(algorithm, status), daemon=True)
    t.start()
    return t, status


def count_pkl_files(data_path: str) -> int:
    """Count .pkl files in a directory tree."""
    return len(glob.glob(os.path.join(data_path, "**", "*.pkl"), recursive=True))


def run_batch_experiment(batch_exp, n_runs, max_workers, status: BatchStatus,
                         data_path: str, run_analysis: bool = True,
                         analysis_kwargs: dict = None):
    """Run BatchExperiment.run() in a thread, updating status."""
    status.running = True
    status.start_time = time.time()
    status.current_phase = "running"
    try:
        batch_exp.run(n_runs=n_runs, max_workers=max_workers)

        if status.cancelled:
            status.error = "Cancelled by user"
            status.finished = True
            return

        status.current_phase = "analyzing"

        if run_analysis:
            from ddmtolab.Methods.data_analysis import DataAnalyzer
            kwargs = analysis_kwargs or {}
            analyzer = DataAnalyzer(data_path=data_path, **kwargs)
            status.analysis_result = analyzer.run()

        status.current_phase = "complete"
        status.finished = True
    except Exception as e:
        status.error = str(e)
        status.finished = True
    finally:
        status.running = False
        status.elapsed = time.time() - status.start_time


def start_batch_thread(batch_exp, n_runs, max_workers, total_tasks, data_path,
                       run_analysis=True, analysis_kwargs=None) -> tuple:
    """Start batch experiment in a background thread. Returns (thread, status)."""
    status = BatchStatus(total_tasks=total_tasks)
    t = threading.Thread(
        target=run_batch_experiment,
        args=(batch_exp, n_runs, max_workers, status, data_path,
              run_analysis, analysis_kwargs),
        daemon=True,
    )
    t.start()
    return t, status


def get_default_workers() -> int:
    """Get default number of workers (CPU count - 1, min 1)."""
    return max(1, cpu_count() - 1)
