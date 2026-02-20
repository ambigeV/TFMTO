"""Background runners for optimization tasks."""

import glob
import os
from dataclasses import dataclass
from typing import Optional
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


def count_pkl_files(data_path: str) -> int:
    """Count .pkl files in a directory tree."""
    return len(glob.glob(os.path.join(data_path, "**", "*.pkl"), recursive=True))


def get_default_workers() -> int:
    """Get default number of workers (CPU count - 1, min 1)."""
    return max(1, cpu_count() - 1)
