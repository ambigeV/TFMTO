"""Settings builder for metric calculation."""

import numpy as np
from typing import Dict, List, Optional, Union
from pathlib import Path


class SettingsBuilder:
    """Build SETTINGS dict for metric calculation."""

    METRICS = {
        'IGD': {'direction': 'minimize', 'sign': -1, 'requires_ref': True},
        'HV': {'direction': 'maximize', 'sign': 1, 'requires_ref': True},
        'IGDp': {'direction': 'minimize', 'sign': -1, 'requires_ref': True},
        'GD': {'direction': 'minimize', 'sign': -1, 'requires_ref': True},
        'DeltaP': {'direction': 'minimize', 'sign': -1, 'requires_ref': True},
        'Spacing': {'direction': 'minimize', 'sign': -1, 'requires_ref': False},
        'Spread': {'direction': 'minimize', 'sign': -1, 'requires_ref': True},
    }

    def __init__(self):
        self.metric: Optional[str] = None
        self.problem_refs: Dict[str, Dict[str, any]] = {}  # {problem_name: {task_name: ref_data}}
        self.n_ref: int = 10000
        self.ref_path: str = './MOReference'

    def set_metric(self, metric: str):
        """Set the metric type."""
        if metric and metric not in self.METRICS:
            raise ValueError(f"Unknown metric: {metric}. Available: {list(self.METRICS.keys())}")
        self.metric = metric

    def add_reference(self, problem: str, task: str, ref: Union[np.ndarray, str, List]):
        """
        Add reference for a problem/task combination.

        Args:
            problem: Problem name (e.g., 'P1')
            task: Task name (e.g., 'T1')
            ref: Reference data - can be:
                 - np.ndarray: Pareto front points
                 - str: Path to reference file
                 - List: HV reference point values
        """
        if problem not in self.problem_refs:
            self.problem_refs[problem] = {}
        self.problem_refs[problem][task] = ref

    def clear_references(self):
        """Clear all reference data."""
        self.problem_refs = {}

    def needs_configuration(self, category: str) -> bool:
        """Check if SETTINGS is required for this problem category."""
        return category in ['STMO', 'MTMO']

    def has_valid_settings(self) -> bool:
        """Check if current settings are valid for building."""
        if not self.metric:
            return False
        if self.METRICS[self.metric]['requires_ref'] and not self.problem_refs:
            return False
        return True

    def build(self, problem_names: List[str] = None) -> Optional[dict]:
        """
        Build the SETTINGS dictionary.

        Args:
            problem_names: List of problem names (for 'problems' key)

        Returns:
            Settings dictionary or None if no metric set.
        """
        if not self.metric:
            return None

        settings = {
            'metric': self.metric,
            'n_ref': self.n_ref,
            'ref_path': self.ref_path,
        }

        if problem_names:
            settings['problems'] = problem_names

        # Add problem references
        settings.update(self.problem_refs)

        return settings

    def build_for_suite(self, suite: str, methods: List[str], n_tasks: int = 2) -> Optional[dict]:
        """
        Build settings for a specific problem suite.

        This is a convenience method that auto-generates task references
        based on suite naming conventions.

        Args:
            suite: Problem suite name (e.g., 'CEC17-MTMO')
            methods: List of problem methods (e.g., ['P1', 'P2'])
            n_tasks: Number of tasks per problem

        Returns:
            Settings dictionary or None if no metric set.
        """
        if not self.metric:
            return None

        settings = {
            'metric': self.metric,
            'n_ref': self.n_ref,
            'ref_path': self.ref_path,
            'problems': [f'{suite}_{m}' for m in methods],
        }

        # Add empty refs that will be filled by DataAnalyzer
        for method in methods:
            problem_key = f'{suite}_{method}'
            settings[problem_key] = {
                f'T{i+1}': None for i in range(n_tasks)
            }

        return settings

    @classmethod
    def get_metric_info(cls, metric: str) -> dict:
        """Get information about a metric."""
        return cls.METRICS.get(metric, {})

    @classmethod
    def get_available_metrics(cls) -> List[str]:
        """Get list of available metrics."""
        return list(cls.METRICS.keys())

    @classmethod
    def metric_requires_reference(cls, metric: str) -> bool:
        """Check if a metric requires reference data."""
        info = cls.METRICS.get(metric, {})
        return info.get('requires_ref', True)
