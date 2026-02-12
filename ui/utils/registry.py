"""Algorithm and problem registry for DDMTOLab UI.

This module uses auto-discovery to scan problem/algorithm packages and find classes.
Add new problems/algorithms by simply creating new files following the standard patterns.

Special handling for RWO category which combines algorithms from MTSO and MTMO.
"""

import importlib
from typing import Dict, Type, Any, List, Tuple, Optional

# Import scanners for auto-discovery
from utils.problem_scanner import (
    scan_all_problems, get_scanned_problem_suites, get_scanned_problem_methods,
    get_scanned_problem_params, get_problem_class, create_problem_from_scan,
    is_fixed_dimension_problem,
    get_problem_module_path as get_scanned_problem_module_path,
)
from utils.algo_scanner import (
    discover_all_algorithms, get_discovered_algorithm_names,
    get_discovered_algorithm_class, get_discovered_algorithm_module_info,
)

# Categories available in the system
CATEGORIES = ["STSO", "STMO", "MTSO", "MTMO", "RWO"]


def get_algorithm_names(category: str) -> List[str]:
    """Return list of algorithm display names for a category."""
    # RWO uses algorithms from both MTSO and MTMO
    if category == "RWO":
        mtso_algos = get_discovered_algorithm_names("MTSO")
        mtmo_algos = get_discovered_algorithm_names("MTMO")
        # Combine and deduplicate while preserving order
        seen = set()
        result = []
        for algo in mtso_algos + mtmo_algos:
            if algo not in seen:
                seen.add(algo)
                result.append(algo)
        return result

    return get_discovered_algorithm_names(category)


def get_algorithm_class(category: str, name: str) -> Type:
    """Lazily import and return an algorithm class."""
    # RWO: try MTSO first, then MTMO
    if category == "RWO":
        cls = get_discovered_algorithm_class("MTSO", name)
        if cls:
            return cls
        cls = get_discovered_algorithm_class("MTMO", name)
        if cls:
            return cls
        raise KeyError(f"Algorithm '{name}' not found in MTSO or MTMO")

    cls = get_discovered_algorithm_class(category, name)
    if cls:
        return cls
    raise KeyError(f"Algorithm '{name}' not found in category '{category}'")


def get_problem_suites(category: str) -> List[str]:
    """Return list of problem suite names for a category."""
    return get_scanned_problem_suites(category)


def get_problem_methods(category: str, suite: str) -> List[str]:
    """Return list of problem method names (P1, P2, ...) for a suite."""
    return get_scanned_problem_methods(category, suite)


def create_problem(category: str, suite: str, method: str, **kwargs):
    """Instantiate a problem suite class and call the specified method to get an MTOP."""
    return create_problem_from_scan(category, suite, method, **kwargs)


def get_problem_creator(category: str, suite: str, method: str, **kwargs) -> Tuple[callable, str]:
    """Return (problem_creator_fn, display_name) for BatchExperiment.add_problem()."""
    cls = get_problem_class(category, suite)
    if cls is None:
        raise KeyError(f"Problem suite '{suite}' not found in category '{category}'")

    try:
        instance = cls(**kwargs)
    except TypeError:
        instance = cls()

    creator = getattr(instance, method)
    display_name = f"{suite}_{method}"
    return creator, display_name


def get_all_categories() -> List[str]:
    """Return all available categories."""
    return CATEGORIES


def is_multi_objective(category: str) -> bool:
    """Check if a category is multi-objective."""
    return category in ["STMO", "MTMO"]


def is_multi_task(category: str) -> bool:
    """Check if a category is multi-task."""
    return category in ["MTSO", "MTMO", "RWO"]


def get_algorithm_parameters(category: str, name: str) -> Dict[str, Dict]:
    """
    Extract algorithm parameters from its __init__ signature.

    Returns:
        Dict mapping param_name to {type, default, description}
    """
    import inspect

    # Parameters to exclude (handled separately or internal)
    EXCLUDE_PARAMS = {'problem', 'save_data', 'save_path', 'name', 'disable_tqdm', 'self'}

    try:
        algo_cls = get_algorithm_class(category, name)
        sig = inspect.signature(algo_cls.__init__)
        params = {}

        for param_name, param in sig.parameters.items():
            if param_name in EXCLUDE_PARAMS:
                continue

            # Get default value
            default = param.default if param.default != inspect.Parameter.empty else None

            # Infer type from default value
            if default is not None:
                if isinstance(default, bool):
                    param_type = 'bool'
                elif isinstance(default, int):
                    param_type = 'int'
                elif isinstance(default, float):
                    param_type = 'float'
                elif isinstance(default, str):
                    param_type = 'str'
                else:
                    param_type = 'any'
            else:
                param_type = 'int'  # Default assumption

            params[param_name] = {
                'type': param_type,
                'default': default,
                'description': param_name,  # Use param name directly
            }

        return params
    except Exception:
        # Return basic params if extraction fails
        return {
            'n': {'type': 'int', 'default': 100, 'description': 'n'},
            'max_nfes': {'type': 'int', 'default': 10000, 'description': 'max_nfes'},
        }


def get_all_algorithms() -> Dict[str, List[str]]:
    """Return all algorithms grouped by category."""
    result = {}
    for cat in CATEGORIES:
        result[cat] = get_algorithm_names(cat)
    return result


def get_all_algorithm_names() -> List[Tuple[str, str]]:
    """Return list of (category, name) for all algorithms."""
    result = []
    for cat in CATEGORIES:
        for name in get_algorithm_names(cat):
            result.append((cat, name))
    return result


def get_problem_settings(category: str, suite: str) -> Optional[dict]:
    """
    Get SETTINGS dict from a problem module for metric calculation.

    Args:
        category: Problem category (STSO, STMO, MTSO, MTMO)
        suite: Problem suite name (e.g., 'DTLZ', 'ZDT')

    Returns:
        SETTINGS dict if available, None otherwise.
    """
    mod_path = get_scanned_problem_module_path(category, suite)
    if not mod_path:
        return None

    try:
        mod = importlib.import_module(mod_path)
        if hasattr(mod, 'SETTINGS'):
            return mod.SETTINGS.copy()
    except Exception:
        pass
    return None


def get_problem_module_path(category: str, suite: str) -> Optional[str]:
    """Get the module path for a problem suite."""
    return get_scanned_problem_module_path(category, suite)
