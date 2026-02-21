# Metric List Design Proposal

## Overview

Allow the `metric` field in settings to be either a **string** (current behavior) or a **list** (per-task specification). This enables different tasks within a multi-task problem to use different metrics.

---

## 1. SETTINGS Structure

### Current

```python
settings = {'metric': 'IGD', 'n_ref': 10000, 'DTLZ1': {'T1': PF}, ...}
```

### Proposed

`metric` accepts `str`, `list`, or `None`:

```python
# String: all MO tasks use IGD, SO tasks auto min
settings = {'metric': 'IGD', 'n_ref': 10000, ...}

# List: per-task specification
settings = {'metric': ['IGD', 'HV', None], 'n_ref': 10000, ...}
```

### Per-problem override (optional)

```python
settings = {
    'metric': 'IGD',              # Global default
    'n_ref': 10000,
    'DTLZ1': {
        'T1': PF,                 # No 'metric' key → use global default
    },
    'P1': {
        'metric': ['IGD', 'HV'],  # Override: T1=IGD, T2=HV
        'T1': PF1,
        'T2': PF2,
    },
}
```

---

## 2. Input Rules

| Input | Meaning |
|-------|---------|
| `'IGD'` | All MO tasks use IGD; SO tasks auto min(obj) |
| `['IGD']` | Same as `'IGD'` |
| `['IGD', 'HV', None]` | T1=IGD, T2=HV, T3=min(obj) |
| `['IGD', 'HV']` with 3 tasks | T1=IGD, T2=HV, T3=HV (repeat last element) |
| `[None, None, 'IGD']` | T1=min, T2=min, T3=IGD |
| `None` | All tasks use min(obj) |

**`None` meaning**: Force min(obj) for that task regardless of objective count M.

---

## 3. Metric Properties Reference

| Metric | Type | Direction | Needs Reference | Needs n_ref |
|--------|------|-----------|-----------------|-------------|
| IGD | MO | minimize (-1) | Yes (PF) | Yes (when PF is callable) |
| IGDp | MO | minimize (-1) | Yes (PF) | Yes |
| HV | MO | maximize (+1) | Yes (PF or ref point) | Yes |
| GD | MO | minimize (-1) | Yes (PF) | Yes |
| DeltaP | MO | minimize (-1) | Yes (PF) | Yes |
| Spread | MO | minimize (-1) | Yes (PF) | Yes |
| Spacing | MO | minimize (-1) | No | No |
| FR | Constraint | maximize (+1) | No (needs constraint data) | No |
| CV | Constraint | minimize (-1) | No (needs constraint data) | No |
| None | SO/forced | minimize (-1) | No | No |

---

## 4. Resolution Logic

### 4.1 Determine metric for a specific task

```
_get_task_metric(problem_name, task_index):
    1. Check settings[problem_name]['metric'] (per-problem override)
    2. If not found, use settings['metric'] (global default)
    3. If result is a list:
       - index = min(task_index, len(list) - 1)
       - return list[index]
    4. If result is a string or None:
       - return as-is
```

### 4.2 Calculate metric for a task/generation

```
For each task t, generation g:
    task_metric = _get_task_metric(prob, t)
    M = objectives count

    Case 1: task_metric is None OR M == 1
        → value = min(obj[:, 0])
        → sign = -1

    Case 2: task_metric in ('Spacing',)
        → No reference needed
        → value = Spacing().calculate(objs)
        → sign = -1

    Case 3: task_metric in ('FR', 'CV')
        → No reference needed, needs constraint data
        → value = FR().calculate(cons) or CV().calculate(cons)
        → sign = +1 (FR) or -1 (CV)

    Case 4: task_metric in ('IGD', 'HV', 'IGDp', 'GD', 'DeltaP', 'Spread')
        → reference = load_reference(settings, prob, task, M, D, C)
        → If reference is None:
            → Fallback to Spacing + print warning
        → Else:
            → value = metric.calculate(objs, reference)
            → sign = metric.sign
```

### 4.3 Edge cases

| Situation | Behavior |
|-----------|----------|
| M==1 but metric='IGD' | Auto downgrade to min(obj), print info |
| M>1 but metric=None | Force min(sum(obj, axis=1)) |
| metric='FR' but no constraint data | Raise ValueError |
| metric='IGD' but no reference data | Fallback to Spacing + warning |
| List shorter than task count | Repeat last element |
| List longer than task count | Extra elements ignored |

---

## 5. UI Changes

### Batch Mode & Test Mode

Replace the metric **Combo** with a **text input**:

```
Before:  Combo [IGD ▼]
After:   Input [IGD, HV, None        ]
```

Default value: `"IGD"`

### Parse logic

```python
def parse_metric_input(text: str):
    """Parse metric text input into str, list, or None."""
    text = text.strip()
    if not text:
        return 'IGD'

    parts = [p.strip() for p in text.split(',')]

    if len(parts) == 1:
        # Single value: return as string or None
        val = parts[0]
        return None if val.lower() == 'none' else val

    # Multiple values: return as list
    result = []
    for p in parts:
        result.append(None if p.lower() == 'none' else p)
    return result
```

### n_ref visibility

Keep `n_ref` input always visible. When no metric needs it, the value is simply ignored.

---

## 6. Backend Changes

### 6.1 `data_analysis.py`

Add method to `DataAnalyzer`:

```python
def _get_task_metric(self, prob: str, task_idx: int) -> Optional[str]:
    """Determine which metric to use for a specific problem/task."""
    if self.settings is None:
        return None

    # Check per-problem override
    prob_st = self.settings.get(prob, {})
    if isinstance(prob_st, dict) and 'metric' in prob_st:
        metric_cfg = prob_st['metric']
    else:
        metric_cfg = self.settings.get('metric', 'IGD')

    # Resolve list vs scalar
    if isinstance(metric_cfg, list):
        idx = min(task_idx, len(metric_cfg) - 1)
        return metric_cfg[idx]
    return metric_cfg
```

Modify `_get_single_run_metric_value`:

```python
for t in range(n_tasks):
    task_metric = self._get_task_metric(prob, t)
    # ... use task_metric instead of global self.settings['metric']
```

### 6.2 `test_data_analysis.py`

Same changes as `data_analysis.py`.

### 6.3 `batch_mode.py` / `test_mode.py`

- Replace metric Combo with text input (`dpg.add_input_text`)
- Add `parse_metric_input()` to parse user input
- Pass parsed metric to settings dict

### 6.4 `constants.py`

No structural change needed. Keep `METRICS` list for reference/validation.

---

## 7. SETTINGS Building in UI

### _run_clicked (batch_mode.py)

```python
metric_text = dpg.get_value("batch_metric_input")
metric = parse_metric_input(metric_text)

settings = {'metric': metric, 'n_ref': n_ref}

if is_multi_objective(prob_cat):
    # Load reference data as before
    for i, (suite, method) in enumerate(selected_probs):
        ...
```

### _load_data_clicked

```python
metric_text = dpg.get_value("batch_metric_input")
metric = parse_metric_input(metric_text)

settings = {'metric': metric, 'n_ref': n_ref}
# Load all MO suite references as before
```

---

## 8. Backward Compatibility

| Existing usage | Still works? |
|----------------|-------------|
| `settings = {'metric': 'IGD', ...}` | Yes, unchanged |
| `settings = None` | Yes, SO mode |
| Problem SETTINGS with `'metric': 'IGD'` | Yes, treated as global string |
| `DataAnalyzer(settings={'metric': 'IGD', ...})` | Yes, unchanged |
| Scripts using `analyzer.run()` | Yes, unchanged |

---

## 9. Files to Modify

| File | Change |
|------|--------|
| `src/ddmtolab/Methods/data_analysis.py` | Add `_get_task_metric()`, modify metric calculation loop |
| `src/ddmtolab/Methods/test_data_analysis.py` | Same as above |
| `ui/pages/batch_mode.py` | Combo → text input, add `parse_metric_input()` |
| `ui/pages/test_mode.py` | Combo → text input, add `parse_metric_input()` |
| `ui/config/constants.py` | Optional: keep METRICS list for validation |

---

## 10. Example Scenarios

### Scenario A: Standard STMO experiment (DTLZ1, DTLZ2, DTLZ3)

```
UI input: "IGD"
Each problem has 1 task, all multi-objective.
→ All use IGD with PF reference data from DTLZ SETTINGS.
```

### Scenario B: MTMO experiment with mixed objectives

```
Problem P1 has 3 tasks: T1(3-obj), T2(2-obj), T3(1-obj)
UI input: "IGD, HV, None"
→ T1: IGD with reference, T2: HV with reference, T3: min(obj)
```

### Scenario C: Constrained MO experiment

```
Problem CF1 has 1 task, 2-obj with constraints.
UI input: "FR"
→ T1: Feasible Rate (uses constraint data, no reference needed)
```

### Scenario D: Mixed batch experiment

```
Problems: DTLZ1 (1 task, 3-obj), CEC17_P1 (2 tasks, mixed)
UI input: "IGD"
→ DTLZ1 T1: IGD, CEC17_P1 T1: IGD, CEC17_P1 T2: IGD
All MO tasks use IGD; if any task is SO, auto min(obj).
```

### Scenario E: Per-problem override via SETTINGS

```python
settings = {
    'metric': 'IGD',
    'n_ref': 10000,
    'DTLZ1': {'T1': PF},
    'CF1': {
        'metric': 'FR',       # Override for this problem
        'T1': CF1_PF,
    },
}
# DTLZ1 uses IGD (global default), CF1 uses FR (per-problem override)
```
