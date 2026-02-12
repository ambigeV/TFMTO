"""UI constants, colors, and sizes."""

# Window dimensions
WINDOW_WIDTH = 1550
WINDOW_HEIGHT = 900
MIN_WINDOW_WIDTH = 1400
MIN_WINDOW_HEIGHT = 800

# Panel widths
LEFT_PANEL_WIDTH = 280
MIDDLE_PANEL_WIDTH = 300

# Colors (RGBA tuples, 0-255) - designed to work in both dark and light modes
COLOR_TITLE = (50, 120, 200, 255)      # Blue for titles
COLOR_SECTION = (60, 140, 180, 255)    # Teal for sections
COLOR_SUCCESS = (80, 255, 80, 255)
COLOR_ERROR = (255, 80, 80, 255)
COLOR_WARNING = (255, 200, 80, 255)
COLOR_HIGHLIGHT = (255, 255, 100, 255)

# Font sizes
FONT_SIZE_DEFAULT = 18
FONT_SIZE_TITLE = 24
FONT_SIZE_SECTION = 20

# Categories
CATEGORIES = ["STSO", "STMO", "MTSO", "MTMO", "RWO"]

# Metrics for multi-objective problems
METRICS = ["IGD", "HV", "IGDp", "GD", "DeltaP", "Spacing", "Spread"]

# Metric descriptions
METRIC_INFO = {
    "IGD": {"direction": "minimize", "requires_ref": True, "description": "Inverted Generational Distance"},
    "HV": {"direction": "maximize", "requires_ref": True, "description": "Hypervolume"},
    "IGDp": {"direction": "minimize", "requires_ref": True, "description": "IGD Plus"},
    "GD": {"direction": "minimize", "requires_ref": True, "description": "Generational Distance"},
    "DeltaP": {"direction": "minimize", "requires_ref": True, "description": "Delta_p"},
    "Spacing": {"direction": "minimize", "requires_ref": False, "description": "Spacing"},
    "Spread": {"direction": "minimize", "requires_ref": True, "description": "Spread"},
}

# Table formats
TABLE_FORMATS = ["excel", "latex"]

# Figure formats
FIGURE_FORMATS = ["png", "pdf", "svg"]

# Statistic types
STATISTIC_TYPES = ["mean", "median", "max", "min"]

# Paths (relative to tests folder)
DEFAULT_DATA_PATH = "Data"
DEFAULT_RESULTS_PATH = "Results"
DEFAULT_BACKUP_PATH = "backup"
