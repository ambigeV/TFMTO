"""Default parameters for problems."""

# Problem-specific parameters (fallback when auto-scan fails)
# Only type and default needed, description = param name
PROBLEM_PARAMS = {
    "ZDT": {"D": {"type": "int", "default": 30}},
    "DTLZ": {"M": {"type": "int", "default": 3}, "D": {"type": "int", "default": 12}},
    "WFG": {"M": {"type": "int", "default": 3}, "Kp": {"type": "int", "default": 4}},
    "UF": {"D": {"type": "int", "default": 30}},
    "CF": {"D": {"type": "int", "default": 10}},
    "MW": {"D": {"type": "int", "default": 15}},
    "MTMO-DTLZ": {"M": {"type": "int", "default": 3}, "D": {"type": "int", "default": 12}},
    "Classical SO": {"D": {"type": "int", "default": 50}},
    "STSO Test": {"D": {"type": "int", "default": 50}},
    "CEC19-MaTSO": {"K": {"type": "int", "default": 10}},
    "STOP": {"K": {"type": "int", "default": 10}},
    "CMT": {"D": {"type": "int", "default": 50}},
    "CEC19-MaTMO": {"K": {"type": "int", "default": 10}},
}

# Suites with fixed dimensions (no D parameter)
FIXED_DIMENSION_SUITES = [
    "CEC17-MTSO", "CEC17-MTSO-10D", "CEC19-MaTSO", "STOP",
    "CEC17-MTMO", "CEC19-MTMO", "CEC19-MaTMO", "CEC21-MTMO",
    "PEPVM", "SOPM",
]

# Suites with fixed objectives (no M parameter)
# CF: CF1-CF7=2obj, CF8-CF10=3obj; ZDT: all 2obj; UF: UF1-7=2obj, UF8-10=3obj
FIXED_OBJECTIVES_SUITES = ["ZDT", "CF", "UF", "MW"]
