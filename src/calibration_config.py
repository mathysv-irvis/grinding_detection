import json
from pathlib import Path


CONFIG_PATH = Path(__file__).parent / "calibration_config.json"

DEFAULT_VALUES = {
    "intensity": 40,
    "K": 5,
    "max_component": 2000,
    "threshold": 100,
    "kernel_size": 7,
}


def load_values():
    if not CONFIG_PATH.exists():
        save_values(DEFAULT_VALUES)
        return DEFAULT_VALUES.copy()

    try:
        with open(CONFIG_PATH, "r") as f:
            values = json.load(f)
    except (json.JSONDecodeError, OSError):
        return DEFAULT_VALUES.copy()

    result = DEFAULT_VALUES.copy()
    result.update(values)

    return result


def save_values(values):
    with open(CONFIG_PATH, "w") as f:
        json.dump(
            values,
            f,
            indent=4,
        )
