import json
import os
from pathlib import Path


CONFIG_PATH = Path(__file__).parent / "calibration_config.json"


DEFAULT_VALUES = {
    "plate_margin": 20,
    "intensity_min": 20,
    "intensity_max": 80,
    "merge_distance": 0,
    "detection_mode": "intensity",
    "contrast_threshold": 25,
}


def load_values():
    if not CONFIG_PATH.exists():
        save_values(DEFAULT_VALUES)
        return DEFAULT_VALUES.copy()

    try:
        with open(
            CONFIG_PATH,
            "r",
        ) as f:
            values = json.load(f)

    except (
        json.JSONDecodeError,
        OSError,
        TypeError,
    ):
        return DEFAULT_VALUES.copy()

    result = DEFAULT_VALUES.copy()

    for key in DEFAULT_VALUES:
        if key in values:
            result[key] = values[key]

    return result


def save_values(values):
    clean_values = DEFAULT_VALUES.copy()

    for key in DEFAULT_VALUES:
        if key in values:
            clean_values[key] = values[key]

    temp_path = CONFIG_PATH.with_suffix(".tmp")

    with open(
        temp_path,
        "w",
    ) as f:
        json.dump(
            clean_values,
            f,
            indent=4,
        )

    os.replace(
        temp_path,
        CONFIG_PATH,
    )
