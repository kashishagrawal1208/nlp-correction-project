"""
config.py
Loads config/config.yaml once and shares it with every module.
"""

from functools import lru_cache

import yaml

CONFIG_PATH = "config/config.yaml"


@lru_cache(maxsize=1)
def load_config():
    """Read the YAML settings file (only the first time; later calls reuse it)."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)