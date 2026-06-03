from pathlib import Path

import yaml


def load_config() -> dict:
    config_path = Path("configs/config.yaml")

    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)