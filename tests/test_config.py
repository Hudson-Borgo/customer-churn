from src.utils.config import load_config


def test_load_config_has_paths_key():
    config = load_config()
    assert "paths" in config
