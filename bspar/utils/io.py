"""File I/O and checkpoint utilities."""

import os
import json
import torch
import yaml


def load_yaml_config(path: str) -> dict:
    """Load a YAML config file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(data, path: str):
    """Save data as JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: str):
    """Load JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_checkpoint(state: dict, path: str):
    """Save a training checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, device="cpu"):
    """Load a training checkpoint."""
    return torch.load(path, map_location=device, weights_only=False)
