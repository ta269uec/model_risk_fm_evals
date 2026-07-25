from __future__ import annotations
import yaml
from pathlib import Path
from .base import ModelAdapter
from .openai_compat import OpenAICompatAdapter
from .anthropic_adapter import AnthropicAdapter
from .google_adapter import GoogleAdapter
from .mock import MockAdapter

ADAPTERS = {
    "openai_compat": OpenAICompatAdapter,
    "anthropic": AnthropicAdapter,
    "google": GoogleAdapter,
    "mock": MockAdapter,
}


def load_model(name: str, config_dir: str = "config/models") -> ModelAdapter:
    path = Path(config_dir) / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No model config at {path}")
    cfg = yaml.safe_load(path.read_text())
    adapter_key = cfg.get("adapter")
    if adapter_key not in ADAPTERS:
        raise ValueError(f"Unknown adapter '{adapter_key}'. Known: {list(ADAPTERS)}")
    return ADAPTERS[adapter_key](cfg)
