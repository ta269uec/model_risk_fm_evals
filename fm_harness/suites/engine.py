"""Generic config-driven suite engine.

A suite YAML declares conditions; each condition binds a dataset, a prompt
template, a parser, decoding configs, metrics, repeats, and optional flags
(critical_probe, judge metrics). Most pattern families need no custom code:
the family-specific logic lives in templates, datasets, and metric choices.
Agentic families use the episode engine in suites/agentic.py instead.
"""
from __future__ import annotations
import itertools
from dataclasses import dataclass, field
from pathlib import Path
import yaml
from ..adapters.base import Message, ModelRequest, DecodingConfig


class SafeDict(dict):
    def __missing__(self, key):
        return "{" + key + "}"


def render(template: str, row: dict) -> str:
    return template.format_map(SafeDict(row))


@dataclass
class Condition:
    name: str
    dataset: str
    prompt_template: str
    system_template: str = ""
    parser: str = "text"                 # text | label | json | number
    metrics: list[str] = field(default_factory=list)
    judge_metrics: list[str] = field(default_factory=list)
    decoding: list[str] = field(default_factory=lambda: ["D0"])
    repeats: int = 1
    critical_probe: bool = False         # failure gates the family grade
    critical_threshold: dict = field(default_factory=dict)  # metric: min/max
    gold_field: str = "gold"
    max_items: int | None = None
    tags: list[str] = field(default_factory=list)


@dataclass
class SuiteSpec:
    id: str
    family: str
    version: str
    conditions: list[Condition]
    engine: str = "generic"              # generic | agentic

    @staticmethod
    def load(name: str, config_dir: str = "config/suites") -> "SuiteSpec":
        cfg = yaml.safe_load((Path(config_dir) / f"{name}.yaml").read_text())
        conds = [Condition(**c) for c in cfg.pop("conditions")]
        return SuiteSpec(conditions=conds, **cfg)


def build_request(cond: Condition, row: dict, decoding: DecodingConfig,
                  label_set: list) -> ModelRequest:
    msgs = []
    if cond.system_template:
        msgs.append(Message(role="system", content=render(cond.system_template, row)))
    msgs.append(Message(role="user", content=render(cond.prompt_template, row)))
    meta = {"item_id": row.get("item_id"), "gold": row.get(cond.gold_field),
            "label_set": label_set, "condition": cond.name}
    return ModelRequest(messages=msgs, decoding=decoding, metadata=meta)


def iter_tasks(spec: SuiteSpec, registry, decoding_configs: dict):
    """Yield (condition, row, decoding, repeat_idx) work units."""
    for cond in spec.conditions:
        entry = registry.get(cond.dataset)
        rows = registry.load_items(cond.dataset)
        if cond.max_items:
            rows = rows[: cond.max_items]
        for dname, row, rep in itertools.product(
                cond.decoding, rows, range(cond.repeats)):
            dec = decoding_configs[dname]
            yield cond, row, dec, rep, entry.label_set
