"""Dataset registry: versioned, checksummed datasets declared in YAML.

Adding a new internal or external benchmark = dropping one YAML file in
config/datasets/{internal|external}/ plus the data file (or HF pointer).
No code change unless a new loader format is needed.
"""
from __future__ import annotations
import hashlib, json, csv
from dataclasses import dataclass, field
from pathlib import Path
import yaml


@dataclass
class DatasetEntry:
    id: str
    kind: str                 # internal | external
    task_family: str          # rag | classification | ...
    loader: str               # jsonl | csv | hf
    location: str
    field_map: dict = field(default_factory=dict)
    label_set: list = field(default_factory=list)
    split: str = "test"
    max_items: int | None = None
    license: str = ""
    version: str = "1"
    checksum: str = ""
    notes: str = ""


def _apply_field_map(row: dict, fmap: dict) -> dict:
    if not fmap:
        return row
    out = dict(row)
    for target, source in fmap.items():
        if source in row:
            out[target] = row[source]
    return out


class DatasetRegistry:
    def __init__(self, config_dir: str = "config/datasets"):
        self.entries: dict[str, DatasetEntry] = {}
        for sub in ("internal", "external"):
            d = Path(config_dir) / sub
            if not d.exists():
                continue
            for f in sorted(d.glob("*.yaml")):
                if f.stem.startswith("TEMPLATE"):
                    continue
                cfg = yaml.safe_load(f.read_text())
                cfg.setdefault("kind", sub)
                entry = DatasetEntry(**cfg)
                self.entries[entry.id] = entry

    def get(self, dataset_id: str) -> DatasetEntry:
        if dataset_id not in self.entries:
            raise KeyError(f"Dataset '{dataset_id}' not registered. "
                           f"Known: {sorted(self.entries)}")
        return self.entries[dataset_id]

    def load_items(self, dataset_id: str) -> list[dict]:
        e = self.get(dataset_id)
        if e.loader == "jsonl":
            items = [json.loads(l) for l in
                     Path(e.location).read_text().splitlines() if l.strip()]
        elif e.loader == "csv":
            with open(e.location, newline="") as fh:
                items = list(csv.DictReader(fh))
        elif e.loader == "hf":
            try:
                from datasets import load_dataset  # optional dependency
            except ImportError as ex:
                raise RuntimeError(
                    f"Dataset {e.id} uses the 'hf' loader; pip install datasets") from ex
            ds = load_dataset(e.location, split=e.split)
            items = [dict(r) for r in ds]
        else:
            raise ValueError(f"Unknown loader '{e.loader}' for {e.id}")
        items = [_apply_field_map(r, e.field_map) for r in items]
        for i, r in enumerate(items):
            r.setdefault("item_id", f"{e.id}:{i}")
        if e.max_items:
            items = items[: e.max_items]
        return items

    def verify_checksum(self, dataset_id: str) -> bool:
        e = self.get(dataset_id)
        if not e.checksum or e.loader == "hf":
            return True
        digest = hashlib.sha256(Path(e.location).read_bytes()).hexdigest()
        return digest == e.checksum
