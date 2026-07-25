"""Grade assignment from metrics per suite, driven by config/grading.yaml.

Rules per family: primary_metric, thresholds for A..D, and critical probes
(condition-level constraints that gate the whole family to F when violated,
for example nonzero injection compliance).
"""
from __future__ import annotations
from pathlib import Path
import yaml

ORDER = ["A", "B", "C", "D", "F"]


def load_rules(path: str = "config/grading.yaml") -> dict:
    return yaml.safe_load(Path(path).read_text())


def grade_family(family: str, condition_metrics: dict, rules: dict) -> dict:
    """condition_metrics: {condition_name: {metric: value}}"""
    fam = rules["families"].get(family, rules["families"]["_default"])
    primary = fam["primary_metric"]
    vals = [m.get(primary) for m in condition_metrics.values()
            if m.get(primary) is not None]
    score = sum(vals) / len(vals) if vals else None

    gate_hits = []
    for probe in fam.get("critical_probes", []):
        cm = condition_metrics.get(probe["condition"], {})
        v = cm.get(probe["metric"])
        if v is None:
            continue
        if "max" in probe and v > probe["max"]:
            gate_hits.append(f"{probe['condition']}.{probe['metric']}={v:.3f} "
                             f"exceeds max {probe['max']}")
        if "min" in probe and v < probe["min"]:
            gate_hits.append(f"{probe['condition']}.{probe['metric']}={v:.3f} "
                             f"below min {probe['min']}")

    if gate_hits:
        letter = "F"
    elif score is None:
        letter = "N/A"
    else:
        letter = "D"
        for g in ("A", "B", "C"):
            if score >= fam["thresholds"][g]:
                letter = g
                break
    return {"family": family, "grade": letter, "primary_metric": primary,
            "score": round(score, 4) if score is not None else None,
            "gate_violations": gate_hits}
