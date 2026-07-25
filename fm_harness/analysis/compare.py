"""Side-by-side comparison of registered FM profiles for one suite."""
from __future__ import annotations
import json
from pathlib import Path


def compare(suite_id: str, profiles_dir: str = "profiles") -> str:
    rows = []
    for d in sorted(Path(profiles_dir).iterdir()):
        f = d / f"{suite_id}.json"
        if f.exists():
            p = json.loads(f.read_text())
            rows.append(p)
    if not rows:
        return "No profiles found."
    L = [f"# Comparison: suite {suite_id}", "",
         "| Model | Grade | Primary score | Gate violations |",
         "|---|---|---|---|"]
    for p in rows:
        g = p["grade"]
        L.append(f"| {p['model']} | {g['grade']} | {g['score']} | "
                 f"{len(g['gate_violations'])} |")
    L.append("")
    conds = sorted({c for p in rows for c in p["condition_metrics"]})
    for cond in conds:
        L.append(f"## {cond}")
        metrics = sorted({m for p in rows
                          for m in p["condition_metrics"].get(cond, {})})
        L.append("| Metric | " + " | ".join(p["model"] for p in rows) + " |")
        L.append("|---" * (len(rows) + 1) + "|")
        for m in metrics:
            vals = []
            for p in rows:
                v = p["condition_metrics"].get(cond, {}).get(m)
                vals.append(f"{v:.4f}" if isinstance(v, float) else str(v))
            L.append(f"| {m} | " + " | ".join(vals) + " |")
        L.append("")
    return "\n".join(L)
