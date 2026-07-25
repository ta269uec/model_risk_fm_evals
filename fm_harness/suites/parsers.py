from __future__ import annotations
import json, re


def parse_label(text: str, label_set: list[str]) -> str | None:
    """Match model output to a label. Exact, then case-insensitive, then
    unique-substring. None means invalid label (a tracked metric)."""
    t = text.strip().strip('"').strip(".")
    if t in label_set:
        return t
    lowered = {l.lower(): l for l in label_set}
    if t.lower() in lowered:
        return lowered[t.lower()]
    hits = [l for l in label_set if l.lower() in t.lower()]
    return hits[0] if len(hits) == 1 else None


def parse_json(text: str):
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n?", "", t)
        t = re.sub(r"\n?```$", "", t)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", t, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
        return None


def parse_number(text: str) -> float | None:
    m = re.findall(r"-?\d[\d,]*\.?\d*", text.replace("$", ""))
    if not m:
        return None
    try:
        return float(m[-1].replace(",", ""))
    except ValueError:
        return None


PARSERS = {"text": lambda t, ls: t.strip(),
           "label": parse_label,
           "json": lambda t, ls: parse_json(t),
           "number": lambda t, ls: parse_number(t)}
