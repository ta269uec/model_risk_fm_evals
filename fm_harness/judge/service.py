"""LLM-as-judge service. Judge model, version, and rubrics are pinned in
config/judge.yaml; every score records the judge identity for auditability.
"""
from __future__ import annotations
import re
from pathlib import Path
import yaml
from ..adapters.registry import load_model
from ..adapters.base import Message, ModelRequest, DecodingConfig

RUBRICS = {
    "correctness": (
        "You are grading an answer against a reference.\n"
        "Question or task:\n{question}\n\nReference answer:\n{gold}\n\n"
        "Model answer:\n{answer}\n\n"
        "Score 0 to 10 for factual correctness against the reference. "
        "Penalize wrong facts heavily; ignore style. "
        "Reply exactly as: SCORE: <number> RATIONALE: <one sentence>"),
    "faithfulness": (
        "You are checking whether an answer is fully supported by the provided context.\n"
        "Context:\n{context}\n\nAnswer:\n{answer}\n\n"
        "Score 0 to 10, where 10 means every claim is supported by the context and "
        "0 means the answer is largely unsupported. "
        "Reply exactly as: SCORE: <number> RATIONALE: <one sentence>"),
    "fact_recall": (
        "Critical facts that must appear in the summary:\n{critical_facts}\n\n"
        "Summary:\n{answer}\n\n"
        "Score 0 to 10 proportional to the fraction of critical facts present. "
        "Reply exactly as: SCORE: <number> RATIONALE: <one sentence>"),
}


class JudgeService:
    def __init__(self, config_path: str = "config/judge.yaml",
                 model_dir: str = "config/models"):
        cfg = yaml.safe_load(Path(config_path).read_text())
        self.judge_model_name = cfg["judge_model"]
        self.judge = load_model(self.judge_model_name, model_dir)
        self.rubric_version = cfg.get("rubric_version", "1")
        self.rubrics = dict(RUBRICS)
        self.rubrics.update(cfg.get("rubrics", {}))

    def score(self, metric: str, fields: dict) -> tuple[float | None, str]:
        prompt = self.rubrics[metric].format_map(
            {k: str(v)[:8000] for k, v in fields.items()})
        req = ModelRequest(
            messages=[Message(role="user", content=prompt)],
            decoding=DecodingConfig(name="judge", temperature=0.0, max_tokens=200),
            metadata={"item_id": f"judge:{metric}", "gold": None})
        resp = self.judge.complete(req)
        if resp.error:
            return None, resp.error
        m = re.search(r"SCORE:\s*([0-9.]+)", resp.text)
        return (min(float(m.group(1)), 10.0) / 10.0 if m else None), resp.text[:300]
