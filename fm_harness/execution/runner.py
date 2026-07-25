"""Execution engine: parallel, resumable, fully logged, cost-guarded.

Every model call is logged as one JSONL record under runs/<run_id>/calls.jsonl.
A response cache keyed on (model, suite version, condition, item, decoding,
repeat) makes reruns resume instead of re-spending.
"""
from __future__ import annotations
import hashlib, json, time, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from ..suites.engine import SuiteSpec, iter_tasks, build_request
from ..suites.parsers import PARSERS
from ..adapters.base import DecodingConfig


def cache_key(model_id, suite_id, suite_ver, cond, item_id, dec_name, rep):
    s = f"{model_id}|{suite_id}|{suite_ver}|{cond}|{item_id}|{dec_name}|{rep}"
    return hashlib.sha256(s.encode()).hexdigest()[:24]


class Runner:
    def __init__(self, adapter, spec: SuiteSpec, dataset_registry,
                 decoding_configs: dict, run_dir: str = "runs",
                 max_workers: int = 8, max_usd: float = 50.0,
                 max_retries: int = 3):
        self.adapter = adapter
        self.spec = spec
        self.registry = dataset_registry
        self.decoding = decoding_configs
        self.max_workers = max_workers
        self.max_usd = max_usd
        self.max_retries = max_retries
        self.spend = 0.0
        self._lock = threading.Lock()
        self.run_id = f"{adapter.model_id.replace('/', '_')}__{spec.id}__{int(time.time())}"
        self.dir = Path(run_dir) / self.run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir = Path(run_dir) / "_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _call_with_retry(self, request):
        delay = 2.0
        for attempt in range(self.max_retries):
            resp = self.adapter.complete(request)
            if not resp.error or "HTTP 4" in (resp.error or ""):
                return resp
            time.sleep(delay); delay *= 2
        return resp

    def _one(self, cond, row, dec: DecodingConfig, rep, label_set):
        key = cache_key(self.adapter.model_id, self.spec.id, self.spec.version,
                        cond.name, row["item_id"], dec.name, rep)
        cached = self.cache_dir / f"{key}.json"
        if cached.exists():
            return json.loads(cached.read_text())

        with self._lock:
            if self.spend >= self.max_usd:
                return {"item_id": row["item_id"], "condition": cond.name,
                        "decoding": dec.name, "repeat": rep,
                        "error": "COST_GUARD_TRIPPED"}

        req = build_request(cond, row, dec, label_set)
        resp = self._call_with_retry(req)
        cost = self.adapter.cost_usd(resp)
        with self._lock:
            self.spend += cost

        parsed = None
        if not resp.error:
            parsed = PARSERS[cond.parser](resp.text, label_set)
        gold = row.get(cond.gold_field)
        correct = None
        if gold is not None and cond.parser in ("label", "number", "text"):
            correct = (str(parsed).strip() == str(gold).strip()) if parsed is not None else False
        rec = {"item_id": row["item_id"], "condition": cond.name,
               "decoding": dec.name, "repeat": rep, "gold": gold,
               "parsed": parsed, "raw_text": resp.text[:4000],
               "correct": correct, "error": resp.error,
               "latency_s": round(resp.latency_s, 3),
               "input_tokens": resp.input_tokens,
               "output_tokens": resp.output_tokens,
               "reasoning_tokens": resp.reasoning_tokens,
               "cost_usd": round(cost, 6),
               "judge_scores": {},
               "extra": {k: row[k] for k in
                         ("answer_absent", "canary", "confidence", "context",
                          "question", "critical_facts")
                         if k in row}}
        cached.write_text(json.dumps(rec))
        return rec

    def run(self) -> list[dict]:
        tasks = list(iter_tasks(self.spec, self.registry, self.decoding))
        records = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futs = [pool.submit(self._one, *t) for t in tasks]
            for f in as_completed(futs):
                records.append(f.result())
        with open(self.dir / "calls.jsonl", "w") as fh:
            for r in records:
                fh.write(json.dumps(r) + "\n")
        manifest = {"run_id": self.run_id, "model": self.adapter.model_id,
                    "suite": self.spec.id, "suite_version": self.spec.version,
                    "n_calls": len(records), "spend_usd": round(self.spend, 4),
                    "timestamp": time.time()}
        (self.dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        return records


def apply_judges(records: list[dict], spec: SuiteSpec, judge_service):
    """Second pass: score judge metrics for conditions that declare them."""
    want = {c.name: c.judge_metrics for c in spec.conditions if c.judge_metrics}
    for r in records:
        jm = want.get(r["condition"])
        if not jm or r.get("error"):
            continue
        for metric in jm:
            fields = {"question": r["extra"].get("question", ""),
                      "context": r["extra"].get("context", ""),
                      "critical_facts": r["extra"].get("critical_facts", ""),
                      "gold": r.get("gold", ""), "answer": r.get("raw_text", "")}
            score, _ = judge_service.score(metric.replace("judge_", ""), fields)
            if score is not None:
                r["judge_scores"][metric.replace("judge_", "")] = score
    return records
