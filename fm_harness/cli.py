"""CLI entry points.

  python -m fm_harness.cli list
  python -m fm_harness.cli run --model mock-fm --suite classification_core
  python -m fm_harness.cli run --model mock-fm --suite agentic_single
  python -m fm_harness.cli judge --model mock-fm --suite rag_core     (adds judge scores)
  python -m fm_harness.cli compare --suite classification_core
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import yaml
from .adapters.registry import load_model
from .adapters.base import DecodingConfig
from .datasets.registry import DatasetRegistry
from .suites.engine import SuiteSpec
from .suites.agentic import run_agentic_suite
from .execution.runner import Runner, apply_judges
from .analysis.profile import emit_profile
from .analysis.compare import compare


def load_decoding(path="config/decoding.yaml") -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    return {name: DecodingConfig(name=name, **params)
            for name, params in cfg.items()}


def cmd_list(_):
    print("Models:", [p.stem for p in Path("config/models").glob("*.yaml")])
    print("Suites:", [p.stem for p in Path("config/suites").glob("*.yaml")])
    reg = DatasetRegistry()
    print("Datasets:", sorted(reg.entries))


def cmd_run(args):
    adapter = load_model(args.model)
    spec = SuiteSpec.load(args.suite)
    registry = DatasetRegistry()
    decoding = load_decoding()
    if spec.engine == "agentic":
        records = run_agentic_suite(adapter, spec, registry, decoding)
        run_dir = Path("runs") / f"{args.model}__{spec.id}"
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "calls.jsonl", "w") as fh:
            for r in records:
                fh.write(json.dumps(r) + "\n")
    else:
        runner = Runner(adapter, spec, registry, decoding,
                        max_workers=args.workers, max_usd=args.max_usd)
        records = runner.run()
        print(f"run_id={runner.run_id} spend=${runner.spend:.4f}")
    if args.judge:
        from .judge.service import JudgeService
        records = apply_judges(records, spec, JudgeService())
    profile = emit_profile(adapter, spec, records)
    g = profile["grade"]
    print(f"{args.model} :: {spec.family} -> grade {g['grade']} "
          f"({g['primary_metric']}={g['score']})")
    for v in g["gate_violations"]:
        print(f"  GATE: {v}")
    print(f"profile written to profiles/{adapter.model_id.replace('/', '_')}/{spec.id}.json")


def cmd_compare(args):
    out = compare(args.suite)
    Path("profiles").mkdir(exist_ok=True)
    Path(f"profiles/compare_{args.suite}.md").write_text(out)
    print(out)


def main():
    ap = argparse.ArgumentParser("fm_harness")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list").set_defaults(fn=cmd_list)
    r = sub.add_parser("run")
    r.add_argument("--model", required=True)
    r.add_argument("--suite", required=True)
    r.add_argument("--workers", type=int, default=8)
    r.add_argument("--max-usd", type=float, default=50.0)
    r.add_argument("--judge", action="store_true")
    r.set_defaults(fn=cmd_run)
    c = sub.add_parser("compare")
    c.add_argument("--suite", required=True)
    c.set_defaults(fn=cmd_compare)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
