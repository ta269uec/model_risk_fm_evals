# FM Testing Harness

Automated profiling harness for validating foundation models as registered
components. Implements the Foundation Model Testing Program playbook:
config-driven suites per task pattern family, standardized decoding
configurations, metric registry, LLM-as-judge scoring, rule-based grading with
critical-probe gates, limitation drafting, and cross-model comparison.

## Install

    pip install -r requirements.txt
    # optional, only for Hugging Face external benchmarks:
    pip install datasets

## Quickstart (no network, mock model)

    python -m fm_harness.cli list
    python -m fm_harness.cli run --model mock-fm --suite classification_core
    python -m fm_harness.cli run --model mock-fm --suite rag_core --judge
    python -m fm_harness.cli run --model mock-fm --suite agentic_single
    python -m fm_harness.cli compare --suite classification_core

## Real model

    export ANTHROPIC_API_KEY=...
    python -m fm_harness.cli run --model claude-fable-5 --suite classification_core --max-usd 25

## Layout

    fm_harness/adapters/    provider adapters (openai_compat, anthropic, google, mock)
    fm_harness/suites/      generic config-driven engine + agentic episode engine + parsers
    fm_harness/metrics/     metric registry (add a function, reference it in YAML)
    fm_harness/judge/       pinned LLM-as-judge service
    fm_harness/execution/   parallel resumable runner with cache and cost guard
    fm_harness/sandbox/     deterministic banking tool sandbox for agentic suites
    fm_harness/analysis/    grading rules, profile emission, comparison
    config/models/          one YAML per FM (the onboarding artifact)
    config/suites/          one YAML per pattern family suite
    config/datasets/        internal and external benchmark registrations
    config/decoding.yaml    D0..D3 and DR_* standard configurations
    config/grading.yaml     grade thresholds and critical probes per family
    profiles/               emitted profiles (JSON + markdown) per model
    runs/                   call logs, manifests, response cache

See the accompanying configuration guide (Word document) for onboarding a new
FM and registering new benchmarks.
