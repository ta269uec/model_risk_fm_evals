"""Deterministic mock adapter for harness self-tests and CI.

Answers correctly on a fixed fraction of items (by content hash) so metric,
grading, and profile code paths can be exercised without network access.
"""
from __future__ import annotations
import hashlib, json, random
from .base import ModelAdapter, ModelRequest, ModelResponse, ToolCall


class MockAdapter(ModelAdapter):
    def complete(self, request: ModelRequest) -> ModelResponse:
        meta = request.metadata
        gold = meta.get("gold")
        prompt = request.messages[-1].content if request.messages else ""
        if "SCORE:" in prompt:  # acting as a mock judge
            h0 = int(hashlib.md5(prompt.encode()).hexdigest(), 16)
            return ModelResponse(text=f"SCORE: {7 + h0 % 4} RATIONALE: mock judge",
                                 input_tokens=300, output_tokens=15)
        h = int(hashlib.md5(str(meta.get("item_id", "")).encode()).hexdigest(), 16)
        correct = (h % 100) < int(self.config.get("accuracy_pct", 85))

        if request.tools and meta.get("gold_tool"):
            if correct:
                calls = [ToolCall(name=meta["gold_tool"], arguments=meta.get("gold_args", {}))]
            else:
                calls = [ToolCall(name=request.tools[h % len(request.tools)].name, arguments={})]
            return ModelResponse(text="", tool_calls=calls, input_tokens=200, output_tokens=30)

        if gold is not None:
            if correct:
                text = gold if isinstance(gold, str) else json.dumps(gold)
            else:
                labels = meta.get("label_set") or ["WRONG_ANSWER"]
                pool = [l for l in labels if l != gold] or labels
                text = random.Random(h).choice(pool)
                if not isinstance(gold, str): text = json.dumps({"error": "wrong"})
        else:
            text = "Mock response."
        return ModelResponse(text=text, input_tokens=150, output_tokens=40, latency_s=0.01)
