"""Adapter for the Anthropic Messages API (Claude family)."""
from __future__ import annotations
import os, time
import httpx
from .base import ModelAdapter, ModelRequest, ModelResponse, ToolCall


class AnthropicAdapter(ModelAdapter):
    def __init__(self, model_config: dict):
        super().__init__(model_config)
        self.base_url = model_config.get("base_url", "https://api.anthropic.com").rstrip("/")
        key = os.environ.get(model_config.get("api_key_env", "ANTHROPIC_API_KEY"), "")
        self.headers = {"x-api-key": key, "anthropic-version": "2023-06-01",
                        "Content-Type": "application/json"}
        self.timeout = model_config.get("timeout_s", 300)

    def complete(self, request: ModelRequest) -> ModelResponse:
        system = "\n".join(m.content for m in request.messages if m.role == "system")
        msgs = []
        for m in request.messages:
            if m.role == "system":
                continue
            if m.role == "tool":
                msgs.append({"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": m.tool_call_id, "content": m.content}]})
            elif m.role == "assistant" and m.tool_calls:
                blocks = ([{"type": "text", "text": m.content}] if m.content else [])
                blocks += [{"type": "tool_use", "id": c.call_id or f"tu_{i}",
                            "name": c.name, "input": c.arguments}
                           for i, c in enumerate(m.tool_calls)]
                msgs.append({"role": "assistant", "content": blocks})
            else:
                msgs.append({"role": m.role, "content": m.content})

        body = {"model": self.model_id, "messages": msgs,
                "max_tokens": request.decoding.max_tokens}
        if system: body["system"] = system
        d = request.decoding
        if d.reasoning_effort and self.capabilities.supports_reasoning_modes:
            budgets = self.config.get("thinking_budgets",
                                      {"low": 2048, "medium": 8192, "high": 32768})
            body["thinking"] = {"type": "enabled", "budget_tokens": budgets[d.reasoning_effort]}
            # temperature is provider-constrained under extended thinking
        elif d.temperature is not None:
            body["temperature"] = d.temperature
        if d.top_p is not None and "thinking" not in body: body["top_p"] = d.top_p
        if d.stop: body["stop_sequences"] = d.stop
        if request.tools:
            body["tools"] = [{"name": t.name, "description": t.description,
                              "input_schema": t.parameters} for t in request.tools]

        t0 = time.time()
        r = httpx.post(f"{self.base_url}/v1/messages", headers=self.headers,
                       json=body, timeout=self.timeout)
        latency = time.time() - t0
        if r.status_code != 200:
            return ModelResponse(text="", error=f"HTTP {r.status_code}: {r.text[:500]}",
                                 latency_s=latency)
        data = r.json()
        text, calls, rtoks = "", [], 0
        for block in data.get("content", []):
            if block["type"] == "text":
                text += block["text"]
            elif block["type"] == "tool_use":
                calls.append(ToolCall(name=block["name"], arguments=block["input"],
                                      call_id=block["id"]))
            elif block["type"] == "thinking":
                rtoks += len(block.get("thinking", "")) // 4
        usage = data.get("usage", {})
        return ModelResponse(text=text, tool_calls=calls,
                             input_tokens=usage.get("input_tokens", 0),
                             output_tokens=usage.get("output_tokens", 0),
                             reasoning_tokens=rtoks, latency_s=latency, raw=data)
