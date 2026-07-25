"""Adapter for any OpenAI-compatible chat completions endpoint.

Covers OpenAI, Azure OpenAI, Mistral, Qwen and DeepSeek hosted APIs, and
self-hosted Llama via vLLM or TGI. Configure base_url and api_key_env.
"""
from __future__ import annotations
import os, json, time
import httpx
from .base import ModelAdapter, ModelRequest, ModelResponse, ToolCall


class OpenAICompatAdapter(ModelAdapter):
    def __init__(self, model_config: dict):
        super().__init__(model_config)
        self.base_url = model_config["base_url"].rstrip("/")
        key = os.environ.get(model_config.get("api_key_env", "OPENAI_API_KEY"), "")
        self.headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        self.extra_body = model_config.get("extra_body", {})
        self.timeout = model_config.get("timeout_s", 300)

    def complete(self, request: ModelRequest) -> ModelResponse:
        body = {
            "model": self.model_id,
            "messages": self._messages(request),
            "max_tokens": request.decoding.max_tokens,
        }
        d = request.decoding
        if d.temperature is not None: body["temperature"] = d.temperature
        if d.top_p is not None: body["top_p"] = d.top_p
        if d.seed is not None and self.capabilities.supports_seed: body["seed"] = d.seed
        if d.stop: body["stop"] = d.stop
        if d.reasoning_effort and self.capabilities.supports_reasoning_modes:
            body["reasoning_effort"] = d.reasoning_effort
        if request.tools:
            body["tools"] = [{"type": "function", "function":
                              {"name": t.name, "description": t.description,
                               "parameters": t.parameters}} for t in request.tools]
        if request.response_schema and self.capabilities.supports_structured_output:
            body["response_format"] = {"type": "json_schema",
                                       "json_schema": {"name": "out",
                                                       "schema": request.response_schema}}
        body.update(self.extra_body)

        t0 = time.time()
        r = httpx.post(f"{self.base_url}/chat/completions", headers=self.headers,
                       json=body, timeout=self.timeout)
        latency = time.time() - t0
        if r.status_code != 200:
            return ModelResponse(text="", error=f"HTTP {r.status_code}: {r.text[:500]}",
                                 latency_s=latency)
        data = r.json()
        msg = data["choices"][0]["message"]
        calls = [ToolCall(name=c["function"]["name"],
                          arguments=json.loads(c["function"]["arguments"] or "{}"),
                          call_id=c.get("id", ""))
                 for c in msg.get("tool_calls") or []]
        usage = data.get("usage", {})
        return ModelResponse(
            text=msg.get("content") or "", tool_calls=calls,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            reasoning_tokens=(usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0),
            latency_s=latency, raw=data)

    def _messages(self, request: ModelRequest) -> list[dict]:
        out = []
        for m in request.messages:
            if m.role == "tool":
                out.append({"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content})
            elif m.role == "assistant" and m.tool_calls:
                out.append({"role": "assistant", "content": m.content or None,
                            "tool_calls": [{"id": c.call_id or f"call_{i}", "type": "function",
                                            "function": {"name": c.name,
                                                         "arguments": json.dumps(c.arguments)}}
                                           for i, c in enumerate(m.tool_calls)]})
            else:
                out.append({"role": m.role, "content": m.content})
        return out
