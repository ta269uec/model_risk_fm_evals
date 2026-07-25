"""Adapter for the Google Gemini API (generateContent)."""
from __future__ import annotations
import os, time
import httpx
from .base import ModelAdapter, ModelRequest, ModelResponse, ToolCall


class GoogleAdapter(ModelAdapter):
    def __init__(self, model_config: dict):
        super().__init__(model_config)
        self.base_url = model_config.get(
            "base_url", "https://generativelanguage.googleapis.com/v1beta").rstrip("/")
        self.key = os.environ.get(model_config.get("api_key_env", "GOOGLE_API_KEY"), "")
        self.timeout = model_config.get("timeout_s", 300)

    def complete(self, request: ModelRequest) -> ModelResponse:
        system = "\n".join(m.content for m in request.messages if m.role == "system")
        contents = []
        for m in request.messages:
            if m.role == "system":
                continue
            if m.role == "tool":
                contents.append({"role": "user", "parts": [{"functionResponse": {
                    "name": m.tool_call_id, "response": {"content": m.content}}}]})
            elif m.role == "assistant" and m.tool_calls:
                parts = ([{"text": m.content}] if m.content else [])
                parts += [{"functionCall": {"name": c.name, "args": c.arguments}}
                          for c in m.tool_calls]
                contents.append({"role": "model", "parts": parts})
            else:
                role = "model" if m.role == "assistant" else "user"
                contents.append({"role": role, "parts": [{"text": m.content}]})

        d = request.decoding
        gen = {"maxOutputTokens": d.max_tokens}
        if d.temperature is not None: gen["temperature"] = d.temperature
        if d.top_p is not None: gen["topP"] = d.top_p
        if d.top_k is not None: gen["topK"] = d.top_k
        if d.stop: gen["stopSequences"] = d.stop
        if d.reasoning_effort and self.capabilities.supports_reasoning_modes:
            budgets = self.config.get("thinking_budgets",
                                      {"low": 1024, "medium": 8192, "high": 24576})
            gen["thinkingConfig"] = {"thinkingBudget": budgets[d.reasoning_effort]}
        body = {"contents": contents, "generationConfig": gen}
        if system: body["systemInstruction"] = {"parts": [{"text": system}]}
        if request.tools:
            body["tools"] = [{"functionDeclarations": [
                {"name": t.name, "description": t.description, "parameters": t.parameters}
                for t in request.tools]}]

        t0 = time.time()
        r = httpx.post(f"{self.base_url}/models/{self.model_id}:generateContent",
                       params={"key": self.key}, json=body, timeout=self.timeout)
        latency = time.time() - t0
        if r.status_code != 200:
            return ModelResponse(text="", error=f"HTTP {r.status_code}: {r.text[:500]}",
                                 latency_s=latency)
        data = r.json()
        try:
            parts = data["candidates"][0]["content"]["parts"]
        except (KeyError, IndexError):
            return ModelResponse(text="", error=f"empty candidates: {str(data)[:300]}",
                                 latency_s=latency)
        text, calls = "", []
        for p in parts:
            if "text" in p: text += p["text"]
            if "functionCall" in p:
                calls.append(ToolCall(name=p["functionCall"]["name"],
                                      arguments=p["functionCall"].get("args", {})))
        usage = data.get("usageMetadata", {})
        return ModelResponse(text=text, tool_calls=calls,
                             input_tokens=usage.get("promptTokenCount", 0),
                             output_tokens=usage.get("candidatesTokenCount", 0),
                             reasoning_tokens=usage.get("thoughtsTokenCount", 0),
                             latency_s=latency, raw=data)
