"""Model adapter interface. One adapter per provider API shape.

Every suite talks only to this interface. Onboarding a new FM means either
reusing an existing adapter (openai_compat covers most vendors and vLLM/TGI
hosted models) or writing one small subclass.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict  # JSON schema


@dataclass
class ToolCall:
    name: str
    arguments: dict
    call_id: str = ""


@dataclass
class Message:
    role: str                      # system | user | assistant | tool
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str = ""         # for role=tool results


@dataclass
class DecodingConfig:
    name: str = "D0"
    temperature: Optional[float] = 0.0
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    seed: Optional[int] = None
    max_tokens: int = 2048
    reasoning_effort: Optional[str] = None   # provider thinking/effort level
    stop: list[str] = field(default_factory=list)


@dataclass
class ModelRequest:
    messages: list[Message]
    decoding: DecodingConfig
    tools: list[ToolSpec] = field(default_factory=list)
    response_schema: Optional[dict] = None   # native structured output if supported
    metadata: dict = field(default_factory=dict)


@dataclass
class ModelResponse:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    latency_s: float = 0.0
    ttft_s: Optional[float] = None
    raw: Any = None
    error: Optional[str] = None


@dataclass
class Capabilities:
    """Declared at model registration; suites consult these to skip or adapt."""
    supports_tools: bool = True
    supports_structured_output: bool = False
    supports_seed: bool = False
    supports_logprobs: bool = False
    supports_reasoning_modes: bool = False
    reasoning_levels: list[str] = field(default_factory=list)
    supports_vision: bool = False
    max_context: int = 128000
    max_output: int = 8192
    locked_temperature_in_reasoning: bool = False


class ModelAdapter(ABC):
    def __init__(self, model_config: dict):
        self.config = model_config
        self.model_id = model_config["model_string"]
        self.capabilities = Capabilities(**model_config.get("capabilities", {}))
        self.pricing = model_config.get("pricing", {})  # usd per 1M tokens

    @abstractmethod
    def complete(self, request: ModelRequest) -> ModelResponse:
        ...

    def cost_usd(self, resp: ModelResponse) -> float:
        pin = self.pricing.get("input_per_mtok", 0.0)
        pout = self.pricing.get("output_per_mtok", 0.0)
        return (resp.input_tokens * pin + (resp.output_tokens + resp.reasoning_tokens) * pout) / 1e6
