"""
LLM backend abstraction for the research agent.

Two implementations:
  - OpenAIBackend  — works with OpenAI API and any OpenAI-compatible server
                     (Ollama, vLLM, LM Studio, etc.)
  - ClaudeBackend  — Anthropic Claude API

Both implement the same interface so the Agent is backend-agnostic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Message:
    role: str   # "user" | "assistant" | "tool" | "system"
    content: str


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class BackendResponse:
    text: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: Any = None  # original response object for debugging


class LLMBackend(ABC):
    @abstractmethod
    def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
        system: str,
    ) -> BackendResponse:
        """Send messages and return the model response."""

    @abstractmethod
    def tool_result_message(self, tool_call: ToolCall, result: str) -> Message:
        """Format a tool result as a Message in the format the backend expects."""


# ---------------------------------------------------------------------------
# OpenAI-compatible backend (OpenAI, Ollama, vLLM, LM Studio, …)
# ---------------------------------------------------------------------------

class OpenAIBackend(LLMBackend):
    """
    Backend for any OpenAI-compatible API.

    For OpenAI:
        OpenAIBackend(api_key="sk-...", model="gpt-4o")

    For local Ollama:
        OpenAIBackend(base_url="http://localhost:11434/v1", api_key="ollama",
                      model="llama3.1")
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        from openai import OpenAI
        self._model = model
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
        system: str,
    ) -> BackendResponse:
        openai_messages = [{"role": "system", "content": system}]
        for m in messages:
            openai_messages.append({"role": m.role, "content": m.content})

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": openai_messages,
        }
        if tools:
            kwargs["tools"] = [{"type": "function", "function": t} for t in tools]
            kwargs["tool_choice"] = "auto"

        response = self._client.chat.completions.create(**kwargs)
        choice = response.choices[0].message

        tool_calls = []
        if choice.tool_calls:
            import json
            for tc in choice.tool_calls:
                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=json.loads(tc.function.arguments),
                ))

        return BackendResponse(
            text=choice.content,
            tool_calls=tool_calls,
            raw=response,
        )

    def tool_result_message(self, tool_call: ToolCall, result: str) -> Message:
        # OpenAI expects role="tool" with tool_call_id; we encode it in content
        # because our Message type is simple. We handle this specially in the agent.
        return Message(role="tool", content=result)


# ---------------------------------------------------------------------------
# Anthropic Claude backend
# ---------------------------------------------------------------------------

class ClaudeBackend(LLMBackend):
    """
    Backend for the Anthropic Claude API.

        ClaudeBackend(api_key="sk-ant-...", model="claude-sonnet-4-6")
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        api_key: str | None = None,
    ):
        from anthropic import Anthropic
        self._model = model
        self._client = Anthropic(api_key=api_key)

    def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
        system: str,
    ) -> BackendResponse:
        import json

        claude_messages = []
        for m in messages:
            if m.role == "tool":
                # tool results in Claude format come from the agent layer
                claude_messages.append(m.content)  # pre-formatted by tool_result_message
            else:
                claude_messages.append({"role": m.role, "content": m.content})

        claude_tools = [
            {
                "name": t["name"],
                "description": t.get("description", ""),
                "input_schema": t.get("parameters", {"type": "object", "properties": {}}),
            }
            for t in tools
        ]

        response = self._client.messages.create(
            model=self._model,
            max_tokens=4096,
            system=system,
            messages=claude_messages,
            tools=claude_tools if tools else [],
        )

        text = None
        tool_calls = []
        for block in response.content:
            if block.type == "text":
                text = block.text
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=block.input,
                ))

        return BackendResponse(text=text, tool_calls=tool_calls, raw=response)

    def tool_result_message(self, tool_call: ToolCall, result: str) -> Message:
        # Claude requires a specific structure; we pass the full dict as content
        # and the agent handles it
        content = {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": tool_call.id, "content": result}],
        }
        return Message(role="tool", content=content)
