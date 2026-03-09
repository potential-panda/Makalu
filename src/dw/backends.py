"""
LLM backend abstraction for the research agent.

Each backend owns its own conversation history in its native format, which
avoids the impedance mismatch between OpenAI's tool_call/tool_result format
and Claude's tool_use/tool_result format.

Two implementations:
  - OpenAIBackend  — works with OpenAI API and any OpenAI-compatible server
                     (Ollama, vLLM, LM Studio, etc.)
  - ClaudeBackend  — Anthropic Claude API
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class BackendResponse:
    text: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)


class LLMBackend(ABC):
    @abstractmethod
    def reset(self) -> None:
        """Clear conversation history."""

    @abstractmethod
    def step(
        self,
        user_message: str | None,
        system: str,
        tools: list[dict[str, Any]],
    ) -> BackendResponse:
        """
        Optionally add a user message then run one completion step.
        Pass user_message=None for continuation steps (tool results already recorded).
        """

    @abstractmethod
    def record_tool_result(self, tool_call: ToolCall, result: str) -> None:
        """Record a tool result into history after executing a tool call."""


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
        self._history: list[dict[str, Any]] = []

    def reset(self) -> None:
        self._history = []

    def step(
        self,
        user_message: str | None,
        system: str,
        tools: list[dict[str, Any]],
    ) -> BackendResponse:
        if user_message is not None:
            self._history.append({"role": "user", "content": user_message})

        messages = [{"role": "system", "content": system}] + self._history
        kwargs: dict[str, Any] = {"model": self._model, "messages": messages}
        if tools:
            kwargs["tools"] = [{"type": "function", "function": t} for t in tools]
            kwargs["tool_choice"] = "auto"

        response = self._client.chat.completions.create(**kwargs)
        choice = response.choices[0].message

        tool_calls = []
        if choice.tool_calls:
            for tc in choice.tool_calls:
                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=json.loads(tc.function.arguments),
                ))
            # Store assistant message with the full tool_calls structure OpenAI requires
            self._history.append({
                "role": "assistant",
                "content": choice.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in choice.tool_calls
                ],
            })
        else:
            self._history.append({"role": "assistant", "content": choice.content})

        return BackendResponse(text=choice.content, tool_calls=tool_calls)

    def record_tool_result(self, tool_call: ToolCall, result: str) -> None:
        self._history.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": result,
        })


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
        self._history: list[dict[str, Any]] = []
        # Track pending tool uses so we can group results correctly
        self._pending_tool_uses: list[dict[str, Any]] = []

    def reset(self) -> None:
        self._history = []
        self._pending_tool_uses = []

    def step(
        self,
        user_message: str | None,
        system: str,
        tools: list[dict[str, Any]],
    ) -> BackendResponse:
        if user_message is not None:
            self._history.append({"role": "user", "content": user_message})

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
            messages=self._history,
            tools=claude_tools if tools else [],
        )

        text = None
        tool_calls = []
        self._pending_tool_uses = []

        for block in response.content:
            if block.type == "text":
                text = block.text
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=block.input,
                ))
                self._pending_tool_uses.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })

        if tool_calls:
            # Store the full assistant message with all content blocks
            content_blocks: list[dict[str, Any]] = []
            if text:
                content_blocks.append({"type": "text", "text": text})
            content_blocks.extend(self._pending_tool_uses)
            self._history.append({"role": "assistant", "content": content_blocks})
        else:
            self._history.append({"role": "assistant", "content": text or ""})

        return BackendResponse(text=text, tool_calls=tool_calls)

    def record_tool_result(self, tool_call: ToolCall, result: str) -> None:
        # Claude batches all tool results for a turn into a single user message.
        # If the last history entry is already a tool-result user message, append to it.
        tool_result_block = {
            "type": "tool_result",
            "tool_use_id": tool_call.id,
            "content": result,
        }
        if (
            self._history
            and self._history[-1]["role"] == "user"
            and isinstance(self._history[-1]["content"], list)
            and self._history[-1]["content"]
            and self._history[-1]["content"][0].get("type") == "tool_result"
        ):
            self._history[-1]["content"].append(tool_result_block)
        else:
            self._history.append({"role": "user", "content": [tool_result_block]})
