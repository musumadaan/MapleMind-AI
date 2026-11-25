"""Reusable agent skills, implemented as LangChain tools.

Each skill is registered as a `langchain_core.tools.StructuredTool` (typed args schema inferred from the function
signature + description). Agents call skills ONLY through `tool.invoke(args)`, so the same objects can be handed
unchanged to a tool-calling LLM (`llm.bind_tools(registry.tools())`) or a LangGraph `ToolNode`. Every call is timed
by the tracer as `skill:<name>`.
"""
from __future__ import annotations

from typing import Callable

from langchain_core.tools import StructuredTool


class SkillRegistry:
    def __init__(self, tracer):
        self.tracer = tracer
        self.skills: dict[str, StructuredTool] = {}

    def register(self, name: str, description: str, fn: Callable):
        self.skills[name] = StructuredTool.from_function(func=fn, name=name, description=description)

    def call(self, name: str, **kwargs):
        with self.tracer.span(f"skill:{name}"):
            return self.skills[name].invoke(kwargs)

    def tools(self) -> list[StructuredTool]:
        return list(self.skills.values())

    def describe(self) -> list[dict]:
        return [{"name": t.name, "description": t.description, "args": list(t.args)} for t in self.skills.values()]

    def as_langchain_tools(self):
        return self.tools()
