"""Tool registry for assembling LangChain tools by group and profile."""

from __future__ import annotations

from langchain_core.tools import BaseTool


class ToolRegistry:
    """Organize tools into named groups and reusable profiles."""

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}
        self._groups: dict[str, list[str]] = {}
        self._profiles: dict[str, list[str]] = {}

    def register(self, tool: BaseTool, *, group: str) -> None:
        self._tools[tool.name] = tool
        tool_names = self._groups.setdefault(group, [])
        if tool.name not in tool_names:
            tool_names.append(tool.name)

    def define_profile(self, profile: str, *, groups: list[str]) -> None:
        self._profiles[profile] = list(groups)

    def get_tools(self, *, groups: list[str]) -> list[BaseTool]:
        seen: set[str] = set()
        tools: list[BaseTool] = []
        for group in groups:
            for name in self._groups.get(group, []):
                if name in seen:
                    continue
                tool = self._tools.get(name)
                if tool is None:
                    continue
                seen.add(name)
                tools.append(tool)
        return tools

    def get_tools_for_profile(self, profile: str) -> list[BaseTool]:
        groups = self._profiles.get(profile)
        if not groups:
            return []
        return self.get_tools(groups=groups)

    def list_groups(self) -> list[str]:
        return sorted(self._groups)

    def list_profiles(self) -> list[str]:
        return sorted(self._profiles)
