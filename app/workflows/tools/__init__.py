"""Tool registry for LangGraph workflow.

This module provides a registry pattern for managing multiple tools
that can be used in the RAG workflow.

Usage:
    from app.workflows.tools import tool_registry, get_all_tools

    # Register a custom tool
    @tool_registry.register
    def my_custom_tool(query: str) -> str:
        '''My custom tool description.'''
        return "result"

    # Or register a LangChain tool
    tool_registry.add_tool(my_langchain_tool)

    # Get all registered tools (including retriever)
    all_tools = get_all_tools(retriever_tool)
"""

from typing import List, Callable, Optional
from langchain_core.tools import BaseTool, tool


class ToolRegistry:
    """Central registry for all workflow tools.

    Provides a clean way to register, manage, and retrieve tools
    for use in the LangGraph workflow.
    """

    def __init__(self):
        self._tools: List[BaseTool] = []
        self._tool_functions: List[Callable] = []

    def register(self, func: Callable) -> Callable:
        """Decorator to register a function as a tool.

        Example:
            @tool_registry.register
            def calculate(expression: str) -> str:
                '''Calculate a mathematical expression.'''
                return str(eval(expression))
        """
        # Convert function to LangChain tool
        langchain_tool = tool(func)
        self._tools.append(langchain_tool)
        self._tool_functions.append(func)
        return func

    def add_tool(self, t: BaseTool) -> None:
        """Add an existing LangChain tool to the registry.

        Use this for tools created externally (like retriever tools).
        """
        if t not in self._tools:
            self._tools.append(t)

    def get_tools(self) -> List[BaseTool]:
        """Get all registered tools."""
        return list(self._tools)

    def get_tool_by_name(self, name: str) -> Optional[BaseTool]:
        """Get a specific tool by name."""
        for t in self._tools:
            if t.name == name:
                return t
        return None

    def clear(self) -> None:
        """Clear all registered tools (useful for testing)."""
        self._tools.clear()
        self._tool_functions.clear()

    def __len__(self) -> int:
        return len(self._tools)

    def __iter__(self):
        return iter(self._tools)


# Global tool registry instance
tool_registry = ToolRegistry()


def get_all_tools(retriever_tool: BaseTool = None) -> List[BaseTool]:
    """Get all tools including the retriever tool.

    Args:
        retriever_tool: The RAG retriever tool (optional, added first if provided)

    Returns:
        List of all tools for the workflow
    """
    tools = []

    # Add retriever tool first (primary tool)
    if retriever_tool is not None:
        tools.append(retriever_tool)

    # Add all registered tools
    for t in tool_registry.get_tools():
        if t not in tools:
            tools.append(t)

    return tools


# Import tools to auto-register them
# Add your tool modules here to register them automatically
from app.workflows.tools import example_tools  # noqa: E402, F401
