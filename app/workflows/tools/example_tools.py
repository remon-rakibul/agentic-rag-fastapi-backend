"""Example tools for the RAG workflow.

This file demonstrates how to create and register tools.
Add your custom tools here or create new files in this directory.

Each tool should:
1. Have a clear docstring (becomes the tool description for the LLM)
2. Have typed parameters with descriptions
3. Return a string result

To create a new tool file:
1. Create a new .py file in this directory (e.g., my_tools.py)
2. Import and use the registry: from app.workflows.tools import tool_registry
3. Add the import to __init__.py
"""

from datetime import datetime
from typing import Annotated
from app.workflows.tools import tool_registry


@tool_registry.register
def get_current_datetime(
    format: Annotated[str, "Date format string (e.g., '%Y-%m-%d %H:%M:%S'). Defaults to ISO format."] = None
) -> str:
    """Get the current date and time. Use this when the user asks about today's date, current time, or needs timestamp information."""
    now = datetime.now()
    print(f"Current date and time: {now}")
    if format:
        try:
            return now.strftime(format)
        except ValueError:
            return now.isoformat()
    return now.isoformat()


@tool_registry.register
def calculate(
    expression: Annotated[str, "A mathematical expression to evaluate (e.g., '2 + 2', '100 * 0.15', 'sqrt(16)')"]
) -> str:
    """Perform mathematical calculations. Use this tool when the user asks to calculate, compute, or do math operations. Supports basic arithmetic (+, -, *, /, **) and common functions (sqrt, sin, cos, abs, round, etc.)."""
    import math

    # Safe math functions to allow
    safe_dict = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
        "pow": pow,
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "log10": math.log10,
        "exp": math.exp,
        "pi": math.pi,
        "e": math.e,
        "floor": math.floor,
        "ceil": math.ceil,
    }

    try:
        # Evaluate expression in a restricted namespace
        result = eval(expression, {"__builtins__": {}}, safe_dict)
        return str(result)
    except Exception as e:
        return f"Error calculating '{expression}': {str(e)}"
