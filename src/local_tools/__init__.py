# src/local_tools/__init__.py
from typing import List, Callable, Any

_REGISTERED_TOOLS: List[Callable] = []

def register_tool(func: Callable) -> Callable:
    """A decorator to register a function as an available tool."""
    _REGISTERED_TOOLS.append(func)
    return func

def get_registered_tools() -> List[Callable]:
    """Returns a list of all tools registered with the @register_tool decorator."""
    return _REGISTERED_TOOLS

# Import all tool modules here to ensure their tools are registered on startup.
from . import file_system_tool
from . import regnavigator_tool
from . import web_search_tool