"""
Tool registry for dynamic tool discovery and management.
"""

import inspect
import importlib
import pkgutil
from typing import Dict, List, Type, Optional

from .base import BaseTool
from src.utils.logging import mcp_logger


class ToolRegistry:
    """Registry for managing and discovering MCP tools."""

    def __init__(self):
        """Initialize the tool registry."""
        self._tools: Dict[str, BaseTool] = {}
        self._tool_classes: Dict[str, Type[BaseTool]] = {}

    def register_tool(self, tool_class: Type[BaseTool], config: Optional[Dict] = None) -> None:
        """
        Register a tool class with optional configuration.

        Args:
            tool_class: The tool class to register
            config: Optional configuration for the tool instance

        Raises:
            ValueError: If tool name is already registered
        """
        # Create tool instance
        tool_instance = tool_class(config)
        tool_name = tool_instance.name

        if tool_name in self._tools:
            raise ValueError(f"Tool '{tool_name}' is already registered")

        self._tools[tool_name] = tool_instance
        self._tool_classes[tool_name] = tool_class

        mcp_logger.info(f"Registered tool: {tool_name}")

    def unregister_tool(self, tool_name: str) -> None:
        """
        Unregister a tool by name.

        Args:
            tool_name: Name of the tool to unregister

        Raises:
            KeyError: If tool is not registered
        """
        if tool_name not in self._tools:
            raise KeyError(f"Tool '{tool_name}' is not registered")

        del self._tools[tool_name]
        del self._tool_classes[tool_name]

        mcp_logger.info(f"Unregistered tool: {tool_name}")

    def get_tool(self, tool_name: str) -> BaseTool:
        """
        Get a registered tool by name.

        Args:
            tool_name: Name of the tool to retrieve

        Returns:
            The tool instance

        Raises:
            KeyError: If tool is not registered
        """
        if tool_name not in self._tools:
            raise KeyError(f"Tool '{tool_name}' is not registered")

        return self._tools[tool_name]

    def list_tools(self) -> List[str]:
        """
        Get a list of all registered tool names.

        Returns:
            List of tool names
        """
        return list(self._tools.keys())

    def get_all_tools(self) -> Dict[str, BaseTool]:
        """
        Get all registered tools.

        Returns:
            Dictionary mapping tool names to tool instances
        """
        return self._tools.copy()

    def get_mcp_tool_definitions(self) -> List[Dict]:
        """
        Get MCP tool definitions for all registered tools.

        Returns:
            List of MCP tool definition dictionaries
        """
        return [tool.get_mcp_tool_definition() for tool in self._tools.values()]

    def auto_discover_tools(self, package_name: str = 'tools') -> int:
        """
        Automatically discover and register tools from a package.

        Args:
            package_name: Name of the package to search for tools

        Returns:
            Number of tools discovered and registered
        """
        discovered_count = 0

        try:
            # Import the tools package
            tools_package = importlib.import_module(f'src.{package_name}')
            package_path = tools_package.__path__

            # Iterate through all modules in the package
            for _, module_name, _ in pkgutil.iter_modules(package_path):
                if module_name.startswith('_'):  # Skip private modules
                    continue

                try:
                    # Import the module
                    module = importlib.import_module(f'src.{package_name}.{module_name}')

                    # Find all BaseTool subclasses in the module
                    for name, obj in inspect.getmembers(module):
                        if (inspect.isclass(obj) and
                            issubclass(obj, BaseTool) and
                            obj is not BaseTool and
                            not inspect.isabstract(obj)):

                            # Register the tool
                            try:
                                self.register_tool(obj)
                                discovered_count += 1
                                mcp_logger.info(f"Auto-discovered tool: {obj.__name__}")
                            except ValueError as e:
                                mcp_logger.warning(f"Failed to register tool {obj.__name__}: {e}")

                except ImportError as e:
                    mcp_logger.warning(f"Failed to import module {module_name}: {e}")

        except ImportError as e:
            mcp_logger.error(f"Failed to import package {package_name}: {e}")

        mcp_logger.info(f"Auto-discovery completed. Registered {discovered_count} tools.")
        return discovered_count

    async def execute_tool(self, tool_name: str, **kwargs) -> Dict:
        """
        Execute a tool by name with the provided arguments.

        Args:
            tool_name: Name of the tool to execute
            **kwargs: Arguments to pass to the tool

        Returns:
            Tool execution result

        Raises:
            KeyError: If tool is not registered
        """
        tool = self.get_tool(tool_name)
        return await tool.safe_execute(**kwargs)


# Global registry instance
registry = ToolRegistry()