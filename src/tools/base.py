"""
Base tool interface for all MCP tools.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from src.utils.logging import mcp_logger


class BaseTool(ABC):
    """Base interface for all MCP tools."""

    def __init__(self, config: Optional[Dict] = None):
        """Initialize the tool with optional configuration."""
        self.config = config or {}
        self.logger = mcp_logger

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name - must be unique across all tools."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description for the MCP client."""
        pass

    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        """
        Execute the tool with the provided arguments.

        Args:
            **kwargs: Tool-specific arguments

        Returns:
            Tool execution result

        Raises:
            ValueError: For invalid input arguments
            RuntimeError: For tool execution errors
        """
        pass

    @abstractmethod
    def get_schema(self) -> Dict:
        """
        Get the JSON schema for tool input parameters.

        Returns:
            JSON schema dictionary defining input parameters
        """
        pass

    def validate_input(self, **kwargs) -> Dict:
        """
        Validate input arguments against the tool schema.

        Args:
            **kwargs: Input arguments to validate

        Returns:
            Validated and potentially transformed arguments

        Raises:
            ValueError: If validation fails
        """
        # Basic validation - subclasses can override for custom validation
        schema = self.get_schema()
        properties = schema.get('properties', {})
        required = schema.get('required', [])

        # Check required parameters
        for param in required:
            if param not in kwargs:
                raise ValueError(f"Missing required parameter: {param}")

        # Check for unknown parameters
        for param in kwargs:
            if param not in properties:
                raise ValueError(f"Unknown parameter: {param}")

        return kwargs

    async def safe_execute(self, **kwargs) -> Dict[str, Any]:
        """
        Safely execute the tool with error handling and logging.

        Args:
            **kwargs: Tool arguments

        Returns:
            Dictionary with 'success', 'result', and optional 'error' keys
        """
        try:
            self.logger.info(f"Executing tool '{self.name}' with args: {kwargs}")

            # Validate input
            validated_kwargs = self.validate_input(**kwargs)

            # Execute tool
            result = await self.execute(**validated_kwargs)

            self.logger.info(f"Tool '{self.name}' executed successfully")
            return {
                'success': True,
                'result': result
            }

        except Exception as e:
            self.logger.error(f"Tool '{self.name}' execution failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'result': None
            }

    def get_mcp_tool_definition(self) -> Dict:
        """
        Get the MCP tool definition for registration.

        Returns:
            MCP tool definition dictionary
        """
        return {
            'name': self.name,
            'description': self.description,
            'inputSchema': {
                'type': 'object',
                **self.get_schema()
            }
        }