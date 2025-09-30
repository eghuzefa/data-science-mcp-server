"""
Logging utilities for Engineer Your Data MCP.
"""

import logging
import sys
from typing import Optional
from pathlib import Path


class MCPLogger:
    """Enhanced logger for MCP server operations."""

    def __init__(self, name: str = "engineer-your-data", level: str = "INFO"):
        """Initialize the logger."""
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))

        # Prevent duplicate handlers
        if not self.logger.handlers:
            self._setup_handlers()

    def _setup_handlers(self) -> None:
        """Setup console and file handlers."""
        # Console handler - use stderr to avoid interfering with MCP protocol on stdout
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(logging.INFO)

        # Create formatters
        console_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

    def setup_file_logging(self, log_file: Optional[str] = None) -> None:
        """Setup file logging if needed."""
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.FileHandler(log_path)
            file_handler.setLevel(logging.DEBUG)

            file_formatter = logging.Formatter(
                '%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(funcName)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )

            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)

    def log_tool_execution(self, tool_name: str, operation: str, **kwargs) -> None:
        """Log tool execution with structured data."""
        extra_info = " | ".join([f"{k}={v}" for k, v in kwargs.items()])
        self.logger.info(f"TOOL_EXEC | {tool_name} | {operation} | {extra_info}")

    def log_data_operation(self, operation: str, records_count: int,
                          execution_time: float, **kwargs) -> None:
        """Log data operation metrics."""
        extra_info = " | ".join([f"{k}={v}" for k, v in kwargs.items()])
        self.logger.info(
            f"DATA_OP | {operation} | records={records_count} | "
            f"time={execution_time:.3f}s | {extra_info}"
        )

    def log_error_with_context(self, error: Exception, context: dict) -> None:
        """Log error with additional context."""
        context_str = " | ".join([f"{k}={v}" for k, v in context.items()])
        self.logger.error(f"ERROR | {type(error).__name__}: {str(error)} | {context_str}")

    def log_performance_metric(self, metric_name: str, value: float, unit: str = "") -> None:
        """Log performance metrics."""
        self.logger.info(f"METRIC | {metric_name} | {value} {unit}")

    def info(self, message: str, **kwargs) -> None:
        """Log info message."""
        self.logger.info(message, extra=kwargs)

    def debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        self.logger.debug(message, extra=kwargs)

    def warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        self.logger.warning(message, extra=kwargs)

    def error(self, message: str, **kwargs) -> None:
        """Log error message."""
        self.logger.error(message, extra=kwargs)


# Global logger instance
mcp_logger = MCPLogger()