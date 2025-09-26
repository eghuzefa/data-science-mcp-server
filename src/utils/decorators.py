"""
Common decorators for Engineer Your Data MCP.
"""

import time
import asyncio
import functools
from typing import Any, Callable, Dict, Optional
from .logging import mcp_logger


def log_execution_time(operation_name: Optional[str] = None):
    """Decorator to log execution time of functions."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            op_name = operation_name or f"{func.__name__}"

            try:
                result = await func(*args, **kwargs)
                execution_time = time.time() - start_time
                mcp_logger.log_performance_metric(f"{op_name}_execution_time", execution_time, "seconds")
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                mcp_logger.log_error_with_context(e, {
                    "function": op_name,
                    "execution_time": execution_time,
                    "args_count": len(args),
                    "kwargs_keys": list(kwargs.keys())
                })
                raise

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            op_name = operation_name or f"{func.__name__}"

            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                mcp_logger.log_performance_metric(f"{op_name}_execution_time", execution_time, "seconds")
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                mcp_logger.log_error_with_context(e, {
                    "function": op_name,
                    "execution_time": execution_time,
                    "args_count": len(args),
                    "kwargs_keys": list(kwargs.keys())
                })
                raise

        # Return appropriate wrapper based on function type
        if hasattr(func, '__code__') and 'await' in func.__code__.co_names:
            return async_wrapper
        return sync_wrapper

    return decorator





