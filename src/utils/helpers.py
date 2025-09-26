"""
Helper functions for Engineer Your Data MCP.
Only contains actually used functions.
"""

from pathlib import Path
from typing import Union


def format_bytes(bytes_value: int) -> str:
    """Format bytes into human readable format."""
    if bytes_value == 0:
        return "0.0 B"

    units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
    size = float(bytes_value)
    unit_index = 0

    while size >= 1024.0 and unit_index < len(units) - 1:
        size /= 1024.0
        unit_index += 1

    return f"{size:.1f} {units[unit_index]}"


def detect_encoding(file_path: Union[str, Path]) -> str:
    """Simple encoding detection - defaults to utf-8."""
    # Simple approach without external dependencies
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            f.read(100)  # Try to read a bit
        return 'utf-8'
    except UnicodeDecodeError:
        return 'latin-1'  # Fallback encoding
    except Exception:
        return 'utf-8'  # Default fallback