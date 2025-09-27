"""
File operations tools for reading, writing, and managing files.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd

from .base import BaseTool
from src.utils.logging import mcp_logger
from src.utils.decorators import log_execution_time
from src.utils.helpers import detect_encoding, format_bytes


class ReadFileTool(BaseTool):
    """Tool for reading various file formats."""

    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return "Read data from various file formats (CSV, JSON, Parquet, Excel)"

    def get_schema(self) -> Dict:
        return {
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to read"
                },
                "file_type": {
                    "type": "string",
                    "enum": ["csv", "json", "parquet", "excel", "auto"],
                    "description": "File format (auto-detect if not specified)",
                    "default": "auto"
                },
                "options": {
                    "type": "object",
                    "description": "Format-specific options",
                    "properties": {
                        "encoding": {"type": "string", "default": "utf-8"},
                        "delimiter": {"type": "string", "default": ","},
                        "sheet_name": {"type": ["string", "integer"], "default": 0},
                        "header": {"type": ["integer", "boolean"], "default": 0}
                    }
                }
            },
            "required": ["file_path"]
        }

    @log_execution_time("read_file")
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the read file operation."""
        file_path_str = kwargs.get("file_path")
        if not file_path_str:
            raise ValueError("file_path is required")
        file_type = kwargs.get("file_type", "auto")
        options = kwargs.get("options") or {}

        # Use utils for validation
        file_path = Path(file_path_str)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")


        mcp_logger.log_tool_execution("read_file", "start", file_path=str(file_path), file_type=file_type)

        # Auto-detect file type if needed
        if file_type == "auto":
            file_type = self._detect_file_type(file_path)

        # Read file based on type
        try:
            if file_type == "csv":
                data = self._read_csv(file_path, options)
            elif file_type == "json":
                data = self._read_json(file_path, options)
            elif file_type == "parquet":
                data = self._read_parquet(file_path, options)
            elif file_type == "excel":
                data = self._read_excel(file_path, options)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")

            result = {
                "file_path": str(file_path),
                "file_type": file_type,
                "data": data,
                "shape": self._get_data_shape(data),
                "columns": self._get_columns(data),
                "file_size": format_bytes(file_path.stat().st_size),
                "encoding": detect_encoding(str(file_path)) if file_type in ["csv", "json"] else "binary"
            }

            mcp_logger.log_tool_execution("read_file", "success",
                                        records_read=len(data) if hasattr(data, '__len__') else 0,
                                        file_size=file_path.stat().st_size)
            return result

        except Exception as e:
            mcp_logger.log_error_with_context(e, {"tool": "read_file", "file_path": str(file_path), "file_type": file_type})
            raise RuntimeError(f"Failed to read file {file_path}: {str(e)}")

    def _detect_file_type(self, file_path: Path) -> str:
        """Auto-detect file type from extension."""
        suffix = file_path.suffix.lower()
        if suffix == ".csv":
            return "csv"
        elif suffix == ".json":
            return "json"
        elif suffix == ".parquet":
            return "parquet"
        elif suffix in [".xlsx", ".xls"]:
            return "excel"
        else:
            raise ValueError(f"Cannot auto-detect file type for: {suffix}")

    def _read_csv(self, file_path: Path, options: Dict) -> List[Dict]:
        """Read CSV file."""
        encoding = options.get("encoding", "utf-8")
        delimiter = options.get("delimiter", ",")
        nrows = options.get("nrows")

        df = pd.read_csv(file_path, encoding=encoding, delimiter=delimiter, nrows=nrows)
        return df.to_dict(orient="records")

    def _read_json(self, file_path: Path, options: Dict) -> Union[Dict, List]:
        """Read JSON file."""
        encoding = options.get("encoding", "utf-8")

        with open(file_path, "r", encoding=encoding) as f:
            return json.load(f)

    def _read_parquet(self, file_path: Path, options: Dict) -> List[Dict]:
        """Read Parquet file."""
        df = pd.read_parquet(file_path)
        return df.to_dict(orient="records")

    def _read_excel(self, file_path: Path, options: Dict) -> List[Dict]:
        """Read Excel file."""
        sheet_name = options.get("sheet_name", 0)
        header = options.get("header", 0)

        df = pd.read_excel(file_path, sheet_name=sheet_name, header=header)
        return df.to_dict(orient="records")

    def _get_data_shape(self, data: Any) -> Optional[List[int]]:
        """Get shape of the data if applicable."""
        if isinstance(data, list) and data:
            if isinstance(data[0], dict):
                return [len(data), len(data[0].keys())]
            return [len(data)]
        elif isinstance(data, dict):
            return [1, len(data.keys())]
        return None

    def _get_columns(self, data: Any) -> Optional[List[str]]:
        """Get column names if applicable."""
        if isinstance(data, list) and data and isinstance(data[0], dict):
            return list(data[0].keys())
        elif isinstance(data, dict):
            return list(data.keys())
        return None


class WriteFileTool(BaseTool):
    """Tool for writing data to various file formats."""

    @property
    def name(self) -> str:
        return "write_file"

    @property
    def description(self) -> str:
        return "Write data to various file formats (CSV, JSON, Parquet, Excel)"

    def get_schema(self) -> Dict:
        return {
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path where the file will be written"
                },
                "data": {
                    "type": ["array", "object"],
                    "description": "Data to write (list of dicts for tabular data, or any JSON-serializable object)"
                },
                "file_type": {
                    "type": "string",
                    "enum": ["csv", "json", "parquet", "excel", "auto"],
                    "description": "File format (auto-detect from extension if not specified)",
                    "default": "auto"
                },
                "options": {
                    "type": "object",
                    "description": "Format-specific options",
                    "properties": {
                        "encoding": {"type": "string", "default": "utf-8"},
                        "delimiter": {"type": "string", "default": ","},
                        "sheet_name": {"type": "string", "default": "Sheet1"},
                        "index": {"type": "boolean", "default": False}
                    }
                }
            },
            "required": ["file_path", "data"]
        }

    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the write file operation."""
        file_path_str = kwargs.get("file_path")
        if not file_path_str:
            raise ValueError("file_path is required")
        data = kwargs.get("data")
        if data is None:
            raise ValueError("data is required")
        file_type = kwargs.get("file_type", "auto")
        options = kwargs.get("options") or {}

        file_path = Path(file_path_str)

        # Create directory if it doesn't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Auto-detect file type if needed
        if file_type == "auto":
            file_type = self._detect_file_type(file_path)

        # Write file based on type
        try:
            if file_type == "csv":
                self._write_csv(file_path, data, options)
            elif file_type == "json":
                self._write_json(file_path, data, options)
            elif file_type == "parquet":
                self._write_parquet(file_path, data, options)
            elif file_type == "excel":
                self._write_excel(file_path, data, options)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")

            return {
                "file_path": str(file_path),
                "file_type": file_type,
                "file_size": file_path.stat().st_size,
                "records_written": len(data) if isinstance(data, list) else 1
            }

        except Exception as e:
            raise RuntimeError(f"Failed to write file {file_path}: {str(e)}")

    def _detect_file_type(self, file_path: Path) -> str:
        """Auto-detect file type from extension."""
        suffix = file_path.suffix.lower()
        if suffix == ".csv":
            return "csv"
        elif suffix == ".json":
            return "json"
        elif suffix == ".parquet":
            return "parquet"
        elif suffix in [".xlsx", ".xls"]:
            return "excel"
        else:
            raise ValueError(f"Cannot auto-detect file type for: {suffix}")

    def _write_csv(self, file_path: Path, data: Any, options: Dict) -> None:
        """Write CSV file."""
        encoding = options.get("encoding", "utf-8")
        delimiter = options.get("delimiter", ",")
        index = options.get("index", False)

        df = pd.DataFrame(data)
        df.to_csv(file_path, encoding=encoding, sep=delimiter, index=index)

    def _write_json(self, file_path: Path, data: Any, options: Dict) -> None:
        """Write JSON file."""
        encoding = options.get("encoding", "utf-8")

        with open(file_path, "w", encoding=encoding) as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _write_parquet(self, file_path: Path, data: Any, options: Dict) -> None:
        """Write Parquet file."""
        df = pd.DataFrame(data)
        df.to_parquet(file_path, index=False)

    def _write_excel(self, file_path: Path, data: Any, options: Dict) -> None:
        """Write Excel file."""
        sheet_name = options.get("sheet_name", "Sheet1")
        index = options.get("index", False)

        df = pd.DataFrame(data)
        df.to_excel(file_path, sheet_name=sheet_name, index=index)


class ListFilesTool(BaseTool):
    """Tool for listing files in directories with filtering options."""

    @property
    def name(self) -> str:
        return "list_files"

    @property
    def description(self) -> str:
        return "List files in a directory with optional filtering by extension, size, or date"

    def get_schema(self) -> Dict:
        return {
            "properties": {
                "directory": {
                    "type": "string",
                    "description": "Directory path to list files from",
                    "default": "."
                },
                "pattern": {
                    "type": "string",
                    "description": "File pattern to match (e.g., '*.csv', '*.json')"
                },
                "recursive": {
                    "type": "boolean",
                    "description": "Search recursively in subdirectories",
                    "default": False
                },
                "include_hidden": {
                    "type": "boolean",
                    "description": "Include hidden files (starting with '.')",
                    "default": False
                },
                "min_size": {
                    "type": "integer",
                    "description": "Minimum file size in bytes"
                },
                "max_size": {
                    "type": "integer",
                    "description": "Maximum file size in bytes"
                }
            },
            "required": []
        }

    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the list files operation."""
        directory = kwargs.get("directory", ".")
        pattern = kwargs.get("pattern")
        recursive = kwargs.get("recursive", False)
        include_hidden = kwargs.get("include_hidden", False)
        min_size = kwargs.get("min_size")
        max_size = kwargs.get("max_size")

        directory_path = Path(directory)

        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        if not directory_path.is_dir():
            raise ValueError(f"Path is not a directory: {directory}")

        files = []
        search_pattern = pattern or "*"

        try:
            if recursive:
                file_paths = directory_path.rglob(search_pattern)
            else:
                file_paths = directory_path.glob(search_pattern)

            for file_path in file_paths:
                if not file_path.is_file():
                    continue

                # Skip hidden files if not requested
                if not include_hidden and file_path.name.startswith('.'):
                    continue

                # Get file stats
                stat = file_path.stat()

                # Apply size filters
                if min_size is not None and stat.st_size < min_size:
                    continue
                if max_size is not None and stat.st_size > max_size:
                    continue

                files.append({
                    "name": file_path.name,
                    "path": str(file_path),
                    "size": stat.st_size,
                    "modified": stat.st_mtime,
                    "extension": file_path.suffix.lower()
                })

            # Sort by name
            files.sort(key=lambda x: x["name"])

            return {
                "directory": str(directory_path),
                "total_files": len(files),
                "files": files,
                "filters_applied": {
                    "pattern": pattern,
                    "recursive": recursive,
                    "include_hidden": include_hidden,
                    "min_size": min_size,
                    "max_size": max_size
                }
            }

        except Exception as e:
            raise RuntimeError(f"Failed to list files in {directory}: {str(e)}")


class FileInfoTool(BaseTool):
    """Tool for getting detailed information about a file."""

    @property
    def name(self) -> str:
        return "file_info"

    @property
    def description(self) -> str:
        return "Get file metadata and detailed information about a file including basic statistics"

    def get_schema(self) -> Dict:
        return {
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to analyze"
                },
                "include_preview": {
                    "type": "boolean",
                    "description": "Include a preview of file contents",
                    "default": True
                },
                "preview_lines": {
                    "type": "integer",
                    "description": "Number of lines/records to include in preview",
                    "default": 5,
                    "minimum": 1
                }
            },
            "required": ["file_path"]
        }

    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the file info operation."""
        file_path_str = kwargs.get("file_path")
        if not file_path_str:
            raise ValueError("file_path is required")
        include_preview = kwargs.get("include_preview", True)
        preview_lines = kwargs.get("preview_lines", 5)

        file_path = Path(file_path_str)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not file_path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")

        try:
            # Get basic file info
            stat = file_path.stat()
            info = {
                "name": file_path.name,
                "path": str(file_path),
                "size": stat.st_size,
                "size_human": self._format_file_size(stat.st_size),
                "extension": file_path.suffix.lower(),
                "created": stat.st_ctime,
                "modified": stat.st_mtime,
                "permissions": oct(stat.st_mode)[-3:]
            }

            # Try to detect file type and get additional info
            file_type = self._detect_file_type(file_path)

            # Add encoding and line count for text files
            if file_type in ["csv", "json", "text"]:
                try:
                    info["encoding"] = detect_encoding(str(file_path))
                    with open(file_path, 'r', encoding=info["encoding"]) as f:
                        info["line_count"] = sum(1 for _ in f)
                except Exception:
                    info["encoding"] = "unknown"
                    info["line_count"] = "unknown"
            if file_type:
                info["detected_type"] = file_type

                # Get format-specific info
                if include_preview:
                    try:
                        preview_info = await self._get_file_preview(file_path, file_type, preview_lines)
                        info.update(preview_info)
                    except Exception as e:
                        info["preview_error"] = str(e)

            return info

        except Exception as e:
            raise RuntimeError(f"Failed to get file info for {file_path}: {str(e)}")

    def _detect_file_type(self, file_path: Path) -> Optional[str]:
        """Detect file type from extension."""
        suffix = file_path.suffix.lower()
        type_map = {
            ".csv": "csv",
            ".json": "json",
            ".parquet": "parquet",
            ".xlsx": "excel",
            ".xls": "excel",
            ".txt": "text",
            ".log": "text"
        }
        return type_map.get(suffix)

    async def _get_file_preview(self, file_path: Path, file_type: str, preview_lines: int) -> Dict:
        """Get preview information for the file."""
        preview_info = {}

        try:
            if file_type == "csv":
                df = pd.read_csv(file_path, nrows=preview_lines)
                preview_info.update({
                    "row_count_sample": len(df),
                    "column_count": len(df.columns),
                    "columns": df.columns.tolist(),
                    "data_types": df.dtypes.to_dict(),
                    "preview": df.to_dict(orient="records")
                })

            elif file_type == "json":
                with open(file_path, "r") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        preview_info.update({
                            "total_records": len(data),
                            "preview": data[:preview_lines]
                        })
                    else:
                        preview_info["preview"] = data

            elif file_type == "parquet":
                df = pd.read_parquet(file_path)
                preview_info.update({
                    "row_count": len(df),
                    "column_count": len(df.columns),
                    "columns": df.columns.tolist(),
                    "data_types": df.dtypes.to_dict(),
                    "preview": df.head(preview_lines).to_dict(orient="records")
                })

            elif file_type == "excel":
                df = pd.read_excel(file_path, nrows=preview_lines)
                preview_info.update({
                    "row_count_sample": len(df),
                    "column_count": len(df.columns),
                    "columns": df.columns.tolist(),
                    "data_types": df.dtypes.to_dict(),
                    "preview": df.to_dict(orient="records")
                })

            elif file_type == "text":
                with open(file_path, "r") as f:
                    lines = [f.readline().strip() for _ in range(preview_lines)]
                    lines = [line for line in lines if line]  # Remove empty lines
                preview_info["preview"] = lines

        except Exception as e:
            preview_info["preview_error"] = str(e)

        return preview_info

    def _format_file_size(self, size_bytes: int) -> str:
        """Format file size in human-readable format."""
        if size_bytes == 0:
            return "0 B"

        size_names = ["B", "KB", "MB", "GB", "TB"]
        i = 0
        size_float = float(size_bytes)
        while size_float >= 1024 and i < len(size_names) - 1:
            size_float /= 1024.0
            i += 1

        return f"{size_float:.1f} {size_names[i]}"