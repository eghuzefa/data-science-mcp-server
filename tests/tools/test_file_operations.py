"""
Comprehensive tests for file operations tools.
"""

import pytest
import tempfile
import json
import csv
import os
from pathlib import Path
from unittest.mock import patch, mock_open
import pandas as pd

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from tools.file_operations import ReadFileTool, WriteFileTool, ListFilesTool, FileInfoTool


class TestReadFileTool:
    """Test suite for ReadFileTool."""

    @pytest.fixture
    def tool(self):
        """Create a ReadFileTool instance."""
        return ReadFileTool()

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_tool_properties(self, tool):
        """Test basic tool properties."""
        assert tool.name == "read_file"
        assert "read data from various file formats" in tool.description.lower()
        assert isinstance(tool.get_schema(), dict)
        assert "file_path" in tool.get_schema()["properties"]
        assert "file_path" in tool.get_schema()["required"]

    def test_schema_validation(self, tool):
        """Test input schema validation."""
        schema = tool.get_schema()
        properties = schema["properties"]

        # Check file types
        file_types = properties["file_type"]["enum"]
        expected_types = ["csv", "json", "parquet", "excel", "auto"]
        assert all(file_type in file_types for file_type in expected_types)

        # Check default values
        assert properties["file_type"]["default"] == "auto"

    @pytest.mark.asyncio
    async def test_read_csv_file(self, tool, temp_dir):
        """Test reading CSV files."""
        # Create test CSV file
        csv_path = os.path.join(temp_dir, "test.csv")
        test_data = [
            ["name", "age", "city"],
            ["Alice", "25", "New York"],
            ["Bob", "30", "San Francisco"]
        ]

        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(test_data)

        result = await tool.safe_execute(file_path=csv_path, file_type="csv")

        assert result["success"] is True
        assert "data" in result["result"]
        assert len(result["result"]["data"]) == 2  # Excluding header
        assert result["result"]["data"][0]["name"] == "Alice"

    @pytest.mark.asyncio
    async def test_read_json_file(self, tool, temp_dir):
        """Test reading JSON files."""
        # Create test JSON file
        json_path = os.path.join(temp_dir, "test.json")
        test_data = {
            "users": [
                {"name": "Alice", "age": 25},
                {"name": "Bob", "age": 30}
            ]
        }

        with open(json_path, 'w') as f:
            json.dump(test_data, f)

        result = await tool.safe_execute(file_path=json_path, file_type="json")

        assert result["success"] is True
        assert result["result"]["data"] == test_data

    @pytest.mark.asyncio
    async def test_auto_detect_file_type(self, tool, temp_dir):
        """Test automatic file type detection."""
        # Create test CSV file with .csv extension
        csv_path = os.path.join(temp_dir, "autodetect.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows([["col1", "col2"], ["val1", "val2"]])

        result = await tool.safe_execute(file_path=csv_path, file_type="auto")

        assert result["success"] is True
        assert "data" in result["result"]

    @pytest.mark.asyncio
    async def test_read_nonexistent_file(self, tool):
        """Test reading a non-existent file."""
        result = await tool.safe_execute(file_path="/nonexistent/path.csv")

        assert result["success"] is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_read_with_custom_options(self, tool, temp_dir):
        """Test reading with custom options."""
        # Create CSV with semicolon delimiter
        csv_path = os.path.join(temp_dir, "custom.csv")
        with open(csv_path, 'w') as f:
            f.write("name;age\nAlice;25\nBob;30")

        result = await tool.safe_execute(
            file_path=csv_path,
            file_type="csv",
            options={"delimiter": ";"}
        )

        assert result["success"] is True
        assert len(result["result"]["data"]) == 2

    @pytest.mark.asyncio
    async def test_read_large_file_limit(self, tool, temp_dir):
        """Test reading large files with row limits."""
        # Create large CSV file
        csv_path = os.path.join(temp_dir, "large.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["id", "value"])
            for i in range(1000):
                writer.writerow([i, f"value_{i}"])

        result = await tool.safe_execute(
            file_path=csv_path,
            file_type="csv",
            options={"nrows": 10}
        )

        assert result["success"] is True
        assert len(result["result"]["data"]) == 10


class TestWriteFileTool:
    """Test suite for WriteFileTool."""

    @pytest.fixture
    def tool(self):
        """Create a WriteFileTool instance."""
        return WriteFileTool()

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_tool_properties(self, tool):
        """Test basic tool properties."""
        assert tool.name == "write_file"
        assert "write data to various file formats" in tool.description.lower()
        assert isinstance(tool.get_schema(), dict)
        assert "file_path" in tool.get_schema()["properties"]
        assert "data" in tool.get_schema()["properties"]

    @pytest.mark.asyncio
    async def test_write_csv_file(self, tool, temp_dir):
        """Test writing CSV files."""
        csv_path = os.path.join(temp_dir, "output.csv")
        test_data = [
            {"name": "Alice", "age": 25},
            {"name": "Bob", "age": 30}
        ]

        result = await tool.safe_execute(
            file_path=csv_path,
            data=test_data,
            file_type="csv"
        )

        assert result["success"] is True
        assert os.path.exists(csv_path)

        # Verify file contents
        with open(csv_path, 'r') as f:
            content = f.read()
            assert "Alice" in content
            assert "Bob" in content

    @pytest.mark.asyncio
    async def test_write_json_file(self, tool, temp_dir):
        """Test writing JSON files."""
        json_path = os.path.join(temp_dir, "output.json")
        test_data = {"users": [{"name": "Alice", "age": 25}]}

        result = await tool.safe_execute(
            file_path=json_path,
            data=test_data,
            file_type="json"
        )

        assert result["success"] is True
        assert os.path.exists(json_path)

        # Verify file contents
        with open(json_path, 'r') as f:
            loaded_data = json.load(f)
            assert loaded_data == test_data

    @pytest.mark.asyncio
    async def test_write_with_auto_detect(self, tool, temp_dir):
        """Test writing with automatic file type detection."""
        json_path = os.path.join(temp_dir, "auto.json")
        test_data = {"test": "data"}

        result = await tool.safe_execute(
            file_path=json_path,
            data=test_data,
            file_type="auto"
        )

        assert result["success"] is True
        assert os.path.exists(json_path)

    @pytest.mark.asyncio
    async def test_write_invalid_directory(self, tool):
        """Test writing to invalid directory."""
        result = await tool.safe_execute(
            file_path="/invalid/directory/file.csv",
            data=[{"test": "data"}],
            file_type="csv"
        )

        assert result["success"] is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_write_with_options(self, tool, temp_dir):
        """Test writing with custom options."""
        csv_path = os.path.join(temp_dir, "custom.csv")
        test_data = [{"name": "Alice", "age": 25}]

        result = await tool.safe_execute(
            file_path=csv_path,
            data=test_data,
            file_type="csv",
            options={"index": False}
        )

        assert result["success"] is True


class TestListFilesTool:
    """Test suite for ListFilesTool."""

    @pytest.fixture
    def tool(self):
        """Create a ListFilesTool instance."""
        return ListFilesTool()

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory with test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            Path(tmpdir, "file1.csv").touch()
            Path(tmpdir, "file2.json").touch()
            Path(tmpdir, "file3.txt").touch()
            Path(tmpdir, "subdir").mkdir()
            Path(tmpdir, "subdir", "file4.csv").touch()
            yield tmpdir

    def test_tool_properties(self, tool):
        """Test basic tool properties."""
        assert tool.name == "list_files"
        assert "list files" in tool.description.lower()
        assert isinstance(tool.get_schema(), dict)
        assert "directory" in tool.get_schema()["properties"]

    @pytest.mark.asyncio
    async def test_list_files_basic(self, tool, temp_dir):
        """Test basic file listing."""
        result = await tool.safe_execute(directory=temp_dir)

        assert result["success"] is True
        assert "files" in result["result"]
        files = result["result"]["files"]

        # Should find at least the files we created
        file_names = [f["name"] for f in files]
        assert "file1.csv" in file_names
        assert "file2.json" in file_names

    @pytest.mark.asyncio
    async def test_list_files_with_filter(self, tool, temp_dir):
        """Test file listing with extension filter."""
        result = await tool.safe_execute(
            directory=temp_dir,
            pattern="*.csv"
        )

        assert result["success"] is True
        files = result["result"]["files"]

        # Should only find CSV files
        for file_info in files:
            assert file_info["name"].endswith(".csv")

    @pytest.mark.asyncio
    async def test_list_files_recursive(self, tool, temp_dir):
        """Test recursive file listing."""
        result = await tool.safe_execute(
            directory=temp_dir,
            recursive=True
        )

        assert result["success"] is True
        files = result["result"]["files"]

        # Should find files in subdirectories
        file_paths = [f["path"] for f in files]
        subdir_files = [p for p in file_paths if "subdir" in p]
        assert len(subdir_files) > 0

    @pytest.mark.asyncio
    async def test_list_nonexistent_directory(self, tool):
        """Test listing files in non-existent directory."""
        result = await tool.safe_execute(directory="/nonexistent/directory")

        assert result["success"] is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_list_files_with_details(self, tool, temp_dir):
        """Test file listing with detailed information."""
        result = await tool.safe_execute(
            directory=temp_dir
        )

        assert result["success"] is True
        files = result["result"]["files"]

        # Check that detailed info is included
        if files:
            file_info = files[0]
            assert "size" in file_info
            assert "modified" in file_info


class TestFileInfoTool:
    """Test suite for FileInfoTool."""

    @pytest.fixture
    def tool(self):
        """Create a FileInfoTool instance."""
        return FileInfoTool()

    @pytest.fixture
    def temp_file(self):
        """Create a temporary file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("name,age\nAlice,25\nBob,30")
            temp_path = f.name

        yield temp_path

        # Cleanup
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    def test_tool_properties(self, tool):
        """Test basic tool properties."""
        assert tool.name == "file_info"
        assert "get file metadata" in tool.description.lower()
        assert isinstance(tool.get_schema(), dict)
        assert "file_path" in tool.get_schema()["properties"]

    @pytest.mark.asyncio
    async def test_get_file_info_basic(self, tool, temp_file):
        """Test getting basic file information."""
        result = await tool.safe_execute(file_path=temp_file)

        assert result["success"] is True
        info = result["result"]

        assert "name" in info
        assert "size" in info
        assert "extension" in info
        assert "modified" in info
        assert info["extension"] == ".csv"

    @pytest.mark.asyncio
    async def test_get_file_info_detailed(self, tool, temp_file):
        """Test getting detailed file information."""
        result = await tool.safe_execute(
            file_path=temp_file,
            include_preview=True
        )

        assert result["success"] is True
        info = result["result"]

        # Should include detailed stats
        assert "encoding" in info
        assert "line_count" in info

    @pytest.mark.asyncio
    async def test_get_file_info_nonexistent(self, tool):
        """Test getting info for non-existent file."""
        result = await tool.safe_execute(file_path="/nonexistent/file.csv")

        assert result["success"] is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_get_file_info_with_preview(self, tool, temp_file):
        """Test getting file info with content preview."""
        result = await tool.safe_execute(
            file_path=temp_file,
            include_preview=True
        )

        assert result["success"] is True
        info = result["result"]

        # Should include content preview
        assert "preview" in info
        assert len(info["preview"]) > 0
        # Check if Alice is in the preview data structure
        preview_str = str(info["preview"])
        assert "Alice" in preview_str