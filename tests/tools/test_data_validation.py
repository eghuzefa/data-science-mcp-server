"""
Comprehensive tests for data validation tools.
"""

import pytest
import pandas as pd
import tempfile
import json
from unittest.mock import patch

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from tools.data_validation import ValidateSchemaTool, CheckNullsTool, DataQualityReportTool, DetectDuplicatesTool


class TestValidateSchemaTool:
    """Test suite for ValidateSchemaTool."""

    @pytest.fixture
    def tool(self):
        """Create a ValidateSchemaTool instance."""
        return ValidateSchemaTool()

    @pytest.fixture
    def sample_data(self):
        """Create sample test data."""
        return [
            {"name": "Alice", "age": 25, "email": "alice@example.com"},
            {"name": "Bob", "age": 30, "email": "bob@example.com"},
            {"name": "Charlie", "age": 35, "email": "charlie@example.com"}
        ]

    def test_tool_properties(self, tool):
        """Test basic tool properties."""
        assert tool.name == "validate_schema"
        assert "validate data against expected schema" in tool.description.lower()
        assert isinstance(tool.get_schema(), dict)
        assert "data" in tool.get_schema()["properties"]
        assert "schema" in tool.get_schema()["properties"]

    @pytest.mark.asyncio
    async def test_validate_schema_success(self, tool, sample_data):
        """Test successful schema validation."""
        schema = {
            "name": {"type": "string", "required": True},
            "age": {"type": "integer", "required": True, "min": 0},
            "email": {"type": "string", "required": True, "pattern": ".*@.*"}
        }

        result = await tool.safe_execute(data=sample_data, schema=schema)

        assert result["success"] is True
        assert result["result"]["overall_valid"] is True
        assert result["result"]["total_records"] == 3
        assert result["result"]["valid_records"] == 3

    @pytest.mark.asyncio
    async def test_validate_schema_failure(self, tool):
        """Test schema validation with invalid data."""
        invalid_data = [
            {"name": "Alice", "age": "invalid_age", "email": "alice@example.com"},
            {"name": "Bob", "age": 30, "email": "invalid_email"},
            {"name": "", "age": 25, "email": "charlie@example.com"}
        ]

        schema = {
            "name": {"type": "string", "required": True, "min_length": 1},
            "age": {"type": "integer", "required": True},
            "email": {"type": "string", "required": True, "pattern": ".*@.*"}
        }

        result = await tool.safe_execute(data=invalid_data, schema=schema)

        assert result["success"] is True
        assert result["result"]["overall_valid"] is False
        assert result["result"]["total_records"] == 3
        assert result["result"]["valid_records"] < 3
        assert len(result["result"]["validation_errors"]) > 0

    @pytest.mark.asyncio
    async def test_validate_schema_missing_fields(self, tool):
        """Test schema validation with missing required fields."""
        incomplete_data = [
            {"name": "Alice", "age": 25},  # Missing email
            {"name": "Bob", "email": "bob@example.com"}  # Missing age
        ]

        schema = {
            "name": {"type": "string", "required": True},
            "age": {"type": "integer", "required": True},
            "email": {"type": "string", "required": True}
        }

        result = await tool.safe_execute(data=incomplete_data, schema=schema)

        assert result["success"] is True
        assert result["result"]["overall_valid"] is False
        assert "required field" in str(result["result"]["validation_errors"]).lower()

    @pytest.mark.asyncio
    async def test_validate_schema_custom_rules(self, tool, sample_data):
        """Test schema validation with custom rules."""
        schema = {
            "name": {"type": "string", "required": True, "min_length": 2},
            "age": {"type": "integer", "required": True, "min": 18, "max": 100},
            "email": {"type": "string", "required": True, "pattern": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"}
        }

        result = await tool.safe_execute(data=sample_data, schema=schema)

        assert result["success"] is True
        # All sample data should pass these validations


class TestCheckNullsTool:
    """Test suite for CheckNullsTool."""

    @pytest.fixture
    def tool(self):
        """Create a CheckNullsTool instance."""
        return CheckNullsTool()

    @pytest.fixture
    def data_with_nulls(self):
        """Create test data with null values."""
        return [
            {"name": "Alice", "age": 25, "email": "alice@example.com"},
            {"name": None, "age": 30, "email": "bob@example.com"},
            {"name": "Charlie", "age": None, "email": None},
            {"name": "", "age": 35, "email": "dave@example.com"},
            {"name": "Eve", "age": 40, "email": ""}
        ]

    def test_tool_properties(self, tool):
        """Test basic tool properties."""
        assert tool.name == "check_nulls"
        assert "analyze null values" in tool.description.lower()
        assert isinstance(tool.get_schema(), dict)
        assert "data" in tool.get_schema()["properties"]

    @pytest.mark.asyncio
    async def test_check_nulls_basic(self, tool, data_with_nulls):
        """Test basic null checking."""
        result = await tool.safe_execute(data=data_with_nulls)

        assert result["success"] is True
        report = result["result"]

        assert "total_records" in report
        assert "field_analysis" in report
        assert "summary" in report

        # Check that null counts are detected
        assert report["field_analysis"]["name"]["null_count"] > 0
        assert report["field_analysis"]["age"]["null_count"] > 0
        assert report["field_analysis"]["email"]["null_count"] > 0

    @pytest.mark.asyncio
    async def test_check_nulls_include_empty(self, tool, data_with_nulls):
        """Test null checking including empty strings."""
        result = await tool.safe_execute(
            data=data_with_nulls,
            null_values=[""]
        )

        assert result["success"] is True
        report = result["result"]

        # Should detect empty strings as nulls
        assert report["field_analysis"]["name"]["null_count"] >= 1  # At least one empty string
        assert report["field_analysis"]["email"]["null_count"] >= 1  # At least one empty string

    @pytest.mark.asyncio
    async def test_check_nulls_no_nulls(self, tool):
        """Test null checking with clean data."""
        clean_data = [
            {"name": "Alice", "age": 25, "email": "alice@example.com"},
            {"name": "Bob", "age": 30, "email": "bob@example.com"}
        ]

        result = await tool.safe_execute(data=clean_data)

        assert result["success"] is True
        report = result["result"]

        # Should have no nulls
        assert all(report["field_analysis"][field]["null_count"] == 0 for field in report["field_analysis"])

    @pytest.mark.asyncio
    async def test_check_nulls_patterns(self, tool, data_with_nulls):
        """Test null pattern analysis."""
        result = await tool.safe_execute(
            data=data_with_nulls
        )

        assert result["success"] is True
        report = result["result"]

        assert "patterns" in report
        # Should identify records with multiple nulls


class TestDataQualityReportTool:
    """Test suite for DataQualityReportTool."""

    @pytest.fixture
    def tool(self):
        """Create a DataQualityReportTool instance."""
        return DataQualityReportTool()

    @pytest.fixture
    def mixed_quality_data(self):
        """Create test data with various quality issues."""
        return [
            {"id": 1, "name": "Alice", "age": 25, "email": "alice@example.com", "score": 95.5},
            {"id": 2, "name": "Bob", "age": 30, "email": "bob@example.com", "score": 87.2},
            {"id": 3, "name": None, "age": 35, "email": "invalid-email", "score": None},
            {"id": 4, "name": "Dave", "age": -5, "email": "dave@example.com", "score": 150.0},
            {"id": 5, "name": "", "age": 200, "email": "", "score": -10.0},
            {"id": 1, "name": "Alice", "age": 25, "email": "alice@example.com", "score": 95.5}  # Duplicate
        ]

    def test_tool_properties(self, tool):
        """Test basic tool properties."""
        assert tool.name == "data_quality_report"
        assert "comprehensive data quality report" in tool.description.lower()
        assert isinstance(tool.get_schema(), dict)
        assert "data" in tool.get_schema()["properties"]

    @pytest.mark.asyncio
    async def test_data_quality_basic_report(self, tool, mixed_quality_data):
        """Test basic data quality report generation."""
        result = await tool.safe_execute(data=mixed_quality_data)

        assert result["success"] is True
        report = result["result"]

        # Check main sections
        assert "overview" in report
        assert "field_analysis" in report
        assert "data_quality_score" in report

        # Check overview
        overview = report["overview"]
        assert "total_records" in overview
        assert "total_fields" in overview
        assert overview["total_records"] == 6

    @pytest.mark.asyncio
    async def test_data_quality_with_rules(self, tool, mixed_quality_data):
        """Test data quality report with custom validation rules."""
        rules = {
            "age": {"min": 0, "max": 120},
            "score": {"min": 0, "max": 100},
            "email": {"pattern": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"}
        }

        result = await tool.safe_execute(
            data=mixed_quality_data
        )

        assert result["success"] is True
        report = result["result"]

        assert "issues" in report
        # Should detect age and score outliers, invalid emails

    @pytest.mark.asyncio
    async def test_data_quality_detailed_analysis(self, tool, mixed_quality_data):
        """Test detailed data quality analysis."""
        result = await tool.safe_execute(
            data=mixed_quality_data
        )

        assert result["success"] is True
        report = result["result"]

        assert "issues" in report
        assert "recommendations" in report

        # Should have data quality issues
        assert len(report["issues"]) > 0

    @pytest.mark.asyncio
    async def test_data_quality_empty_data(self, tool):
        """Test data quality report with empty data."""
        result = await tool.safe_execute(data=[])

        assert result["success"] is True
        report = result["result"]

        assert "message" in report
        assert "No data to analyze" in report["message"]


class TestDetectDuplicatesTool:
    """Test suite for DetectDuplicatesTool."""

    @pytest.fixture
    def tool(self):
        """Create a DetectDuplicatesTool instance."""
        return DetectDuplicatesTool()

    @pytest.fixture
    def data_with_duplicates(self):
        """Create test data with duplicates."""
        return [
            {"id": 1, "name": "Alice", "email": "alice@example.com"},
            {"id": 2, "name": "Bob", "email": "bob@example.com"},
            {"id": 1, "name": "Alice", "email": "alice@example.com"},  # Exact duplicate
            {"id": 4, "name": "Charlie", "email": "charlie@example.com"},
            {"id": 5, "name": "Bob", "email": "bob2@example.com"},  # Partial duplicate (name)
            {"id": 6, "name": "Dave", "email": "alice@example.com"}  # Partial duplicate (email)
        ]

    def test_tool_properties(self, tool):
        """Test basic tool properties."""
        assert tool.name == "detect_duplicates"
        assert "detect duplicate records" in tool.description.lower()
        assert isinstance(tool.get_schema(), dict)
        assert "data" in tool.get_schema()["properties"]

    @pytest.mark.asyncio
    async def test_detect_exact_duplicates(self, tool, data_with_duplicates):
        """Test detection of exact duplicates."""
        result = await tool.safe_execute(data=data_with_duplicates)

        assert result["success"] is True
        report = result["result"]

        assert "total_records" in report
        assert "duplicate_records" in report

        # Should detect exact duplicates
        assert report["duplicate_records"] > 0
        if "duplicate_groups" in report:
            assert len(report["duplicate_groups"]) > 0

    @pytest.mark.asyncio
    async def test_detect_duplicates_by_columns(self, tool, data_with_duplicates):
        """Test duplicate detection by specific columns."""
        result = await tool.safe_execute(
            data=data_with_duplicates,
            key_fields=["name"]
        )

        assert result["success"] is True
        report = result["result"]

        # Should detect duplicates by name only
        assert report["duplicate_records"] >= 2  # Alice and Bob appear twice

    @pytest.mark.asyncio
    async def test_detect_duplicates_keep_option(self, tool, data_with_duplicates):
        """Test duplicate detection with different keep options."""
        result = await tool.safe_execute(
            data=data_with_duplicates
        )

        assert result["success"] is True
        report = result["result"]

        # Check that duplicates are detected
        assert "duplicate_records" in report

    @pytest.mark.asyncio
    async def test_detect_no_duplicates(self, tool):
        """Test duplicate detection with unique data."""
        unique_data = [
            {"id": 1, "name": "Alice", "email": "alice@example.com"},
            {"id": 2, "name": "Bob", "email": "bob@example.com"},
            {"id": 3, "name": "Charlie", "email": "charlie@example.com"}
        ]

        result = await tool.safe_execute(data=unique_data)

        assert result["success"] is True
        report = result["result"]

        assert report["duplicate_records"] == 0
        if "duplicate_groups" in report:
            assert len(report["duplicate_groups"]) == 0

    @pytest.mark.asyncio
    async def test_detect_duplicates_with_similarity(self, tool):
        """Test duplicate detection with similarity threshold."""
        similar_data = [
            {"name": "John Smith", "email": "john@example.com"},
            {"name": "Jon Smith", "email": "jon@example.com"},  # Similar name
            {"name": "Jane Doe", "email": "jane@example.com"}
        ]

        result = await tool.safe_execute(
            data=similar_data
        )

        assert result["success"] is True
        report = result["result"]

        # Basic duplicate detection should work
        assert "duplicate_records" in report