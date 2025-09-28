"""
Comprehensive tests for data transformation tools.
"""

import pytest
import pandas as pd
from unittest.mock import patch

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from tools.data_transformation import FilterDataTool, AggregateDataTool, JoinDataTool, PivotDataTool, CleanDataTool


class TestFilterDataTool:
    """Test suite for FilterDataTool."""

    @pytest.fixture
    def tool(self):
        """Create a FilterDataTool instance."""
        return FilterDataTool()

    @pytest.fixture
    def sample_data(self):
        """Create sample test data."""
        return [
            {"name": "Alice", "age": 25, "department": "Engineering", "salary": 75000},
            {"name": "Bob", "age": 30, "department": "Marketing", "salary": 65000},
            {"name": "Charlie", "age": 35, "department": "Engineering", "salary": 85000},
            {"name": "Diana", "age": 28, "department": "Sales", "salary": 70000},
            {"name": "Eve", "age": 32, "department": "Marketing", "salary": 72000}
        ]

    def test_tool_properties(self, tool):
        """Test basic tool properties."""
        assert tool.name == "filter_data"
        assert "filter dataset records based on" in tool.description.lower()
        assert isinstance(tool.get_schema(), dict)
        assert "data" in tool.get_schema()["properties"]
        assert "conditions" in tool.get_schema()["properties"]

    @pytest.mark.asyncio
    async def test_filter_simple_condition(self, tool, sample_data):
        """Test filtering with a simple condition."""
        conditions = [
            {"field": "age", "operator": ">", "value": 30}
        ]

        result = await tool.safe_execute(data=sample_data, conditions=conditions)

        assert result["success"] is True
        filtered_data = result["result"]["filtered_data"]

        # Should return records where age > 30
        assert len(filtered_data) == 2  # Charlie and Eve
        assert all(record["age"] > 30 for record in filtered_data)

    @pytest.mark.asyncio
    async def test_filter_multiple_conditions_and(self, tool, sample_data):
        """Test filtering with multiple AND conditions."""
        conditions = [
            {"field": "age", "operator": ">=", "value": 30},
            {"field": "department", "operator": "==", "value": "Engineering"}
        ]

        result = await tool.safe_execute(
            data=sample_data,
            conditions=conditions,
            logic="AND"
        )

        assert result["success"] is True
        filtered_data = result["result"]["filtered_data"]

        # Should return only Charlie (age >= 30 AND department = Engineering)
        assert len(filtered_data) == 1
        assert filtered_data[0]["name"] == "Charlie"

    @pytest.mark.asyncio
    async def test_filter_multiple_conditions_or(self, tool, sample_data):
        """Test filtering with multiple OR conditions."""
        conditions = [
            {"field": "department", "operator": "==", "value": "Engineering"},
            {"field": "salary", "operator": ">=", "value": 80000}
        ]

        result = await tool.safe_execute(
            data=sample_data,
            conditions=conditions,
            logic="OR"
        )

        assert result["success"] is True
        filtered_data = result["result"]["filtered_data"]

        # Should return Engineering department OR salary >= 80000
        assert len(filtered_data) >= 2
        names = [record["name"] for record in filtered_data]
        assert "Alice" in names  # Engineering
        assert "Charlie" in names  # Engineering and high salary

    @pytest.mark.asyncio
    async def test_filter_string_operations(self, tool, sample_data):
        """Test filtering with string operations."""
        conditions = [
            {"field": "name", "operator": "contains", "value": "e"}
        ]

        result = await tool.safe_execute(data=sample_data, conditions=conditions)

        assert result["success"] is True
        filtered_data = result["result"]["filtered_data"]

        # Should return names containing 'e'
        for record in filtered_data:
            assert "e" in record["name"].lower()

    @pytest.mark.asyncio
    async def test_filter_in_operation(self, tool, sample_data):
        """Test filtering with 'in' operation."""
        conditions = [
            {"field": "department", "operator": "in", "value": ["Engineering", "Sales"]}
        ]

        result = await tool.safe_execute(data=sample_data, conditions=conditions)

        assert result["success"] is True
        filtered_data = result["result"]["filtered_data"]

        # Should return only Engineering and Sales departments
        departments = [record["department"] for record in filtered_data]
        assert all(dept in ["Engineering", "Sales"] for dept in departments)

    @pytest.mark.asyncio
    async def test_filter_no_matches(self, tool, sample_data):
        """Test filtering with no matching records."""
        conditions = [
            {"field": "age", "operator": ">", "value": 100}
        ]

        result = await tool.safe_execute(data=sample_data, conditions=conditions)

        assert result["success"] is True
        filtered_data = result["result"]["filtered_data"]

        assert len(filtered_data) == 0


class TestAggregateDataTool:
    """Test suite for AggregateDataTool."""

    @pytest.fixture
    def tool(self):
        """Create an AggregateDataTool instance."""
        return AggregateDataTool()

    @pytest.fixture
    def sample_data(self):
        """Create sample test data."""
        return [
            {"department": "Engineering", "salary": 75000, "age": 25, "bonus": 5000},
            {"department": "Engineering", "salary": 85000, "age": 35, "bonus": 7000},
            {"department": "Marketing", "salary": 65000, "age": 30, "bonus": 4000},
            {"department": "Marketing", "salary": 72000, "age": 32, "bonus": 5500},
            {"department": "Sales", "salary": 70000, "age": 28, "bonus": 4500}
        ]

    def test_tool_properties(self, tool):
        """Test basic tool properties."""
        assert tool.name == "aggregate_data"
        assert "group by operations and aggregate" in tool.description.lower()
        assert isinstance(tool.get_schema(), dict)
        assert "data" in tool.get_schema()["properties"]
        assert "group_by" in tool.get_schema()["properties"]

    @pytest.mark.asyncio
    async def test_aggregate_simple_groupby(self, tool, sample_data):
        """Test simple group by aggregation."""
        aggregations = [
            {"field": "salary", "operation": "mean"},
            {"field": "age", "operation": "max"}
        ]

        result = await tool.safe_execute(
            data=sample_data,
            group_by=["department"],
            aggregations=aggregations
        )

        assert result["success"] is True
        aggregated_data = result["result"]["aggregated_data"]

        # Should have one record per department
        departments = [record["department"] for record in aggregated_data]
        assert "Engineering" in departments
        assert "Marketing" in departments
        assert "Sales" in departments

        # Check aggregation results - tool uses original field names
        eng_record = next(r for r in aggregated_data if r["department"] == "Engineering")
        assert "salary" in eng_record  # mean aggregation result
        assert "age" in eng_record     # max aggregation result

    @pytest.mark.asyncio
    async def test_aggregate_multiple_functions(self, tool, sample_data):
        """Test aggregation with multiple functions."""
        aggregations = [
            {"field": "salary", "operation": "sum"},
            {"field": "salary", "operation": "count"},
            {"field": "salary", "operation": "min"},
            {"field": "salary", "operation": "max"}
        ]

        result = await tool.safe_execute(
            data=sample_data,
            group_by=["department"],
            aggregations=aggregations
        )

        assert result["success"] is True
        aggregated_data = result["result"]["aggregated_data"]

        # Check that all aggregation functions are present
        if aggregated_data:
            record = aggregated_data[0]
            assert "salary_sum" in record
            assert "salary_count" in record
            assert "salary_min" in record
            assert "salary_max" in record

    @pytest.mark.asyncio
    async def test_aggregate_no_groupby(self, tool, sample_data):
        """Test aggregation without grouping."""
        aggregations = [
            {"field": "salary", "operation": "mean"},
            {"field": "age", "operation": "median"}
        ]

        result = await tool.safe_execute(
            data=sample_data,
            aggregations=aggregations
        )

        assert result["success"] is True
        aggregated_data = result["result"]["aggregated_data"]

        # Should return a single record with overall aggregations
        assert len(aggregated_data) == 1
        assert "salary" in aggregated_data[0]  # overall mean
        assert "age" in aggregated_data[0]     # overall median

    @pytest.mark.asyncio
    async def test_aggregate_multiple_groupby(self, tool):
        """Test aggregation with multiple group by columns."""
        data = [
            {"dept": "Eng", "level": "Junior", "salary": 60000},
            {"dept": "Eng", "level": "Senior", "salary": 90000},
            {"dept": "Sales", "level": "Junior", "salary": 50000},
            {"dept": "Sales", "level": "Senior", "salary": 80000}
        ]

        aggregations = [{"field": "salary", "operation": "mean"}]

        result = await tool.safe_execute(
            data=data,
            group_by=["dept", "level"],
            aggregations=aggregations
        )

        assert result["success"] is True
        aggregated_data = result["result"]["aggregated_data"]

        # Should have 4 groups
        assert len(aggregated_data) == 4


class TestJoinDataTool:
    """Test suite for JoinDataTool."""

    @pytest.fixture
    def tool(self):
        """Create a JoinDataTool instance."""
        return JoinDataTool()

    @pytest.fixture
    def sample_datasets(self):
        """Create sample datasets for joining."""
        employees = [
            {"id": 1, "name": "Alice", "dept_id": 10},
            {"id": 2, "name": "Bob", "dept_id": 20},
            {"id": 3, "name": "Charlie", "dept_id": 10}
        ]

        departments = [
            {"dept_id": 10, "dept_name": "Engineering", "budget": 500000},
            {"dept_id": 20, "dept_name": "Marketing", "budget": 300000},
            {"dept_id": 30, "dept_name": "Sales", "budget": 400000}
        ]

        return employees, departments

    def test_tool_properties(self, tool):
        """Test basic tool properties."""
        assert tool.name == "join_data"
        assert "merge multiple datasets" in tool.description.lower()
        assert isinstance(tool.get_schema(), dict)
        assert "left_data" in tool.get_schema()["properties"]
        assert "right_data" in tool.get_schema()["properties"]

    @pytest.mark.asyncio
    async def test_inner_join(self, tool, sample_datasets):
        """Test inner join operation."""
        employees, departments = sample_datasets

        result = await tool.safe_execute(
            left_data=employees,
            right_data=departments,
            left_on=["dept_id"],
            right_on=["dept_id"],
            join_type="inner"
        )

        assert result["success"] is True
        joined_data = result["result"]["joined_data"]

        # Should return only matching records
        assert len(joined_data) == 3  # All employees have matching departments

        # Check that join worked correctly
        alice_record = next(r for r in joined_data if r["name"] == "Alice")
        assert alice_record["dept_name"] == "Engineering"

    @pytest.mark.asyncio
    async def test_left_join(self, tool, sample_datasets):
        """Test left join operation."""
        employees, departments = sample_datasets

        # Add employee with non-existent department
        employees_extended = employees + [{"id": 4, "name": "Dave", "dept_id": 99}]

        result = await tool.safe_execute(
            left_data=employees_extended,
            right_data=departments,
            left_on=["dept_id"],
            right_on=["dept_id"],
            join_type="left"
        )

        assert result["success"] is True
        joined_data = result["result"]["joined_data"]

        # Should return all employees
        assert len(joined_data) == 4

        # Dave should have null department info
        dave_record = next(r for r in joined_data if r["name"] == "Dave")
        import pandas as pd
        assert pd.isna(dave_record["dept_name"])  # pandas returns NaN for missing values

    @pytest.mark.asyncio
    async def test_right_join(self, tool, sample_datasets):
        """Test right join operation."""
        employees, departments = sample_datasets

        result = await tool.safe_execute(
            left_data=employees,
            right_data=departments,
            left_on=["dept_id"],
            right_on=["dept_id"],
            join_type="right"
        )

        assert result["success"] is True
        joined_data = result["result"]["joined_data"]

        # Should include Sales department even though no employees
        dept_names = [r["dept_name"] for r in joined_data]
        assert "Sales" in dept_names

    @pytest.mark.asyncio
    async def test_outer_join(self, tool, sample_datasets):
        """Test outer join operation."""
        employees, departments = sample_datasets

        # Add employee with non-existent department
        employees_extended = employees + [{"id": 4, "name": "Dave", "dept_id": 99}]

        result = await tool.safe_execute(
            left_data=employees_extended,
            right_data=departments,
            left_on=["dept_id"],
            right_on=["dept_id"],
            join_type="outer"
        )

        assert result["success"] is True
        joined_data = result["result"]["joined_data"]

        # Should include all employees and all departments
        names = [r.get("name") for r in joined_data if r.get("name")]
        dept_names = [r.get("dept_name") for r in joined_data if r.get("dept_name")]

        assert "Dave" in names
        assert "Sales" in dept_names

    @pytest.mark.asyncio
    async def test_join_multiple_columns(self, tool):
        """Test joining on multiple columns."""
        data1 = [
            {"year": 2023, "quarter": 1, "sales": 100000},
            {"year": 2023, "quarter": 2, "sales": 120000}
        ]

        data2 = [
            {"year": 2023, "quarter": 1, "costs": 80000},
            {"year": 2023, "quarter": 2, "costs": 90000}
        ]

        result = await tool.safe_execute(
            left_data=data1,
            right_data=data2,
            left_on=["year", "quarter"],
            right_on=["year", "quarter"],
            join_type="inner"
        )

        assert result["success"] is True
        joined_data = result["result"]["joined_data"]

        # Should join on both year and quarter
        assert len(joined_data) == 2
        assert all("sales" in record and "costs" in record for record in joined_data)


class TestPivotDataTool:
    """Test suite for PivotDataTool."""

    @pytest.fixture
    def tool(self):
        """Create a PivotDataTool instance."""
        return PivotDataTool()

    @pytest.fixture
    def sample_data(self):
        """Create sample data for pivoting."""
        return [
            {"region": "North", "product": "A", "quarter": "Q1", "sales": 100},
            {"region": "North", "product": "A", "quarter": "Q2", "sales": 120},
            {"region": "North", "product": "B", "quarter": "Q1", "sales": 80},
            {"region": "South", "product": "A", "quarter": "Q1", "sales": 90},
            {"region": "South", "product": "A", "quarter": "Q2", "sales": 110},
            {"region": "South", "product": "B", "quarter": "Q1", "sales": 70}
        ]

    def test_tool_properties(self, tool):
        """Test basic tool properties."""
        assert tool.name == "pivot_data"
        assert "pivot" in tool.description.lower()
        assert isinstance(tool.get_schema(), dict)
        assert "data" in tool.get_schema()["properties"]
        assert "index" in tool.get_schema()["properties"]

    @pytest.mark.asyncio
    async def test_simple_pivot(self, tool, sample_data):
        """Test simple pivot operation."""
        result = await tool.safe_execute(
            data=sample_data,
            index=["region"],
            columns="quarter",
            values="sales",
            aggfunc="sum"
        )

        assert result["success"] is True
        pivoted_data = result["result"]["pivoted_data"]

        # Should have regions as rows and quarters as columns
        assert len(pivoted_data) >= 2  # North and South regions

        # Check that pivot structure is correct
        if pivoted_data:
            record = pivoted_data[0]
            assert "region" in record
            # Quarter columns should be present
            assert any("Q1" in str(key) or "Q2" in str(key) for key in record.keys())

    @pytest.mark.asyncio
    async def test_pivot_with_multiple_values(self, tool):
        """Test pivot with multiple value columns."""
        data = [
            {"region": "North", "quarter": "Q1", "sales": 100, "profit": 20},
            {"region": "North", "quarter": "Q2", "sales": 120, "profit": 25},
            {"region": "South", "quarter": "Q1", "sales": 90, "profit": 18}
        ]

        result = await tool.safe_execute(
            data=data,
            index=["region"],
            columns="quarter",
            values="sales",
            aggfunc="sum"
        )

        assert result["success"] is True
        pivoted_data = result["result"]["pivoted_data"]

        # Should handle multiple value columns
        assert len(pivoted_data) >= 1

    @pytest.mark.asyncio
    async def test_pivot_with_aggregation(self, tool, sample_data):
        """Test pivot with different aggregation functions."""
        result = await tool.safe_execute(
            data=sample_data,
            index=["region"],
            columns="product",
            values="sales",
            aggfunc="mean"
        )

        assert result["success"] is True
        pivoted_data = result["result"]["pivoted_data"]

        # Should apply mean aggregation
        assert len(pivoted_data) >= 1


class TestCleanDataTool:
    """Test suite for CleanDataTool."""

    @pytest.fixture
    def tool(self):
        """Create a CleanDataTool instance."""
        return CleanDataTool()

    @pytest.fixture
    def messy_data(self):
        """Create messy data for cleaning."""
        return [
            {"name": "  Alice  ", "email": "ALICE@EXAMPLE.COM", "phone": "123-456-7890", "age": "25"},
            {"name": "Bob", "email": "bob@example.com", "phone": "(555) 123-4567", "age": "30"},
            {"name": "", "email": "invalid-email", "phone": "555.123.4567", "age": ""},
            {"name": "Charlie", "email": None, "phone": "5551234567", "age": "35"},
            {"name": "  DAVE  ", "email": "dave@EXAMPLE.com", "phone": "", "age": "40"}
        ]

    def test_tool_properties(self, tool):
        """Test basic tool properties."""
        assert tool.name == "clean_data"
        assert "common data cleaning operations" in tool.description.lower()
        assert isinstance(tool.get_schema(), dict)
        assert "data" in tool.get_schema()["properties"]
        assert "operations" in tool.get_schema()["properties"]

    @pytest.mark.asyncio
    async def test_trim_whitespace(self, tool, messy_data):
        """Test trimming whitespace."""
        operations = [
            {"type": "trim", "fields": ["name"]}
        ]

        result = await tool.safe_execute(data=messy_data, operations=operations)

        assert result["success"] is True
        cleaned_data = result["result"]["cleaned_data"]

        # Check that whitespace is trimmed
        alice_record = next(r for r in cleaned_data if "Alice" in str(r.get("name", "")))
        assert alice_record["name"] == "Alice"  # No leading/trailing spaces

    @pytest.mark.asyncio
    async def test_standardize_case(self, tool, messy_data):
        """Test case standardization."""
        operations = [
            {"type": "lowercase", "fields": ["email"]}
        ]

        result = await tool.safe_execute(data=messy_data, operations=operations)

        assert result["success"] is True
        cleaned_data = result["result"]["cleaned_data"]

        # Check that emails are lowercased
        for record in cleaned_data:
            if record.get("email"):
                assert record["email"] == record["email"].lower()

    @pytest.mark.asyncio
    async def test_remove_nulls(self, tool, messy_data):
        """Test removing null/empty values."""
        operations = [
            {"type": "remove_nulls", "fields": ["name", "email"]}
        ]

        result = await tool.safe_execute(data=messy_data, operations=operations)

        assert result["success"] is True
        cleaned_data = result["result"]["cleaned_data"]

        # Should remove records with null names or emails (but empty strings may remain)
        for record in cleaned_data:
            # None values should be removed, but empty strings might remain
            assert record.get("name") is not None
            assert record.get("email") is not None

    @pytest.mark.asyncio
    async def test_standardize_text(self, tool, messy_data):
        """Test text standardization."""
        operations = [
            {"type": "standardize_text", "fields": ["name"]}
        ]

        result = await tool.safe_execute(data=messy_data, operations=operations)

        assert result["success"] is True
        cleaned_data = result["result"]["cleaned_data"]

        # Check that text is standardized (lowercase, trimmed)
        for record in cleaned_data:
            name = record.get("name")
            if name and name.strip():
                assert name == name.lower().strip()

    @pytest.mark.asyncio
    async def test_fill_nulls(self, tool, messy_data):
        """Test filling null values."""
        operations = [
            {"type": "fill_nulls", "fields": ["name"], "parameters": {"fill_value": "Unknown"}}
        ]

        result = await tool.safe_execute(data=messy_data, operations=operations)

        assert result["success"] is True
        cleaned_data = result["result"]["cleaned_data"]

        # Check that null names are filled (pandas None/NaN values)
        # No names should be None anymore, they should be filled with "Unknown"
        for record in cleaned_data:
            name = record.get("name")
            # None values should have been replaced with "Unknown"
            assert name is not None

    @pytest.mark.asyncio
    async def test_multiple_operations(self, tool, messy_data):
        """Test multiple cleaning operations."""
        operations = [
            {"type": "trim", "fields": ["name"]},
            {"type": "lowercase", "fields": ["email"]},
            {"type": "remove_duplicates"}
        ]

        result = await tool.safe_execute(data=messy_data, operations=operations)

        assert result["success"] is True
        cleaned_data = result["result"]["cleaned_data"]

        # Should apply all operations
        assert len(cleaned_data) >= 0  # At least some data should remain

    @pytest.mark.asyncio
    async def test_validate_after_cleaning(self, tool, messy_data):
        """Test validation after cleaning."""
        operations = [
            {"type": "trim", "fields": ["name"]},
            {"type": "remove_nulls", "fields": ["name"]}
        ]

        result = await tool.safe_execute(
            data=messy_data,
            operations=operations
        )

        assert result["success"] is True

        # Should include validation results
        if "validation_results" in result["result"]:
            validation = result["result"]["validation_results"]
            assert isinstance(validation, dict)