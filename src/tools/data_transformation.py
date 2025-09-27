"""
Data transformation tools for filtering, aggregation, and manipulation.
"""

from typing import Any, Dict, List
import pandas as pd
import re

from .base import BaseTool


class FilterDataTool(BaseTool):
    """Tool for filtering datasets based on conditions."""

    @property
    def name(self) -> str:
        return "filter_data"

    @property
    def description(self) -> str:
        return "Filter dataset records based on specified conditions and criteria"

    def get_schema(self) -> Dict:
        return {
            "properties": {
                "data": {
                    "type": "array",
                    "description": "Data to filter (list of dictionaries)"
                },
                "conditions": {
                    "type": "array",
                    "description": "List of filter conditions",
                    "items": {
                        "type": "object",
                        "properties": {
                            "field": {
                                "type": "string",
                                "description": "Field name to filter on"
                            },
                            "operator": {
                                "type": "string",
                                "enum": ["==", "!=", ">", ">=", "<", "<=", "in", "not_in", "contains", "starts_with", "ends_with",
                                        "is_null", "is_not_null", "regex", "not_regex", "between", "not_between",
                                        "length_eq", "length_gt", "length_lt"],
                                "description": "Comparison operator"
                            },
                            "value": {
                                "description": "Value to compare against (not needed for is_null/is_not_null)"
                            }
                        },
                        "required": ["field", "operator"]
                    }
                },
                "logic": {
                    "type": "string",
                    "enum": ["AND", "OR"],
                    "description": "Logic to combine multiple conditions",
                    "default": "AND"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of records to return",
                    "minimum": 1
                },
                "sort_by": {
                    "type": "string",
                    "description": "Field to sort results by"
                },
                "sort_order": {
                    "type": "string",
                    "enum": ["asc", "desc"],
                    "description": "Sort order",
                    "default": "asc"
                },
                "validate_output": {
                    "type": "boolean",
                    "description": "Whether to validate output data quality",
                    "default": False
                }
            },
            "required": ["data", "conditions"]
        }

    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute data filtering operation."""
        data = kwargs.get("data")
        conditions = kwargs.get("conditions")
        logic = kwargs.get("logic", "AND")
        limit = kwargs.get("limit")
        sort_by = kwargs.get("sort_by")
        sort_order = kwargs.get("sort_order", "asc")
        validate_output = kwargs.get("validate_output", False)

        if not isinstance(data, list):
            raise ValueError("Data must be a list of dictionaries")

        if not data:
            return {
                "filtered_data": [],
                "original_count": 0,
                "filtered_count": 0,
                "conditions_applied": conditions,
                "logic": logic
            }

        if not conditions:
            raise ValueError("At least one condition must be provided")

        # Convert to DataFrame for easier filtering
        df = pd.DataFrame(data)

        # Apply filters
        try:
            filtered_df = self._apply_filters(df, conditions, logic)

            # Sort if requested
            if sort_by and sort_by in filtered_df.columns:
                ascending = sort_order == "asc"
                filtered_df = filtered_df.sort_values(by=sort_by, ascending=ascending)

            # Apply limit if specified
            if limit:
                filtered_df = filtered_df.head(limit)

            # Convert back to list of dictionaries
            filtered_data = filtered_df.to_dict(orient="records")

            result = {
                "filtered_data": filtered_data,
                "original_count": len(data),
                "filtered_count": len(filtered_data),
                "conditions_applied": conditions,
                "logic": logic,
                "filters_summary": self._generate_filter_summary(conditions, logic, len(data), len(filtered_data))
            }

            # Add validation if requested
            if validate_output:
                validation_result = self._validate_data_quality(filtered_data)
                result["validation"] = validation_result

            return result

        except Exception as e:
            raise RuntimeError(f"Failed to filter data: {str(e)}")

    def _apply_filters(self, df: pd.DataFrame, conditions: List[Dict], logic: str) -> pd.DataFrame:
        """Apply filter conditions to the DataFrame."""
        if not conditions:
            return df

        # Build individual condition masks
        masks = []
        for condition in conditions:
            mask = self._build_condition_mask(df, condition)
            masks.append(mask)

        # Combine masks based on logic
        if logic == "AND":
            combined_mask = masks[0]
            for mask in masks[1:]:
                combined_mask = combined_mask & mask
        else:  # OR
            combined_mask = masks[0]
            for mask in masks[1:]:
                combined_mask = combined_mask | mask

        return df[combined_mask]

    def _build_condition_mask(self, df: pd.DataFrame, condition: Dict) -> pd.Series:
        """Build a boolean mask for a single condition."""
        field = condition["field"]
        operator = condition["operator"]
        value = condition.get("value")

        if field not in df.columns:
            raise ValueError(f"Field '{field}' not found in data")

        series = df[field]

        if operator == "==":
            return series == value
        elif operator == "!=":
            return series != value
        elif operator == ">":
            return series > value
        elif operator == ">=":
            return series >= value
        elif operator == "<":
            return series < value
        elif operator == "<=":
            return series <= value
        elif operator == "in":
            if not isinstance(value, list):
                raise ValueError("Value for 'in' operator must be a list")
            return series.isin(value)
        elif operator == "not_in":
            if not isinstance(value, list):
                raise ValueError("Value for 'not_in' operator must be a list")
            return ~series.isin(value)
        elif operator == "contains":
            return series.astype(str).str.contains(str(value), na=False)
        elif operator == "starts_with":
            return series.astype(str).str.startswith(str(value), na=False)
        elif operator == "ends_with":
            return series.astype(str).str.endswith(str(value), na=False)
        elif operator == "is_null":
            return series.isnull()
        elif operator == "is_not_null":
            return series.notnull()
        elif operator == "regex":
            return series.astype(str).str.contains(str(value), regex=True, na=False)
        elif operator == "not_regex":
            return ~series.astype(str).str.contains(str(value), regex=True, na=False)
        elif operator == "between":
            if not isinstance(value, list) or len(value) != 2:
                raise ValueError("Value for 'between' operator must be a list of two values [min, max]")
            return (series >= value[0]) & (series <= value[1])
        elif operator == "not_between":
            if not isinstance(value, list) or len(value) != 2:
                raise ValueError("Value for 'not_between' operator must be a list of two values [min, max]")
            return ~((series >= value[0]) & (series <= value[1]))
        elif operator == "length_eq":
            return series.astype(str).str.len() == value
        elif operator == "length_gt":
            return series.astype(str).str.len() > value
        elif operator == "length_lt":
            return series.astype(str).str.len() < value
        else:
            raise ValueError(f"Unsupported operator: {operator}")

    def _generate_filter_summary(self, conditions: List[Dict], logic: str,
                                original_count: int, filtered_count: int) -> Dict:
        """Generate a summary of the filtering operation."""
        reduction_percentage = ((original_count - filtered_count) / original_count * 100) if original_count > 0 else 0

        return {
            "conditions_count": len(conditions),
            "logic_operator": logic,
            "records_removed": original_count - filtered_count,
            "records_kept": filtered_count,
            "reduction_percentage": round(reduction_percentage, 2),
            "conditions_summary": [
                f"{cond['field']} {cond['operator']} {cond.get('value', '')}"
                for cond in conditions
            ]
        }

    def _validate_data_quality(self, data: List[Dict]) -> Dict:
        """Validate data quality and return quality metrics."""
        if not data:
            return {
                "quality_score": 1.0,
                "total_records": 0,
                "issues": [],
                "recommendations": []
            }

        df = pd.DataFrame(data)
        issues = []
        recommendations = []

        # Check for null values
        null_counts = df.isnull().sum()
        null_fields = null_counts[null_counts > 0].to_dict()
        if null_fields:
            issues.append({
                "type": "null_values",
                "description": f"Found null values in fields: {list(null_fields.keys())}",
                "details": null_fields
            })
            recommendations.append("Consider filtering out or filling null values")

        # Check for duplicate records
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            issues.append({
                "type": "duplicate_records",
                "description": f"Found {duplicates} duplicate records",
                "count": duplicates
            })
            recommendations.append("Consider removing duplicate records")

        # Check for empty strings
        empty_string_counts = {}
        for col in df.select_dtypes(include=['object']).columns:
            empty_count = (df[col] == '').sum()
            if empty_count > 0:
                empty_string_counts[col] = empty_count

        if empty_string_counts:
            issues.append({
                "type": "empty_strings",
                "description": f"Found empty strings in fields: {list(empty_string_counts.keys())}",
                "details": empty_string_counts
            })
            recommendations.append("Consider treating empty strings as null values")

        # Calculate quality score (1.0 = perfect, 0.0 = very poor)
        total_cells = len(data) * len(df.columns) if len(df.columns) > 0 else 1
        problem_cells = sum(null_counts) + sum(empty_string_counts.values()) + duplicates * len(df.columns)
        quality_score = max(0.0, 1.0 - (problem_cells / total_cells))

        return {
            "quality_score": round(quality_score, 3),
            "total_records": len(data),
            "total_fields": len(df.columns),
            "issues": issues,
            "recommendations": recommendations,
            "summary": {
                "null_values": len(null_fields),
                "duplicates": duplicates,
                "empty_strings": len(empty_string_counts)
            }
        }


class AggregateDataTool(BaseTool):
    """Tool for aggregating data with groupby operations."""

    @property
    def name(self) -> str:
        return "aggregate_data"

    @property
    def description(self) -> str:
        return "Perform group by operations and aggregate calculations on datasets"

    def get_schema(self) -> Dict:
        return {
            "properties": {
                "data": {
                    "type": "array",
                    "description": "Data to aggregate (list of dictionaries)"
                },
                "group_by": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Fields to group by"
                },
                "aggregations": {
                    "type": "array",
                    "description": "Aggregation operations to perform",
                    "items": {
                        "type": "object",
                        "properties": {
                            "field": {
                                "type": "string",
                                "description": "Field to aggregate"
                            },
                            "operation": {
                                "type": "string",
                                "enum": ["sum", "mean", "median", "min", "max", "count", "std", "var", "first", "last",
                                        "percentile_25", "percentile_50", "percentile_75", "percentile_90", "percentile_95",
                                        "mode", "nunique", "skew", "kurt", "range"],
                                "description": "Aggregation operation"
                            },
                            "alias": {
                                "type": "string",
                                "description": "Alias for the aggregated field (optional)"
                            }
                        },
                        "required": ["field", "operation"]
                    }
                },
                "sort_by": {
                    "type": "string",
                    "description": "Field to sort results by"
                },
                "sort_order": {
                    "type": "string",
                    "enum": ["asc", "desc"],
                    "description": "Sort order",
                    "default": "asc"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of groups to return",
                    "minimum": 1
                }
            },
            "required": ["data", "aggregations"]
        }

    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute data aggregation operation."""
        data = kwargs.get("data")
        group_by = kwargs.get("group_by", [])
        aggregations = kwargs.get("aggregations")
        sort_by = kwargs.get("sort_by")
        sort_order = kwargs.get("sort_order", "asc")
        limit = kwargs.get("limit")

        if not isinstance(data, list):
            raise ValueError("Data must be a list of dictionaries")

        if not data:
            return {
                "aggregated_data": [],
                "original_count": 0,
                "group_count": 0,
                "group_by_fields": group_by,
                "aggregations_applied": aggregations
            }

        if not aggregations:
            raise ValueError("At least one aggregation must be provided")

        # Convert to DataFrame for easier aggregation
        df = pd.DataFrame(data)

        # Validate group_by fields if provided
        if group_by:
            missing_fields = [field for field in group_by if field not in df.columns]
            if missing_fields:
                raise ValueError(f"Group by fields not found in data: {missing_fields}")

        # Validate aggregation fields
        agg_fields = [agg["field"] for agg in aggregations]
        missing_agg_fields = [field for field in agg_fields if field not in df.columns]
        if missing_agg_fields:
            raise ValueError(f"Aggregation fields not found in data: {missing_agg_fields}")

        try:
            # Perform aggregation
            aggregated_df = self._perform_aggregation(df, group_by, aggregations)

            # Sort if requested
            if sort_by and sort_by in aggregated_df.columns:
                ascending = sort_order == "asc"
                aggregated_df = aggregated_df.sort_values(by=sort_by, ascending=ascending)

            # Apply limit if specified
            if limit:
                aggregated_df = aggregated_df.head(limit)

            # Reset index to make group_by fields regular columns (only if grouped)
            if group_by:
                aggregated_df = aggregated_df.reset_index()

            # Convert to list of dictionaries
            aggregated_data = aggregated_df.to_dict(orient="records")

            return {
                "aggregated_data": aggregated_data,
                "original_count": len(data),
                "group_count": len(aggregated_data),
                "group_by_fields": group_by,
                "aggregations_applied": aggregations,
                "aggregation_summary": self._generate_aggregation_summary(
                    group_by, aggregations, len(data), len(aggregated_data)
                )
            }

        except Exception as e:
            raise RuntimeError(f"Failed to aggregate data: {str(e)}")

    def _perform_aggregation(self, df: pd.DataFrame, group_by: List[str], aggregations: List[Dict]) -> pd.DataFrame:
        """Perform the aggregation operations."""
        # Create aggregation dictionary for pandas
        agg_dict = {}

        for agg in aggregations:
            field = agg["field"]
            operation = agg["operation"]
            alias = agg.get("alias", f"{field}_{operation}")

            # Map operations to pandas functions
            if operation == "mean":
                agg_func = "mean"
            elif operation == "median":
                agg_func = "median"
            elif operation == "std":
                agg_func = "std"
            elif operation == "var":
                agg_func = "var"
            elif operation == "first":
                agg_func = "first"
            elif operation == "last":
                agg_func = "last"
            elif operation == "mode":
                agg_func = lambda x: x.mode().iloc[0] if not x.mode().empty else None
            elif operation == "nunique":
                agg_func = "nunique"
            elif operation == "skew":
                agg_func = "skew"
            elif operation == "kurt":
                agg_func = lambda x: x.kurtosis()
            elif operation == "range":
                agg_func = lambda x: x.max() - x.min()
            elif operation.startswith("percentile_"):
                percentile = int(operation.split("_")[1])
                agg_func = lambda x, p=percentile: x.quantile(p/100)
            else:
                agg_func = operation  # sum, min, max, count

            if field in agg_dict:
                # Multiple operations on same field
                if isinstance(agg_dict[field], list):
                    agg_dict[field].append(agg_func)
                else:
                    agg_dict[field] = [agg_dict[field], agg_func]
            else:
                agg_dict[field] = agg_func

        # Perform groupby and aggregation
        if group_by:
            grouped = df.groupby(group_by)
            result = grouped.agg(agg_dict)
        else:
            # Overall aggregation without grouping
            result = df.agg(agg_dict)
            # Convert to DataFrame if Series (single row result)
            if isinstance(result, pd.Series):
                result = result.to_frame().T

        # Flatten column names if multi-level
        if isinstance(result.columns, pd.MultiIndex):
            # Create new column names based on aliases or default naming
            new_columns = []
            for agg in aggregations:
                field = agg["field"]
                operation = agg["operation"]
                alias = agg.get("alias", f"{field}_{operation}")
                new_columns.append(alias)

            result.columns = new_columns
        else:
            # Single level columns - apply aliases if provided
            column_mapping = {}
            for agg in aggregations:
                field = agg["field"]
                operation = agg["operation"]
                alias = agg.get("alias")
                if alias:
                    # Find the matching column (could be field or field_operation)
                    if field in result.columns:
                        column_mapping[field] = alias
                    elif f"{field}_{operation}" in result.columns:
                        column_mapping[f"{field}_{operation}"] = alias

            result = result.rename(columns=column_mapping)

        return result

    def _generate_aggregation_summary(self, group_by: List[str], aggregations: List[Dict],
                                    original_count: int, group_count: int) -> Dict:
        """Generate a summary of the aggregation operation."""
        return {
            "group_by_fields": group_by,
            "group_by_count": len(group_by),
            "aggregations_count": len(aggregations),
            "original_records": original_count,
            "result_groups": group_count,
            "reduction_ratio": round(original_count / group_count, 2) if group_count > 0 else 0,
            "aggregation_operations": [
                f"{agg['operation']}({agg['field']})" + (f" as {agg['alias']}" if agg.get('alias') else "")
                for agg in aggregations
            ]
        }


class JoinDataTool(BaseTool):
    """Tool for joining/merging multiple datasets."""

    @property
    def name(self) -> str:
        return "join_data"

    @property
    def description(self) -> str:
        return "Join/merge multiple datasets using various join types and key fields"

    def get_schema(self) -> Dict:
        return {
            "properties": {
                "left_data": {
                    "type": "array",
                    "description": "Left dataset (list of dictionaries)"
                },
                "right_data": {
                    "type": "array",
                    "description": "Right dataset (list of dictionaries)"
                },
                "join_type": {
                    "type": "string",
                    "enum": ["inner", "left", "right", "outer"],
                    "description": "Type of join to perform",
                    "default": "inner"
                },
                "left_on": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Fields to join on from left dataset"
                },
                "right_on": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Fields to join on from right dataset (if different from left_on)"
                },
                "suffixes": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Suffixes for overlapping column names [left_suffix, right_suffix]",
                    "default": ["_left", "_right"]
                }
            },
            "required": ["left_data", "right_data", "left_on"]
        }

    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the join operation."""
        left_data = kwargs.get("left_data")
        right_data = kwargs.get("right_data")
        left_on = kwargs.get("left_on")
        join_type = kwargs.get("join_type", "inner")
        right_on = kwargs.get("right_on")
        suffixes = kwargs.get("suffixes", ["_left", "_right"])

        # Validate required parameters
        if not left_data or not isinstance(left_data, list):
            raise ValueError("left_data must be a non-empty list of dictionaries")
        if not right_data or not isinstance(right_data, list):
            raise ValueError("right_data must be a non-empty list of dictionaries")
        if not left_on:
            raise ValueError("left_on is required")

        # Use right_on if provided, otherwise use left_on for both sides
        if right_on is None:
            right_on = left_on

        # Validate join keys exist
        left_df = pd.DataFrame(left_data)
        right_df = pd.DataFrame(right_data)

        missing_left = [key for key in left_on if key not in left_df.columns]
        missing_right = [key for key in right_on if key not in right_df.columns]

        if missing_left:
            raise ValueError(f"Left join keys not found: {missing_left}")
        if missing_right:
            raise ValueError(f"Right join keys not found: {missing_right}")

        try:
            # Perform the join
            result_df = left_df.merge(
                right_df,
                left_on=left_on,
                right_on=right_on,
                how=join_type,
                suffixes=suffixes
            )

            # Convert back to list of dictionaries
            joined_data = result_df.to_dict(orient="records")

            return {
                "joined_data": joined_data,
                "left_count": len(left_data),
                "right_count": len(right_data),
                "result_count": len(joined_data),
                "join_type": join_type,
                "join_keys": {"left": left_on, "right": right_on},
                "join_summary": self._generate_join_summary(
                    len(left_data), len(right_data), len(joined_data), join_type
                )
            }

        except Exception as e:
            raise RuntimeError(f"Failed to join datasets: {str(e)}")

    def _generate_join_summary(self, left_count: int, right_count: int,
                              result_count: int, join_type: str) -> Dict:
        """Generate summary of the join operation."""
        return {
            "join_type": join_type,
            "input_records": {"left": left_count, "right": right_count},
            "output_records": result_count,
            "join_efficiency": {
                "left_match_rate": (result_count / left_count * 100) if left_count > 0 else 0,
                "right_match_rate": (result_count / right_count * 100) if right_count > 0 else 0
            }
        }


class PivotDataTool(BaseTool):
    """Tool for pivoting datasets (reshape from long to wide format)."""

    @property
    def name(self) -> str:
        return "pivot_data"

    @property
    def description(self) -> str:
        return "Pivot/reshape data from long to wide format using index, columns, and values"

    def get_schema(self) -> Dict:
        return {
            "properties": {
                "data": {
                    "type": "array",
                    "description": "Data to pivot (list of dictionaries)"
                },
                "index": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Fields to use as row index"
                },
                "columns": {
                    "type": "string",
                    "description": "Field to use as column headers"
                },
                "values": {
                    "type": "string",
                    "description": "Field to use as values in the pivot table"
                },
                "aggfunc": {
                    "type": "string",
                    "enum": ["sum", "mean", "count", "min", "max", "first", "last"],
                    "description": "Aggregation function for duplicate combinations",
                    "default": "sum"
                },
                "fill_value": {
                    "description": "Value to replace missing entries",
                    "default": 0
                }
            },
            "required": ["data", "index", "columns", "values"]
        }

    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the pivot operation."""
        data = kwargs.get("data")
        index = kwargs.get("index")
        columns = kwargs.get("columns")
        values = kwargs.get("values")
        aggfunc = kwargs.get("aggfunc", "sum")
        fill_value = kwargs.get("fill_value", 0)

        # Validate required parameters
        if not isinstance(data, list):
            raise ValueError("Data must be a list of dictionaries")
        if not data:
            return {
                "pivoted_data": [],
                "original_count": 0,
                "pivoted_shape": [0, 0],
                "pivot_columns": []
            }
        if not index:
            raise ValueError("index is required")
        if not columns:
            raise ValueError("columns is required")
        if not values:
            raise ValueError("values is required")

        df = pd.DataFrame(data)

        # Validate required columns exist
        required_fields = index + [columns, values]
        missing_fields = [field for field in required_fields if field not in df.columns]
        if missing_fields:
            raise ValueError(f"Required fields not found: {missing_fields}")

        try:
            # Perform pivot operation
            pivot_df = df.pivot_table(
                index=index,
                columns=columns,
                values=values,
                aggfunc=aggfunc,
                fill_value=fill_value
            )

            # Reset index to make index fields regular columns
            pivot_df = pivot_df.reset_index()

            # Flatten column names if they are multi-level
            if isinstance(pivot_df.columns, pd.MultiIndex):
                pivot_df.columns = [f"{col[0]}_{col[1]}" if col[1] != '' else col[0]
                                   for col in pivot_df.columns]
            else:
                pivot_df.columns = [str(col) for col in pivot_df.columns]

            # Convert to list of dictionaries
            pivoted_data = pivot_df.to_dict(orient="records")

            return {
                "pivoted_data": pivoted_data,
                "original_count": len(data),
                "pivoted_shape": list(pivot_df.shape),
                "pivot_columns": list(pivot_df.columns),
                "pivot_summary": {
                    "index_fields": index,
                    "column_field": columns,
                    "value_field": values,
                    "aggregation": aggfunc,
                    "unique_column_values": df[columns].nunique(),
                    "data_reduction_ratio": len(data) / len(pivoted_data) if len(pivoted_data) > 0 else 0
                }
            }

        except Exception as e:
            raise RuntimeError(f"Failed to pivot data: {str(e)}")


class CleanDataTool(BaseTool):
    """Tool for common data cleaning operations."""

    @property
    def name(self) -> str:
        return "clean_data"

    @property
    def description(self) -> str:
        return "Perform common data cleaning operations like trimming, standardizing, and removing duplicates"

    def get_schema(self) -> Dict:
        return {
            "properties": {
                "data": {
                    "type": "array",
                    "description": "Data to clean (list of dictionaries)"
                },
                "operations": {
                    "type": "array",
                    "description": "List of cleaning operations to perform",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {
                                "type": "string",
                                "enum": ["trim", "lowercase", "uppercase", "remove_nulls", "fill_nulls",
                                        "remove_duplicates", "standardize_text", "remove_special_chars"],
                                "description": "Type of cleaning operation"
                            },
                            "fields": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Fields to apply operation to (if not specified, applies to all applicable fields)"
                            },
                            "parameters": {
                                "type": "object",
                                "description": "Additional parameters for the operation"
                            }
                        },
                        "required": ["type"]
                    }
                }
            },
            "required": ["data", "operations"]
        }

    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute data cleaning operations."""
        data = kwargs.get("data")
        operations = kwargs.get("operations")

        if not isinstance(data, list):
            raise ValueError("Data must be a list of dictionaries")

        if not data:
            return {
                "cleaned_data": [],
                "original_count": 0,
                "cleaned_count": 0,
                "operations_applied": []
            }

        if not operations:
            return {
                "cleaned_data": data,
                "original_count": len(data),
                "cleaned_count": len(data),
                "operations_applied": []
            }

        df = pd.DataFrame(data)
        original_count = len(df)
        operations_applied = []

        try:
            for operation in operations:
                op_type = operation["type"]
                fields = operation.get("fields", [])
                parameters = operation.get("parameters", {})

                # Apply operation
                df, op_summary = self._apply_cleaning_operation(df, op_type, fields, parameters)
                operations_applied.append({
                    "type": op_type,
                    "fields": fields,
                    "summary": op_summary
                })

            # Convert back to list of dictionaries
            cleaned_data = df.to_dict(orient="records")

            return {
                "cleaned_data": cleaned_data,
                "original_count": original_count,
                "cleaned_count": len(cleaned_data),
                "operations_applied": operations_applied,
                "cleaning_summary": {
                    "records_removed": original_count - len(cleaned_data),
                    "operations_count": len(operations_applied),
                    "data_quality_improvement": self._calculate_quality_improvement(data, cleaned_data)
                }
            }

        except Exception as e:
            raise RuntimeError(f"Failed to clean data: {str(e)}")

    def _apply_cleaning_operation(self, df: pd.DataFrame, op_type: str,
                                 fields: List[str], parameters: Dict) -> tuple:
        """Apply a single cleaning operation."""
        original_shape = df.shape

        if op_type == "trim":
            target_fields = fields if fields else df.select_dtypes(include=['object']).columns
            for field in target_fields:
                if field in df.columns:
                    df[field] = df[field].astype(str).str.strip()

        elif op_type == "lowercase":
            target_fields = fields if fields else df.select_dtypes(include=['object']).columns
            for field in target_fields:
                if field in df.columns:
                    df[field] = df[field].astype(str).str.lower()

        elif op_type == "uppercase":
            target_fields = fields if fields else df.select_dtypes(include=['object']).columns
            for field in target_fields:
                if field in df.columns:
                    df[field] = df[field].astype(str).str.upper()

        elif op_type == "remove_nulls":
            if fields:
                df = df.dropna(subset=fields)
            else:
                df = df.dropna()

        elif op_type == "fill_nulls":
            fill_value = parameters.get("fill_value", "")
            target_fields = fields if fields else df.columns
            for field in target_fields:
                if field in df.columns:
                    df[field] = df[field].fillna(fill_value)

        elif op_type == "remove_duplicates":
            subset = fields if fields else None
            df = df.drop_duplicates(subset=subset)

        elif op_type == "standardize_text":
            target_fields = fields if fields else df.select_dtypes(include=['object']).columns
            for field in target_fields:
                if field in df.columns:
                    # Standardize: strip, lowercase, remove extra spaces
                    df[field] = df[field].astype(str).str.strip().str.lower()
                    df[field] = df[field].str.replace(r'\s+', ' ', regex=True)

        elif op_type == "remove_special_chars":
            target_fields = fields if fields else df.select_dtypes(include=['object']).columns
            pattern = parameters.get("pattern", r'[^a-zA-Z0-9\s]')
            replacement = parameters.get("replacement", "")

            for field in target_fields:
                if field in df.columns:
                    df[field] = df[field].astype(str).str.replace(pattern, replacement, regex=True)

        else:
            raise ValueError(f"Unknown cleaning operation: {op_type}")

        # Generate operation summary
        new_shape = df.shape
        summary = {
            "records_before": original_shape[0],
            "records_after": new_shape[0],
            "records_affected": original_shape[0] - new_shape[0],
            "fields_processed": len(fields) if fields else len(df.columns)
        }

        return df, summary

    def _calculate_quality_improvement(self, original_data: List[Dict],
                                     cleaned_data: List[Dict]) -> Dict:
        """Calculate data quality improvement metrics."""
        if not original_data or not cleaned_data:
            return {"improvement_score": 0}

        original_df = pd.DataFrame(original_data)
        cleaned_df = pd.DataFrame(cleaned_data)

        # Calculate basic quality metrics
        original_nulls = original_df.isnull().sum().sum()
        cleaned_nulls = cleaned_df.isnull().sum().sum()

        original_duplicates = len(original_df) - len(original_df.drop_duplicates())
        cleaned_duplicates = len(cleaned_df) - len(cleaned_df.drop_duplicates())

        return {
            "null_reduction": {
                "before": int(original_nulls),
                "after": int(cleaned_nulls),
                "reduction_pct": ((original_nulls - cleaned_nulls) / original_nulls * 100) if original_nulls > 0 else 0
            },
            "duplicate_reduction": {
                "before": original_duplicates,
                "after": cleaned_duplicates,
                "reduction_pct": ((original_duplicates - cleaned_duplicates) / original_duplicates * 100) if original_duplicates > 0 else 0
            },
            "improvement_score": 85.0  # Simplified score - could be more sophisticated
        }