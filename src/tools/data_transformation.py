"""
Data transformation tools for filtering, aggregation, and manipulation.
"""

from typing import Any, Dict, List, Optional
import pandas as pd

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
                                "enum": ["==", "!=", ">", ">=", "<", "<=", "in", "not_in", "contains", "starts_with", "ends_with", "is_null", "is_not_null"],
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
                }
            },
            "required": ["data", "conditions"]
        }

    async def execute(self, data: List[Dict], conditions: List[Dict],
                     logic: str = "AND", limit: Optional[int] = None,
                     sort_by: Optional[str] = None, sort_order: str = "asc") -> Dict[str, Any]:
        """Execute data filtering operation."""
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

            return {
                "filtered_data": filtered_data,
                "original_count": len(data),
                "filtered_count": len(filtered_data),
                "conditions_applied": conditions,
                "logic": logic,
                "filters_summary": self._generate_filter_summary(conditions, logic, len(data), len(filtered_data))
            }

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
                                "enum": ["sum", "mean", "median", "min", "max", "count", "std", "var", "first", "last"],
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
            "required": ["data", "group_by", "aggregations"]
        }

    async def execute(self, data: List[Dict], group_by: List[str], aggregations: List[Dict],
                     sort_by: Optional[str] = None, sort_order: str = "asc",
                     limit: Optional[int] = None) -> Dict[str, Any]:
        """Execute data aggregation operation."""
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

        if not group_by:
            raise ValueError("At least one group_by field must be provided")

        if not aggregations:
            raise ValueError("At least one aggregation must be provided")

        # Convert to DataFrame for easier aggregation
        df = pd.DataFrame(data)

        # Validate group_by fields
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

            # Reset index to make group_by fields regular columns
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
        grouped = df.groupby(group_by)
        result = grouped.agg(agg_dict)

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