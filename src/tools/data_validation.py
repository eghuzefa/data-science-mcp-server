"""
Data validation tools for quality checking and schema validation.
"""

import json
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

from .base import BaseTool


class ValidateSchemaTool(BaseTool):
    """Tool for validating data against a defined schema."""

    @property
    def name(self) -> str:
        return "validate_schema"

    @property
    def description(self) -> str:
        return "Validate data against expected schema with type checking and constraints"

    def get_schema(self) -> Dict:
        return {
            "properties": {
                "data": {
                    "type": "array",
                    "description": "Data to validate (list of dictionaries)"
                },
                "schema": {
                    "type": "object",
                    "description": "Schema definition with field types and constraints",
                    "properties": {
                        "fields": {
                            "type": "object",
                            "description": "Field definitions",
                            "additionalProperties": {
                                "type": "object",
                                "properties": {
                                    "type": {"type": "string", "enum": ["string", "integer", "float", "boolean", "date", "datetime"]},
                                    "required": {"type": "boolean", "default": True},
                                    "min_length": {"type": "integer"},
                                    "max_length": {"type": "integer"},
                                    "min_value": {"type": "number"},
                                    "max_value": {"type": "number"},
                                    "allowed_values": {"type": "array"},
                                    "pattern": {"type": "string"}
                                },
                                "required": ["type"]
                            }
                        }
                    },
                    "required": ["fields"]
                }
            },
            "required": ["data", "schema"]
        }

    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute schema validation."""
        data = kwargs.get("data")
        schema = kwargs.get("schema")

        if not isinstance(data, list):
            raise ValueError("Data must be a list of dictionaries")

        if not isinstance(schema, dict):
            raise ValueError("Schema must be a dictionary")

        # Support both direct field definitions and nested under "fields"
        if "fields" in schema:
            field_definitions = schema["fields"]
        else:
            field_definitions = schema
        validation_results = {
            "total_records": len(data),
            "valid_records": 0,
            "invalid_records": 0,
            "validation_errors": [],
            "field_statistics": {},
            "overall_valid": True
        }

        # Validate each record
        for i, record in enumerate(data):
            record_errors = self._validate_record(record, field_definitions, i)
            if record_errors:
                validation_results["validation_errors"].extend(record_errors)
                validation_results["invalid_records"] += 1
            else:
                validation_results["valid_records"] += 1

        # Generate field statistics
        validation_results["field_statistics"] = self._generate_field_statistics(data, field_definitions)

        # Overall validation status
        validation_results["overall_valid"] = validation_results["invalid_records"] == 0

        return validation_results

    def _validate_record(self, record: Dict, field_definitions: Dict, record_index: int) -> List[Dict]:
        """Validate a single record against the schema."""
        errors = []

        # Check for required fields
        for field_name, field_def in field_definitions.items():
            if field_def.get("required", True) and field_name not in record:
                errors.append({
                    "record_index": record_index,
                    "field": field_name,
                    "error_type": "missing_field",
                    "message": f"Required field '{field_name}' is missing"
                })
                continue

            if field_name not in record:
                continue  # Skip optional missing fields

            value = record[field_name]
            field_errors = self._validate_field_value(value, field_def, field_name, record_index)
            errors.extend(field_errors)

        return errors

    def _validate_field_value(self, value: Any, field_def: Dict, field_name: str, record_index: int) -> List[Dict]:
        """Validate a single field value."""
        errors = []
        field_type = field_def["type"]

        # Handle null values
        if value is None or value == "":
            if field_def.get("required", True):
                errors.append({
                    "record_index": record_index,
                    "field": field_name,
                    "error_type": "null_value",
                    "message": f"Field '{field_name}' cannot be null"
                })
            return errors

        # Type validation
        try:
            converted_value = self._convert_and_validate_type(value, field_type)
        except (ValueError, TypeError) as e:
            errors.append({
                "record_index": record_index,
                "field": field_name,
                "error_type": "type_error",
                "message": f"Field '{field_name}' type error: {str(e)}"
            })
            return errors

        # Constraint validation
        constraint_errors = self._validate_constraints(converted_value, field_def, field_name, record_index)
        errors.extend(constraint_errors)

        return errors

    def _convert_and_validate_type(self, value: Any, expected_type: str) -> Any:
        """Convert and validate value type."""
        if expected_type == "string":
            return str(value)
        elif expected_type == "integer":
            if isinstance(value, bool):
                raise ValueError("Boolean cannot be converted to integer")
            return int(float(value))  # Handle string numbers
        elif expected_type == "float":
            if isinstance(value, bool):
                raise ValueError("Boolean cannot be converted to float")
            return float(value)
        elif expected_type == "boolean":
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                if value.lower() in ["true", "1", "yes", "y"]:
                    return True
                elif value.lower() in ["false", "0", "no", "n"]:
                    return False
            raise ValueError(f"Cannot convert '{value}' to boolean")
        elif expected_type == "date":
            if isinstance(value, str):
                try:
                    return datetime.strptime(value, "%Y-%m-%d").date()
                except ValueError:
                    raise ValueError(f"Date '{value}' does not match format YYYY-MM-DD")
            raise ValueError("Date must be a string in YYYY-MM-DD format")
        elif expected_type == "datetime":
            if isinstance(value, str):
                # Try common datetime formats
                formats = ["%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"]
                for fmt in formats:
                    try:
                        return datetime.strptime(value, fmt)
                    except ValueError:
                        continue
                raise ValueError(f"Datetime '{value}' does not match any expected format")
            raise ValueError("Datetime must be a string")
        else:
            raise ValueError(f"Unknown type: {expected_type}")

    def _validate_constraints(self, value: Any, field_def: Dict, field_name: str, record_index: int) -> List[Dict]:
        """Validate field constraints."""
        errors = []

        # String length constraints
        if isinstance(value, str):
            if "min_length" in field_def and len(value) < field_def["min_length"]:
                errors.append({
                    "record_index": record_index,
                    "field": field_name,
                    "error_type": "min_length",
                    "message": f"Field '{field_name}' length {len(value)} is less than minimum {field_def['min_length']}"
                })

            if "max_length" in field_def and len(value) > field_def["max_length"]:
                errors.append({
                    "record_index": record_index,
                    "field": field_name,
                    "error_type": "max_length",
                    "message": f"Field '{field_name}' length {len(value)} exceeds maximum {field_def['max_length']}"
                })

        # Numeric constraints
        if isinstance(value, (int, float)):
            if "min_value" in field_def and value < field_def["min_value"]:
                errors.append({
                    "record_index": record_index,
                    "field": field_name,
                    "error_type": "min_value",
                    "message": f"Field '{field_name}' value {value} is less than minimum {field_def['min_value']}"
                })

            if "max_value" in field_def and value > field_def["max_value"]:
                errors.append({
                    "record_index": record_index,
                    "field": field_name,
                    "error_type": "max_value",
                    "message": f"Field '{field_name}' value {value} exceeds maximum {field_def['max_value']}"
                })

        # Allowed values constraint
        if "allowed_values" in field_def and value not in field_def["allowed_values"]:
            errors.append({
                "record_index": record_index,
                "field": field_name,
                "error_type": "invalid_value",
                "message": f"Field '{field_name}' value '{value}' is not in allowed values: {field_def['allowed_values']}"
            })

        return errors

    def _generate_field_statistics(self, data: List[Dict], field_definitions: Dict) -> Dict:
        """Generate statistics for each field."""
        statistics = {}

        for field_name in field_definitions:
            field_values = [record.get(field_name) for record in data]
            non_null_values = [v for v in field_values if v is not None and v != ""]

            stats = {
                "total_count": len(field_values),
                "non_null_count": len(non_null_values),
                "null_count": len(field_values) - len(non_null_values),
                "null_percentage": ((len(field_values) - len(non_null_values)) / len(field_values)) * 100 if field_values else 0
            }

            if non_null_values:
                stats["unique_count"] = len(set(non_null_values))
                stats["unique_percentage"] = (stats["unique_count"] / len(non_null_values)) * 100

            statistics[field_name] = stats

        return statistics


class CheckNullsTool(BaseTool):
    """Tool for identifying null values and patterns in data."""

    @property
    def name(self) -> str:
        return "check_nulls"

    @property
    def description(self) -> str:
        return "Analyze null values and missing data patterns in datasets"

    def get_schema(self) -> Dict:
        return {
            "properties": {
                "data": {
                    "type": "array",
                    "description": "Data to analyze (list of dictionaries)"
                },
                "null_values": {
                    "type": "array",
                    "description": "Additional values to treat as null (e.g., ['', 'N/A', 'NULL'])",
                    "items": {"type": "string"},
                    "default": ["", "null", "NULL", "None", "N/A", "n/a", "#N/A"]
                }
            },
            "required": ["data"]
        }

    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute null value analysis."""
        data = kwargs.get("data")
        null_values = kwargs.get("null_values")

        if not isinstance(data, list):
            raise ValueError("Data must be a list of dictionaries")

        if not data:
            return {"message": "No data to analyze"}

        null_values = null_values or ["", "null", "NULL", "None", "N/A", "n/a", "#N/A"]

        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(data)

        # Replace custom null values with actual NaN
        for null_val in null_values:
            df = df.replace(null_val, np.nan)

        analysis_results = {
            "total_records": len(df),
            "total_fields": len(df.columns),
            "field_analysis": {},
            "summary": {},
            "patterns": {}
        }

        # Analyze each field
        for column in df.columns:
            field_analysis = self._analyze_field_nulls(df[column], column)
            analysis_results["field_analysis"][column] = field_analysis

        # Generate summary statistics
        analysis_results["summary"] = self._generate_null_summary(df)

        # Identify patterns
        analysis_results["patterns"] = self._identify_null_patterns(df)

        return analysis_results

    def _analyze_field_nulls(self, series: pd.Series, field_name: str) -> Dict:
        """Analyze null values for a single field."""
        total_count = len(series)
        null_count = series.isnull().sum()
        non_null_count = total_count - null_count

        analysis = {
            "field_name": field_name,
            "total_count": total_count,
            "null_count": int(null_count),
            "non_null_count": int(non_null_count),
            "null_percentage": (null_count / total_count) * 100 if total_count > 0 else 0,
            "null_severity": self._classify_null_severity(null_count / total_count if total_count > 0 else 0)
        }

        # Find null positions
        if null_count > 0:
            null_indices = series[series.isnull()].index.tolist()
            # Include all null positions (LLM can decide how many to use)
            analysis["null_positions"] = null_indices
            analysis["has_leading_nulls"] = series.iloc[:5].isnull().any()
            analysis["has_trailing_nulls"] = series.iloc[-5:].isnull().any()

        return analysis

    def _classify_null_severity(self, null_ratio: float) -> str:
        """Classify the severity of null values."""
        if null_ratio == 0:
            return "none"
        elif null_ratio < 0.05:
            return "low"
        elif null_ratio < 0.20:
            return "moderate"
        elif null_ratio < 0.50:
            return "high"
        else:
            return "critical"

    def _generate_null_summary(self, df: pd.DataFrame) -> Dict:
        """Generate overall null value summary."""
        total_cells = df.size
        total_nulls = df.isnull().sum().sum()

        fields_with_nulls = (df.isnull().sum() > 0).sum()
        fields_all_null = (df.isnull().sum() == len(df)).sum()
        fields_no_nulls = (df.isnull().sum() == 0).sum()

        return {
            "total_cells": int(total_cells),
            "total_nulls": int(total_nulls),
            "overall_null_percentage": (total_nulls / total_cells) * 100 if total_cells > 0 else 0,
            "fields_with_nulls": int(fields_with_nulls),
            "fields_all_null": int(fields_all_null),
            "fields_no_nulls": int(fields_no_nulls),
            "completeness_score": ((total_cells - total_nulls) / total_cells) * 100 if total_cells > 0 else 0
        }

    def _identify_null_patterns(self, df: pd.DataFrame) -> Dict:
        """Identify patterns in null value distribution."""
        patterns = {}

        # Records with no nulls
        complete_records = (~df.isnull().any(axis=1)).sum()
        patterns["complete_records"] = {
            "count": int(complete_records),
            "percentage": (complete_records / len(df)) * 100 if len(df) > 0 else 0
        }

        # Records with all nulls
        empty_records = df.isnull().all(axis=1).sum()
        patterns["empty_records"] = {
            "count": int(empty_records),
            "percentage": (empty_records / len(df)) * 100 if len(df) > 0 else 0
        }

        # Most common null combinations (get all, let LLM decide how many to use)
        null_combinations = df.isnull().value_counts()
        patterns["common_null_combinations"] = []

        for combo, count in null_combinations.items():
            if isinstance(combo, tuple):
                # Handle tuple case (when multiple columns)
                null_fields = [df.columns[i] for i, is_null in enumerate(combo) if is_null]
            else:
                # Handle single boolean case
                null_fields = [df.columns[0]] if combo else []

            patterns["common_null_combinations"].append({
                "fields": null_fields,
                "count": int(count)
            })

        return patterns


class DataQualityReportTool(BaseTool):
    """Tool for generating comprehensive data quality reports."""

    @property
    def name(self) -> str:
        return "data_quality_report"

    @property
    def description(self) -> str:
        return "Generate comprehensive data quality report with statistics, issues, and recommendations"

    def get_schema(self) -> Dict:
        return {
            "properties": {
                "data": {
                    "type": "array",
                    "description": "Data to analyze (list of dictionaries)"
                },
                "include_samples": {
                    "type": "boolean",
                    "description": "Include sample data in the report",
                    "default": True
                },
                "sample_size": {
                    "type": "integer",
                    "description": "Number of sample records to include",
                    "default": 5,
                    "minimum": 1
                }
            },
            "required": ["data"]
        }

    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute data quality analysis."""
        data = kwargs.get("data")
        include_samples = kwargs.get("include_samples", True)
        sample_size = kwargs.get("sample_size", 5)

        if not isinstance(data, list):
            raise ValueError("Data must be a list of dictionaries")

        if not data:
            return {"message": "No data to analyze"}

        df = pd.DataFrame(data)

        report = {
            "overview": self._generate_overview(df),
            "field_analysis": self._analyze_fields(df),
            "data_quality_score": 0,
            "issues": [],
            "recommendations": [],
            "timestamp": datetime.now().isoformat()
        }

        if include_samples:
            report["samples"] = {
                "first_records": data[:min(sample_size, len(data))],
                "random_records": df.sample(min(sample_size, len(df))).to_dict(orient="records") if len(df) >= sample_size else []
            }

        # Calculate overall quality score
        report["data_quality_score"] = self._calculate_quality_score(report)

        # Generate issues and recommendations
        report["issues"] = self._identify_issues(report)
        report["recommendations"] = self._generate_recommendations(report)

        return report

    def _generate_overview(self, df: pd.DataFrame) -> Dict:
        """Generate overview statistics."""
        return {
            "total_records": len(df),
            "total_fields": len(df.columns),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / (1024 * 1024),
            "data_types": df.dtypes.value_counts().to_dict(),
            "shape": list(df.shape)
        }

    def _analyze_fields(self, df: pd.DataFrame) -> Dict:
        """Analyze each field in detail."""
        field_analysis = {}

        for column in df.columns:
            series = df[column]
            analysis = {
                "name": column,
                "data_type": str(series.dtype),
                "total_count": len(series),
                "null_count": int(series.isnull().sum()),
                "null_percentage": (series.isnull().sum() / len(series)) * 100,
                "unique_count": int(series.nunique()),
                "unique_percentage": (series.nunique() / len(series)) * 100 if len(series) > 0 else 0
            }

            # Type-specific analysis
            if pd.api.types.is_numeric_dtype(series):
                analysis.update(self._analyze_numeric_field(series))
            elif pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
                analysis.update(self._analyze_text_field(series))
            elif pd.api.types.is_datetime64_any_dtype(series):
                analysis.update(self._analyze_datetime_field(series))

            field_analysis[column] = analysis

        return field_analysis

    def _analyze_numeric_field(self, series: pd.Series) -> Dict:
        """Analyze numeric field."""
        non_null_series = series.dropna()
        if len(non_null_series) == 0:
            return {"analysis_type": "numeric", "statistics": "No non-null values"}

        stats = {
            "analysis_type": "numeric",
            "statistics": {
                "min": float(non_null_series.min()),
                "max": float(non_null_series.max()),
                "mean": float(non_null_series.mean()),
                "median": float(non_null_series.median()),
                "std": float(non_null_series.std()) if len(non_null_series) > 1 else 0,
                "q25": float(non_null_series.quantile(0.25)),
                "q75": float(non_null_series.quantile(0.75))
            },
            "outliers": self._detect_outliers(non_null_series),
            "zeros_count": int((non_null_series == 0).sum()),
            "negative_count": int((non_null_series < 0).sum())
        }

        return stats

    def _analyze_text_field(self, series: pd.Series) -> Dict:
        """Analyze text field."""
        non_null_series = series.dropna().astype(str)
        if len(non_null_series) == 0:
            return {"analysis_type": "text", "statistics": "No non-null values"}

        lengths = non_null_series.str.len()

        stats = {
            "analysis_type": "text",
            "statistics": {
                "min_length": int(lengths.min()),
                "max_length": int(lengths.max()),
                "avg_length": float(lengths.mean()),
                "empty_strings": int((non_null_series == "").sum()),
                "whitespace_only": int(non_null_series.str.strip().eq("").sum())
            },
            "top_values": non_null_series.value_counts().to_dict(),
            "patterns": self._analyze_text_patterns(non_null_series)
        }

        return stats

    def _analyze_datetime_field(self, series: pd.Series) -> Dict:
        """Analyze datetime field."""
        non_null_series = series.dropna()
        if len(non_null_series) == 0:
            return {"analysis_type": "datetime", "statistics": "No non-null values"}

        stats = {
            "analysis_type": "datetime",
            "statistics": {
                "min_date": non_null_series.min().isoformat(),
                "max_date": non_null_series.max().isoformat(),
                "date_range_days": (non_null_series.max() - non_null_series.min()).days
            }
        }

        return stats

    def _detect_outliers(self, series: pd.Series) -> Dict:
        """Detect outliers using IQR method."""
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = series[(series < lower_bound) | (series > upper_bound)]

        return {
            "count": len(outliers),
            "percentage": (len(outliers) / len(series)) * 100,
            "lower_bound": float(lower_bound),
            "upper_bound": float(upper_bound)
        }

    def _analyze_text_patterns(self, series: pd.Series) -> Dict:
        """Analyze common text patterns."""
        patterns = {
            "all_uppercase": int(series.str.isupper().sum()),
            "all_lowercase": int(series.str.islower().sum()),
            "mixed_case": int((~series.str.isupper() & ~series.str.islower()).sum()),
            "contains_numbers": int(series.str.contains(r'\d', na=False).sum()),
            "contains_special_chars": int(series.str.contains(r'[^\w\s]', na=False).sum())
        }

        return patterns

    def _calculate_quality_score(self, report: Dict) -> float:
        """Calculate overall data quality score (0-100)."""
        field_analysis = report["field_analysis"]

        if not field_analysis:
            return 0

        scores = []

        for field_name, analysis in field_analysis.items():
            field_score = 100  # Start with perfect score

            # Deduct for null values
            null_penalty = analysis["null_percentage"] * 0.5
            field_score -= null_penalty

            # Deduct for low uniqueness (potential duplicates)
            if analysis["unique_percentage"] < 10:
                field_score -= 20

            # Type-specific penalties
            if analysis.get("analysis_type") == "numeric":
                outlier_penalty = analysis.get("outliers", {}).get("percentage", 0) * 0.3
                field_score -= outlier_penalty

            field_score = max(0, field_score)  # Ensure non-negative
            scores.append(field_score)

        return sum(scores) / len(scores) if scores else 0

    def _identify_issues(self, report: Dict) -> List[Dict]:
        """Identify data quality issues."""
        issues = []
        field_analysis = report["field_analysis"]

        for field_name, analysis in field_analysis.items():
            # High null percentage
            if analysis["null_percentage"] > 20:
                issues.append({
                    "field": field_name,
                    "type": "high_null_percentage",
                    "severity": "high" if analysis["null_percentage"] > 50 else "medium",
                    "description": f"Field has {analysis['null_percentage']:.1f}% null values"
                })

            # Low uniqueness
            if analysis["unique_percentage"] < 5 and analysis["unique_count"] > 1:
                issues.append({
                    "field": field_name,
                    "type": "low_uniqueness",
                    "severity": "medium",
                    "description": f"Field has only {analysis['unique_percentage']:.1f}% unique values"
                })

            # Outliers in numeric fields
            if analysis.get("analysis_type") == "numeric":
                outlier_info = analysis.get("outliers", {})
                if outlier_info.get("percentage", 0) > 5:
                    issues.append({
                        "field": field_name,
                        "type": "outliers",
                        "severity": "low",
                        "description": f"Field has {outlier_info['percentage']:.1f}% outliers"
                    })

        return issues

    def _generate_recommendations(self, report: Dict) -> List[str]:
        """Generate data quality improvement recommendations."""
        recommendations = []
        issues = report["issues"]

        if any(issue["type"] == "high_null_percentage" for issue in issues):
            recommendations.append("Consider investigating the source of null values and implement data collection improvements")

        if any(issue["type"] == "low_uniqueness" for issue in issues):
            recommendations.append("Review fields with low uniqueness for potential data entry errors or missing normalization")

        if any(issue["type"] == "outliers" for issue in issues):
            recommendations.append("Investigate outliers to determine if they represent data errors or legitimate extreme values")

        if report["data_quality_score"] < 70:
            recommendations.append("Overall data quality score is below acceptable threshold - consider comprehensive data cleaning")

        if not recommendations:
            recommendations.append("Data quality appears good - maintain current data collection and validation processes")

        return recommendations


class DetectDuplicatesTool(BaseTool):
    """Tool for detecting duplicate records in datasets."""

    @property
    def name(self) -> str:
        return "detect_duplicates"

    @property
    def description(self) -> str:
        return "Detect duplicate records and near-duplicates in datasets with configurable matching criteria"

    def get_schema(self) -> Dict:
        return {
            "properties": {
                "data": {
                    "type": "array",
                    "description": "Data to analyze (list of dictionaries)"
                },
                "key_fields": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Fields to use for duplicate detection (if not provided, all fields are used)"
                },
                "ignore_fields": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Fields to ignore during duplicate detection",
                    "default": []
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Whether string comparison should be case sensitive",
                    "default": False
                },
                "include_duplicates": {
                    "type": "boolean",
                    "description": "Include actual duplicate records in the response",
                    "default": True
                }
            },
            "required": ["data"]
        }

    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute duplicate detection."""
        data = kwargs.get("data")
        key_fields = kwargs.get("key_fields")
        ignore_fields = kwargs.get("ignore_fields") or []
        case_sensitive = kwargs.get("case_sensitive", False)
        include_duplicates = kwargs.get("include_duplicates", True)

        if not isinstance(data, list):
            raise ValueError("Data must be a list of dictionaries")

        if not data:
            return {"message": "No data to analyze"}
        df = pd.DataFrame(data)

        # Determine fields to use for comparison
        if key_fields:
            comparison_fields = [f for f in key_fields if f in df.columns]
            if not comparison_fields:
                raise ValueError("None of the specified key_fields exist in the data")
        else:
            comparison_fields = [f for f in df.columns if f not in ignore_fields]

        if not comparison_fields:
            raise ValueError("No fields available for duplicate detection")

        # Prepare data for comparison
        comparison_df = df[comparison_fields].copy()

        # Handle case sensitivity for string fields
        if not case_sensitive:
            for field in comparison_fields:
                if comparison_df[field].dtype == 'object':
                    comparison_df[field] = comparison_df[field].astype(str).str.lower()

        # Find duplicates
        duplicate_mask = comparison_df.duplicated(keep=False)
        duplicate_indices = df.index[duplicate_mask].tolist()

        # Group duplicates
        duplicate_groups = []
        if duplicate_indices:
            for _, group in comparison_df[duplicate_mask].groupby(comparison_fields):
                group_indices = group.index.tolist()
                if len(group_indices) > 1:
                    duplicate_groups.append(group_indices)

        result = {
            "total_records": len(df),
            "unique_records": len(df) - len(duplicate_indices),
            "duplicate_records": len(duplicate_indices),
            "duplicate_percentage": (len(duplicate_indices) / len(df)) * 100 if len(df) > 0 else 0,
            "duplicate_groups_count": len(duplicate_groups),
            "comparison_fields": comparison_fields,
            "analysis_settings": {
                "key_fields": key_fields,
                "ignore_fields": ignore_fields,
                "case_sensitive": case_sensitive
            }
        }

        if include_duplicates and duplicate_groups:
            result["duplicate_groups"] = []
            for i, group_indices in enumerate(duplicate_groups):  # Include all duplicate groups
                group_data = {
                    "group_id": i + 1,
                    "record_count": len(group_indices),
                    "record_indices": group_indices,
                    "records": [data[idx] for idx in group_indices]
                }
                result["duplicate_groups"].append(group_data)

        # Generate recommendations
        result["recommendations"] = self._generate_duplicate_recommendations(result)

        return result

    def _generate_duplicate_recommendations(self, result: Dict) -> List[str]:
        """Generate recommendations for handling duplicates."""
        recommendations = []

        duplicate_percentage = result["duplicate_percentage"]

        if duplicate_percentage == 0:
            recommendations.append("No duplicates found - data appears to be clean")
        elif duplicate_percentage < 5:
            recommendations.append("Low level of duplicates detected - review and remove as needed")
        elif duplicate_percentage < 20:
            recommendations.append("Moderate level of duplicates - implement deduplication process")
        else:
            recommendations.append("High level of duplicates - investigate data collection process and implement comprehensive deduplication")

        if result["duplicate_groups_count"] > 0:
            recommendations.append("Review duplicate groups to determine appropriate merge or removal strategy")
            recommendations.append("Consider implementing unique constraints or validation rules to prevent future duplicates")

        return recommendations