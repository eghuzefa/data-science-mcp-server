"""
Schema introspection tools for analyzing data structure automatically.
"""

from typing import Any, Dict, List
import pandas as pd
import numpy as np
from .base import BaseTool


class DataSchemaAnalyzer(BaseTool):
    """Tool for analyzing data schema and structure automatically."""

    @property
    def name(self) -> str:
        return "analyze_data_schema"

    @property
    def description(self) -> str:
        return "Analyze data structure, infer schema, detect patterns, and suggest optimal data types"

    def get_schema(self) -> Dict:
        return {
            "properties": {
                "data": {
                    "type": "array",
                    "description": "Data to analyze (list of dictionaries)"
                },
                "sample_size": {
                    "type": "integer",
                    "description": "Number of records to sample for analysis (default: all)",
                    "minimum": 1
                },
                "suggest_optimizations": {
                    "type": "boolean",
                    "description": "Whether to suggest data type optimizations",
                    "default": True
                },
                "detect_categories": {
                    "type": "boolean",
                    "description": "Whether to detect categorical fields",
                    "default": True
                },
                "analyze_relationships": {
                    "type": "boolean",
                    "description": "Whether to analyze field relationships",
                    "default": False
                }
            },
            "required": ["data"]
        }

    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute data schema analysis."""
        data = kwargs.get("data")
        sample_size = kwargs.get("sample_size")
        suggest_optimizations = kwargs.get("suggest_optimizations", True)
        detect_categories = kwargs.get("detect_categories", True)
        analyze_relationships = kwargs.get("analyze_relationships", False)

        if not isinstance(data, list):
            raise ValueError("Data must be a list of dictionaries")

        if not data:
            return {
                "schema": {},
                "analysis": {},
                "suggestions": [],
                "summary": {
                    "total_records": 0,
                    "total_fields": 0
                }
            }

        # Sample data if requested
        if sample_size and sample_size < len(data):
            data = data[:sample_size]

        df = pd.DataFrame(data)

        try:
            schema_analysis = self._analyze_schema(df)
            field_analysis = self._analyze_fields(df, detect_categories)

            suggestions = []
            if suggest_optimizations:
                suggestions.extend(self._suggest_optimizations(field_analysis))

            relationships = {}
            if analyze_relationships:
                relationships = self._analyze_relationships(df)

            return {
                "schema": schema_analysis,
                "field_analysis": field_analysis,
                "relationships": relationships,
                "suggestions": suggestions,
                "summary": {
                    "total_records": len(data),
                    "total_fields": len(df.columns),
                    "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
                    "nullable_fields": len([f for f in field_analysis.values() if f["has_nulls"]]),
                    "categorical_fields": len([f for f in field_analysis.values() if f.get("is_categorical", False)])
                }
            }

        except Exception as e:
            raise RuntimeError(f"Failed to analyze schema: {str(e)}")

    def _analyze_schema(self, df: pd.DataFrame) -> Dict:
        """Analyze overall schema structure."""
        schema = {}

        for col in df.columns:
            series = df[col]

            # Infer pandas dtype
            inferred_type = str(series.dtype)

            # Infer semantic type
            semantic_type = self._infer_semantic_type(series)

            # Basic statistics
            null_count = series.isnull().sum()
            unique_count = series.nunique()

            schema[col] = {
                "pandas_dtype": inferred_type,
                "semantic_type": semantic_type,
                "nullable": null_count > 0,
                "unique_values": unique_count,
                "null_percentage": round((null_count / len(series)) * 100, 2),
                "sample_values": series.dropna().head(5).tolist()
            }

        return schema

    def _analyze_fields(self, df: pd.DataFrame, detect_categories: bool = True) -> Dict:
        """Analyze individual fields in detail."""
        field_analysis = {}

        for col in df.columns:
            series = df[col]
            analysis = {
                "field_name": col,
                "data_type": str(series.dtype),
                "total_values": len(series),
                "null_values": series.isnull().sum(),
                "unique_values": series.nunique(),
                "has_nulls": series.isnull().any(),
                "memory_usage_bytes": series.memory_usage(deep=True)
            }

            # Type-specific analysis
            if pd.api.types.is_numeric_dtype(series):
                analysis.update(self._analyze_numeric_field(series))
            elif pd.api.types.is_string_dtype(series) or series.dtype == 'object':
                analysis.update(self._analyze_text_field(series, detect_categories))
            elif pd.api.types.is_datetime64_any_dtype(series):
                analysis.update(self._analyze_datetime_field(series))

            field_analysis[col] = analysis

        return field_analysis

    def _analyze_numeric_field(self, series: pd.Series) -> Dict:
        """Analyze numeric field characteristics."""
        try:
            return {
                "min_value": float(series.min()) if not series.empty else None,
                "max_value": float(series.max()) if not series.empty else None,
                "mean_value": float(series.mean()) if not series.empty else None,
                "median_value": float(series.median()) if not series.empty else None,
                "std_dev": float(series.std()) if not series.empty else None,
                "has_decimals": (series % 1 != 0).any() if not series.empty else False,
                "is_positive_only": (series >= 0).all() if not series.empty else False,
                "potential_id_field": series.nunique() == len(series) and series.dtype in ['int64', 'int32']
            }
        except Exception:
            return {"analysis_error": "Could not analyze numeric field"}

    def _analyze_text_field(self, series: pd.Series, detect_categories: bool) -> Dict:
        """Analyze text field characteristics."""
        try:
            text_series = series.astype(str)
            analysis = {
                "avg_length": round(text_series.str.len().mean(), 2) if not text_series.empty else 0,
                "min_length": text_series.str.len().min() if not text_series.empty else 0,
                "max_length": text_series.str.len().max() if not text_series.empty else 0,
                "has_special_chars": text_series.str.contains('[^a-zA-Z0-9\\s]', na=False).any(),
                "has_numbers": text_series.str.contains('\\d', na=False).any(),
                "has_emails": text_series.str.contains('@', na=False).any(),
                "has_urls": text_series.str.contains('http|www', na=False).any()
            }

            if detect_categories:
                unique_ratio = series.nunique() / len(series)
                analysis["is_categorical"] = unique_ratio < 0.5 and series.nunique() <= 50
                analysis["unique_ratio"] = round(unique_ratio, 3)
                if analysis["is_categorical"]:
                    analysis["categories"] = series.value_counts().head(10).to_dict()

            return analysis
        except Exception:
            return {"analysis_error": "Could not analyze text field"}

    def _analyze_datetime_field(self, series: pd.Series) -> Dict:
        """Analyze datetime field characteristics."""
        try:
            return {
                "min_date": str(series.min()) if not series.empty else None,
                "max_date": str(series.max()) if not series.empty else None,
                "date_range_days": (series.max() - series.min()).days if not series.empty else 0,
                "has_time_component": series.dt.hour.nunique() > 1 if not series.empty else False,
                "timezone_aware": series.dt.tz is not None if not series.empty else False
            }
        except Exception:
            return {"analysis_error": "Could not analyze datetime field"}

    def _infer_semantic_type(self, series: pd.Series) -> str:
        """Infer semantic type beyond pandas dtype."""
        if pd.api.types.is_numeric_dtype(series):
            if series.nunique() == len(series) and series.name and 'id' in str(series.name).lower():
                return "identifier"
            elif series.dtype in ['int64', 'int32'] and (series >= 0).all():
                return "count"
            elif series.dtype in ['float64', 'float32']:
                return "measurement"
            else:
                return "numeric"
        elif pd.api.types.is_string_dtype(series) or series.dtype == 'object':
            sample_str = str(series.dropna().iloc[0]) if not series.dropna().empty else ""
            if '@' in sample_str:
                return "email"
            elif 'http' in sample_str or 'www' in sample_str:
                return "url"
            elif series.nunique() / len(series) < 0.5:
                return "categorical"
            else:
                return "text"
        elif pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"
        else:
            return "unknown"

    def _suggest_optimizations(self, field_analysis: Dict) -> List[Dict]:
        """Suggest data type and structure optimizations."""
        suggestions = []

        for col, analysis in field_analysis.items():
            # Suggest categorical conversion
            if (analysis.get("is_categorical") and
                analysis["data_type"] == "object"):
                suggestions.append({
                    "type": "optimization",
                    "field": col,
                    "suggestion": "Convert to pandas categorical",
                    "reason": f"Low unique ratio ({analysis.get('unique_ratio', 0)}) suggests categorical data",
                    "potential_memory_savings": "Up to 50-90% memory reduction"
                })

            # Suggest integer downcasting
            if (analysis["data_type"] in ["int64"] and
                "min_value" in analysis and "max_value" in analysis):
                if analysis["max_value"] <= 127 and analysis["min_value"] >= -128:
                    suggestions.append({
                        "type": "optimization",
                        "field": col,
                        "suggestion": "Convert to int8",
                        "reason": "Values fit in smaller integer type",
                        "potential_memory_savings": "87.5% memory reduction"
                    })
                elif analysis["max_value"] <= 32767 and analysis["min_value"] >= -32768:
                    suggestions.append({
                        "type": "optimization",
                        "field": col,
                        "suggestion": "Convert to int16",
                        "reason": "Values fit in smaller integer type",
                        "potential_memory_savings": "75% memory reduction"
                    })

            # Suggest handling nulls
            if analysis["has_nulls"] and analysis["null_values"] / analysis["total_values"] > 0.8:
                suggestions.append({
                    "type": "data_quality",
                    "field": col,
                    "suggestion": "Consider removing this field",
                    "reason": f"Over 80% null values ({analysis['null_values']}/{analysis['total_values']})",
                    "impact": "Significant data sparsity"
                })

        return suggestions

    def _analyze_relationships(self, df: pd.DataFrame) -> Dict:
        """Analyze relationships between fields."""
        relationships = {}

        # Find potential foreign key relationships
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].nunique() / len(df) > 0.8:  # High uniqueness suggests ID field
                relationships[col] = {
                    "type": "potential_primary_key",
                    "uniqueness": df[col].nunique() / len(df)
                }

        # Find correlated numeric fields
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            high_correlations = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if pd.notna(corr_val) and isinstance(corr_val, (int, float)) and abs(float(corr_val)) > 0.7:
                        high_correlations.append({
                            "field1": corr_matrix.columns[i],
                            "field2": corr_matrix.columns[j],
                            "correlation": round(float(corr_val), 3)
                        })

            if high_correlations:
                relationships["correlations"] = high_correlations

        return relationships