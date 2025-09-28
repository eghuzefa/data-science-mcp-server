"""
Data visualization tools for creating charts and statistical summaries.
"""

from typing import Any, Dict, List
import pandas as pd
import numpy as np
import json
from .base import BaseTool


class CreateChartTool(BaseTool):
    """Tool for creating various types of charts from data."""

    @property
    def name(self) -> str:
        return "create_chart"

    @property
    def description(self) -> str:
        return "Create charts and visualizations from data with customizable options"

    def get_schema(self) -> Dict:
        return {
            "properties": {
                "data": {
                    "type": "array",
                    "description": "Data to visualize (list of dictionaries)"
                },
                "chart_type": {
                    "type": "string",
                    "enum": ["bar", "line", "scatter", "histogram", "box", "pie", "heatmap"],
                    "description": "Type of chart to create"
                },
                "x_axis": {
                    "type": "string",
                    "description": "Field for x-axis"
                },
                "y_axis": {
                    "type": "string",
                    "description": "Field for y-axis (not needed for pie charts)"
                },
                "title": {
                    "type": "string",
                    "description": "Chart title",
                    "default": "Data Visualization"
                },
                "width": {
                    "type": "integer",
                    "description": "Chart width in pixels",
                    "default": 800,
                    "minimum": 200
                },
                "height": {
                    "type": "integer",
                    "description": "Chart height in pixels",
                    "default": 600,
                    "minimum": 200
                },
                "color_scheme": {
                    "type": "string",
                    "enum": ["default", "viridis", "plasma", "cool", "warm"],
                    "description": "Color scheme for the chart",
                    "default": "default"
                },
                "group_by": {
                    "type": "string",
                    "description": "Field to group data by (for multi-series charts)"
                },
                "aggregation": {
                    "type": "string",
                    "enum": ["sum", "mean", "count", "min", "max"],
                    "description": "Aggregation function when grouping data",
                    "default": "sum"
                }
            },
            "required": ["data", "chart_type", "x_axis"]
        }

    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute chart creation."""
        data = kwargs.get("data")
        chart_type = kwargs.get("chart_type", "bar")
        x_axis = kwargs.get("x_axis", "")
        y_axis = kwargs.get("y_axis", "")
        title = kwargs.get("title", "Data Visualization")
        width = kwargs.get("width", 800)
        height = kwargs.get("height", 600)
        color_scheme = kwargs.get("color_scheme", "default")
        group_by = kwargs.get("group_by")
        aggregation = kwargs.get("aggregation", "sum")

        if not isinstance(data, list):
            raise ValueError("Data must be a list of dictionaries")

        if not data:
            return {
                "chart_data": {},
                "chart_config": {},
                "message": "No data provided for visualization"
            }

        # Validate required parameters
        if not x_axis:
            raise ValueError("x_axis is required")

        if chart_type not in ["pie", "histogram"] and not y_axis:
            raise ValueError("y_axis is required for non-pie and non-histogram charts")

        try:
            df = pd.DataFrame(data)

            # Validate required fields
            if x_axis not in df.columns:
                raise ValueError(f"X-axis field '{x_axis}' not found in data")

            if chart_type != "pie" and y_axis and y_axis not in df.columns:
                raise ValueError(f"Y-axis field '{y_axis}' not found in data")

            # Prepare data based on chart type
            chart_data = self._prepare_chart_data(df, chart_type, x_axis, y_axis, group_by, aggregation)

            # Generate chart configuration
            chart_config = self._generate_chart_config(
                chart_type, x_axis, y_axis, title, width, height, color_scheme, group_by
            )

            # Generate insights
            insights = self._generate_insights(df, chart_type, x_axis, y_axis)

            return {
                "chart_data": chart_data,
                "chart_config": chart_config,
                "insights": insights,
                "data_summary": {
                    "total_records": len(df),
                    "fields_used": [f for f in [x_axis, y_axis, group_by] if f],
                    "chart_type": chart_type
                }
            }

        except Exception as e:
            raise RuntimeError(f"Failed to create chart: {str(e)}")

    def _prepare_chart_data(self, df: pd.DataFrame, chart_type: str, x_axis: str,
                          y_axis: str, group_by: str | None, aggregation: str) -> Dict:
        """Prepare data for different chart types."""

        if chart_type == "pie":
            # For pie charts, use value counts or aggregation
            if group_by:
                grouped = df.groupby(x_axis)[group_by].agg(aggregation)
            else:
                grouped = df[x_axis].value_counts()

            return {
                "labels": grouped.index.tolist(),
                "values": grouped.values.tolist()
            }

        elif chart_type in ["bar", "line"]:
            if group_by:
                # Multi-series chart
                pivot_df = df.pivot_table(
                    values=y_axis,
                    index=x_axis,
                    columns=group_by,
                    aggfunc=aggregation,
                    fill_value=0
                )
                return {
                    "x_values": pivot_df.index.tolist(),
                    "series": [
                        {
                            "name": col,
                            "values": pivot_df[col].tolist()
                        }
                        for col in pivot_df.columns
                    ]
                }
            else:
                # Single series
                if pd.api.types.is_numeric_dtype(df[x_axis]):
                    # For numeric x-axis, sort by x values
                    sorted_df = df.sort_values(x_axis)
                    return {
                        "x_values": sorted_df[x_axis].tolist(),
                        "y_values": sorted_df[y_axis].tolist()
                    }
                else:
                    # For categorical x-axis, aggregate
                    grouped = df.groupby(x_axis)[y_axis].agg(aggregation)
                    return {
                        "x_values": grouped.index.tolist(),
                        "y_values": grouped.values.tolist()
                    }

        elif chart_type == "scatter":
            return {
                "x_values": df[x_axis].tolist(),
                "y_values": df[y_axis].tolist(),
                "point_labels": df.index.tolist()
            }

        elif chart_type == "histogram":
            values = df[x_axis].dropna()
            hist, bins = np.histogram(values, bins=20)
            return {
                "bins": bins.tolist(),
                "frequencies": hist.tolist(),
                "bin_centers": ((bins[:-1] + bins[1:]) / 2).tolist()
            }

        elif chart_type == "box":
            if group_by:
                # Box plot by group
                groups = {}
                for group in df[group_by].unique():
                    group_data = df[df[group_by] == group][y_axis].dropna()
                    groups[str(group)] = self._calculate_box_plot_stats(group_data)
                return {"groups": groups}
            else:
                # Single box plot
                stats = self._calculate_box_plot_stats(df[y_axis].dropna())
                return {"single_group": stats}

        elif chart_type == "heatmap":
            # Create correlation matrix if both axes are numeric
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) >= 2:
                corr_matrix = df[numeric_cols].corr()
                return {
                    "matrix": corr_matrix.values.tolist(),
                    "x_labels": corr_matrix.columns.tolist(),
                    "y_labels": corr_matrix.index.tolist()
                }
            else:
                # Pivot table heatmap
                if group_by and y_axis:
                    pivot_df = df.pivot_table(
                        values=y_axis,
                        index=x_axis,
                        columns=group_by,
                        aggfunc=aggregation,
                        fill_value=0
                    )
                    return {
                        "matrix": pivot_df.values.tolist(),
                        "x_labels": pivot_df.columns.tolist(),
                        "y_labels": pivot_df.index.tolist()
                    }

        return {}

    def _calculate_box_plot_stats(self, data: pd.Series) -> Dict:
        """Calculate box plot statistics."""
        if len(data) == 0:
            return {}

        q1 = data.quantile(0.25)
        median = data.median()
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr

        outliers = data[(data < lower_fence) | (data > upper_fence)]

        return {
            "min": float(data.min()),
            "q1": float(q1),
            "median": float(median),
            "q3": float(q3),
            "max": float(data.max()),
            "outliers": outliers.tolist()
        }

    def _generate_chart_config(self, chart_type: str, x_axis: str, y_axis: str,
                             title: str, width: int, height: int, color_scheme: str, group_by: str | None) -> Dict:
        """Generate chart configuration."""

        color_schemes = {
            "default": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"],
            "viridis": ["#440154", "#31688e", "#35b779", "#fde725"],
            "plasma": ["#0d0887", "#7e03a8", "#cc4778", "#f89441", "#f0f921"],
            "cool": ["#6e8cd5", "#95b3e8", "#c4daf7", "#e8f4fd"],
            "warm": ["#ff6b6b", "#ffa726", "#ffcc02", "#66bb6a"]
        }

        return {
            "type": chart_type,
            "title": title,
            "width": width,
            "height": height,
            "colors": color_schemes.get(color_scheme, color_schemes["default"]),
            "axes": {
                "x_axis": {
                    "field": x_axis,
                    "label": x_axis.replace("_", " ").title()
                },
                "y_axis": {
                    "field": y_axis,
                    "label": y_axis.replace("_", " ").title() if y_axis else ""
                }
            },
            "grouping": {
                "field": group_by,
                "label": group_by.replace("_", " ").title() if group_by else None
            }
        }

    def _generate_insights(self, df: pd.DataFrame, chart_type: str, x_axis: str, y_axis: str) -> List[str]:
        """Generate insights about the data visualization."""
        insights = []

        # Data size insights
        if len(df) > 1000:
            insights.append(f"Large dataset with {len(df):,} records - consider sampling for better performance")

        # Chart-specific insights
        if chart_type == "bar" and y_axis:
            top_value = df.loc[df[y_axis].idxmax(), x_axis]
            insights.append(f"Highest value: {top_value}")

        elif chart_type == "line" and y_axis:
            if pd.api.types.is_numeric_dtype(df[y_axis]):
                index_series = pd.Series(range(len(df)))
                trend = "increasing" if df[y_axis].corr(index_series) > 0.1 else "decreasing" if df[y_axis].corr(index_series) < -0.1 else "stable"
                insights.append(f"Overall trend appears to be {trend}")

        elif chart_type == "scatter" and y_axis:
            if pd.api.types.is_numeric_dtype(df[x_axis]) and pd.api.types.is_numeric_dtype(df[y_axis]):
                correlation = df[x_axis].corr(df[y_axis])
                if abs(correlation) > 0.7:
                    insights.append(f"Strong {'positive' if correlation > 0 else 'negative'} correlation detected (r={correlation:.2f})")
                elif abs(correlation) > 0.3:
                    insights.append(f"Moderate {'positive' if correlation > 0 else 'negative'} correlation detected (r={correlation:.2f})")

        elif chart_type == "histogram":
            mean_val = df[x_axis].mean()
            median_val = df[x_axis].median()
            if abs(mean_val - median_val) / median_val > 0.1:
                skew_direction = "right" if mean_val > median_val else "left"
                insights.append(f"Distribution appears to be {skew_direction}-skewed")

        # Missing data insights
        if y_axis and df[y_axis].isnull().sum() > 0:
            missing_pct = (df[y_axis].isnull().sum() / len(df)) * 100
            insights.append(f"{missing_pct:.1f}% of {y_axis} values are missing")

        return insights


class DataSummaryTool(BaseTool):
    """Tool for creating comprehensive statistical summaries of data."""

    @property
    def name(self) -> str:
        return "data_summary"

    @property
    def description(self) -> str:
        return "Generate comprehensive statistical summaries and descriptive analytics for datasets"

    def get_schema(self) -> Dict:
        return {
            "properties": {
                "data": {
                    "type": "array",
                    "description": "Data to summarize (list of dictionaries)"
                },
                "include_correlations": {
                    "type": "boolean",
                    "description": "Whether to include correlation analysis",
                    "default": True
                },
                "include_distributions": {
                    "type": "boolean",
                    "description": "Whether to include distribution analysis",
                    "default": True
                },
                "group_by": {
                    "type": "string",
                    "description": "Field to group summary statistics by"
                },
                "focus_fields": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Specific fields to focus analysis on (if not provided, analyzes all)"
                }
            },
            "required": ["data"]
        }

    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute data summary generation."""
        data = kwargs.get("data")
        include_correlations = kwargs.get("include_correlations", True)
        include_distributions = kwargs.get("include_distributions", True)
        group_by = kwargs.get("group_by")
        focus_fields = kwargs.get("focus_fields")

        if not isinstance(data, list):
            raise ValueError("Data must be a list of dictionaries")

        if not data:
            return {
                "summary": "No data provided",
                "statistics": {},
                "insights": []
            }

        try:
            df = pd.DataFrame(data)

            # Filter to focus fields if specified
            if focus_fields:
                available_fields = [f for f in focus_fields if f in df.columns]
                if available_fields:
                    df = df[available_fields + ([group_by] if group_by and group_by not in available_fields else [])]

            # Generate comprehensive summary
            summary = {
                "dataset_overview": self._generate_dataset_overview(df),
                "field_summaries": self._generate_field_summaries(df),
                "data_quality": self._assess_data_quality(df)
            }

            # Add correlations if requested
            if include_correlations:
                summary["correlations"] = self._analyze_correlations(df)

            # Add distribution analysis if requested
            if include_distributions:
                summary["distributions"] = self._analyze_distributions(df)

            # Add grouped analysis if requested
            if group_by and group_by in df.columns:
                summary["grouped_analysis"] = self._generate_grouped_summary(df, group_by)

            # Generate insights
            insights = self._generate_summary_insights(summary)

            return {
                "summary": summary,
                "insights": insights,
                "recommendations": self._generate_recommendations(summary)
            }

        except Exception as e:
            raise RuntimeError(f"Failed to generate data summary: {str(e)}")

    def _generate_dataset_overview(self, df: pd.DataFrame) -> Dict:
        """Generate high-level dataset overview."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        datetime_cols = df.select_dtypes(include=['datetime64']).columns

        return {
            "total_records": len(df),
            "total_fields": len(df.columns),
            "field_types": {
                "numeric": len(numeric_cols),
                "categorical": len(categorical_cols),
                "datetime": len(datetime_cols)
            },
            "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
            "missing_values": {
                "total_missing": df.isnull().sum().sum(),
                "percentage": round((df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100, 2)
            }
        }

    def _generate_field_summaries(self, df: pd.DataFrame) -> Dict:
        """Generate detailed summaries for each field."""
        summaries = {}

        for col in df.columns:
            series = df[col]
            summary = {
                "data_type": str(series.dtype),
                "non_null_count": series.count(),
                "null_count": series.isnull().sum(),
                "unique_values": series.nunique()
            }

            if pd.api.types.is_numeric_dtype(series):
                summary.update({
                    "mean": float(series.mean()) if not series.empty else None,
                    "median": float(series.median()) if not series.empty else None,
                    "std": float(series.std()) if not series.empty else None,
                    "min": float(series.min()) if not series.empty else None,
                    "max": float(series.max()) if not series.empty else None,
                    "quartiles": {
                        "q1": float(series.quantile(0.25)) if not series.empty else None,
                        "q3": float(series.quantile(0.75)) if not series.empty else None
                    }
                })

            elif pd.api.types.is_string_dtype(series) or series.dtype == 'object':
                value_counts = series.value_counts().head(5)
                summary.update({
                    "most_common": value_counts.to_dict(),
                    "avg_length": round(series.astype(str).str.len().mean(), 2) if not series.empty else 0
                })

            summaries[col] = summary

        return summaries

    def _assess_data_quality(self, df: pd.DataFrame) -> Dict:
        """Assess overall data quality."""
        total_cells = len(df) * len(df.columns)
        missing_cells = df.isnull().sum().sum()

        # Check for duplicates
        duplicate_rows = df.duplicated().sum()

        # Check for empty strings in object columns
        empty_strings = 0
        for col in df.select_dtypes(include=['object']).columns:
            empty_strings += (df[col] == '').sum()

        quality_score = max(0, 1 - ((missing_cells + empty_strings + duplicate_rows * len(df.columns)) / total_cells))

        return {
            "quality_score": round(quality_score, 3),
            "issues": {
                "missing_values": missing_cells,
                "duplicate_rows": duplicate_rows,
                "empty_strings": empty_strings
            },
            "completeness": round((1 - missing_cells / total_cells) * 100, 2) if total_cells > 0 else 100
        }

    def _analyze_correlations(self, df: pd.DataFrame) -> Dict:
        """Analyze correlations between numeric fields."""
        numeric_df = df.select_dtypes(include=[np.number])

        if len(numeric_df.columns) < 2:
            return {"message": "Not enough numeric fields for correlation analysis"}

        corr_matrix = numeric_df.corr()

        # Find strong correlations
        strong_correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if pd.notna(corr_val) and isinstance(corr_val, (int, float)) and abs(float(corr_val)) > 0.7:
                    corr_float = float(corr_val)
                    strong_correlations.append({
                        "field1": corr_matrix.columns[i],
                        "field2": corr_matrix.columns[j],
                        "correlation": round(corr_float, 3),
                        "strength": "strong positive" if corr_float > 0.7 else "strong negative"
                    })

        return {
            "correlation_matrix": corr_matrix.round(3).to_dict(),
            "strong_correlations": strong_correlations
        }

    def _analyze_distributions(self, df: pd.DataFrame) -> Dict:
        """Analyze distributions of numeric fields."""
        distributions = {}

        for col in df.select_dtypes(include=[np.number]).columns:
            series = df[col].dropna()
            if len(series) == 0:
                continue

            # Calculate distribution statistics
            mean_val = series.mean()
            median_val = series.median()
            std_val = series.std()

            # Assess skewness
            skewness = "right-skewed" if mean_val > median_val * 1.1 else "left-skewed" if mean_val < median_val * 0.9 else "approximately normal"

            distributions[col] = {
                "distribution_type": skewness,
                "outlier_count": len(series[(series < (series.quantile(0.25) - 1.5 * (series.quantile(0.75) - series.quantile(0.25)))) |
                                         (series > (series.quantile(0.75) + 1.5 * (series.quantile(0.75) - series.quantile(0.25))))]),
                "coefficient_of_variation": round((std_val / mean_val) * 100, 2) if mean_val != 0 else None
            }

        return distributions

    def _generate_grouped_summary(self, df: pd.DataFrame, group_by: str) -> Dict:
        """Generate summary statistics grouped by a categorical field."""
        if group_by not in df.columns:
            return {"error": f"Group by field '{group_by}' not found"}

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return {"message": "No numeric fields available for grouped analysis"}

        grouped_stats = {}
        for group in df[group_by].unique():
            if pd.isna(group):
                continue

            group_data = df[df[group_by] == group][numeric_cols]
            grouped_stats[str(group)] = {
                "count": len(group_data),
                "means": group_data.mean().round(2).to_dict(),
                "medians": group_data.median().round(2).to_dict()
            }

        return grouped_stats

    def _generate_summary_insights(self, summary: Dict) -> List[str]:
        """Generate insights from the data summary."""
        insights = []

        # Dataset size insights
        total_records = summary["dataset_overview"]["total_records"]
        if total_records > 10000:
            insights.append(f"Large dataset with {total_records:,} records")
        elif total_records < 100:
            insights.append(f"Small dataset with only {total_records} records - consider gathering more data")

        # Data quality insights
        quality_score = summary["data_quality"]["quality_score"]
        if quality_score < 0.7:
            insights.append("Data quality is concerning - significant cleaning may be needed")
        elif quality_score > 0.95:
            insights.append("Excellent data quality with minimal issues")

        # Missing data insights
        missing_pct = summary["dataset_overview"]["missing_values"]["percentage"]
        if missing_pct > 20:
            insights.append(f"High percentage of missing values ({missing_pct:.1f}%) - investigate data collection process")

        # Correlation insights
        if "correlations" in summary and "strong_correlations" in summary["correlations"]:
            strong_corrs = len(summary["correlations"]["strong_correlations"])
            if strong_corrs > 0:
                insights.append(f"Found {strong_corrs} strong correlations - potential for feature engineering or redundancy")

        return insights

    def _generate_recommendations(self, summary: Dict) -> List[str]:
        """Generate actionable recommendations based on the analysis."""
        recommendations = []

        # Data quality recommendations
        quality_issues = summary["data_quality"]["issues"]
        if quality_issues["missing_values"] > 0:
            recommendations.append("Consider implementing data imputation strategies for missing values")

        if quality_issues["duplicate_rows"] > 0:
            recommendations.append("Remove duplicate rows to improve data quality")

        # Performance recommendations
        memory_mb = summary["dataset_overview"]["memory_usage_mb"]
        if memory_mb > 100:
            recommendations.append("Large dataset - consider data sampling or optimization for better performance")

        # Analysis recommendations
        numeric_fields = summary["dataset_overview"]["field_types"]["numeric"]
        if numeric_fields >= 3:
            recommendations.append("Multiple numeric fields available - consider correlation and clustering analysis")

        return recommendations


class ExportVisualizationTool(BaseTool):
    """Tool for exporting visualizations and summaries to various formats."""

    @property
    def name(self) -> str:
        return "export_visualization"

    @property
    def description(self) -> str:
        return "Export charts, summaries, and analysis results to files in various formats"

    def get_schema(self) -> Dict:
        return {
            "properties": {
                "data": {
                    "type": "array",
                    "description": "Raw data to export (list of dictionaries)"
                },
                "content": {
                    "type": "object",
                    "description": "Content to export (chart data, summary, etc.)"
                },
                "format": {
                    "type": "string",
                    "enum": ["json", "csv", "html", "markdown"],
                    "description": "Output format"
                },
                "filename": {
                    "type": "string",
                    "description": "Output filename (without extension)"
                },
                "include_metadata": {
                    "type": "boolean",
                    "description": "Whether to include metadata in export",
                    "default": True
                },
                "template": {
                    "type": "string",
                    "enum": ["minimal", "detailed", "report"],
                    "description": "Export template style",
                    "default": "detailed"
                }
            },
            "required": ["content", "format", "filename"]
        }

    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute visualization export."""
        data = kwargs.get("data")
        content = kwargs.get("content")
        format_type = kwargs.get("format")
        filename = kwargs.get("filename")
        include_metadata = kwargs.get("include_metadata", True)
        template = kwargs.get("template", "detailed")

        if not content:
            raise ValueError("Content is required for export")

        try:
            # Generate export content based on format
            if format_type == "json":
                exported_content = self._export_json(content, include_metadata)
                full_filename = f"{filename}.json"

            elif format_type == "csv":
                exported_content = self._export_csv(content, data)
                full_filename = f"{filename}.csv"

            elif format_type == "html":
                exported_content = self._export_html(content, template, data)
                full_filename = f"{filename}.html"

            elif format_type == "markdown":
                exported_content = self._export_markdown(content, template, data)
                full_filename = f"{filename}.md"

            else:
                raise ValueError(f"Unsupported format: {format_type}")

            return {
                "exported_content": exported_content,
                "filename": full_filename,
                "format": format_type,
                "size_bytes": len(exported_content.encode('utf-8')),
                "export_summary": {
                    "content_type": self._detect_content_type(content),
                    "template_used": template,
                    "metadata_included": include_metadata,
                    "total_records": len(data) if data else 0,
                    "total_fields": len(data[0].keys()) if data and data else 0,
                    "timestamp": pd.Timestamp.now().isoformat(),
                    "success": True
                }
            }

        except Exception as e:
            raise RuntimeError(f"Failed to export visualization: {str(e)}")

    def _export_json(self, content: Dict, include_metadata: bool) -> str:
        """Export content as JSON."""
        export_data = content.copy()

        if include_metadata:
            export_data["export_metadata"] = {
                "exported_at": pd.Timestamp.now().isoformat(),
                "format": "json",
                "version": "1.0"
            }

        return json.dumps(export_data, indent=2, default=str)

    def _export_csv(self, content: Dict, data: List[Dict] | None = None) -> str:
        """Export content as CSV (for tabular data)."""
        # If raw data is provided, export that first
        if data and isinstance(data, list) and data:
            df = pd.DataFrame(data)
            return df.to_csv(index=False)

        # Try to find tabular data in the content
        if "chart_data" in content:
            chart_data = content["chart_data"]
            if "x_values" in chart_data and "y_values" in chart_data:
                df = pd.DataFrame({
                    "x": chart_data["x_values"],
                    "y": chart_data["y_values"]
                })
                return df.to_csv(index=False)

        elif "summary" in content and "field_summaries" in content["summary"]:
            # Export field summaries as CSV
            summaries = content["summary"]["field_summaries"]
            rows = []
            for field, stats in summaries.items():
                row = {"field": field}
                row.update(stats)
                rows.append(row)
            df = pd.DataFrame(rows)
            return df.to_csv(index=False)

        # Fallback: convert to simple key-value CSV
        rows = []
        self._flatten_dict(content, "", rows)
        df = pd.DataFrame(rows)
        return df.to_csv(index=False)

    def _export_html(self, content: Dict, template: str, data: List[Dict] | None = None) -> str:
        """Export content as HTML."""
        if template == "minimal":
            return self._generate_minimal_html(content, data)
        elif template == "report":
            return self._generate_report_html(content, data)
        else:  # detailed
            return self._generate_detailed_html(content, data)

    def _export_markdown(self, content: Dict, template: str, data: List[Dict] | None = None) -> str:
        """Export content as Markdown."""
        if template == "minimal":
            return self._generate_minimal_markdown(content, data)
        elif template == "report":
            return self._generate_report_markdown(content, data)
        else:  # detailed
            return self._generate_detailed_markdown(content, data)

    def _generate_detailed_html(self, content: Dict, data: List[Dict] | None = None) -> str:
        """Generate detailed HTML report."""
        title = content.get("title", "Data Analysis Report")
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1, h2, h3 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .insight {{ background-color: #e7f3ff; padding: 10px; margin: 10px 0; border-left: 4px solid #2196F3; }}
            </style>
        </head>
        <body>
            <h1>{title}</h1>
        """

        # Add data table if provided
        if data and isinstance(data, list) and data:
            html += "<h2>Data Table</h2>"
            html += self._generate_html_table(data)

        if "chart_config" in content:
            html += f"<h2>Chart: {content['chart_config'].get('title', 'Visualization')}</h2>"
            html += f"<p>Type: {content['chart_config'].get('type', 'Unknown')}</p>"

        if "insights" in content:
            html += "<h2>Insights</h2>"
            for insight in content["insights"]:
                html += f'<div class="insight">{insight}</div>'

        if "summary" in content:
            html += "<h2>Summary Statistics</h2>"
            html += self._dict_to_html_table(content["summary"])

        html += "</body></html>"
        return html

    def _generate_detailed_markdown(self, content: Dict, data: List[Dict] | None = None) -> str:
        """Generate detailed Markdown report."""
        title = content.get("title", "Data Analysis Report")
        md = f"# {title}\n\n"

        # Add data table if provided
        if data and isinstance(data, list) and data:
            md += "## Data Table\n\n"
            md += self._generate_markdown_table(data)
            md += "\n"

        if "chart_config" in content:
            md += f"## Chart: {content['chart_config'].get('title', 'Visualization')}\n"
            md += f"**Type:** {content['chart_config'].get('type', 'Unknown')}\n\n"

        if "insights" in content:
            md += "## Insights\n\n"
            for insight in content["insights"]:
                md += f"- {insight}\n"
            md += "\n"

        if "summary" in content:
            md += "## Summary Statistics\n\n"
            md += self._dict_to_markdown_table(content["summary"])

        return md

    def _generate_html_table(self, data: List[Dict]) -> str:
        """Generate an HTML table from data."""
        if not data:
            return ""

        html = "<table>\n<thead>\n<tr>"
        for key in data[0].keys():
            html += f"<th>{key}</th>"
        html += "</tr>\n</thead>\n<tbody>\n"

        for row in data:
            html += "<tr>"
            for value in row.values():
                html += f"<td>{value}</td>"
            html += "</tr>\n"
        html += "</tbody>\n</table>\n"
        return html

    def _generate_markdown_table(self, data: List[Dict]) -> str:
        """Generate a Markdown table from data."""
        if not data:
            return ""

        # Get headers
        headers = list(data[0].keys())

        # Create table header
        md = "|" + "|".join(headers) + "|\n"
        md += "|" + "|".join(["---"] * len(headers)) + "|\n"

        # Add rows
        for row in data:
            md += "|" + "|".join(str(row.get(h, "")) for h in headers) + "|\n"

        return md

    def _generate_minimal_html(self, content: Dict, data: List[Dict] | None = None) -> str:
        """Generate minimal HTML output."""
        return f"<html><body><pre>{json.dumps(content, indent=2)}</pre></body></html>"

    def _generate_minimal_markdown(self, content: Dict, data: List[Dict] | None = None) -> str:
        """Generate minimal Markdown output."""
        return f"```json\n{json.dumps(content, indent=2)}\n```"

    def _generate_report_html(self, content: Dict, data: List[Dict] | None = None) -> str:
        """Generate comprehensive report-style HTML."""
        # This would be a more elaborate report format
        return self._generate_detailed_html(content, data)  # For now, same as detailed

    def _generate_report_markdown(self, content: Dict, data: List[Dict] | None = None) -> str:
        """Generate comprehensive report-style Markdown."""
        # This would be a more elaborate report format
        return self._generate_detailed_markdown(content, data)  # For now, same as detailed

    def _dict_to_html_table(self, data: Dict, max_depth: int = 2) -> str:
        """Convert dictionary to HTML table."""
        html = "<table>"
        for key, value in data.items():
            html += f"<tr><th>{key}</th><td>"
            if isinstance(value, dict) and max_depth > 0:
                html += self._dict_to_html_table(value, max_depth - 1)
            else:
                html += str(value)
            html += "</td></tr>"
        html += "</table>"
        return html

    def _dict_to_markdown_table(self, data: Dict) -> str:
        """Convert dictionary to Markdown table."""
        if not data:
            return ""

        md = "| Key | Value |\n|-----|-------|\n"
        for key, value in data.items():
            if isinstance(value, dict):
                value_str = json.dumps(value, indent=2)
            else:
                value_str = str(value)
            md += f"| {key} | {value_str} |\n"
        return md

    def _flatten_dict(self, d: Dict, prefix: str, rows: List) -> None:
        """Flatten nested dictionary for CSV export."""
        for key, value in d.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                self._flatten_dict(value, full_key, rows)
            else:
                rows.append({"key": full_key, "value": str(value)})

    def _detect_content_type(self, content: Dict) -> str:
        """Detect the type of content being exported."""
        if "chart_data" in content and "chart_config" in content:
            return "chart"
        elif "summary" in content and "insights" in content:
            return "data_summary"
        else:
            return "generic"