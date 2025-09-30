# Engineer Your Data

[![MCP Registry](https://img.shields.io/badge/MCP%20Registry-Published-blue)](https://registry.modelcontextprotocol.io/v0/servers?search=engineer-your-data)
[![PyPI](https://img.shields.io/pypi/v/engineer-your-data)](https://pypi.org/project/engineer-your-data/)

A Model Context Protocol (MCP) server designed specifically for **data engineers** and **business intelligence professionals**. Transform your data pipelines and BI workflows with AI-assisted data engineering capabilities that run locally without internet dependency.

## Why Engineer Your Data?

Built from the ground up for data engineering teams and BI analysts who need:
- **Pipeline Development** - Build and test ETL/ELT transformations
- **Data Quality Assurance** - Profile and validate data sources
- **Business Intelligence** - Create analytics models and dashboard visualizations
- **Local Control** - Keep sensitive data on-premises with no cloud dependencies

## 🚀 Quick Start

**New to Engineer Your Data?** Start with these 5 essential operations:

1. **Check Data Quality**: `"Generate a data quality report for my sales.csv file"`
2. **Find Issues**: `"Check for null values in the customer_data.csv"`
3. **Transform Data**: `"Filter the orders.csv for rows where status is 'completed'"`
4. **Visualize**: `"Create a bar chart showing sales by region from revenue.csv"`
5. **Summarize**: `"Give me a statistical summary of the dataset"`

These cover 80% of daily data engineering tasks. Explore the full capabilities below!

## Core Capabilities

🚀 **File Operations**:
- `read_file` - Read data files from local filesystem
- `write_file` - Write processed data to files
- `list_files` - Browse and discover data files
- `file_info` - Get metadata about data files

📊 **Data Validation & Quality**:
- `validate_schema` - Validate data against expected schemas
- `check_nulls` - Analyze null values and missing data patterns
- `data_quality_report` - Comprehensive data quality assessment
- `detect_duplicates` - Find duplicate records with configurable matching

🔄 **Data Transformation**:
- `filter_data` - Filter datasets based on conditions
- `aggregate_data` - Group and aggregate data with statistical functions
- `join_data` - Join multiple datasets with flexible join types
- `pivot_data` - Reshape data from long to wide format
- `clean_data` - Clean and standardize data values

📈 **Visualization & Analysis**:
- `create_chart` - Generate bar, pie, line, scatter, histogram, box, and heatmap charts
- `data_summary` - Create comprehensive dataset summaries with statistics
- `export_visualization` - Export charts and data to JSON, CSV, HTML, Markdown

🌐 **API Integration**:
- `fetch_api_data` - Retrieve data from REST APIs
- `monitor_api` - Monitor API endpoints for health and performance
- `batch_api_calls` - Execute multiple API calls efficiently
- `api_auth` - Manage API authentication

🔧 **Utilities**:
- `chain_tools` - Execute multiple tools in sequence
- `analyze_schema` - Analyze and understand data schemas

## Quick Start for Data Teams

### Installation

```bash
# Option 1: Install from PyPI (recommended)
pip install engineer-your-data

# Option 2: Install from source
git clone https://github.com/eghuzefa/engineer-your-data-mcp.git
cd engineer-your-data-mcp
pip install -e .
```

### Configure for Your Data Environment

**For PyPI Installation:**
Add to your Claude Desktop configuration (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "engineer-your-data": {
      "command": "python",
      "args": ["-m", "src.server"],
      "env": {
        "WORKSPACE_PATH": "/path/to/your/data/workspace"
      }
    }
  }
}
```

**For Source Installation:**
Add to your Claude Desktop configuration (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "engineer-your-data": {
      "command": "python",
      "args": ["/path/to/engineer-your-data-mcp/src/server.py"],
      "env": {
        "WORKSPACE_PATH": "/path/to/your/data/workspace"
      }
    }
  }
}
```

### Data Engineering Examples

**Data Quality Analysis:**
```
"Check the customer data for null values and duplicates"
"Generate a comprehensive data quality report for the sales dataset"
"Validate this CSV file against our customer schema"
```

**Data Transformation:**
```
"Filter the orders data for customers in the US region"
"Aggregate sales data by month and calculate total revenue"
"Join customer data with order data on customer_id"
"Pivot the sales data to show products as columns"
```

**Visualization & Reporting:**
```
"Create a bar chart showing revenue by department"
"Generate a summary of the dataset with key statistics"
"Export the sales analysis as an HTML report"
```

**API Data Integration:**
```
"Fetch customer data from the CRM API"
"Monitor the data pipeline API for health status"
"Authenticate with the analytics API using OAuth"
```

## Architecture for Data Teams

```
Claude Desktop → MCP Protocol → Engineer Your Data → Local Python Environment
                                        ↓
                    pandas + numpy + requests + matplotlib
                                        ↓
                         Local Files + APIs + Data Sources
```

## Testing & Quality

- **161 comprehensive tests** with 100% pass rate
- **Async/await support** for high-performance operations
- **Error handling** with detailed logging and debugging
- **Type safety** with proper schema validation

```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=src

# Run specific tool tests
python -m pytest tests/tools/test_visualization.py
```

## Available Tools (17 Total)

### File Operations (4 tools)
| Tool | Description |
|------|-------------|
| `read_file` | Read and parse data files (CSV, JSON, etc.) |
| `write_file` | Write data to files with format options |
| `list_files` | Directory browsing and file discovery |
| `file_info` | File metadata and basic statistics |

### Data Validation (4 tools)
| Tool | Description |
|------|-------------|
| `validate_schema` | Schema validation with custom rules |
| `check_nulls` | Null value analysis and patterns |
| `data_quality_report` | Comprehensive quality assessment |
| `detect_duplicates` | Duplicate detection with flexible matching |

### Data Transformation (5 tools)
| Tool | Description |
|------|-------------|
| `filter_data` | Advanced filtering with conditions |
| `aggregate_data` | Grouping and statistical aggregation |
| `join_data` | Multi-dataset joins (inner, outer, left, right) |
| `pivot_data` | Data reshaping and pivoting |
| `clean_data` | Data cleaning and standardization |

### Visualization (3 tools)
| Tool | Description |
|------|-------------|
| `create_chart` | 7 chart types with customization |
| `data_summary` | Statistical summaries and insights |
| `export_visualization` | Multi-format export capabilities |

### API Integration (4 tools)
| Tool | Description |
|------|-------------|
| `fetch_api_data` | REST API data retrieval |
| `monitor_api` | API health monitoring |
| `batch_api_calls` | Efficient bulk API operations |
| `api_auth` | Authentication management |

## Data Engineering Best Practices

- **Sandboxed Execution** - Safe environment for testing transformations
- **Local Data Control** - Keep sensitive data on your infrastructure
- **Comprehensive Testing** - All tools thoroughly tested and validated
- **Enterprise Security** - No external API calls for core functionality
- **Performance Optimized** - Async operations and efficient data processing

## Integration with Your Stack

Works seamlessly alongside:
- **dbt** - Use for complex transformation logic development
- **Airflow/Prefect** - Incorporate into existing workflow orchestration
- **Jupyter/Notebooks** - Prototype and iterate on data transformations
- **BI Tools** - Generate data and visualizations for Tableau, Power BI, etc.
- **APIs** - Integrate with REST APIs and microservices

## Contributing

Data engineers and BI professionals welcome! Please read our contributing guidelines and submit PRs for new data connectors, transformations, or BI features.

## MCP Registry

<!-- MCP name format: mcp-name: io.github.eghuzefa/engineer-your-data -->

This server is available in the official [Model Context Protocol Registry](https://registry.modelcontextprotocol.io).

## License

MIT License - see LICENSE file for details.
