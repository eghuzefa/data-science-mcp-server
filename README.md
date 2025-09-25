# Engineer Your Data

A Model Context Protocol (MCP) server designed specifically for **data engineers** and **business intelligence professionals**. Transform your data pipelines and BI workflows with AI-assisted data engineering capabilities that run locally without internet dependency.

## Why Engineer Your Data?

Built from the ground up for data engineering teams and BI analysts who need:
- **Pipeline Development** - Build and test ETL/ELT transformations
- **Data Quality Assurance** - Profile and validate data sources
- **Business Intelligence** - Create analytics models and dashboard visualizations
- **Local Control** - Keep sensitive data on-premises with no cloud dependencies

## Core Capabilities

🚀 **Data Engineering Tools**:
- `ingest_data_source` - Connect to and ingest from multiple data sources
- `transform_dataset` - Apply data transformations and cleaning operations
- `profile_data_source` - Comprehensive data quality profiling and validation
- `export_results` - Output processed data to various formats

📊 **Business Intelligence Tools**:
- `run_analytics_model` - Execute statistical analysis and predictive models
- `compute_metrics` - Calculate KPIs and business metrics
- `create_dashboard_chart` - Generate publication-ready charts and visualizations

🎯 **Target Audience**:
- **Data Engineers** - ETL pipeline development and data integration
- **BI Analysts** - Dashboard creation and business reporting
- **Analytics Engineers** - dbt-style transformations and modeling
- **Data Platform Teams** - Data quality monitoring and governance

## Quick Start for Data Teams

### Installation

```bash
# Clone the repository
git clone https://github.com/eghuzefa/engineer-your-data.git
cd engineer-your-data

# Install dependencies
pip install -r requirements.txt
```

### Configure for Your Data Environment

Add to your Claude Desktop configuration (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "engineer-your-data": {
      "command": "python",
      "args": ["/path/to/engineer-your-data/src/server.py"],
      "env": {
        "WORKSPACE_PATH": "/path/to/your/data/workspace"
      }
    }
  }
}
```

### Data Engineering Examples

**ETL Pipeline Development:**
```
"Ingest the customer_orders.csv file and profile the data quality"
"Transform the dataset to clean null values in the email column"
"Export the cleaned data as parquet format for the data warehouse"
```

**Business Intelligence Analysis:**
```
"Create a dashboard chart showing monthly revenue trends"
"Run analytics model to identify customer segments using clustering"
"Compute key metrics like customer lifetime value from this dataset"
```

**Data Quality Monitoring:**
```
"Profile this data source and identify any data quality issues"
"Check for duplicate records and missing values across all columns"
"Generate a data quality report for the monthly data refresh"
```

## Architecture for Data Teams

```
Claude Desktop → MCP Protocol → Engineer Your Data → Local Python Environment
                                        ↓
                          pandas (ETL) + scikit-learn (Analytics)
                                        ↓
                              Your Data Warehouse/Lake
```

## Available Tools

### Data Ingestion & Integration
- **`ingest_data_source`** - Load from CSV, Excel, Parquet, and database sources
- **`profile_data_source`** - Data profiling, schema detection, and quality assessment

### Data Transformation & Processing
- **`transform_dataset`** - ETL operations, data cleaning, and feature engineering
- **`compute_metrics`** - Business KPIs, aggregations, and calculations

### Analytics & Business Intelligence
- **`run_analytics_model`** - Statistical analysis, ML models, and predictive analytics
- **`create_dashboard_chart`** - BI visualizations, reports, and dashboard components
- **`export_results`** - Output to data warehouse, BI tools, or reporting formats

## Data Engineering Best Practices

- **Sandboxed Execution** - Safe environment for testing transformations
- **Local Data Control** - Keep sensitive data on your infrastructure
- **Version Control Ready** - All operations logged and reproducible
- **Enterprise Security** - No external API calls or data sharing

## Integration with Your Stack

Works seamlessly alongside:
- **dbt** - Use for complex transformation logic development
- **Airflow/Prefect** - Incorporate into existing workflow orchestration
- **Jupyter/Notebooks** - Prototype and iterate on data transformations
- **BI Tools** - Generate data and visualizations for Tableau, Power BI, etc.

## Contributing

Data engineers and BI professionals welcome! Please read our contributing guidelines and submit PRs for new data connectors, transformations, or BI features.

## License

MIT License - see LICENSE file for details.