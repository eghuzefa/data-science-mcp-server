# Data Science MCP Server

A Model Context Protocol (MCP) server that provides local data science capabilities to AI assistants. Execute pandas, sklearn, numpy operations directly without internet dependency.

## Features

🔧 **Core Tools**:
- Execute pandas operations on local datasets
- Run scikit-learn ML analyses 
- Perform numpy mathematical computations
- Generate matplotlib/seaborn visualizations
- Load/save data in multiple formats (CSV, Excel, Parquet)
- Get dataset metadata and profiling

🎯 **Target Users**:
- Data Engineers
- Data Analysts  
- Business Intelligence Analysts
- Machine Learning Engineers

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/eghuzefa/data-science-mcp-server.git
cd data-science-mcp-server

# Install dependencies
pip install -r requirements.txt
```

### Usage

1. **Configure Claude Desktop** (add to `claude_desktop_config.json`):
```json
{
  "mcpServers": {
    "data-science": {
      "command": "python",
      "args": ["/path/to/data-science-mcp-server/src/server.py"],
      "env": {
        "WORKSPACE_PATH": "/Users/huzaifashaikh/Local Documents"
      }
    }
  }
}
```

2. **Start using in Claude**:
```
"Load my sales_data.csv and show me the correlation matrix"
"Run a quick data quality check on this dataset"
"Create clusters from this customer data using K-means"
```

## Architecture

```
Claude Desktop → MCP Protocol → Data Science Server → Local Python Environment
                                      ↓
                               pandas/sklearn/numpy
                                      ↓
                                Local File System
```

## Available Tools

- `load_dataset` - Load CSV, Excel, Parquet files
- `execute_pandas` - Run pandas operations
- `execute_sklearn` - ML analysis and modeling
- `execute_numpy` - Mathematical computations
- `create_visualization` - Generate plots
- `save_results` - Export analysis results
- `get_dataset_info` - Dataset profiling and metadata

## Security

- Sandboxed execution environment
- Restricted file system access to configured workspace
- No internet access required
- Code execution logging

## Contributing

Contributions welcome! Please read our contributing guidelines and submit PRs.

## License

MIT License - see LICENSE file for details.
