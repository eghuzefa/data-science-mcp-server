# How to Use Engineer Your Data with Claude Desktop

## Simple 3-Step Setup

### Step 1: Install the Package

```bash
pip install engineer-your-data
```

That's it! No cloning repos, no virtual environments needed.

### Step 2: Configure Claude Desktop

**Find your Claude Desktop config file:**
- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

**Add this configuration:**

```json
{
  "mcpServers": {
    "engineer-your-data": {
      "command": "python3",
      "args": [
        "-m",
        "src"
      ],
      "env": {
        "WORKSPACE_PATH": "~/Documents"
      }
    }
  }
}
```

**Change the WORKSPACE_PATH** to wherever you keep your data files.

### Step 3: Restart Claude Desktop

- Quit Claude Desktop completely (Cmd + Q on Mac)
- Reopen Claude Desktop
- Done!

## Test It Works

Try these commands in Claude:

```
List the files in my workspace
```

```
Create a sample CSV with product, price, quantity columns
```

```
Generate a data quality report for that file
```

## Common Issues

### "spawn python3 ENOENT" Error

Your system doesn't have `python3`. Try finding your Python path:

```bash
which python3
# or
which python
```

Then use the full path in your config:

```json
{
  "mcpServers": {
    "engineer-your-data": {
      "command": "/usr/bin/python3",
      "args": ["-m", "src"],
      "env": {
        "WORKSPACE_PATH": "~/Documents"
      }
    }
  }
}
```

### Module Not Found Errors

Make sure you installed the package:

```bash
pip install --upgrade engineer-your-data
```

If using a virtual environment, use the full path to that Python:

```json
{
  "command": "/path/to/your/venv/bin/python3"
}
```

### Still Not Working?

Check the Claude Desktop logs:
```bash
# macOS
tail -f ~/Library/Logs/Claude/mcp*.log
```

Look for error messages and share them in a GitHub issue.

## What You Can Do

- **Data Quality**: Check nulls, find duplicates, validate schemas
- **Transform Data**: Filter, aggregate, join, pivot, clean
- **Visualize**: Create charts and statistical summaries
- **APIs**: Fetch data from REST APIs, monitor endpoints

## Need Help?

- **GitHub Issues**: https://github.com/eghuzefa/engineer-your-data-mcp/issues
- **Documentation**: Check the main README for all available tools
