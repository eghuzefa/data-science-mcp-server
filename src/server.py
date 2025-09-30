#!/usr/bin/env python3

"""
Engineer Your Data MCP Server

Provides data engineering and BI capabilities through MCP protocol.
Enables AI assistants to ingest, transform, and analyze data for business intelligence.
"""

import asyncio
import json
import os
from typing import Any, Dict, List

from .utils.logging import mcp_logger
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
    ImageContent
)

# Import our modular tools
from .tools.registry import registry
from .tools.file_operations import ReadFileTool, WriteFileTool, ListFilesTool, FileInfoTool
from .tools.data_validation import ValidateSchemaTool, CheckNullsTool, DataQualityReportTool, DetectDuplicatesTool
from .tools.data_transformation import FilterDataTool, AggregateDataTool, JoinDataTool, PivotDataTool, CleanDataTool
from .tools.tool_chaining import ToolChainExecutor
from .tools.schema_introspection import DataSchemaAnalyzer
from .tools.api_client import FetchApiDataTool, MonitorApiTool, BatchApiCallsTool, ApiAuthTool
from .tools.visualization import CreateChartTool, DataSummaryTool, ExportVisualizationTool


# Initialize configuration
WORKSPACE_PATH = os.getenv("WORKSPACE_PATH", os.path.expanduser("~/Documents"))

# Create MCP server
server = Server("engineer-your-data")

# Initialize tool registry
def initialize_tools():
    """Initialize and register all available tools."""
    mcp_logger.info("Initializing tools...")

    # Define tools to register
    tools_to_register = [
        # File Operations
        ReadFileTool,
        WriteFileTool,
        ListFilesTool,
        FileInfoTool,
        # Data Validation
        ValidateSchemaTool,
        CheckNullsTool,
        DataQualityReportTool,
        DetectDuplicatesTool,
        # Data Transformation
        FilterDataTool,
        AggregateDataTool,
        JoinDataTool,
        PivotDataTool,
        CleanDataTool,
        # Tool Chaining
        ToolChainExecutor,
        # Schema Introspection
        DataSchemaAnalyzer,
        # API Client Tools
        FetchApiDataTool,
        MonitorApiTool,
        BatchApiCallsTool,
        ApiAuthTool,
        # Visualization Tools
        CreateChartTool,
        DataSummaryTool,
        ExportVisualizationTool
    ]

    # Register tools if not already registered
    for tool_class in tools_to_register:
        tool_instance = tool_class()
        tool_name = tool_instance.name

        if tool_name not in registry.list_tools():
            registry.register_tool(tool_class)
        else:
            mcp_logger.debug(f"Tool '{tool_name}' already registered, skipping")

    mcp_logger.info(f"Registered {len(registry.list_tools())} tools: {', '.join(registry.list_tools())}")

# Initialize tools at startup
initialize_tools()


@server.list_tools()
async def list_tools() -> List[Tool]:
    """
    List available data engineering and BI tools.
    """
    # Get tool definitions from registry
    tool_definitions = registry.get_mcp_tool_definitions()

    # Convert to MCP Tool objects
    tools = []
    for tool_def in tool_definitions:
        tools.append(Tool(
            name=tool_def["name"],
            description=tool_def["description"],
            inputSchema=tool_def["inputSchema"]
        ))

    return tools

@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent | ImageContent]:
    """
    Execute the requested tool with given arguments.
    """
    try:
        mcp_logger.log_tool_execution(name, "start", arguments=arguments)

        # Execute tool using registry
        result = await registry.execute_tool(name, **arguments)

        # Format result as JSON for better readability
        formatted_result = json.dumps(result, indent=2, default=str)

        mcp_logger.log_tool_execution(name, "success", result_size=len(formatted_result))
        return [TextContent(type="text", text=formatted_result)]

    except KeyError:
        error_msg = f"Tool '{name}' not found. Available tools: {', '.join(registry.list_tools())}"
        mcp_logger.error(error_msg)
        return [TextContent(type="text", text=json.dumps({"error": error_msg}, indent=2))]

    except Exception as e:
        error_msg = f"Error executing {name}: {str(e)}"
        mcp_logger.log_error_with_context(e, {"tool": name, "arguments": arguments})
        return [TextContent(type="text", text=json.dumps({"error": error_msg}, indent=2))]

async def main():
    """
    Main entry point for the MCP server.
    """
    mcp_logger.info("Starting Engineer Your Data MCP Server")
    mcp_logger.info(f"Workspace path: {WORKSPACE_PATH}")

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())
