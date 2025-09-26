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

from loguru import logger
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

# Initialize configuration
WORKSPACE_PATH = os.getenv("WORKSPACE_PATH", "/Users/huzaifashaikh/Local Documents")

# Create MCP server
server = Server("engineer-your-data")

# Initialize tool registry
def initialize_tools():
    """Initialize and register all available tools."""
    logger.info("Initializing tools...")

    # Define tools to register
    tools_to_register = [
        ReadFileTool,
        WriteFileTool,
        ListFilesTool,
        FileInfoTool,
        ValidateSchemaTool,
        CheckNullsTool,
        DataQualityReportTool,
        DetectDuplicatesTool
    ]

    # Register tools if not already registered
    for tool_class in tools_to_register:
        tool_instance = tool_class()
        tool_name = tool_instance.name

        if tool_name not in registry.list_tools():
            registry.register_tool(tool_class)
        else:
            logger.debug(f"Tool '{tool_name}' already registered, skipping")

    logger.info(f"Registered {len(registry.list_tools())} tools: {', '.join(registry.list_tools())}")

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
        logger.info(f"Executing tool: {name} with args: {arguments}")

        # Execute tool using registry
        result = await registry.execute_tool(name, **arguments)

        # Format result as JSON for better readability
        formatted_result = json.dumps(result, indent=2, default=str)

        logger.info(f"Tool {name} executed successfully")
        return [TextContent(type="text", text=formatted_result)]

    except KeyError:
        error_msg = f"Tool '{name}' not found. Available tools: {', '.join(registry.list_tools())}"
        logger.error(error_msg)
        return [TextContent(type="text", text=json.dumps({"error": error_msg}, indent=2))]

    except Exception as e:
        error_msg = f"Error executing {name}: {str(e)}"
        logger.error(error_msg)
        return [TextContent(type="text", text=json.dumps({"error": error_msg}, indent=2))]

async def main():
    """
    Main entry point for the MCP server.
    """
    logger.info("Starting Engineer Your Data MCP Server")
    logger.info(f"Workspace path: {WORKSPACE_PATH}")
    
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())
