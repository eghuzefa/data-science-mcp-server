"""
Tool chaining support for executing multiple data tools in sequence.
"""

from typing import Any, Dict, List
from .base import BaseTool
from .data_transformation import FilterDataTool, AggregateDataTool, JoinDataTool, PivotDataTool, CleanDataTool


class ToolChainExecutor(BaseTool):
    """Tool for executing multiple data transformation tools in sequence."""

    @property
    def name(self) -> str:
        return "execute_tool_chain"

    @property
    def description(self) -> str:
        return "Execute multiple data transformation tools in sequence, passing output from one to the next"

    def get_schema(self) -> Dict:
        return {
            "properties": {
                "data": {
                    "type": "array",
                    "description": "Initial data to process (list of dictionaries)"
                },
                "chain": {
                    "type": "array",
                    "description": "List of tools to execute in sequence",
                    "items": {
                        "type": "object",
                        "properties": {
                            "tool": {
                                "type": "string",
                                "enum": ["filter_data", "aggregate_data", "join_data", "pivot_data", "clean_data"],
                                "description": "Tool to execute"
                            },
                            "parameters": {
                                "type": "object",
                                "description": "Parameters for the tool (data will be auto-injected)"
                            },
                            "output_key": {
                                "type": "string",
                                "description": "Key to extract data from tool output (auto-detected if not specified)"
                            }
                        },
                        "required": ["tool", "parameters"]
                    }
                },
                "validate_each_step": {
                    "type": "boolean",
                    "description": "Whether to validate data quality at each step",
                    "default": False
                }
            },
            "required": ["data", "chain"]
        }

    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute tool chain operation."""
        data = kwargs.get("data")
        chain = kwargs.get("chain")
        validate_each_step = kwargs.get("validate_each_step", False)

        if not isinstance(data, list):
            raise ValueError("Data must be a list of dictionaries")

        if not chain:
            raise ValueError("Chain must contain at least one tool")

        # Tool registry
        tools = {
            "filter_data": FilterDataTool(),
            "aggregate_data": AggregateDataTool(),
            "join_data": JoinDataTool(),
            "pivot_data": PivotDataTool(),
            "clean_data": CleanDataTool()
        }

        # Output key mapping for each tool
        output_keys = {
            "filter_data": "filtered_data",
            "aggregate_data": "aggregated_data",
            "join_data": "joined_data",
            "pivot_data": "pivoted_data",
            "clean_data": "cleaned_data"
        }

        current_data = data
        step_results = []

        try:
            for i, step in enumerate(chain):
                tool_name = step["tool"]
                parameters = step["parameters"].copy()
                custom_output_key = step.get("output_key")

                if tool_name not in tools:
                    raise ValueError(f"Unknown tool: {tool_name}")

                tool = tools[tool_name]

                # Inject current data
                parameters["data"] = current_data

                # Add validation if requested
                if validate_each_step and tool_name in ["filter_data"]:
                    parameters["validate_output"] = True

                # Execute tool
                result = await tool.safe_execute(**parameters)

                if not result["success"]:
                    raise RuntimeError(f"Tool {tool_name} failed at step {i+1}: {result.get('error', 'Unknown error')}")

                # Extract data for next step
                output_key = custom_output_key or output_keys.get(tool_name)
                if output_key and output_key in result["result"]:
                    current_data = result["result"][output_key]
                else:
                    # Fallback: try to find data in result
                    if "data" in result["result"]:
                        current_data = result["result"]["data"]
                    else:
                        raise RuntimeError(f"Could not extract data from {tool_name} output")

                step_results.append({
                    "step": i + 1,
                    "tool": tool_name,
                    "parameters": parameters,
                    "result_summary": {
                        "success": result["success"],
                        "records_out": len(current_data) if isinstance(current_data, list) else 1,
                        "output_key": output_key
                    },
                    "validation": result["result"].get("validation") if validate_each_step else None
                })

            return {
                "final_data": current_data,
                "original_count": len(data),
                "final_count": len(current_data) if isinstance(current_data, list) else 1,
                "steps_executed": len(chain),
                "step_results": step_results,
                "chain_summary": {
                    "tools_used": [step["tool"] for step in chain],
                    "data_reduction": len(data) - (len(current_data) if isinstance(current_data, list) else 1),
                    "success": True
                }
            }

        except Exception as e:
            raise RuntimeError(f"Tool chain execution failed: {str(e)}")