"""
API client tools for external data fetching and monitoring.
"""

import asyncio
import json
import time
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urljoin, urlparse

import aiohttp
from src.tools.base import BaseTool
from src.utils.decorators import log_execution_time
from src.utils.logging import mcp_logger


class FetchApiDataTool(BaseTool):
    """Fetch data from external APIs with configurable parameters."""

    @property
    def name(self) -> str:
        return "fetch_api_data"

    @property
    def description(self) -> str:
        return "Fetch data from external APIs with support for various HTTP methods, headers, and authentication"

    def get_schema(self) -> Dict:
        return {
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The API endpoint URL to fetch data from"
                },
                "method": {
                    "type": "string",
                    "enum": ["GET", "POST", "PUT", "PATCH", "DELETE"],
                    "default": "GET",
                    "description": "HTTP method to use"
                },
                "headers": {
                    "type": "object",
                    "description": "Request headers as key-value pairs",
                    "additionalProperties": {"type": "string"}
                },
                "params": {
                    "type": "object",
                    "description": "URL query parameters as key-value pairs",
                    "additionalProperties": {"type": "string"}
                },
                "data": {
                    "type": "object",
                    "description": "Request body data (for POST/PUT/PATCH)"
                },
                "timeout": {
                    "type": "number",
                    "default": 30,
                    "description": "Request timeout in seconds"
                },
                "auth_type": {
                    "type": "string",
                    "enum": ["none", "bearer", "basic", "api_key"],
                    "default": "none",
                    "description": "Authentication type"
                },
                "auth_token": {
                    "type": "string",
                    "description": "Authentication token/key (required if auth_type is not 'none')"
                },
                "auth_username": {
                    "type": "string",
                    "description": "Username for basic auth"
                },
                "auth_password": {
                    "type": "string",
                    "description": "Password for basic auth"
                },
                "api_key_header": {
                    "type": "string",
                    "default": "X-API-Key",
                    "description": "Header name for API key authentication"
                },
                "verify_ssl": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether to verify SSL certificates"
                }
            },
            "required": ["url"]
        }

    def validate_input(self, **kwargs) -> Dict:
        """Enhanced validation for API requests."""
        validated = super().validate_input(**kwargs)

        # Validate URL
        try:
            parsed = urlparse(validated["url"])
            if not all([parsed.scheme, parsed.netloc]):
                raise ValueError("Invalid URL format")
        except Exception:
            raise ValueError("Invalid URL format")

        # Validate auth requirements
        auth_type = validated.get("auth_type", "none")
        if auth_type == "bearer" and not validated.get("auth_token"):
            raise ValueError("auth_token is required for bearer authentication")
        elif auth_type == "basic" and not all([validated.get("auth_username"), validated.get("auth_password")]):
            raise ValueError("auth_username and auth_password are required for basic authentication")
        elif auth_type == "api_key" and not validated.get("auth_token"):
            raise ValueError("auth_token is required for API key authentication")

        return validated

    @log_execution_time("fetch_api_data")
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the API request."""
        url = kwargs["url"]
        method = kwargs.get("method", "GET").upper()
        headers = kwargs.get("headers", {}).copy()
        params = kwargs.get("params", {})
        data = kwargs.get("data")
        timeout = kwargs.get("timeout", 30)
        auth_type = kwargs.get("auth_type", "none")
        verify_ssl = kwargs.get("verify_ssl", True)

        # Setup authentication
        auth = None
        if auth_type == "bearer":
            headers["Authorization"] = f"Bearer {kwargs['auth_token']}"
        elif auth_type == "basic":
            auth = aiohttp.BasicAuth(kwargs["auth_username"], kwargs["auth_password"])
        elif auth_type == "api_key":
            api_key_header = kwargs.get("api_key_header", "X-API-Key")
            headers[api_key_header] = kwargs["auth_token"]

        # Prepare request data
        request_kwargs = {
            "timeout": aiohttp.ClientTimeout(total=timeout),
            "headers": headers,
            "params": params,
            "auth": auth,
            "ssl": verify_ssl
        }

        if method in ["POST", "PUT", "PATCH"] and data:
            if isinstance(data, dict):
                request_kwargs["json"] = data
                headers.setdefault("Content-Type", "application/json")
            else:
                request_kwargs["data"] = data

        self.logger.info(f"Making {method} request to {url}")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.request(method, url, **request_kwargs) as response:
                    # Get response data
                    response_text = await response.text()

                    # Try to parse as JSON, fallback to text
                    try:
                        response_data = json.loads(response_text)
                    except json.JSONDecodeError:
                        response_data = response_text

                    result = {
                        "status_code": response.status,
                        "headers": dict(response.headers),
                        "data": response_data,
                        "url": str(response.url),
                        "method": method,
                        "success": 200 <= response.status < 300
                    }

                    if not result["success"]:
                        self.logger.warning(f"API request failed with status {response.status}")

                    return result

        except asyncio.TimeoutError:
            raise RuntimeError(f"Request timeout after {timeout} seconds")
        except aiohttp.ClientError as e:
            raise RuntimeError(f"Request failed: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error: {str(e)}")


class MonitorApiTool(BaseTool):
    """Monitor API health and performance metrics."""

    @property
    def name(self) -> str:
        return "monitor_api"

    @property
    def description(self) -> str:
        return "Monitor API endpoints for health, response times, and availability"

    def get_schema(self) -> Dict:
        return {
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The API endpoint URL to monitor"
                },
                "method": {
                    "type": "string",
                    "enum": ["GET", "POST", "HEAD"],
                    "default": "GET",
                    "description": "HTTP method for health check"
                },
                "headers": {
                    "type": "object",
                    "description": "Request headers for the health check",
                    "additionalProperties": {"type": "string"}
                },
                "timeout": {
                    "type": "number",
                    "default": 10,
                    "description": "Request timeout in seconds"
                },
                "expected_status": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "default": [200],
                    "description": "List of expected successful status codes"
                },
                "check_content": {
                    "type": "boolean",
                    "default": False,
                    "description": "Whether to include response content in monitoring"
                },
                "auth_type": {
                    "type": "string",
                    "enum": ["none", "bearer", "api_key"],
                    "default": "none",
                    "description": "Authentication type"
                },
                "auth_token": {
                    "type": "string",
                    "description": "Authentication token/key"
                },
                "api_key_header": {
                    "type": "string",
                    "default": "X-API-Key",
                    "description": "Header name for API key authentication"
                }
            },
            "required": ["url"]
        }

    @log_execution_time("monitor_api")
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the API monitoring check."""
        url = kwargs["url"]
        method = kwargs.get("method", "GET").upper()
        headers = kwargs.get("headers", {}).copy()
        timeout = kwargs.get("timeout", 10)
        expected_status = kwargs.get("expected_status", [200])
        check_content = kwargs.get("check_content", False)
        auth_type = kwargs.get("auth_type", "none")

        # Setup authentication
        if auth_type == "bearer" and kwargs.get("auth_token"):
            headers["Authorization"] = f"Bearer {kwargs['auth_token']}"
        elif auth_type == "api_key" and kwargs.get("auth_token"):
            api_key_header = kwargs.get("api_key_header", "X-API-Key")
            headers[api_key_header] = kwargs["auth_token"]

        start_time = time.time()

        try:
            async with aiohttp.ClientSession() as session:
                async with session.request(
                    method,
                    url,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=timeout)
                ) as response:
                    end_time = time.time()
                    response_time = (end_time - start_time) * 1000  # Convert to milliseconds

                    # Get content if requested
                    content = None
                    content_length = 0
                    if check_content:
                        content = await response.text()
                        content_length = len(content)
                    else:
                        content_length = int(response.headers.get('content-length', 0))

                    is_healthy = response.status in expected_status

                    result = {
                        "url": url,
                        "status_code": response.status,
                        "response_time_ms": round(response_time, 2),
                        "is_healthy": is_healthy,
                        "content_length": content_length,
                        "headers": dict(response.headers),
                        "timestamp": time.time()
                    }

                    if check_content and content:
                        result["content"] = content[:1000]  # Limit content to first 1000 chars

                    self.logger.info(f"API monitor: {url} - Status: {response.status}, Response time: {response_time:.2f}ms")

                    return result

        except asyncio.TimeoutError:
            end_time = time.time()
            response_time = (end_time - start_time) * 1000

            return {
                "url": url,
                "status_code": None,
                "response_time_ms": round(response_time, 2),
                "is_healthy": False,
                "error": f"Timeout after {timeout} seconds",
                "timestamp": time.time()
            }

        except Exception as e:
            end_time = time.time()
            response_time = (end_time - start_time) * 1000

            return {
                "url": url,
                "status_code": None,
                "response_time_ms": round(response_time, 2),
                "is_healthy": False,
                "error": str(e),
                "timestamp": time.time()
            }


class BatchApiCallsTool(BaseTool):
    """Execute multiple API calls in batch with concurrency control and flexible configuration."""

    @property
    def name(self) -> str:
        return "batch_api_calls"

    @property
    def description(self) -> str:
        return "Execute multiple API calls in batch with concurrency control, flexible request configuration, and aggregated results analysis"

    def get_schema(self) -> Dict:
        return {
            "properties": {
                "requests": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {
                                "type": "string",
                                "description": "Unique identifier for this request"
                            },
                            "url": {
                                "type": "string",
                                "description": "The API endpoint URL"
                            },
                            "method": {
                                "type": "string",
                                "enum": ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"],
                                "default": "GET",
                                "description": "HTTP method"
                            },
                            "headers": {
                                "type": "object",
                                "description": "Request headers",
                                "additionalProperties": {"type": "string"}
                            },
                            "params": {
                                "type": "object",
                                "description": "URL query parameters",
                                "additionalProperties": {"type": "string"}
                            },
                            "data": {
                                "description": "Request body data (any format)"
                            },
                            "timeout": {
                                "type": "number",
                                "description": "Custom timeout for this specific request"
                            },
                            "expected_status": {
                                "type": "array",
                                "items": {"type": "integer"},
                                "description": "Expected status codes for this request"
                            }
                        },
                        "required": ["id", "url"]
                    },
                    "description": "List of API requests to execute"
                },
                "max_concurrent": {
                    "type": "integer",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 50,
                    "description": "Maximum number of concurrent requests"
                },
                "default_timeout": {
                    "type": "number",
                    "default": 30,
                    "description": "Default timeout in seconds for requests without custom timeout"
                },
                "auth_type": {
                    "type": "string",
                    "enum": ["none", "bearer", "basic", "api_key"],
                    "default": "none",
                    "description": "Authentication type (applied to all requests)"
                },
                "auth_token": {
                    "type": "string",
                    "description": "Authentication token/key"
                },
                "auth_username": {
                    "type": "string",
                    "description": "Username for basic auth"
                },
                "auth_password": {
                    "type": "string",
                    "description": "Password for basic auth"
                },
                "api_key_header": {
                    "type": "string",
                    "default": "X-API-Key",
                    "description": "Header name for API key authentication"
                },
                "stop_on_error": {
                    "type": "boolean",
                    "default": False,
                    "description": "Whether to stop batch execution on first error"
                },
                "retry_failed": {
                    "type": "boolean",
                    "default": False,
                    "description": "Whether to retry failed requests once"
                },
                "aggregate_results": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether to provide aggregated statistics and analysis"
                },
                "include_response_data": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether to include full response data (set false for large responses)"
                }
            },
            "required": ["requests"]
        }

    async def _execute_single_request(self, request: Dict, auth_config: Dict, default_timeout: float) -> Dict:
        """Execute a single API request with full flexibility."""
        req_id = request["id"]
        url = request["url"]
        method = request.get("method", "GET").upper()
        headers = request.get("headers", {}).copy()
        params = request.get("params", {})
        data = request.get("data")
        timeout = request.get("timeout", default_timeout)
        expected_status = request.get("expected_status", [200])

        # Apply authentication
        auth = None
        auth_type = auth_config.get("auth_type", "none")
        if auth_type == "bearer" and auth_config.get("auth_token"):
            headers["Authorization"] = f"Bearer {auth_config['auth_token']}"
        elif auth_type == "basic" and auth_config.get("auth_username") and auth_config.get("auth_password"):
            auth = aiohttp.BasicAuth(auth_config["auth_username"], auth_config["auth_password"])
        elif auth_type == "api_key" and auth_config.get("auth_token"):
            api_key_header = auth_config.get("api_key_header", "X-API-Key")
            headers[api_key_header] = auth_config["auth_token"]

        start_time = time.time()

        try:
            async with aiohttp.ClientSession() as session:
                request_kwargs = {
                    "timeout": aiohttp.ClientTimeout(total=timeout),
                    "headers": headers,
                    "params": params,
                    "auth": auth
                }

                if method in ["POST", "PUT", "PATCH"] and data is not None:
                    if isinstance(data, dict):
                        request_kwargs["json"] = data
                    else:
                        request_kwargs["data"] = data

                async with session.request(method, url, **request_kwargs) as response:
                    response_time = (time.time() - start_time) * 1000

                    # Get response data
                    try:
                        if response.content_type == 'application/json':
                            response_data = await response.json()
                        else:
                            response_data = await response.text()
                    except Exception:
                        response_data = await response.text()

                    is_success = response.status in expected_status

                    return {
                        "id": req_id,
                        "url": url,
                        "method": method,
                        "status_code": response.status,
                        "success": is_success,
                        "data": response_data,
                        "response_time_ms": round(response_time, 2),
                        "headers": dict(response.headers),
                        "error": None
                    }

        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return {
                "id": req_id,
                "url": url,
                "method": method,
                "status_code": None,
                "success": False,
                "data": None,
                "response_time_ms": round(response_time, 2),
                "headers": {},
                "error": str(e)
            }

    @log_execution_time("batch_api_calls")
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute batch API requests with maximum flexibility."""
        requests = kwargs["requests"]
        max_concurrent = kwargs.get("max_concurrent", 5)
        default_timeout = kwargs.get("default_timeout", 30)
        stop_on_error = kwargs.get("stop_on_error", False)
        retry_failed = kwargs.get("retry_failed", False)
        aggregate_results = kwargs.get("aggregate_results", True)
        include_response_data = kwargs.get("include_response_data", True)

        # Prepare auth configuration
        auth_config = {
            "auth_type": kwargs.get("auth_type", "none"),
            "auth_token": kwargs.get("auth_token"),
            "auth_username": kwargs.get("auth_username"),
            "auth_password": kwargs.get("auth_password"),
            "api_key_header": kwargs.get("api_key_header", "X-API-Key")
        }

        self.logger.info(f"Executing batch of {len(requests)} API requests with max_concurrent={max_concurrent}")

        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)

        async def bounded_request(request):
            async with semaphore:
                return await self._execute_single_request(request, auth_config, default_timeout)

        # Execute requests
        results = []
        start_time = time.time()

        try:
            if stop_on_error:
                # Execute one by one and stop on first error
                for request in requests:
                    result = await bounded_request(request)
                    results.append(result)
                    if not result["success"]:
                        self.logger.warning(f"Stopping batch execution due to error in request {result['id']}")
                        break
            else:
                # Execute all requests concurrently
                tasks = [bounded_request(request) for request in requests]
                results = await asyncio.gather(*tasks, return_exceptions=False)

            # Retry failed requests if requested
            if retry_failed:
                failed_requests = [r for r in results if not r.get("success", False)]
                if failed_requests:
                    self.logger.info(f"Retrying {len(failed_requests)} failed requests")
                    retry_tasks = []
                    for failed_result in failed_requests:
                        # Find original request
                        original_request = next(r for r in requests if r["id"] == failed_result["id"])
                        retry_tasks.append(bounded_request(original_request))

                    retry_results = await asyncio.gather(*retry_tasks, return_exceptions=False)

                    # Replace failed results with retry results
                    for i, result in enumerate(results):
                        if not result.get("success", False):
                            retry_result = retry_results.pop(0)
                            if retry_result.get("success", False):
                                results[i] = retry_result

            total_time = (time.time() - start_time) * 1000

            # Process results for output
            final_results = results.copy()
            if not include_response_data:
                for result in final_results:
                    if "data" in result:
                        result["data"] = f"<data truncated - {len(str(result['data']))} chars>"

            response: Dict[str, Any] = {"results": final_results}

            # Add aggregated analysis if requested
            if aggregate_results:
                successful = [r for r in results if r.get("success", False)]
                failed = [r for r in results if not r.get("success", False)]

                status_codes = {}
                for result in results:
                    status = result.get("status_code")
                    if status:
                        status_codes[status] = status_codes.get(status, 0) + 1

                avg_response_time = sum(r.get("response_time_ms", 0) for r in results) / len(results) if results else 0
                max_response_time = max((r.get("response_time_ms", 0) for r in results), default=0)
                min_response_time = min((r.get("response_time_ms", 0) for r in results if r.get("response_time_ms", 0) > 0), default=0)

                summary = {
                    "total_requests": len(requests),
                    "executed_requests": len(results),
                    "successful_requests": len(successful),
                    "failed_requests": len(failed),
                    "success_rate": len(successful) / len(results) if results else 0,
                    "total_time_ms": round(total_time, 2),
                    "average_response_time_ms": round(avg_response_time, 2),
                    "max_response_time_ms": round(max_response_time, 2),
                    "min_response_time_ms": round(min_response_time, 2),
                    "status_code_distribution": status_codes,
                    "error_summary": [{"id": r["id"], "error": r["error"]} for r in failed if r.get("error")]
                }

                response["summary"] = summary

            return response

        except Exception as e:
            self.logger.error(f"Batch execution failed: {str(e)}")
            raise RuntimeError(f"Batch execution failed: {str(e)}")


class ApiAuthTool(BaseTool):
    """Comprehensive API authentication testing and management tool."""

    @property
    def name(self) -> str:
        return "api_auth"

    @property
    def description(self) -> str:
        return "Test, validate, and manage various API authentication methods with comprehensive analysis and token management"

    def get_schema(self) -> Dict:
        return {
            "properties": {
                "url": {
                    "type": "string",
                    "description": "API endpoint URL to test authentication against"
                },
                "auth_type": {
                    "type": "string",
                    "enum": ["bearer", "basic", "api_key", "oauth2", "custom_header", "query_param"],
                    "description": "Type of authentication to test"
                },
                "auth_token": {
                    "type": "string",
                    "description": "Authentication token (for bearer, api_key, oauth2)"
                },
                "auth_username": {
                    "type": "string",
                    "description": "Username (for basic auth)"
                },
                "auth_password": {
                    "type": "string",
                    "description": "Password (for basic auth)"
                },
                "api_key_header": {
                    "type": "string",
                    "default": "X-API-Key",
                    "description": "Header name for API key authentication"
                },
                "custom_header_name": {
                    "type": "string",
                    "description": "Custom header name for custom_header auth type"
                },
                "custom_header_value": {
                    "type": "string",
                    "description": "Custom header value for custom_header auth type"
                },
                "query_param_name": {
                    "type": "string",
                    "default": "api_key",
                    "description": "Query parameter name for query_param auth type"
                },
                "oauth2_client_id": {
                    "type": "string",
                    "description": "OAuth2 client ID"
                },
                "oauth2_client_secret": {
                    "type": "string",
                    "description": "OAuth2 client secret"
                },
                "oauth2_token_url": {
                    "type": "string",
                    "description": "OAuth2 token endpoint URL"
                },
                "oauth2_scope": {
                    "type": "string",
                    "description": "OAuth2 scope (optional)"
                },
                "test_endpoint": {
                    "type": "string",
                    "description": "Specific endpoint to test (if different from url)"
                },
                "test_methods": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["GET", "POST", "PUT", "DELETE", "PATCH"]},
                    "default": ["GET"],
                    "description": "HTTP methods to test with authentication"
                },
                "expected_status": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "default": [200],
                    "description": "Expected successful status codes"
                },
                "test_without_auth": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether to test the same endpoint without authentication for comparison"
                },
                "validate_token_expiry": {
                    "type": "boolean",
                    "default": False,
                    "description": "Whether to validate token expiry (for supported auth types)"
                },
                "detailed_analysis": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether to provide detailed authentication analysis"
                }
            },
            "required": ["url", "auth_type"]
        }

    async def _test_oauth2_auth(self, **kwargs) -> Dict:
        """Test OAuth2 authentication by getting a token."""
        token_url = kwargs["oauth2_token_url"]
        client_id = kwargs["oauth2_client_id"]
        client_secret = kwargs["oauth2_client_secret"]
        scope = kwargs.get("oauth2_scope", "")

        try:
            async with aiohttp.ClientSession() as session:
                data = {
                    "grant_type": "client_credentials",
                    "client_id": client_id,
                    "client_secret": client_secret
                }
                if scope:
                    data["scope"] = scope

                async with session.post(
                    token_url,
                    data=data,
                    headers={"Content-Type": "application/x-www-form-urlencoded"}
                ) as response:
                    if response.status == 200:
                        token_data = await response.json()
                        return {
                            "success": True,
                            "token_type": token_data.get("token_type", "bearer"),
                            "access_token": token_data.get("access_token", ""),
                            "expires_in": token_data.get("expires_in"),
                            "scope": token_data.get("scope"),
                            "raw_response": token_data
                        }
                    else:
                        error_text = await response.text()
                        return {
                            "success": False,
                            "error": f"Token request failed with status {response.status}: {error_text}"
                        }

        except Exception as e:
            return {
                "success": False,
                "error": f"OAuth2 authentication failed: {str(e)}"
            }

    async def _test_endpoint_with_auth(self, url: str, method: str, headers: Dict, auth, expected_status: List[int]) -> Dict:
        """Test endpoint with authentication."""
        start_time = time.time()

        try:
            async with aiohttp.ClientSession() as session:
                async with session.request(
                    method,
                    url,
                    headers=headers,
                    auth=auth,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    response_time = (time.time() - start_time) * 1000

                    # Get response info
                    try:
                        if response.content_type == 'application/json':
                            response_data = await response.json()
                        else:
                            response_data = await response.text()
                    except Exception:
                        response_data = await response.text()

                    is_authenticated = response.status in expected_status

                    return {
                        "method": method,
                        "status_code": response.status,
                        "is_authenticated": is_authenticated,
                        "response_time_ms": round(response_time, 2),
                        "response_headers": dict(response.headers),
                        "response_data": response_data[:500] if isinstance(response_data, str) else response_data,
                        "error": None
                    }

        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return {
                "method": method,
                "status_code": None,
                "is_authenticated": False,
                "response_time_ms": round(response_time, 2),
                "response_headers": {},
                "response_data": None,
                "error": str(e)
            }

    @log_execution_time("api_auth")
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Test the API authentication configuration with comprehensive analysis."""
        url = kwargs["url"]
        auth_type = kwargs["auth_type"]
        test_endpoint = kwargs.get("test_endpoint", url)
        test_methods = kwargs.get("test_methods", ["GET"])
        expected_status = kwargs.get("expected_status", [200])
        test_without_auth = kwargs.get("test_without_auth", True)
        validate_token_expiry = kwargs.get("validate_token_expiry", False)
        detailed_analysis = kwargs.get("detailed_analysis", True)

        self.logger.info(f"Testing {auth_type} authentication for {url}")

        results = {
            "auth_type": auth_type,
            "test_url": test_endpoint,
            "timestamp": time.time(),
            "auth_tests": [],
            "oauth_details": None,
            "analysis": {}
        }

        # Handle OAuth2 token acquisition first
        oauth_result = None
        if auth_type == "oauth2":
            oauth_result = await self._test_oauth2_auth(**kwargs)
            results["oauth_details"] = oauth_result
            if not oauth_result["success"]:
                return results

        # Prepare authentication headers and auth objects
        auth_headers = {}
        auth = None

        if auth_type == "bearer":
            if not kwargs.get("auth_token"):
                results["analysis"]["error"] = "auth_token is required for bearer authentication"
                return results
            auth_headers["Authorization"] = f"Bearer {kwargs['auth_token']}"

        elif auth_type == "basic":
            if not all([kwargs.get("auth_username"), kwargs.get("auth_password")]):
                results["analysis"]["error"] = "auth_username and auth_password are required for basic authentication"
                return results
            auth = aiohttp.BasicAuth(kwargs["auth_username"], kwargs["auth_password"])

        elif auth_type == "api_key":
            if not kwargs.get("auth_token"):
                results["analysis"]["error"] = "auth_token is required for API key authentication"
                return results
            api_key_header = kwargs.get("api_key_header", "X-API-Key")
            auth_headers[api_key_header] = kwargs["auth_token"]

        elif auth_type == "custom_header":
            if not all([kwargs.get("custom_header_name"), kwargs.get("custom_header_value")]):
                results["analysis"]["error"] = "custom_header_name and custom_header_value are required"
                return results
            auth_headers[kwargs["custom_header_name"]] = kwargs["custom_header_value"]

        elif auth_type == "query_param":
            if not kwargs.get("auth_token"):
                results["analysis"]["error"] = "auth_token is required for query parameter authentication"
                return results
            param_name = kwargs.get("query_param_name", "api_key")
            test_endpoint = f"{test_endpoint}{'&' if '?' in test_endpoint else '?'}{param_name}={kwargs['auth_token']}"

        elif auth_type == "oauth2" and oauth_result:
            auth_headers["Authorization"] = f"Bearer {oauth_result['access_token']}"

        # Test without authentication first (if requested)
        if test_without_auth:
            for method in test_methods:
                no_auth_result = await self._test_endpoint_with_auth(
                    test_endpoint, method, {}, None, expected_status
                )
                no_auth_result["auth_used"] = "none"
                results["auth_tests"].append(no_auth_result)

        # Test with authentication
        for method in test_methods:
            auth_result = await self._test_endpoint_with_auth(
                test_endpoint, method, auth_headers, auth, expected_status
            )
            auth_result["auth_used"] = auth_type
            results["auth_tests"].append(auth_result)

        # Provide detailed analysis if requested
        if detailed_analysis:
            auth_tests = [t for t in results["auth_tests"] if t["auth_used"] == auth_type]
            no_auth_tests = [t for t in results["auth_tests"] if t["auth_used"] == "none"]

            analysis = {
                "authentication_working": any(t["is_authenticated"] for t in auth_tests),
                "methods_tested": test_methods,
                "successful_methods": [t["method"] for t in auth_tests if t["is_authenticated"]],
                "failed_methods": [t["method"] for t in auth_tests if not t["is_authenticated"]],
                "average_response_time_ms": sum(t["response_time_ms"] for t in auth_tests) / len(auth_tests) if auth_tests else 0,
            }

            if no_auth_tests:
                analysis["requires_authentication"] = not any(t["is_authenticated"] for t in no_auth_tests)
                analysis["auth_improves_access"] = (
                    any(t["is_authenticated"] for t in auth_tests) and
                    not any(t["is_authenticated"] for t in no_auth_tests)
                )

            # Token validation for supported types
            if validate_token_expiry and auth_type in ["bearer", "oauth2"]:
                analysis["token_validation"] = self._validate_token_format(kwargs.get("auth_token", ""))

            results["analysis"] = analysis

        return results

    def _validate_token_format(self, token: str) -> Dict:
        """Basic token format validation."""
        validation = {
            "token_length": len(token),
            "appears_base64": token.replace("-", "+").replace("_", "/").isalnum(),
            "contains_periods": "." in token,  # Might be JWT
            "estimated_type": "unknown"
        }

        if validation["contains_periods"] and len(token.split(".")) == 3:
            validation["estimated_type"] = "jwt"
        elif validation["appears_base64"] and validation["token_length"] > 20:
            validation["estimated_type"] = "bearer_token"
        elif validation["token_length"] < 20:
            validation["estimated_type"] = "api_key"

        return validation