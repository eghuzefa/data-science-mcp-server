"""
Comprehensive tests for API client tools.
"""

import pytest
import asyncio
import aiohttp
from aioresponses import aioresponses

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from tools.api_client import FetchApiDataTool, MonitorApiTool, BatchApiCallsTool, ApiAuthTool


class TestFetchApiDataTool:
    """Test suite for FetchApiDataTool."""

    @pytest.fixture
    def tool(self):
        """Create a FetchApiDataTool instance."""
        return FetchApiDataTool()

    def test_tool_properties(self, tool):
        """Test basic tool properties."""
        assert tool.name == "fetch_api_data"
        assert "fetch data from external" in tool.description.lower()
        assert isinstance(tool.get_schema(), dict)
        assert "url" in tool.get_schema()["properties"]
        assert "url" in tool.get_schema()["required"]

    def test_schema_validation(self, tool):
        """Test input schema validation."""
        schema = tool.get_schema()
        properties = schema["properties"]

        # Check required fields
        assert "url" in schema["required"]

        # Check method enum
        assert properties["method"]["enum"] == ["GET", "POST", "PUT", "PATCH", "DELETE"]
        assert properties["method"]["default"] == "GET"

        # Check auth types
        assert "bearer" in properties["auth_type"]["enum"]
        assert "basic" in properties["auth_type"]["enum"]
        assert "api_key" in properties["auth_type"]["enum"]

    def test_validate_input_valid_url(self, tool):
        """Test input validation with valid URL."""
        valid_input = tool.validate_input(url="https://api.example.com/data")
        assert valid_input["url"] == "https://api.example.com/data"

    def test_validate_input_invalid_url(self, tool):
        """Test input validation with invalid URL."""
        with pytest.raises(ValueError, match="Invalid URL format"):
            tool.validate_input(url="not-a-url")

    def test_validate_input_missing_auth_token_bearer(self, tool):
        """Test validation fails when bearer auth lacks token."""
        with pytest.raises(ValueError, match="auth_token is required for bearer authentication"):
            tool.validate_input(
                url="https://api.example.com",
                auth_type="bearer"
            )

    def test_validate_input_missing_basic_auth_credentials(self, tool):
        """Test validation fails when basic auth lacks credentials."""
        with pytest.raises(ValueError, match="auth_username and auth_password are required"):
            tool.validate_input(
                url="https://api.example.com",
                auth_type="basic",
                auth_username="user"
                # Missing auth_password
            )

    @pytest.mark.asyncio
    async def test_execute_get_request_success(self, tool):
        """Test successful GET request."""
        with aioresponses() as m:
            url = "https://api.example.com/data"
            mock_data = {"id": 1, "name": "test"}

            m.get(url, payload=mock_data, status=200)

            result = await tool.execute(url=url, method="GET")

            assert result["success"] is True
            assert result["status_code"] == 200
            assert result["data"] == mock_data
            assert result["method"] == "GET"
            assert result["url"] == url

    @pytest.mark.asyncio
    async def test_execute_post_request_with_json_data(self, tool):
        """Test POST request with JSON data."""
        with aioresponses() as m:
            url = "https://api.example.com/data"
            request_data = {"name": "new_item"}
            response_data = {"id": 2, "name": "new_item"}

            m.post(url, payload=response_data, status=201)

            result = await tool.execute(
                url=url,
                method="POST",
                data=request_data
            )

            assert result["success"] is True
            assert result["status_code"] == 201
            assert result["data"] == response_data
            assert result["method"] == "POST"

    @pytest.mark.asyncio
    async def test_execute_with_bearer_auth(self, tool):
        """Test request with bearer authentication."""
        with aioresponses() as m:
            url = "https://api.example.com/secure"
            mock_data = {"secure": "data"}

            m.get(url, payload=mock_data, status=200)

            result = await tool.execute(
                url=url,
                auth_type="bearer",
                auth_token="test-token"
            )

            assert result["success"] is True
            assert result["data"] == mock_data

    @pytest.mark.asyncio
    async def test_execute_with_api_key_auth(self, tool):
        """Test request with API key authentication."""
        with aioresponses() as m:
            url = "https://api.example.com/secure"
            mock_data = {"secure": "data"}

            m.get(url, payload=mock_data, status=200)

            result = await tool.execute(
                url=url,
                auth_type="api_key",
                auth_token="test-api-key"
            )

            assert result["success"] is True
            assert result["data"] == mock_data

    @pytest.mark.asyncio
    async def test_execute_with_custom_headers(self, tool):
        """Test request with custom headers."""
        with aioresponses() as m:
            url = "https://api.example.com/data"

            m.get(url, payload={}, status=200)

            result = await tool.execute(
                url=url,
                headers={
                    "Custom-Header": "custom-value",
                    "Accept": "application/json"
                }
            )

            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_execute_with_query_params(self, tool):
        """Test request with query parameters."""
        with aioresponses() as m:
            base_url = "https://api.example.com/data"
            url_with_params = "https://api.example.com/data?filter=active&limit=10"

            # Match the exact URL with query parameters
            m.get(url_with_params, payload={"filtered": "data"})

            result = await tool.execute(
                url=base_url,
                params={"filter": "active", "limit": "10"}
            )

            assert result["success"] is True
            assert result["data"] == {"filtered": "data"}

    @pytest.mark.asyncio
    async def test_execute_error_response(self, tool):
        """Test handling of error HTTP status codes."""
        with aioresponses() as m:
            url = "https://api.example.com/notfound"

            m.get(url, status=404, payload={"error": "Not found"})

            result = await tool.execute(url=url)

            assert result["success"] is False
            assert result["status_code"] == 404
            assert result["data"] == {"error": "Not found"}

    @pytest.mark.asyncio
    async def test_execute_non_json_response(self, tool):
        """Test handling of non-JSON response."""
        with aioresponses() as m:
            url = "https://api.example.com/text"
            text_response = "Plain text response"

            m.get(url, body=text_response, content_type='text/plain')

            result = await tool.execute(url=url)

            assert result["success"] is True
            assert result["data"] == text_response

    @pytest.mark.asyncio
    async def test_execute_timeout_error(self, tool):
        """Test handling of timeout errors."""
        with aioresponses() as m:
            url = "https://api.example.com/slow"

            m.get(url, exception=asyncio.TimeoutError())

            with pytest.raises(RuntimeError, match="Request timeout"):
                await tool.execute(url=url, timeout=1)

    @pytest.mark.asyncio
    async def test_execute_connection_error(self, tool):
        """Test handling of connection errors."""
        with aioresponses() as m:
            url = "https://api.example.com/unreachable"

            m.get(url, exception=aiohttp.ClientError("Connection failed"))

            with pytest.raises(RuntimeError, match="Request failed"):
                await tool.execute(url=url)


class TestMonitorApiTool:
    """Test suite for MonitorApiTool."""

    @pytest.fixture
    def tool(self):
        """Create a MonitorApiTool instance."""
        return MonitorApiTool()

    def test_tool_properties(self, tool):
        """Test basic tool properties."""
        assert tool.name == "monitor_api"
        assert "monitor api" in tool.description.lower()
        assert isinstance(tool.get_schema(), dict)
        assert "url" in tool.get_schema()["properties"]
        assert "url" in tool.get_schema()["required"]

    def test_schema_defaults(self, tool):
        """Test schema default values."""
        schema = tool.get_schema()
        properties = schema["properties"]

        assert properties["method"]["default"] == "GET"
        assert properties["timeout"]["default"] == 10
        assert properties["expected_status"]["default"] == [200]
        assert properties["check_content"]["default"] is False

    @pytest.mark.asyncio
    async def test_execute_healthy_api(self, tool):
        """Test monitoring a healthy API."""
        with aioresponses() as m:
            url = "https://api.example.com/health"

            m.get(url, status=200, payload={"status": "ok"})

            result = await tool.execute(url=url)

            assert result["url"] == url
            assert result["status_code"] == 200
            assert result["is_healthy"] is True
            assert result["response_time_ms"] > 0
            assert "timestamp" in result
            assert "headers" in result

    @pytest.mark.asyncio
    async def test_execute_unhealthy_api(self, tool):
        """Test monitoring an unhealthy API."""
        with aioresponses() as m:
            url = "https://api.example.com/health"

            m.get(url, status=500, payload={"error": "Internal server error"})

            result = await tool.execute(url=url)

            assert result["status_code"] == 500
            assert result["is_healthy"] is False
            assert result["response_time_ms"] > 0

    @pytest.mark.asyncio
    async def test_execute_with_custom_expected_status(self, tool):
        """Test monitoring with custom expected status codes."""
        with aioresponses() as m:
            url = "https://api.example.com/redirect"

            m.get(url, status=301)

            result = await tool.execute(
                url=url,
                expected_status=[200, 301, 302]
            )

            assert result["status_code"] == 301
            assert result["is_healthy"] is True

    @pytest.mark.asyncio
    async def test_execute_with_content_check(self, tool):
        """Test monitoring with content checking enabled."""
        with aioresponses() as m:
            url = "https://api.example.com/status"
            response_body = "API is running normally"

            m.get(url, body=response_body, status=200)

            result = await tool.execute(
                url=url,
                check_content=True
            )

            assert result["is_healthy"] is True
            assert "content" in result
            assert result["content"] == response_body
            assert result["content_length"] == len(response_body)

    @pytest.mark.asyncio
    async def test_execute_with_auth(self, tool):
        """Test monitoring with authentication."""
        with aioresponses() as m:
            url = "https://api.example.com/secure/health"

            m.get(url, status=200, payload={"status": "healthy"})

            result = await tool.execute(
                url=url,
                auth_type="bearer",
                auth_token="health-token"
            )

            assert result["is_healthy"] is True

    @pytest.mark.asyncio
    async def test_execute_timeout_handling(self, tool):
        """Test monitoring timeout handling."""
        with aioresponses() as m:
            url = "https://api.example.com/slow"

            m.get(url, exception=asyncio.TimeoutError())

            result = await tool.execute(url=url, timeout=5)

            assert result["status_code"] is None
            assert result["is_healthy"] is False
            assert "Timeout after 5 seconds" in result["error"]
            assert result["response_time_ms"] > 0

    @pytest.mark.asyncio
    async def test_execute_connection_error_handling(self, tool):
        """Test monitoring connection error handling."""
        with aioresponses() as m:
            url = "https://api.example.com/unreachable"

            m.get(url, exception=aiohttp.ClientError("Connection refused"))

            result = await tool.execute(url=url)

            assert result["status_code"] is None
            assert result["is_healthy"] is False
            assert "error" in result
            assert result["response_time_ms"] >= 0

    @pytest.mark.asyncio
    async def test_execute_head_request(self, tool):
        """Test monitoring using HEAD request method."""
        with aioresponses() as m:
            url = "https://api.example.com/health"

            m.head(url, status=200, headers={'Content-Length': '1024'})

            result = await tool.execute(url=url, method="HEAD")

            assert result["is_healthy"] is True
            assert result["content_length"] == 1024

    @pytest.mark.asyncio
    async def test_response_time_measurement(self, tool):
        """Test that response time is properly measured."""
        with aioresponses() as m:
            url = "https://api.example.com/fast"

            m.get(url, status=200, payload={"fast": "response"})

            result = await tool.execute(url=url)

            # Response time should be a positive number in milliseconds
            assert isinstance(result["response_time_ms"], (int, float))
            assert result["response_time_ms"] >= 0

    @pytest.mark.asyncio
    async def test_content_length_from_headers(self, tool):
        """Test content length extraction from headers when not checking content."""
        with aioresponses() as m:
            url = "https://api.example.com/data"

            m.get(url, status=200, headers={'Content-Length': '2048'}, payload={"data": "test"})

            result = await tool.execute(url=url, check_content=False)

            assert result["content_length"] == 2048
            assert "content" not in result


class TestBatchApiCallsTool:
    """Test suite for BatchApiCallsTool."""

    @pytest.fixture
    def tool(self):
        """Create a BatchApiCallsTool instance."""
        return BatchApiCallsTool()

    def test_tool_properties(self, tool):
        """Test basic tool properties."""
        assert tool.name == "batch_api_calls"
        assert "batch" in tool.description.lower()
        assert isinstance(tool.get_schema(), dict)
        assert "requests" in tool.get_schema()["properties"]
        assert "requests" in tool.get_schema()["required"]

    def test_schema_validation(self, tool):
        """Test input schema validation."""
        schema = tool.get_schema()
        properties = schema["properties"]

        # Check requests array structure
        requests_schema = properties["requests"]
        assert requests_schema["type"] == "array"
        assert "id" in requests_schema["items"]["required"]
        assert "url" in requests_schema["items"]["required"]

        # Check default values
        assert properties["max_concurrent"]["default"] == 5
        assert properties["max_concurrent"]["maximum"] == 50
        assert properties["stop_on_error"]["default"] is False
        assert properties["retry_failed"]["default"] is False

    @pytest.mark.asyncio
    async def test_execute_single_request_batch(self, tool):
        """Test batch execution with single request."""
        with aioresponses() as m:
            url = "https://api.example.com/test"
            mock_data = {"result": "success"}

            m.get(url, payload=mock_data, status=200)

            requests = [{"id": "req1", "url": url, "method": "GET"}]
            result = await tool.execute(requests=requests)

            assert "results" in result
            assert "summary" in result
            assert len(result["results"]) == 1

            req_result = result["results"][0]
            assert req_result["id"] == "req1"
            assert req_result["success"] is True
            assert req_result["data"] == mock_data

            summary = result["summary"]
            assert summary["total_requests"] == 1
            assert summary["successful_requests"] == 1
            assert summary["success_rate"] == 1.0

    @pytest.mark.asyncio
    async def test_execute_multiple_requests_batch(self, tool):
        """Test batch execution with multiple requests."""
        with aioresponses() as m:
            base_url = "https://api.example.com"

            m.get(f"{base_url}/endpoint1", payload={"data": "1"}, status=200)
            m.get(f"{base_url}/endpoint2", payload={"data": "2"}, status=200)
            m.post(f"{base_url}/endpoint3", payload={"created": "3"}, status=201)

            requests = [
                {"id": "req1", "url": f"{base_url}/endpoint1", "method": "GET"},
                {"id": "req2", "url": f"{base_url}/endpoint2", "method": "GET"},
                {"id": "req3", "url": f"{base_url}/endpoint3", "method": "POST", "data": {"name": "test"}, "expected_status": [201]}
            ]

            result = await tool.execute(requests=requests, max_concurrent=2)

            assert len(result["results"]) == 3
            assert result["summary"]["total_requests"] == 3
            assert result["summary"]["successful_requests"] == 3

            # Check individual results
            req1_result = next(r for r in result["results"] if r["id"] == "req1")
            assert req1_result["success"] is True
            assert req1_result["data"] == {"data": "1"}

    @pytest.mark.asyncio
    async def test_execute_with_authentication(self, tool):
        """Test batch execution with authentication."""
        with aioresponses() as m:
            url = "https://api.example.com/secure"

            m.get(url, payload={"secure": "data"}, status=200)

            requests = [{"id": "auth_req", "url": url, "method": "GET"}]
            result = await tool.execute(
                requests=requests,
                auth_type="bearer",
                auth_token="test-token"
            )

            assert result["results"][0]["success"] is True

    @pytest.mark.asyncio
    async def test_execute_stop_on_error(self, tool):
        """Test batch execution with stop_on_error=True."""
        with aioresponses() as m:
            base_url = "https://api.example.com"

            m.get(f"{base_url}/success", payload={"ok": "true"}, status=200)
            m.get(f"{base_url}/error", status=500, payload={"error": "server error"})

            requests = [
                {"id": "req1", "url": f"{base_url}/success", "method": "GET"},
                {"id": "req2", "url": f"{base_url}/error", "method": "GET"},
                {"id": "req3", "url": f"{base_url}/success", "method": "GET"}  # Should not execute
            ]

            result = await tool.execute(requests=requests, stop_on_error=True)

            # Should stop after the error
            assert len(result["results"]) == 2
            assert result["results"][0]["success"] is True
            assert result["results"][1]["success"] is False

    @pytest.mark.asyncio
    async def test_execute_with_custom_timeouts(self, tool):
        """Test batch execution with custom timeouts per request."""
        with aioresponses() as m:
            url1 = "https://api.example.com/fast"
            url2 = "https://api.example.com/slow"

            m.get(url1, payload={"fast": "response"}, status=200)
            m.get(url2, payload={"slow": "response"}, status=200)

            requests = [
                {"id": "fast", "url": url1, "timeout": 1},
                {"id": "slow", "url": url2, "timeout": 60}
            ]

            result = await tool.execute(requests=requests)

            assert len(result["results"]) == 2
            assert all(r["success"] for r in result["results"])

    @pytest.mark.asyncio
    async def test_execute_without_response_data(self, tool):
        """Test batch execution with include_response_data=False."""
        with aioresponses() as m:
            url = "https://api.example.com/large"
            large_data = {"large": "x" * 1000}

            m.get(url, payload=large_data, status=200)

            requests = [{"id": "large_req", "url": url}]
            result = await tool.execute(requests=requests, include_response_data=False)

            req_result = result["results"][0]
            assert req_result["success"] is True
            assert "data truncated" in str(req_result["data"])

    @pytest.mark.asyncio
    async def test_execute_with_retry_failed(self, tool):
        """Test batch execution with retry_failed=True."""
        with aioresponses() as m:
            url = "https://api.example.com/flaky"

            # First call fails, retry succeeds
            m.get(url, exception=aiohttp.ClientError("Connection failed"))
            m.get(url, payload={"retried": "success"}, status=200)

            requests = [{"id": "flaky_req", "url": url}]
            result = await tool.execute(requests=requests, retry_failed=True)

            # Should succeed after retry
            assert result["results"][0]["success"] is True
            assert result["results"][0]["data"] == {"retried": "success"}

    @pytest.mark.asyncio
    async def test_execute_error_handling(self, tool):
        """Test batch execution error handling."""
        with aioresponses() as m:
            url = "https://api.example.com/timeout"

            m.get(url, exception=asyncio.TimeoutError())

            requests = [{"id": "timeout_req", "url": url}]
            result = await tool.execute(requests=requests)

            req_result = result["results"][0]
            assert req_result["success"] is False
            assert "error" in req_result
            assert req_result["status_code"] is None


class TestApiAuthTool:
    """Test suite for ApiAuthTool."""

    @pytest.fixture
    def tool(self):
        """Create an ApiAuthTool instance."""
        return ApiAuthTool()

    def test_tool_properties(self, tool):
        """Test basic tool properties."""
        assert tool.name == "api_auth"
        assert "authentication" in tool.description.lower()
        assert isinstance(tool.get_schema(), dict)
        assert "url" in tool.get_schema()["properties"]
        assert "auth_type" in tool.get_schema()["properties"]

    def test_schema_validation(self, tool):
        """Test input schema validation."""
        schema = tool.get_schema()
        properties = schema["properties"]

        # Check auth types
        auth_types = properties["auth_type"]["enum"]
        expected_types = ["bearer", "basic", "api_key", "oauth2", "custom_header", "query_param"]
        assert all(auth_type in auth_types for auth_type in expected_types)

        # Check default values
        assert properties["test_methods"]["default"] == ["GET"]
        assert properties["expected_status"]["default"] == [200]
        assert properties["test_without_auth"]["default"] is True

    @pytest.mark.asyncio
    async def test_execute_bearer_auth_success(self, tool):
        """Test bearer authentication success."""
        with aioresponses() as m:
            url = "https://api.example.com/protected"

            # Without auth - should fail
            m.get(url, status=401, payload={"error": "Unauthorized"})
            # With auth - should succeed
            m.get(url, status=200, payload={"protected": "data"})

            result = await tool.execute(
                url=url,
                auth_type="bearer",
                auth_token="valid-token"
            )

            assert result["auth_type"] == "bearer"
            assert len(result["auth_tests"]) == 2  # One without auth, one with auth

            # Check analysis
            analysis = result["analysis"]
            assert analysis["authentication_working"] is True
            assert analysis["requires_authentication"] is True

    @pytest.mark.asyncio
    async def test_execute_basic_auth_success(self, tool):
        """Test basic authentication success."""
        with aioresponses() as m:
            url = "https://api.example.com/basic"

            m.get(url, status=401)  # Without auth
            m.get(url, status=200, payload={"authenticated": True})  # With auth

            result = await tool.execute(
                url=url,
                auth_type="basic",
                auth_username="user",
                auth_password="pass"
            )

            assert result["analysis"]["authentication_working"] is True

    @pytest.mark.asyncio
    async def test_execute_api_key_auth_success(self, tool):
        """Test API key authentication success."""
        with aioresponses() as m:
            url = "https://api.example.com/apikey"

            m.get(url, status=403)  # Without auth
            m.get(url, status=200, payload={"api_access": True})  # With auth

            result = await tool.execute(
                url=url,
                auth_type="api_key",
                auth_token="secret-key",
                api_key_header="X-Custom-Key"
            )

            assert result["analysis"]["authentication_working"] is True

    @pytest.mark.asyncio
    async def test_execute_custom_header_auth(self, tool):
        """Test custom header authentication."""
        with aioresponses() as m:
            url = "https://api.example.com/custom"

            m.get(url, status=403)  # Without auth
            m.get(url, status=200, payload={"custom_auth": True})  # With auth

            result = await tool.execute(
                url=url,
                auth_type="custom_header",
                custom_header_name="X-Custom-Auth",
                custom_header_value="custom-value"
            )

            assert result["analysis"]["authentication_working"] is True

    @pytest.mark.asyncio
    async def test_execute_query_param_auth(self, tool):
        """Test query parameter authentication."""
        with aioresponses() as m:
            base_url = "https://api.example.com/query"
            auth_url = f"{base_url}?token=secret"

            m.get(base_url, status=401)  # Without auth
            m.get(auth_url, status=200, payload={"query_auth": True})  # With auth

            result = await tool.execute(
                url=base_url,
                auth_type="query_param",
                auth_token="secret",
                query_param_name="token",
                test_without_auth=False  # Skip testing without auth to avoid confusion
            )

            assert result["analysis"]["authentication_working"] is True

    @pytest.mark.asyncio
    async def test_execute_multiple_methods(self, tool):
        """Test authentication with multiple HTTP methods."""
        with aioresponses() as m:
            url = "https://api.example.com/multi"

            # Mock all methods
            for method in ["GET", "POST", "PUT"]:
                # Without auth
                getattr(m, method.lower())(url, status=401)
                # With auth
                getattr(m, method.lower())(url, status=200, payload={"method": method})

            result = await tool.execute(
                url=url,
                auth_type="bearer",
                auth_token="token",
                test_methods=["GET", "POST", "PUT"]
            )

            auth_tests = [t for t in result["auth_tests"] if t["auth_used"] == "bearer"]
            assert len(auth_tests) == 3
            assert all(t["is_authenticated"] for t in auth_tests)

            analysis = result["analysis"]
            assert len(analysis["successful_methods"]) == 3

    @pytest.mark.asyncio
    async def test_execute_missing_credentials(self, tool):
        """Test authentication with missing credentials."""
        result = await tool.execute(
            url="https://api.example.com/test",
            auth_type="bearer"
            # Missing auth_token
        )

        assert "error" in result["analysis"]
        assert "auth_token is required" in result["analysis"]["error"]

    @pytest.mark.asyncio
    async def test_execute_oauth2_flow(self, tool):
        """Test OAuth2 authentication flow."""
        with aioresponses() as m:
            token_url = "https://auth.example.com/token"
            api_url = "https://api.example.com/oauth"

            # Mock token request
            m.post(token_url, payload={
                "access_token": "oauth-token",
                "token_type": "bearer",
                "expires_in": 3600
            }, status=200)

            # Mock API call
            m.get(api_url, status=401)  # Without auth
            m.get(api_url, status=200, payload={"oauth_data": True})  # With auth

            result = await tool.execute(
                url=api_url,
                auth_type="oauth2",
                oauth2_client_id="client123",
                oauth2_client_secret="secret456",
                oauth2_token_url=token_url
            )

            assert result["oauth_details"]["success"] is True
            assert result["analysis"]["authentication_working"] is True

    @pytest.mark.asyncio
    async def test_execute_detailed_analysis_disabled(self, tool):
        """Test execution with detailed analysis disabled."""
        with aioresponses() as m:
            url = "https://api.example.com/simple"

            m.get(url, status=200, payload={"simple": True})

            result = await tool.execute(
                url=url,
                auth_type="bearer",
                auth_token="token",
                detailed_analysis=False,
                test_without_auth=False
            )

            # Should have minimal analysis
            assert len(result["auth_tests"]) == 1
            assert result["analysis"] == {}

    @pytest.mark.asyncio
    async def test_token_validation(self, tool):
        """Test token format validation."""
        # Test JWT-like token
        jwt_token = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiYWRtaW4iOnRydWV9.TJVA95OrM7E2cBab30RMHrHDcEfxjoYZgeFONFh7HgQ"

        validation = tool._validate_token_format(jwt_token)
        assert validation["estimated_type"] == "jwt"
        assert validation["contains_periods"] is True

        # Test simple API key
        api_key = "abc123"
        validation = tool._validate_token_format(api_key)
        assert validation["estimated_type"] == "api_key"
        assert validation["token_length"] == 6