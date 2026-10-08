from unittest.mock import patch
from careerpilot.framework_tools import framework_career_lookup
from careerpilot.guardrails import (
    is_tool_failure,
    tool_failure_guardrail,
    is_tool_not_found,
    tool_not_found_guardrail,
)


def test_framework_tool_failure_returns_safe_response():
    # Simulate the exact failure result returned by the framework tool
    tool_result = (
        "TOOL_ERROR: Career lookup failed: simulated service failure"
    )

    # Apply the tool-failure guardrail
    response = tool_failure_guardrail(tool_result)

    # The guardrail must stop the agent from presenting guessed information
    assert response is not None
    assert "cannot provide" in response.lower()


def test_framework_lookup_converts_exception_to_tool_error():
    # Force the underlying lookup function to fail
    with patch(
        "careerpilot.framework_tools.lookup",
        side_effect=RuntimeError("simulated service failure"),
    ):
        # Call the actual framework tool
        result = framework_career_lookup(
            {
                "lookup_type": "career",
                "query": "AI Engineer",
            }
        )

    # The framework tool must convert the exception into an explicit failure
    assert result.startswith("TOOL_ERROR:")
    assert "simulated service failure" in result


def test_tool_not_found_is_detected():
    # Simulate a lookup that completed without finding a career
    result = "LOOKUP_NOT_FOUND: Career not found."

    # Verify that the not-found result is recognized
    assert is_tool_not_found(result) is True


def test_tool_not_found_guardrail_prevents_invented_information():
    # Simulate a missing career record
    result = "LOOKUP_NOT_FOUND: Career not found."

    # Apply the not-found guardrail
    response = tool_not_found_guardrail(result)

    # The guardrail should prevent unverified information
    assert response is not None
    assert "unverified" in response.lower()


def test_framework_lookup_returns_not_found_for_unknown_career():
    # Request a career that does not exist in the CareerPilot knowledge base
    result = framework_career_lookup(
        {
            "lookup_type": "career",
            "query": "Quantum Computing Engineer",
        }
    )

    # The framework tool must explicitly identify the missing career
    assert result.startswith("LOOKUP_NOT_FOUND:")
    assert "Career not found" in result