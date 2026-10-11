from careerpilot.agent import CareerPilotAgent
from unittest.mock import Mock, patch
from careerpilot.models import CareerProfile
from careerpilot.guardrails import tool_call_guardrail
from careerpilot.state import AgentState
from careerpilot.guardrails import (
    is_tool_failure,
    tool_failure_guardrail,
)



# Create a reusable candidate profile for the evaluation tests
candidate = CareerProfile(
    name="Michael",
    current_role="Data Analyst",
    years_of_experience=5,
    education=["MSc GIS"],
    skills=["Excel", "SQL", "Power BI", "Python"],
    certifications=["PL-300"],
    industry="Technology",
    career_goal="Transition into AI Engineering",
)


# Test that CareerPilot uses the ask tool for general career questions
def test_general_career_question_uses_ask():
    agent = CareerPilotAgent()

    result = agent.run(
        "What does an AI Engineer do?"
    )

    # Verify that the ask tool was selected
    assert result.tool_steps[0].tool == "ask"

    # Verify that exactly one tool was used
    assert result.tool_call_count == 1


# Test that CareerPilot returns a grounded answer
def test_general_question_returns_answer():
    agent = CareerPilotAgent()

    result = agent.run(
        "What does an AI Engineer do?"
    )

    # Verify that a final response was produced
    assert result.final_response is not None

    # Verify that the response contains the expected topic
    assert "AI Engineer" in result.final_response


# Test that CareerPilot uses the lookup tool for a direct career lookup
def test_career_lookup_uses_lookup():
    agent = CareerPilotAgent()

    result = agent.run(
        "Look up AI Engineer"
    )

    # Verify that the lookup tool was selected
    assert result.tool_steps[0].tool == "lookup"

    # Verify that the career information was found
    assert "AI Engineer" in result.final_response


# Test that CareerPilot uses the full career analysis workflow
def test_career_fit_uses_multiple_tools():
    agent = CareerPilotAgent()

    result = agent.run(
        "How suitable am I for AI Engineer?",
        profile=candidate,
    )

    # Verify that the workflow used all three required tools
    tools_used = [step.tool for step in result.tool_steps]

    assert tools_used == ["lookup", "predict", "ask"]

    # Verify that three tool calls were made
    assert result.tool_call_count == 3

    # Verify that a structured analysis was returned
    assert result.final_response is not None


# Test that CareerPilot refuses requests outside its scope
def test_out_of_scope_request_is_refused():
    agent = CareerPilotAgent()

    result = agent.run(
        "What is the weather today?"
    )

    # Verify that no tools were called
    assert result.tool_call_count == 0

    # Verify that the request was refused safely
    assert "career" in result.final_response.lower()


# Test that CareerPilot safely handles an unknown career
def test_unknown_career_fails_safely():
    agent = CareerPilotAgent()

    result = agent.run(
        "Please lookup Quantum Career Specialist"
    )

    # Verify that the lookup tool was called
    assert result.tool_steps[0].tool == "lookup"

    # Verify that the agent did not invent information
    assert "couldn't find verified information" in result.final_response

    # Verify that the unknown career was identified in the response
    assert "quantum career specialist" in result.final_response.lower()



# Test that CareerPilot handles a failed prediction API request safely
def test_prediction_failure_is_handled_safely():
    agent = CareerPilotAgent()

    # Return a valid career record for the lookup request
    lookup_response = {
        "found": True,
        "career": {
            "role": "AI Engineer",
            "description": "Builds AI systems.",
            "required_skills": ["Python"],
            "preferred_skills": ["Machine Learning"],
            "entry_level": "Intermediate",
            "related_roles": ["Data Scientist"],
        },
    }

    # Create mock responses for lookup success and prediction failure
    lookup_http_response = Mock()
    lookup_http_response.json.return_value = lookup_response
    lookup_http_response.raise_for_status.return_value = None

    prediction_http_response = Mock()
    prediction_http_response.raise_for_status.side_effect = RuntimeError(
        "Prediction service unavailable"
    )

    # Simulate the lookup succeeding before prediction fails
    with patch(
        "careerpilot.agent.httpx.post",
        side_effect=[lookup_http_response, prediction_http_response],
    ):
        result = agent.run(
            "How suitable am I for AI Engineer?",
            profile=candidate,
        )

    # Verify lookup and prediction were attempted in order
    tools_used = [step.tool for step in result.tool_steps]
    assert tools_used == ["lookup", "predict"]

    # Verify the agent returned a safe failure response
    assert "prediction service is currently unavailable" in result.final_response.lower()

# Test that CareerPilot stops when the tool-call limit is reached
def test_tool_call_limit_is_enforced():
    agent = CareerPilotAgent()

    # Create an agent state that has already reached the tool limit
    state = AgentState(
        user_request="What does an AI Engineer do?",
        tool_call_count=5,
    )

    # Verify that the guardrail blocks another tool call
    guardrail = tool_call_guardrail(state.tool_call_count)

    # Verify that the correct limit message is returned
    assert guardrail is not None
    assert "tool-call limit" in guardrail.lower()


# Test that significant career recommendations require human review
def test_low_fit_recommendation_requires_human_review():
    agent = CareerPilotAgent()

    result = agent.run(
        "How suitable am I for AI Engineer?",
        profile=candidate,
    )

    # Verify that human review was triggered
    assert result.human_review_required is True

    # Verify that the career analysis was completed
    assert result.tool_call_count == 3


def test_tool_failure_is_detected():
    # Verify that an explicit tool error is recognized
    result = "TOOL_ERROR: Career lookup failed: service unavailable"

    assert is_tool_failure(result) is True


def test_tool_failure_guardrail_prevents_fallback_answer():
    # Simulate a failed lookup result
    result = "TOOL_ERROR: Career lookup failed: service unavailable"

    # The guardrail should provide a safe response
    response = tool_failure_guardrail(result)

    assert response is not None
    assert "cannot provide" in response.lower()




# Test that analysis fails safely when the lookup API finds no career
def test_missing_career_during_analysis_fails_safely():
    agent = CareerPilotAgent()

    # Mock a successful HTTP response with no matching career
    lookup_http_response = Mock()
    lookup_http_response.raise_for_status.return_value = None
    lookup_http_response.json.return_value = {
        "found": False,
        "career": None,
    }

    # Simulate the lookup API returning no matching career
    with patch(
        "careerpilot.agent.httpx.post",
        return_value=lookup_http_response,
    ):
        result = agent.run(
            "How suitable am I for AI Engineer?",
            profile=candidate,
        )

    # Verify that lookup was attempted but prediction was not
    tools_used = [step.tool for step in result.tool_steps]
    assert tools_used == ["lookup"]

    # Verify that the agent returned a safe failure response
    assert "career path not found" in result.final_response.lower()


# Test that the agent stops before making an API request when its limit is reached
def test_agent_stops_when_tool_call_limit_is_reached():
    agent = CareerPilotAgent()

    # Simulate the guardrail reaching the tool-call limit
    with patch(
        "careerpilot.agent.tool_call_guardrail",
        return_value="Tool-call limit reached. The agent must stop.",
    ), patch(
        "careerpilot.agent.httpx.post",
    ) as mock_post:
        result = agent.run("How suitable am I for AI Engineer?",profile=candidate)

    # Verify that the agent stopped without calling the API
    mock_post.assert_not_called()

    # Verify that the guardrail message was returned
    assert "tool-call limit reached" in result.final_response.lower()



# Test that a misleading keyword does not bypass the scope guardrail
def test_unrelated_request_with_career_keyword_is_refused():
    agent = CareerPilotAgent()

    # Include a career-related word in an unrelated request
    result = agent.run("Explain the history of career criminals.")

    # Verify that the agent refuses the unrelated request
    assert result.tool_call_count == 0
    assert "career and technology navigation agent" in result.final_response.lower()


# Test that lookup failures are handled without exposing internal details
def test_lookup_exception_during_analysis_fails_safely():
    agent = CareerPilotAgent()

    # Simulate the lookup API request failing unexpectedly
    with patch(
        "careerpilot.agent.httpx.post",
        side_effect=RuntimeError("Database connection details"),
    ):
        result = agent.run(
            "How suitable am I for AI Engineer?",
            profile=candidate,
        )

    # Verify that lookup failed before any tool step was recorded
    assert result.tool_steps == []

    # Verify that internal error details are not exposed
    assert "database connection details" not in result.final_response.lower()


    
# Test that an unspecified target career is selected from the profile goal
def test_career_fit_uses_profile_career_goal():
    agent = CareerPilotAgent()

    # Ask for a recommendation without naming a specific target role
    result = agent.run(
        "Which tech career is best for me?",
        profile=candidate,
    )

    # Verify that the stated AI Engineering goal determines the target role
    import json
    analysis = json.loads(result.final_response)

    assert analysis["target_role"] == "AI Engineer"



# Test that an explicitly requested role overrides the profile goal
def test_explicit_career_request_overrides_profile_goal():
    agent = CareerPilotAgent()

    # Explicitly request Data Engineer despite the AI Engineering goal
    result = agent.run(
        "How suitable am I for Data Engineer?",
        profile=candidate,
    )

    # Parse the structured career analysis
    import json
    analysis = json.loads(result.final_response)

    # Verify that the explicitly requested role takes priority
    assert analysis["target_role"] == "Data Engineer"