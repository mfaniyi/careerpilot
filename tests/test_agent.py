from careerpilot.agent import CareerPilotAgent
from unittest.mock import patch
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


# Test that CareerPilot fails safely when the prediction tool fails
def test_prediction_failure_is_handled_safely():
    agent = CareerPilotAgent()

    # Simulate a failure from the prediction tool
    with patch(
        "careerpilot.agent.predict_career_fit",
        side_effect=RuntimeError("Prediction service unavailable"),
    ):
        result = agent.run(
            "How suitable am I for AI Engineer?",
            profile=candidate,
        )

    # Verify that the prediction tool was reached
    tools_used = [step.tool for step in result.tool_steps]

    assert tools_used == ["lookup", "predict"]

    # Verify that the agent returned a safe failure response
    assert "prediction service is currently unavailable" in result.final_response.lower()

    # Verify that the agent did not invent a prediction
    assert result.final_response is not None


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



# Test that career analysis handles a missing career without crashing
def test_missing_career_during_analysis_fails_safely():
    agent = CareerPilotAgent()

    # Simulate the lookup tool returning no matching career
    with patch(
        "careerpilot.agent.lookup",
        return_value=None,
    ):
        result = agent.run(
            "How suitable am I for AI Engineer?",
            profile=candidate,
        )

    # Verify that the lookup step was recorded
    assert [step.tool for step in result.tool_steps] == ["lookup"]

    # Verify that the agent returned a controlled failure response
    assert result.final_response is not None
    assert "career path not found" in result.final_response.lower()



# Test that the agent stops before calling a tool when the limit is reached
def test_agent_stops_when_tool_call_limit_is_reached():
    agent = CareerPilotAgent()

    # Simulate the guardrail reporting that the call limit has been reached
    with patch(
        "careerpilot.agent.tool_call_guardrail",
        return_value="Tool-call limit reached. The agent must stop.",
    ), patch(
        "careerpilot.agent.ask",
    ) as mock_ask:
        result = agent.run("What does an AI Engineer do?")

    # Verify the agent returned the guardrail response
    assert "tool-call limit reached" in result.final_response.lower()

    # Verify no tool call was made
    mock_ask.assert_not_called()
    assert result.tool_call_count == 0



# Test that a misleading keyword does not bypass the scope guardrail
def test_unrelated_request_with_career_keyword_is_refused():
    agent = CareerPilotAgent()

    # Include a career-related word in an unrelated request
    result = agent.run("Explain the history of career criminals.")

    # Verify that the agent refuses the unrelated request
    assert result.tool_call_count == 0
    assert "career and technology navigation agent" in result.final_response.lower()



# Test that an unexpected lookup failure during analysis is handled safely
def test_lookup_exception_during_analysis_fails_safely():
    agent = CareerPilotAgent()

    # Simulate the lookup service raising an unexpected exception
    with patch(
        "careerpilot.agent.lookup",
        side_effect=RuntimeError("Database connection details"),
    ):
        result = agent.run(
            "How suitable am I for AI Engineer?",
            profile=candidate,
        )

    # Verify that the agent returns a response instead of crashing
    assert result.final_response is not None

    # Verify that internal exception details are not exposed
    assert "database connection details" not in result.final_response.lower()

    # Verify that no prediction or knowledge tool ran after lookup failed
    assert [step.tool for step in result.tool_steps] == []