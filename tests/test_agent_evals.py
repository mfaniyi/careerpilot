
import json
from unittest.mock import Mock, patch  # Mock HTTP responses and patch API requests

from careerpilot.agent import CareerPilotAgent
from careerpilot.models import CareerProfile


# Reuse a representative candidate profile for career-fit evaluations
candidate = CareerProfile(
    name="Evaluation Candidate",
    current_role="Data Analyst",
    years_of_experience=5,
    education=["MSc GIS"],
    skills=["Excel", "SQL", "Power BI", "Python"],
    certifications=["PL-300"],
    industry="Technology",
    career_goal="Transition into AI Engineering",
)


# Eval 1: A general career question should select the ask tool
def test_eval_general_question_selects_ask():
    agent = CareerPilotAgent()

    result = agent.run("What does an AI Engineer do?")

    assert [step.tool for step in result.tool_steps] == ["ask"]
    assert result.tool_call_count == 1


# Eval 2: An explicit career lookup should select the lookup tool
def test_eval_lookup_request_selects_lookup():
    agent = CareerPilotAgent()

    result = agent.run("Look up AI Engineer")

    assert [step.tool for step in result.tool_steps] == ["lookup"]
    assert "AI Engineer" in result.final_response


# Eval 3: Career-fit analysis should call tools in the expected order
def test_eval_career_fit_selects_tools_in_order():
    agent = CareerPilotAgent()

    result = agent.run(
        "How suitable am I for AI Engineer?",
        profile=candidate,
    )

    assert [step.tool for step in result.tool_steps] == [
        "lookup",
        "predict",
        "ask",
    ]
    assert result.tool_call_count == 3


# Eval 4: The final career analysis should contain structured JSON fields
def test_eval_career_fit_returns_structured_json():
    agent = CareerPilotAgent()

    result = agent.run(
        "How suitable am I for AI Engineer?",
        profile=candidate,
    )

    # Parse the final response to verify it is valid JSON
    analysis = json.loads(result.final_response)

    # Check the required career-analysis fields
    assert analysis["target_role"] == "AI Engineer"
    assert "career_information" in analysis
    assert "fit_score" in analysis
    assert "matched_skills" in analysis
    assert "missing_skills" in analysis
    assert "recommendation" in analysis


# Eval 5: An out-of-scope request should be refused without tool calls
def test_eval_out_of_scope_request_is_refused():
    agent = CareerPilotAgent()

    result = agent.run("Tell me today's weather.")

    assert result.tool_call_count == 0
    assert "career" in result.final_response.lower()


# Eval 6: A missing career should not trigger invented information
def test_eval_unknown_career_fails_safely():
    agent = CareerPilotAgent()

    result = agent.run(
        "Please lookup Quantum Career Specialist"
    )

    assert [step.tool for step in result.tool_steps] == ["lookup"]
    assert "couldn't find verified information" in result.final_response.lower()



# Eval 7 Test that prediction API failures stop the workflow safely
def test_eval_prediction_failure_stops_workflow():
    agent = CareerPilotAgent()

    # Return a valid career record from the lookup API
    lookup_http_response = Mock()
    lookup_http_response.raise_for_status.return_value = None
    lookup_http_response.json.return_value = {
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

    # Simulate the prediction API failing
    prediction_http_response = Mock()
    prediction_http_response.raise_for_status.side_effect = RuntimeError(
        "Sensitive internal error"
    )

    # Mock lookup success followed by prediction failure
    with patch(
        "careerpilot.agent.httpx.post",
        side_effect=[lookup_http_response, prediction_http_response],
    ):
        result = agent.run(
            "How suitable am I for AI Engineer?",
            profile=candidate,
        )

    # Confirm the workflow stopped safely without exposing internal details
    assert "prediction service is currently unavailable" in result.final_response.lower()
    assert "sensitive internal error" not in result.final_response.lower()


# Eval 8: Test that lookup API failures stop the workflow without exposing internal details
def test_eval_lookup_failure_stops_workflow_safely():
    agent = CareerPilotAgent()

    # Simulate the lookup API request failing unexpectedly
    with patch(
        "careerpilot.agent.httpx.post",
        side_effect=RuntimeError("Sensitive database details"),
    ):
        result = agent.run(
            "How suitable am I for AI Engineer?",
            profile=candidate,
        )

    # Verify that the workflow stopped before recording any tool steps
    assert result.tool_steps == []

    # Verify that internal error details are not exposed
    assert "sensitive database details" not in result.final_response.lower()