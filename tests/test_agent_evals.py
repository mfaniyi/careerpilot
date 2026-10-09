
import json
from unittest.mock import patch

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


# Eval 7: A prediction failure should stop the workflow safely
def test_eval_prediction_failure_stops_workflow():
    agent = CareerPilotAgent()

    # Simulate a prediction service failure without exposing internal details
    with patch(
        "careerpilot.agent.predict_career_fit",
        side_effect=RuntimeError("Sensitive internal error"),
    ):
        result = agent.run(
            "How suitable am I for AI Engineer?",
            profile=candidate,
        )

    assert [step.tool for step in result.tool_steps] == [
        "lookup",
        "predict",
    ]
    assert "prediction service is currently unavailable" in result.final_response.lower()
    assert "sensitive internal error" not in result.final_response.lower()


# Eval 8: An unexpected lookup failure should stop subsequent tool calls
def test_eval_lookup_failure_stops_workflow_safely():
    agent = CareerPilotAgent()

    # Simulate a lookup failure before prediction or explanation can run
    with patch(
        "careerpilot.agent.lookup",
        side_effect=RuntimeError("Sensitive database details"),
    ):
        result = agent.run(
            "How suitable am I for AI Engineer?",
            profile=candidate,
        )

    assert result.tool_steps == []
    assert result.final_response is not None
    assert "sensitive database details" not in result.final_response.lower()