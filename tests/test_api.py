from fastapi.testclient import TestClient
from unittest.mock import Mock, patch  # Mock HTTP responses and patch requests
from careerpilot.api import app


# Create a test client for the CareerPilot API
client = TestClient(app)



# Test that the root endpoint serves the CareerPilot frontend
def test_root_endpoint() -> None:
    """Verify that the frontend is served from the root URL."""

    # Request the root URL
    response = client.get("/")

    # Verify the frontend response
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "<html" in response.text.lower()


# Test that the API health-check endpoint works
def test_health_endpoint() -> None:
    """Verify that the API reports a healthy status."""

    # Request the dedicated health-check endpoint
    response = client.get("/health")

    # Verify the HTTP response
    assert response.status_code == 200

    # Verify the health-check response content
    assert response.json() == {
        "message": "CareerPilot API is running",
        "status": "healthy",
    }


# Test that the /ask endpoint returns a grounded career answer
def test_ask_endpoint() -> None:
    """Verify that the ask endpoint returns a valid answer."""

    # Send a career question to the API
    response = client.post(
        "/ask",
        json={"question": "What does an AI Engineer do?"},
    )

    # Verify the HTTP response
    assert response.status_code == 200

    # Verify that the response contains the expected answer
    assert (
        response.json()["answer"]
        == "An AI Engineer builds and deploys applications that use "
        "artificial intelligence and machine learning."
    )


# Test that the /predict endpoint returns a career-fit prediction
def test_predict_endpoint() -> None:
    """Verify that the predict endpoint returns a valid prediction."""

    # Create a sample career profile for the prediction request
    profile = {
        "name": "Michael",
        "current_role": "Data Analyst",
        "years_of_experience": 5,
        "education": ["MSc GIS"],
        "skills": ["Python"],
        "certifications": ["PL-300"],
        "industry": "Technology",
        "career_goal": "Become an AI Engineer",
    }

    # Send the prediction request to the API
    response = client.post(
        "/predict",
        json={
            "profile": profile,
            "target_role": "AI Engineer",
        },
    )

    # Verify the HTTP response
    assert response.status_code == 200

    # Read the prediction from the response
    result = response.json()

    # Verify the prediction values
    assert result["target_role"] == "AI Engineer"
    assert result["score"] == 20.0
    assert result["matched_preferred_skills"] == []
    assert result["missing_skills"] == [
        "APIs",
        "Data Modelling",
        "Machine Learning",
        "SQL",
    ]


# Test that the /lookup endpoint returns structured career information
def test_lookup_endpoint() -> None:
    """Verify that the lookup endpoint returns a known career."""

    # Send a career lookup request to the API
    response = client.post(
        "/lookup",
        json={
            "lookup_type": "career",
            "query": "AI Engineer",
        },
    )

    # Verify the HTTP response
    assert response.status_code == 200

    # Read the lookup result
    result = response.json()

    # Verify that the career was found
    assert result["found"] is True

    # Verify the updated AI Engineer required skills.
    assert result["career"]["required_skills"] == [
        "Python",
        "Machine Learning",
        "APIs",
        "Data Modelling",
        "SQL",
    ]


# Test that /lookup safely handles an unknown career
def test_lookup_unknown_career() -> None:
    """Verify that lookup does not invent information for an unknown career."""

    # Request a career that is not in the knowledge base
    response = client.post(
        "/lookup",
        json={
            "lookup_type": "career",
            "query": "Quantum Career Specialist",
        },
    )

    # Verify the HTTP response
    assert response.status_code == 200

    # Read the lookup result
    result = response.json()

    # Verify that no career information was invented
    assert result["found"] is False
    assert result["career"] is None


# Test that the /agent endpoint can orchestrate multiple tools
def test_agent_endpoint() -> None:
    """Verify that the agent performs a multi-tool career analysis."""

    # Create a sample career profile
    profile = {
        "name": "Michael",
        "current_role": "Data Analyst",
        "years_of_experience": 5,
        "education": ["MSc GIS"],
        "skills": ["Python"],
        "certifications": ["PL-300"],
        "industry": "Technology",
        "career_goal": "Become an AI Engineer",
    }

    # Send a career-fit request through the agent
    response = client.post(
        "/agent",
        json={
            "request": "How suitable am I for AI Engineer?",
            "profile": profile,
        },
    )

    # Verify the HTTP response
    assert response.status_code == 200

    # Read the agent result
    result = response.json()

    # Verify the agent used the expected number of tools
    assert result["tool_call_count"] == 3

    # Verify that human review was triggered
    assert result["human_review_required"] is True

    
    # Verify that the assessment is awaiting human review.
    assert result["review_status"] == "pending"

    # Verify that the assessment has a unique review identifier.
    assert result["review_id"] is not None

    # Verify that the assessment is preserved for review.
    assert result["pending_review"] is not None

    # Verify the tool execution order
    assert [step["tool"] for step in result["tool_steps"]] == [
        "lookup",
        "predict",
        "ask",
    ]

    # Verify the prediction was included in the final response
    assert "20.0" in result["response"]


# Test that the /agent endpoint safely refuses out-of-scope requests
def test_agent_endpoint_out_of_scope() -> None:
    """Verify that unrelated requests are refused safely."""

    # Send an unrelated request through the agent
    response = client.post(
        "/agent",
        json={
            "request": "What is the weather today?",
        },
    )

    # Verify the HTTP response
    assert response.status_code == 200

    # Read the agent result
    result = response.json()

    # Verify that no tools were called
    assert result["tool_call_count"] == 0

    # Verify that the agent returned its scope message
    assert "career and technology navigation agent" in result["response"]

    # Verify that no human review is required
    assert result["human_review_required"] is False

    # Verify that no tool steps were recorded
    assert result["tool_steps"] == []



# Test that the agent API returns a safe response when prediction fails
def test_agent_endpoint_prediction_failure() -> None:
    # Create a sample career profile
    profile = {
        "name": "Michael",
        "current_role": "Data Analyst",
        "years_of_experience": 5,
        "education": ["MSc GIS"],
        "skills": ["Python"],
        "certifications": ["PL-300"],
        "industry": "Technology",
        "career_goal": "Become an AI Engineer",
    }

    # Mock a successful career lookup response
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

    # Mock a prediction response that raises an error
    prediction_http_response = Mock()
    prediction_http_response.raise_for_status.side_effect = RuntimeError(
        "Prediction service unavailable"
    )

    # Simulate lookup success followed by prediction failure
    with patch(
        "careerpilot.agent.httpx.post",
        side_effect=[lookup_http_response, prediction_http_response],
    ):
        response = client.post(
            "/agent",
            json={
                "request": "How suitable am I for AI Engineer?",
                "profile": profile,
            },
        )

    # Verify the endpoint returns a safe failure message
    assert "prediction service is currently unavailable" in (
        response.json()["response"].lower()
    )

    # Verify internal error details are not exposed
    assert "prediction service unavailable" not in (
        response.json()["response"].lower()
    )


def test_agent_endpoint_rejects_missing_request():
    # Send an invalid request without the required "request" field
    response = client.post(
        "/agent",
        json={
            "profile": {
                "name": "Michael",
                "current_role": "Data Analyst",
                "years_of_experience": 5,
                "education": ["MSc GIS"],
                "skills": ["Python"],
                "certifications": ["PL-300"],
                "industry": "Technology",
                "career_goal": "Become an AI Engineer",
            }
        },
    )

    # FastAPI should reject the request because "request" is required
    assert response.status_code == 422


def test_agent_endpoint_rejects_invalid_profile():
    # Send an invalid profile with a negative experience value
    response = client.post(
        "/agent",
        json={
            "request": "How suitable am I for AI Engineer?",
            "profile": {
                "name": "Michael",
                "current_role": "Data Analyst",
                "years_of_experience": -5,
                "education": ["MSc GIS"],
                "skills": ["Python"],
                "certifications": ["PL-300"],
                "industry": "Technology",
                "career_goal": "Become an AI Engineer",
            },
        },
    )

    # Pydantic should reject the invalid years_of_experience value
    assert response.status_code == 422



# Test that a reviewer can approve an assessment awaiting review.
def test_agent_review_approval() -> None:
    # Create a career-fit assessment that requires human review.
    agent_response = client.post(
        "/agent",
        json={
            "request": "How suitable am I for AI Engineer?",
            "profile": {
                "name": "Michael",
                "current_role": "Data Analyst",
                "years_of_experience": 5,
                "education": ["MSc GIS"],
                "skills": ["Python"],
                "certifications": ["PL-300"],
                "industry": "Technology",
                "career_goal": "Become an AI Engineer",
            },
        },
    )

    # Verify that the assessment was created successfully.
    assert agent_response.status_code == 200

    # Retrieve the unique ID assigned to the pending assessment.
    agent_result = agent_response.json()
    review_id = agent_result["review_id"]

    # Verify that the assessment is waiting for review.
    assert agent_result["review_status"] == "pending"
    assert review_id is not None

    # Submit the reviewer's approval decision.
    review_response = client.post(
        "/agent/review",
        json={
            "review_id": review_id,
            "decision": "approve",
        },
    )

    # Verify that the approval was accepted.
    assert review_response.status_code == 200

    # Verify that the review status and assessment are returned.
    review_result = review_response.json()
    assert review_result["review_status"] == "approved"
    assert review_result["assessment"] is not None
    assert review_result["message"] == "Assessment approved."



# Test that a reviewer can reject an assessment awaiting review.
def test_agent_review_rejection() -> None:
    # Create a career-fit assessment that requires human review.
    agent_response = client.post(
        "/agent",
        json={
            "request": "How suitable am I for AI Engineer?",
            "profile": {
                "name": "Michael",
                "current_role": "Data Analyst",
                "years_of_experience": 5,
                "education": ["MSc GIS"],
                "skills": ["Python"],
                "certifications": ["PL-300"],
                "industry": "Technology",
                "career_goal": "Become an AI Engineer",
            },
        },
    )

    # Verify that the assessment was created successfully.
    assert agent_response.status_code == 200

    # Retrieve the unique ID assigned to the pending assessment.
    agent_result = agent_response.json()
    review_id = agent_result["review_id"]

    # Submit the reviewer's rejection decision.
    review_response = client.post(
        "/agent/review",
        json={
            "review_id": review_id,
            "decision": "reject",
        },
    )

    # Verify that the rejection was accepted.
    assert review_response.status_code == 200

    # Verify that the assessment was rejected and withheld.
    review_result = review_response.json()
    assert review_result["review_status"] == "rejected"
    assert review_result["assessment"] is None
    assert review_result["message"] == "Assessment rejected."



# Test that an unknown review ID returns a 404 response.
def test_agent_review_unknown_id() -> None:
    # Submit a decision for a review ID that does not exist.
    response = client.post(
        "/agent/review",
        json={
            "review_id": "unknown-review-id",
            "decision": "approve",
        },
    )

    # Verify that the API reports the missing review.
    assert response.status_code == 404
    assert response.json()["detail"] == "Review ID not found."



# Test that a reviewed assessment cannot receive a second decision.
def test_agent_review_duplicate_decision() -> None:
    # Create an assessment that requires human review.
    agent_response = client.post(
        "/agent",
        json={
            "request": "How suitable am I for AI Engineer?",
            "profile": {
                "name": "Michael",
                "current_role": "Data Analyst",
                "years_of_experience": 5,
                "education": ["MSc GIS"],
                "skills": ["Python"],
                "certifications": ["PL-300"],
                "industry": "Technology",
                "career_goal": "Become an AI Engineer",
            },
        },
    )

    # Retrieve the ID of the pending assessment.
    assert agent_response.status_code == 200
    review_id = agent_response.json()["review_id"]

    # Submit the first decision and verify it succeeds.
    first_response = client.post(
        "/agent/review",
        json={"review_id": review_id, "decision": "approve"},
    )
    assert first_response.status_code == 200

    # Attempt to reject the same assessment a second time.
    second_response = client.post(
        "/agent/review",
        json={"review_id": review_id, "decision": "reject"},
    )

    # Verify that the API prevents the second decision.
    assert second_response.status_code == 409
    assert (
        second_response.json()["detail"]
        == "This assessment has already been reviewed."
    )



# Verify preferred skills are matched without changing the required-skill score.
def test_predict_endpoint_matches_preferred_skills() -> None:
    # Create a profile with required and preferred AI Engineer skills.
    profile = {
        "name": "Michael",
        "current_role": "Data Analyst",
        "years_of_experience": 5,
        "education": ["MSc GIS"],
        "skills": ["Python", "ETL", "Cloud", "Docker"],
        "certifications": ["PL-300"],
        "industry": "Technology",
        "career_goal": "Become an AI Engineer",
    }

    # Request a career-fit prediction from the API.
    response = client.post(
        "/predict",
        json={
            "profile": profile,
            "target_role": "AI Engineer",
        },
    )

    # Confirm the prediction succeeded.
    assert response.status_code == 200

    # Read the prediction returned by the API.
    result = response.json()

    # Only Python matches the five required skills.
    assert result["score"] == 20.0
    assert result["matched_skills"] == ["Python"]

    # Preferred skills are reported separately in alphabetical order.
    assert result["matched_preferred_skills"] == [
        "Cloud",
        "Docker",
        "ETL",
    ]