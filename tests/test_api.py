from fastapi.testclient import TestClient
from unittest.mock import patch
from careerpilot.api import app


# Create a test client for the CareerPilot API
client = TestClient(app)


# Test that the API health-check endpoint works
def test_root_endpoint() -> None:
    """Verify that the API reports a healthy status."""

    # Send a GET request to the root endpoint
    response = client.get("/")

    # Verify the HTTP response
    assert response.status_code == 200

    # Verify the response content
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
    assert result["score"] == 33.33
    assert result["matched_skills"] == ["Python"]
    assert result["missing_skills"] == ["APIs", "Machine Learning"]


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

    # Verify the returned career
    assert result["career"]["role"] == "AI Engineer"
    assert result["career"]["required_skills"] == [
        "Python",
        "Machine Learning",
        "APIs",
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

    # Verify the tool execution order
    assert [step["tool"] for step in result["tool_steps"]] == [
        "lookup",
        "predict",
        "ask",
    ]

    # Verify the prediction was included in the final response
    assert "33.33" in result["response"]


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


# Test that the /agent endpoint handles prediction failures safely
def test_agent_endpoint_prediction_failure() -> None:
    """Verify that the API returns a safe response when prediction fails."""

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

    # Simulate a prediction service failure
    with patch(
        "careerpilot.agent.predict_career_fit",
        side_effect=Exception("Prediction service unavailable"),
    ):
        # Send the request through the agent API
        response = client.post(
            "/agent",
            json={
                "request": "How suitable am I for AI Engineer?",
                "profile": profile,
            },
        )

    # Verify the API still returns a successful HTTP response
    assert response.status_code == 200

    # Read the agent result
    result = response.json()

    # Verify that the agent returned a safe failure message
    assert "prediction service is currently unavailable" in result["response"]

    # Verify that human review was not incorrectly triggered
    assert result["human_review_required"] is False