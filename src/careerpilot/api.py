from pathlib import Path  # Build a reliable path to the frontend HTML file
from fastapi import FastAPI
from fastapi.responses import FileResponse  # Serve HTML files as HTTP responses
from careerpilot.ask import AskRequest, ask
from careerpilot.predict import CareerFitRequest, predict_career_fit
from careerpilot.models import (
    LookupRequest,
    CareerProfile,
    AgentRequest,
    ReviewDecisionRequest,
)
from careerpilot.tools import lookup
from careerpilot.agent import CareerPilotAgent


# Create the CareerPilot API application
app = FastAPI(
    title="CareerPilot API",
    description="API for the CareerPilot career navigation agent.",
    version="0.1.0",
)


# Store assessments awaiting review, keyed by their unique review ID.
PENDING_REVIEWS: dict[str, dict] = {}

# Locate the frontend HTML file in the project root
FRONTEND_FILE = Path(__file__).resolve().parents[2] / "index.html"


# Serve the CareerPilot frontend at the root URL
@app.get("/", include_in_schema=False)
def serve_frontend() -> FileResponse:
    """Return the CareerPilot web interface."""

    # Serve the HTML file if it exists
    return FileResponse(FRONTEND_FILE, media_type="text/html")


# Keep the API health check available at a separate endpoint
@app.get("/health")
def health_check() -> dict:
    """Return the API health status."""

    # Confirm that the API is running
    return {
        "message": "CareerPilot API is running",
        "status": "healthy",
    }


# Answer a career knowledge question through the API
@app.post("/ask")
def ask_endpoint(request: AskRequest) -> dict:
    """Return a grounded answer from the CareerPilot knowledge base."""

    # Call the existing ask tool
    result = ask(request)

    # Return a structured JSON response
    return {
        "answer": result,
    }


# Calculate career fit through the API
@app.post("/predict")
def predict_endpoint(request: CareerFitRequest) -> dict:
    """Return a structured career-fit prediction."""

    # Call the existing prediction tool
    result = predict_career_fit(request)

    # Return the prediction as structured JSON
    return result.model_dump()


# Look up structured career information through the API
@app.post("/lookup")
def lookup_endpoint(request: LookupRequest) -> dict:
    """Return structured career information from the CareerPilot knowledge base."""

    # Call the existing lookup tool
    result = lookup(request)

    # Return a safe response when no career is found
    if result is None:
        return {
            "found": False,
            "career": None,
        }

    # Return the verified career information
    return {
        "found": True,
        "career": result.model_dump(),
    }


# Run a user request through the CareerPilot agent
@app.post("/agent")
def agent_endpoint(request: AgentRequest) -> dict:
    # Create the CareerPilot agent
    agent = CareerPilotAgent()

    # Run the agent using the validated request and optional profile
    result = agent.run(
        request=request.request,
        profile=request.profile,
        pending_profile=request.pending_profile,
    )

    # Save assessments that are waiting for a human decision.
    if result.review_status == "pending" and result.review_id:
        PENDING_REVIEWS[result.review_id] = {
            "assessment": result.pending_review,
            "status": "pending",
        }

    # Return the agent result and execution information
    return {
        "response": result.final_response,
        "tool_call_count": result.tool_call_count,
        "human_review_required": result.human_review_required,
        "tool_steps": [step.model_dump() for step in result.tool_steps],
        "pending_profile": (
            result.pending_profile.model_dump()
            if result.pending_profile is not None
            else None
        ),
        
        # Expose the current human-review status to API consumers.
        "review_status": result.review_status,

        # Return the assessment awaiting review, if one exists.
        "pending_review": result.pending_review,
        
        # Identify the assessment awaiting human review.
        "review_id": result.review_id,
    }



# Process a human decision for an assessment awaiting review.
@app.post("/agent/review")
def review_agent_assessment(request: ReviewDecisionRequest) -> dict:
    # Retrieve the assessment using its review identifier.
    review = PENDING_REVIEWS.get(request.review_id)

    # Reject requests for review IDs that do not exist.
    if review is None:
        from fastapi import HTTPException

        raise HTTPException(
            status_code=404,
            detail="Review ID not found.",
        )

    # Prevent a second decision on an already reviewed assessment.
    if review["status"] != "pending":
        from fastapi import HTTPException

        raise HTTPException(
            status_code=409,
            detail="This assessment has already been reviewed.",
        )

    # Record the reviewer's decision.
    review["status"] = (
        "approved" if request.decision == "approve" else "rejected"
    )

    # Return the decision and assessment details for the API caller.
    return {
        "review_id": request.review_id,
        "review_status": review["status"],
        "assessment": (
            review["assessment"]
            if request.decision == "approve"
            else None
        ),
        "message": (
            "Assessment approved."
            if request.decision == "approve"
            else "Assessment rejected."
        ),
    }