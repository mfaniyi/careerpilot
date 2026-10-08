from fastapi import FastAPI
from careerpilot.ask import AskRequest, ask
from careerpilot.predict import CareerFitRequest, predict_career_fit
from careerpilot.models import LookupRequest, CareerProfile
from careerpilot.tools import lookup
from careerpilot.agent import CareerPilotAgent


# Create the CareerPilot API application
app = FastAPI(
    title="CareerPilot API",
    description="API for the CareerPilot career navigation agent.",
    version="0.1.0",
)


# Basic health-check endpoint
@app.get("/")
def root() -> dict:
    """Return a simple API status message."""

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
def agent_endpoint(request: dict) -> dict:
    """Process a career request through the CareerPilot agent."""

    # Extract the user's request from the JSON body
    user_request = request.get("request", "")

    # Extract the optional career profile
    profile_data = request.get("profile")

    # Convert the profile dictionary into the validated Pydantic model
    profile = (
        CareerProfile(**profile_data)
        if profile_data is not None
        else None
    )

    # Create the CareerPilot agent
    agent = CareerPilotAgent()

    # Run the agent workflow with the user's profile
    result = agent.run(
        request=user_request,
        profile=profile,
    )

    # Return the agent result and execution state
    return {
        "response": result.final_response,
        "tool_call_count": result.tool_call_count,
        "human_review_required": result.human_review_required,
        "tool_steps": [
            step.model_dump()
            for step in result.tool_steps
        ],
    }