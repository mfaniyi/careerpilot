from pydantic import BaseModel, Field

from careerpilot.models import CareerProfile
from careerpilot.tools import CAREER_PATHS


# Defines the input required by the prediction tool
class CareerFitRequest(BaseModel):
    profile: CareerProfile
    target_role: str


# Defines the prediction returned by the tool
class CareerFitPrediction(BaseModel):
    target_role: str
    score: float = Field(ge=0, le=100)
    matched_skills: list[str]
    missing_skills: list[str]
    recommendation: str


# Calculate how well a career profile matches a target career
def predict_career_fit(request: CareerFitRequest) -> CareerFitPrediction:
    """Calculate a simple career-fit score based on required skills."""

    # Find the requested career
    career = next(
        (
            item
            for item in CAREER_PATHS
            if item.role.lower() == request.target_role.lower()
        ),
        None,
    )

    # Stop if the requested career does not exist
    if career is None:
        raise ValueError(f"Career path not found: {request.target_role}")

    # Normalize skills for case-insensitive comparison
    user_skills = {skill.lower() for skill in request.profile.skills}
    required_skills = {skill.lower() for skill in career.required_skills}

    # Identify skills the candidate already has
    matched_skills = sorted(
        skill for skill in career.required_skills if skill.lower() in user_skills
    )

    # Identify required skills the candidate is missing
    missing_skills = sorted(
        skill for skill in career.required_skills if skill.lower() not in user_skills
    )

    # Calculate the percentage of required skills already matched
    score = round((len(matched_skills) / len(required_skills)) * 100, 2)

    # Generate a simple recommendation based on the score
    if score >= 80:
        recommendation = "Strong fit"
    elif score >= 50:
        recommendation = "Promising fit with some skill gaps"
    else:
        recommendation = "Significant upskilling recommended"

    # Return the structured prediction
    return CareerFitPrediction(
        target_role=career.role,
        score=score,
        matched_skills=matched_skills,
        missing_skills=missing_skills,
        recommendation=recommendation,
    )