from typing import Literal

from pydantic import BaseModel, Field


# Represents the user's career background and current situation
class CareerProfile(BaseModel):
    name: str
    current_role: str
    years_of_experience: int = Field(ge=0)
    education: list[str]
    skills: list[str]
    certifications: list[str]
    industry: str
    career_goal: str


# Represents a technology career that CareerPilot can recommend
class CareerPath(BaseModel):
    role: str
    description: str
    required_skills: list[str]
    preferred_skills: list[str]
    entry_level: str
    related_roles: list[str]


# Defines the supported lookup operation
class LookupRequest(BaseModel):
    lookup_type: Literal["career"]
    query: str


# Represents the combined result of a CareerPilot career analysis
class CareerAnalysis(BaseModel):
    target_role: str
    career_information: CareerPath | None = None
    fit_score: float | None = None
    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    career_explanation: str | None = None
    recommendation: str | None = None


# Defines the input accepted by the CareerPilot agent endpoint
class AgentRequest(BaseModel):
    request: str
    profile: CareerProfile | None = None