# Keywords that indicate CareerPilot-related requests
CAREER_KEYWORDS = [
    "career",
    "job",
    "skill",
    "technology",
    "tech",
    "data",
    "developer",
    "engineer",
    "analyst",
    "learning",
    "course",
    "transition",
]


# Check whether a user request is within CareerPilot's scope
def is_in_scope(request: str) -> bool:
    """Return True when the request is related to careers or technology."""

    # Normalize the request for case-insensitive matching
    request = request.lower()

    # Check whether at least one relevant keyword exists
    return any(keyword in request for keyword in CAREER_KEYWORDS)


# Generate a safe response for out-of-scope requests
def scope_guardrail(request: str) -> str | None:
    """Return a refusal message when the request is outside the agent's scope."""

    # Allow requests that are relevant to CareerPilot
    if is_in_scope(request):
        return None

    # Refuse unrelated requests safely
    return (
        "I'm CareerPilot, a career and technology navigation agent. "
        "I can help with career transitions, technology careers, skills, "
        "job fit, and learning paths."
    )


# Maximum number of tool calls allowed for one agent request
MAX_TOOL_CALLS = 5


# Check whether the agent can make another tool call
def tool_call_guardrail(tool_call_count: int) -> str | None:
    """Return a warning when the agent reaches its tool-call limit."""

    # Allow another tool call while the limit has not been reached
    if tool_call_count < MAX_TOOL_CALLS:
        return None

    # Stop the agent from making additional tool calls
    return (
        "Tool-call limit reached. The agent must stop using tools "
        "and safely complete the request."
    )


# Determine whether a career recommendation should receive human review
def requires_human_review(
    fit_score: float,
    recommendation: str,
) -> bool:
    """Return True when a career recommendation needs human review."""

    # Flag recommendations where the candidate has a significant skill gap
    if fit_score < 50:
        return True

    # Flag recommendations that explicitly indicate substantial upskilling
    if "significant" in recommendation.lower():
        return True

    # Allow stronger recommendations to proceed without review
    return False