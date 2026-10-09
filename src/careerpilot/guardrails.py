
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
    """Return True when the request relates to career or technology guidance."""

    # Normalize the request for case-insensitive matching
    request = request.lower()

    # Reject known unrelated contexts before checking broad keywords
    unrelated_contexts = [
        "career criminal",
        "career criminals",
        "history of",
    ]

    if any(context in request for context in unrelated_contexts):
        return False

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


# Identify explicit tool failures returned by CareerPilot tools
def is_tool_failure(result: str) -> bool:
    """Return True when a tool result represents a failed operation."""

    # Tool failures use a predictable prefix
    return result.startswith("TOOL_ERROR:")


# Generate a safe response when a tool fails
def tool_failure_guardrail(result: str) -> str | None:
    """Return a safe response instead of allowing the agent to guess."""

    # Allow normal tool results to continue
    if not is_tool_failure(result):
        return None

    # Stop the agent from replacing failed data with model-generated information
    return (
        "The CareerPilot lookup service is currently unavailable. "
        "I cannot provide the requested career information without "
        "a successful lookup."
    )


# Identify when a lookup completed without finding the requested career
def is_tool_not_found(result: str) -> bool:
    """Return True when a lookup completed but found no matching record."""

    # Tool not-found results use a predictable prefix
    return result.startswith("LOOKUP_NOT_FOUND:")


# Generate a safe response when structured lookup data is unavailable
def tool_not_found_guardrail(result: str) -> str | None:
    """Return a safe response when the requested career does not exist."""

    # Allow normal tool results to continue
    if not is_tool_not_found(result):
        return None

    # Prevent the model from inventing missing structured information
    return (
        "I couldn't find that career in the CareerPilot knowledge base. "
        "I won't provide unverified career information as if it came "
        "from the CareerPilot database."
    )