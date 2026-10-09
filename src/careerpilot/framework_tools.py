from agent_framework import tool

from careerpilot.models import LookupRequest
from careerpilot.tools import lookup


# Expose the CareerPilot lookup function as an Agent Framework tool
@tool(
    name="career_lookup",
    description="Look up structured information about a technology career.",
    approval_mode="always_require",
    max_invocations=5,
)
def framework_career_lookup(request: LookupRequest) -> str:
    """Return career information through the existing CareerPilot lookup."""

    # Convert the framework input into the CareerPilot Pydantic model
    lookup_request = LookupRequest(**request)

    # Call the existing CareerPilot lookup logic
    try:
        result = lookup(lookup_request)

    except Exception as exc:
        # Return an explicit failure instead of allowing the agent to guess
        return f"TOOL_ERROR: Career lookup failed: {exc}"

    # Safely handle careers that are not found
    if result is None:
        return "LOOKUP_NOT_FOUND: Career not found."

    # Return the structured career information
    return result.model_dump_json()