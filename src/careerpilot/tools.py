from careerpilot.models import CareerPath, LookupRequest


# Store the initial CareerPilot career knowledge
CAREER_PATHS = [
    CareerPath(
        role="Data Analyst",
        description="Analyze data and create insights that support business decisions.",
        required_skills=["Excel", "SQL", "Data Visualization"],
        preferred_skills=["Power BI", "Python", "Statistics"],
        entry_level="Beginner",
        related_roles=["Business Analyst", "BI Analyst", "Data Scientist"],
    ),
    CareerPath(
        role="Data Engineer",
        description="Build and maintain systems that collect, transform, and store data.",
        required_skills=["SQL", "Python", "ETL", "Databases"],
        preferred_skills=["Azure", "Docker", "Spark"],
        entry_level="Intermediate",
        related_roles=["Analytics Engineer", "Cloud Engineer"],
    ),
    CareerPath(
        role="AI Engineer",
        description="Build and deploy AI-powered applications and machine learning systems.",
        required_skills=["Python", "Machine Learning", "APIs"],
        preferred_skills=["Docker", "Cloud", "LLMs"],
        entry_level="Intermediate",
        related_roles=["ML Engineer", "Data Scientist", "AI Developer"],
    ),
    CareerPath(
        role="Business Analyst",
        description="Analyze business processes and requirements to improve organizational decisions.",
        required_skills=["Requirements Analysis", "Excel", "Communication"],
        preferred_skills=["SQL", "Power BI", "Process Modeling"],
        entry_level="Beginner",
        related_roles=["Product Analyst", "Data Analyst", "Product Manager"],
    ),
]


# Look up career information using a structured request
def lookup(request: LookupRequest) -> CareerPath | None:
    """Return structured career information based on a lookup request."""

    # Reject unsupported lookup types explicitly
    if request.lookup_type.lower() != "career":
        raise ValueError(
            f"Unsupported lookup type: {request.lookup_type}"
        )

    # Search the career knowledge base
    for career in CAREER_PATHS:
        if career.role.lower() == request.query.lower():
            return career

    # Return None when no matching career is found
    return None