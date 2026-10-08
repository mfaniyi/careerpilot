from pydantic import BaseModel


# Defines the input required by the ask tool
class AskRequest(BaseModel):
    question: str


# Small knowledge base for CareerPilot
KNOWLEDGE_BASE = {
    "data analyst": (
        "A Data Analyst collects, cleans, analyzes, and visualizes data "
        "to help organizations make better decisions."
    ),
    "ai engineer": (
        "An AI Engineer builds and deploys applications that use "
        "artificial intelligence and machine learning."
    ),
    "data engineer": (
        "A Data Engineer builds and maintains systems for collecting, "
        "transforming, storing, and processing data."
    ),
    "business analyst": (
        "A Business Analyst studies business needs, processes, and requirements "
        "to help organizations improve decisions and operations."
    ),
}


# Answer a question using the local knowledge base
def ask(request: AskRequest) -> str:
    """Return a grounded answer when the topic exists in the knowledge base."""

    # Normalize the question for simple topic matching
    question = request.question.lower()

    # Search for a known career topic
    for topic, answer in KNOWLEDGE_BASE.items():
        if topic in question:
            return answer

    # Return a safe response when the knowledge base has no answer
    return "I don't have enough information in my knowledge base to answer that question."